"""
InterSR-augmented Qwen2 model for SGLang serving.

Extends sglang's Qwen2ForCausalLM with integrated SentenceSplitMLP and
SignalPredictorMLP, using indicator functions to dynamically switch between
System-1 and System-2 reasoning modes during generation.

Usage:
    1. Copy this file to your sglang models directory or set SGLANG_EXTERNAL_MODEL_PACKAGE
    2. Modify the model's config.json: set "architectures": ["Qwen2InterSRForCausalLM"]
    3. Place MLP weights (sentence_split_mlp.pt, signal_predictor_mlp.pt) in the model directory
    4. Launch sglang normally:
       python -m sglang.launch_server --model-path /path/to/intersr_model --port 30000

    Or create InterSR model directory:
       python serving/qwen2_intersr.py --create-model \\
           --base-model model/Qwen/DeepSeek-R1-Distill-Qwen-7B \\
           --split-mlp intersr/signal_predictor/output/.../sentence_split_best.pt \\
           --signal-predictor-mlp intersr/signal_predictor/output/.../signal_predictor_best.pt \\
           --output-dir model/InterSR-DeepSeek-R1-Distill-Qwen-7B
"""

import os
import sys
import json
import shutil
import logging
import argparse
from typing import Iterable, Optional, Dict

import torch
import torch.nn as nn

# sglang imports
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.server_args import get_global_server_args

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from intersr.signal_predictor.architecture import SentenceSplitMLP, SignalPredictorMLP

logger = logging.getLogger(__name__)

# Token IDs for DeepSeek-R1-Distill-Qwen series
THINK_TOKEN_ID = 151648      # <think>
END_THINK_TOKEN_ID = 151649  # </think>


class Qwen2InterSRForCausalLM(Qwen2ForCausalLM):
    """
    InterSR-augmented Qwen2 model for sglang serving.

    Adds SentenceSplitMLP and SignalPredictorMLP as model submodules.
    During forward pass, modifies next-token logits using indicator functions:
        enter_thinking = I(normal) * I(boundary) * I(uncertain) * I(min_tokens)
        exit_thinking  = I(thinking) * I(boundary) * I(certain) * I(min_tokens)
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)

        # InterSR config (read from model config.json)
        intersr_config = getattr(config, 'intersr_config', None) or getattr(config, 'isr_config', {})
        self.think_threshold = intersr_config.get('think_threshold', 0.4)
        self.min_think_tokens = intersr_config.get('min_think_tokens', 10)
        self.min_normal_tokens = intersr_config.get('min_normal_tokens', 10)
        hidden_size = intersr_config.get('hidden_size', config.hidden_size)

        # InterSR MLP modules (weights loaded via load_weights)
        self.sentence_split_mlp = SentenceSplitMLP(hidden_size=hidden_size)
        self.signal_predictor_mlp = SignalPredictorMLP(hidden_size=hidden_size)

        # Per-request InterSR state: {req_pool_idx: (mode, counter)}
        # mode: 0=normal, 1=thinking
        # counter: tokens since last mode switch
        self._intersr_states: Dict[int, tuple] = {}

        logger.info(
            f"InterSR enabled: threshold={self.think_threshold}, "
            f"min_think={self.min_think_tokens}, min_normal={self.min_normal_tokens}"
        )

    def _get_intersr_state(self, req_idx: int) -> tuple:
        """Get InterSR state for a request, initializing if needed."""
        if req_idx not in self._intersr_states:
            # Default: start in thinking mode (model begins with <think>)
            self._intersr_states[req_idx] = (1, 0)
        return self._intersr_states[req_idx]

    def _set_intersr_state(self, req_idx: int, mode: int, counter: int):
        self._intersr_states[req_idx] = (mode, counter)

    def _cleanup_finished_requests(self, active_indices: set):
        """Remove state for finished requests to prevent memory leak."""
        to_remove = [k for k in self._intersr_states if k not in active_indices]
        for k in to_remove:
            del self._intersr_states[k]

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> torch.Tensor:
        # Run base model to get hidden states
        hidden_states = self.model(
            input_ids, positions, forward_batch,
            input_embeds, pp_proxy_tensors=pp_proxy_tensors,
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if not self.pp_group.is_last_rank:
            return hidden_states

        if get_embedding:
            return self.pooler(hidden_states, forward_batch)

        # --- InterSR Logic: modify logits based on indicator functions ---

        # Get logits from the base model's logits_processor
        logits_output = self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states,
        )

        # Apply InterSR to next_token_logits
        if (logits_output.next_token_logits is not None and
                forward_batch.forward_mode.is_decode()):
            self._apply_intersr(
                input_ids, hidden_states, forward_batch, logits_output,
            )

        return logits_output

    def _apply_intersr(self, input_ids, hidden_states, forward_batch, logits_output):
        """Apply InterSR indicator-gated logit modification."""
        batch_size = forward_batch.batch_size
        seq_lens = forward_batch.seq_lens
        req_pool_indices = forward_batch.req_pool_indices
        next_token_logits = logits_output.next_token_logits  # [batch_size, vocab]

        # In decode mode, hidden_states is [batch_size, hidden_dim]
        # (one token per request, already the last token)
        # But it might be [total_tokens, hidden_dim] in some cases
        # We need the last hidden state per request
        if hidden_states.dim() == 2 and hidden_states.shape[0] == batch_size:
            last_hs = hidden_states  # Already [batch_size, hidden_dim]
        else:
            # Need to extract last token per request
            cumsum = torch.cumsum(seq_lens, dim=0)
            last_indices = cumsum - 1
            last_hs = hidden_states[last_indices]

        # Run InterSR MLPs
        split_scores = self.sentence_split_mlp(last_hs)      # [batch_size]
        signal_scores = self.signal_predictor_mlp(last_hs)    # [batch_size]

        # Cleanup old states
        active = set(req_pool_indices.tolist())
        if len(self._intersr_states) > batch_size * 2:
            self._cleanup_finished_requests(active)

        # Apply indicator functions per request
        for b in range(batch_size):
            req_idx = req_pool_indices[b].item()
            token = input_ids[b].item() if input_ids.dim() == 1 else input_ids[b, -1].item()

            # Update mode from last generated token
            mode, counter = self._get_intersr_state(req_idx)
            if token == THINK_TOKEN_ID:
                mode, counter = 1, 0
            elif token == END_THINK_TOKEN_ID:
                mode, counter = 0, 0
            else:
                counter += 1

            # Indicator functions
            I_boundary = float(split_scores[b] > 0)
            I_uncertain = float(signal_scores[b] > self.think_threshold)
            I_certain = 1.0 - I_uncertain

            if mode == 0:  # Normal mode -> check enter thinking
                gate = I_boundary * I_uncertain * float(counter >= self.min_normal_tokens)
                if gate > 0.5:
                    next_token_logits[b, :].fill_(float('-inf'))
                    next_token_logits[b, THINK_TOKEN_ID] = 0.0
                    mode, counter = 1, 0
            else:  # Thinking mode -> check exit thinking
                gate = I_boundary * I_certain * float(counter >= self.min_think_tokens)
                if gate > 0.5:
                    next_token_logits[b, :].fill_(float('-inf'))
                    next_token_logits[b, END_THINK_TOKEN_ID] = 0.0
                    mode, counter = 0, 0

            self._set_intersr_state(req_idx, mode, counter)

    def load_weights(self, weights: Iterable):
        """Load weights including InterSR MLP parameters."""
        # Separate InterSR weights from base model weights
        intersr_weights = {}
        base_weights = []

        for name, loaded_weight in weights:
            if name.startswith("sentence_split_mlp."):
                intersr_weights[name] = loaded_weight
            elif name.startswith("signal_predictor_mlp."):
                intersr_weights[name] = loaded_weight
            else:
                base_weights.append((name, loaded_weight))

        # Load base model weights
        super().load_weights(iter(base_weights))

        # Determine device from model parameters
        device = next(self.model.parameters()).device

        # Load InterSR MLP weights
        if intersr_weights:
            split_state = {}
            signal_predictor_state = {}
            for name, weight in intersr_weights.items():
                if name.startswith("sentence_split_mlp."):
                    key = name[len("sentence_split_mlp."):]
                    split_state[key] = weight
                elif name.startswith("signal_predictor_mlp."):
                    key = name[len("signal_predictor_mlp."):]
                    signal_predictor_state[key] = weight

            if split_state:
                self.sentence_split_mlp.load_state_dict(split_state)
                logger.info(f"Loaded SentenceSplitMLP weights ({len(split_state)} params)")
            if signal_predictor_state:
                self.signal_predictor_mlp.load_state_dict(signal_predictor_state)
                logger.info(f"Loaded SignalPredictorMLP weights ({len(signal_predictor_state)} params)")
        else:
            # Fallback: load from .pt files in model directory
            logger.warning("No InterSR MLP weights found in safetensors! Loading from .pt files...")
            # Get model path from config
            model_path = None
            # Try to find model path from server args
            try:
                server_args = get_global_server_args()
                if server_args:
                    model_path = server_args.model_path
            except Exception:
                pass

            if model_path:
                split_path = os.path.join(model_path, "sentence_split_mlp.pt")
                signal_predictor_path = os.path.join(model_path, "signal_predictor_mlp.pt")
                if os.path.exists(split_path):
                    self.sentence_split_mlp.load_state_dict(
                        torch.load(split_path, map_location="cpu"))
                    logger.info(f"Loaded SentenceSplitMLP from {split_path}")
                if os.path.exists(signal_predictor_path):
                    self.signal_predictor_mlp.load_state_dict(
                        torch.load(signal_predictor_path, map_location="cpu"))
                    logger.info(f"Loaded SignalPredictorMLP from {signal_predictor_path}")

        # Move MLPs to same device as model
        self.sentence_split_mlp = self.sentence_split_mlp.to(device)
        self.signal_predictor_mlp = self.signal_predictor_mlp.to(device)
        self.sentence_split_mlp.eval()
        self.signal_predictor_mlp.eval()
        logger.info(f"InterSR MLPs moved to {device}")


# Required by sglang model registry
EntryClass = Qwen2InterSRForCausalLM


# ------------------------------------------------------------------
# CLI tool to create InterSR model directory for sglang
# ------------------------------------------------------------------

def create_intersr_model_dir(base_model_path, split_mlp_path, signal_predictor_mlp_path,
                             output_dir, think_threshold=0.4,
                             min_think_tokens=10, min_normal_tokens=10):
    """
    Create a sglang-compatible InterSR model directory by:
    1. Symlinking base model files
    2. Merging MLP weights into model.safetensors
    3. Updating config.json with InterSR settings and architecture
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Symlink base model files (except config.json and model weights)
    for fname in os.listdir(base_model_path):
        src = os.path.join(base_model_path, fname)
        dst = os.path.join(output_dir, fname)
        if fname == 'config.json':
            continue  # Will create modified version
        if not os.path.exists(dst):
            os.symlink(src, dst)
            print(f"  Linked: {fname}")

    # Step 2: Copy MLP weights to model directory
    shutil.copy2(split_mlp_path, os.path.join(output_dir, "sentence_split_mlp.pt"))
    shutil.copy2(signal_predictor_mlp_path, os.path.join(output_dir, "signal_predictor_mlp.pt"))
    print(f"  Copied MLP weights")

    # Step 3: Modify config.json
    with open(os.path.join(base_model_path, "config.json")) as f:
        config = json.load(f)

    config["architectures"] = ["Qwen2InterSRForCausalLM"]
    config["intersr_config"] = {
        "think_threshold": think_threshold,
        "min_think_tokens": min_think_tokens,
        "min_normal_tokens": min_normal_tokens,
        "hidden_size": config["hidden_size"],
    }

    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Updated config.json with InterSR settings")

    print(f"\nInterSR model directory created: {output_dir}")
    print(f"\nTo serve with sglang:")
    print(f"  SGLANG_EXTERNAL_MODEL_PACKAGE=serving.qwen2_intersr \\")
    print(f"    python -m sglang.launch_server --model-path {output_dir} --port 30000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-model", action="store_true",
                        help="Create InterSR model directory for sglang")
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--split-mlp", type=str, required=True)
    parser.add_argument("--signal-predictor-mlp", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.4)
    args = parser.parse_args()

    if args.create_model:
        create_intersr_model_dir(
            args.base_model, args.split_mlp, args.signal_predictor_mlp,
            args.output_dir, think_threshold=args.threshold,
        )
