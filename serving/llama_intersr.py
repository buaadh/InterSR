"""
InterSR-augmented Llama model for SGLang serving.
Adapts the Qwen2InterSR approach for Llama architecture (DeepSeek-R1-Distill-Llama-8B).
"""
import os, sys, json, shutil, logging, argparse
from typing import Iterable, Optional, Dict
import torch
import torch.nn as nn

from sglang.srt.models.llama import LlamaForCausalLM
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.server_args import get_global_server_args

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from intersr.signal_predictor.architecture import SentenceSplitMLP, SignalPredictorMLP

logger = logging.getLogger(__name__)

# Token IDs for DeepSeek-R1-Distill-Llama series
THINK_TOKEN_ID = 128013      # <think>
END_THINK_TOKEN_ID = 128014  # </think>


class LlamaInterSRForCausalLM(LlamaForCausalLM):
    """InterSR-augmented Llama model for sglang serving."""

    def __init__(self, config, quant_config: Optional[QuantizationConfig] = None, prefix: str = ""):
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        intersr_config = getattr(config, 'intersr_config', None) or getattr(config, 'isr_config', {})
        self.think_threshold = intersr_config.get('think_threshold', 0.4)
        self.min_think_tokens = intersr_config.get('min_think_tokens', 10)
        self.min_normal_tokens = intersr_config.get('min_normal_tokens', 10)
        hidden_size = intersr_config.get('hidden_size', config.hidden_size)
        self.sentence_split_mlp = SentenceSplitMLP(hidden_size=hidden_size)
        self.signal_predictor_mlp = SignalPredictorMLP(hidden_size=hidden_size)
        self._intersr_states: Dict[int, tuple] = {}
        logger.info(f"InterSR-Llama enabled: threshold={self.think_threshold}")

    def _get_intersr_state(self, req_idx: int) -> tuple:
        if req_idx not in self._intersr_states:
            self._intersr_states[req_idx] = (1, 0)
        return self._intersr_states[req_idx]

    def _set_intersr_state(self, req_idx: int, mode: int, counter: int):
        self._intersr_states[req_idx] = (mode, counter)

    def _cleanup_finished_requests(self, active_indices: set):
        to_remove = [k for k in self._intersr_states if k not in active_indices]
        for k in to_remove:
            del self._intersr_states[k]

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None,
                get_embedding=False, pp_proxy_tensors=None):
        hidden_states = self.model(input_ids, positions, forward_batch,
                                   input_embeds, pp_proxy_tensors=pp_proxy_tensors)
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        if not self.pp_group.is_last_rank:
            return hidden_states
        if get_embedding:
            return self.pooler(hidden_states, forward_batch)

        logits_output = self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states)

        if (logits_output.next_token_logits is not None and
                forward_batch.forward_mode.is_decode()):
            self._apply_intersr(input_ids, hidden_states, forward_batch, logits_output)
        return logits_output

    def _apply_intersr(self, input_ids, hidden_states, forward_batch, logits_output):
        batch_size = forward_batch.batch_size
        seq_lens = forward_batch.seq_lens
        req_pool_indices = forward_batch.req_pool_indices
        next_token_logits = logits_output.next_token_logits

        if hidden_states.dim() == 2 and hidden_states.shape[0] == batch_size:
            last_hs = hidden_states
        else:
            cumsum = torch.cumsum(seq_lens, dim=0)
            last_indices = cumsum - 1
            last_hs = hidden_states[last_indices]

        split_scores = self.sentence_split_mlp(last_hs)
        signal_scores = self.signal_predictor_mlp(last_hs)

        active = set(req_pool_indices.tolist())
        if len(self._intersr_states) > batch_size * 2:
            self._cleanup_finished_requests(active)

        for b in range(batch_size):
            req_idx = req_pool_indices[b].item()
            token = input_ids[b].item() if input_ids.dim() == 1 else input_ids[b, -1].item()
            mode, counter = self._get_intersr_state(req_idx)
            if token == THINK_TOKEN_ID:
                mode, counter = 1, 0
            elif token == END_THINK_TOKEN_ID:
                mode, counter = 0, 0
            else:
                counter += 1

            I_boundary = float(split_scores[b] > 0)
            I_uncertain = float(signal_scores[b] > self.think_threshold)
            I_certain = 1.0 - I_uncertain

            if mode == 0:
                gate = I_boundary * I_uncertain * float(counter >= self.min_normal_tokens)
                if gate > 0.5:
                    next_token_logits[b, :].fill_(float('-inf'))
                    next_token_logits[b, THINK_TOKEN_ID] = 0.0
                    mode, counter = 1, 0
            else:
                gate = I_boundary * I_certain * float(counter >= self.min_think_tokens)
                if gate > 0.5:
                    next_token_logits[b, :].fill_(float('-inf'))
                    next_token_logits[b, END_THINK_TOKEN_ID] = 0.0
                    mode, counter = 0, 0
            self._set_intersr_state(req_idx, mode, counter)

    def load_weights(self, weights: Iterable):
        intersr_weights = {}
        base_weights = []
        for name, loaded_weight in weights:
            if name.startswith("sentence_split_mlp.") or name.startswith("signal_predictor_mlp."):
                intersr_weights[name] = loaded_weight
            else:
                base_weights.append((name, loaded_weight))
        super().load_weights(iter(base_weights))
        device = next(self.model.parameters()).device
        if intersr_weights:
            split_state, signal_predictor_state = {}, {}
            for name, weight in intersr_weights.items():
                if name.startswith("sentence_split_mlp."):
                    split_state[name[len("sentence_split_mlp."):]] = weight
                elif name.startswith("signal_predictor_mlp."):
                    signal_predictor_state[name[len("signal_predictor_mlp."):]] = weight
            if split_state:
                self.sentence_split_mlp.load_state_dict(split_state)
            if signal_predictor_state:
                self.signal_predictor_mlp.load_state_dict(signal_predictor_state)
        else:
            try:
                server_args = get_global_server_args()
                model_path = server_args.model_path if server_args else None
            except:
                model_path = None
            if model_path:
                for name, mlp in [("sentence_split_mlp.pt", self.sentence_split_mlp),
                                  ("signal_predictor_mlp.pt", self.signal_predictor_mlp)]:
                    path = os.path.join(model_path, name)
                    if os.path.exists(path):
                        mlp.load_state_dict(torch.load(path, map_location="cpu"))
                        logger.info(f"Loaded {name}")
        self.sentence_split_mlp = self.sentence_split_mlp.to(device).eval()
        self.signal_predictor_mlp = self.signal_predictor_mlp.to(device).eval()
        logger.info(f"InterSR MLPs loaded on {device}")


EntryClass = LlamaInterSRForCausalLM
