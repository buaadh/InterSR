"""
InterSR Model - Interleaved System-1/System-2 Reasoning as a Model-Integrated Module

Integrates SentenceSplitMLP and SignalPredictorMLP directly into the language model's
forward pass, using indicator functions to dynamically modify logits for <think>
and </think> tokens. This makes InterSR a native part of the model rather than an
external generation hook.

Key design:
    At each generation step, the model's forward pass:
    1. Runs the base LLM -> hidden states + logits
    2. Runs SentenceSplitMLP(h) -> boundary indicator I_b
    3. Runs SignalPredictorMLP(h) -> signal indicator I_s
    4. Determines current reasoning mode from token history
    5. Applies indicator-gated logit modification:
       - Enter thinking: I(normal) * I_b * I_s * I(counter >= min_normal) -> force <think>
       - Exit thinking:  I(thinking) * I_b * (1-I_s) * I(counter >= min_think) -> force </think>

Usage:
    model = InterSRModel.from_pretrained(
        "model/Qwen/DeepSeek-R1-Distill-Qwen-7B",
        sentence_split_path="intersr/signal_predictor/output/.../sentence_split_best.pt",
        signal_predictor_path="intersr/signal_predictor/output/.../signal_predictor_best.pt",
        hidden_size=3584,
        think_threshold=0.4,
    )
    # Works with standard model.generate()
    output = model.generate(input_ids, max_new_tokens=4096, temperature=0.6, top_p=0.95)
"""

import torch
import torch.nn as nn
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from intersr.signal_predictor.architecture import SentenceSplitMLP, SignalPredictorMLP


# Token IDs for Qwen2 / DeepSeek-R1-Distill series
THINK_TOKEN_ID = 151648      # <think>
END_THINK_TOKEN_ID = 151649  # </think>
EOS_TOKEN_ID = 151643


class InterSRModel(nn.Module):
    """
    InterSR-augmented language model with integrated reasoning mode switching.

    Wraps a base causal LM and adds SentenceSplitMLP + SignalPredictorMLP as
    submodules. The forward pass computes indicator-gated logit modifications
    to enable dynamic System-1/System-2 interleaving.
    """

    def __init__(self, base_model, sentence_split_mlp, signal_predictor_mlp,
                 think_threshold=0.4, min_think_tokens=10, min_normal_tokens=10,
                 boost=float('inf')):
        super().__init__()
        self.model = base_model
        self.sentence_split_mlp = sentence_split_mlp
        self.signal_predictor_mlp = signal_predictor_mlp

        self.think_threshold = think_threshold
        self.min_think_tokens = min_think_tokens
        self.min_normal_tokens = min_normal_tokens
        self.boost = boost

        self.think_token_id = THINK_TOKEN_ID
        self.end_think_token_id = END_THINK_TOKEN_ID

        self._mode = 0       # 0 = normal (System-1), 1 = thinking (System-2)
        self._counter = 0
        self._enabled = True

    def reset_intersr_state(self, initial_mode=0):
        """Reset InterSR state before a new generation."""
        self._mode = initial_mode
        self._counter = 0

    def set_intersr_enabled(self, enabled=True):
        """Enable/disable InterSR. When disabled, model behaves as vanilla LLM."""
        self._enabled = enabled

    @property
    def device(self):
        return self.model.device

    @property
    def config(self):
        return self.model.config

    @property
    def generation_config(self):
        return self.model.generation_config

    @generation_config.setter
    def generation_config(self, value):
        self.model.generation_config = value

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def can_generate(self):
        return True

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def _reorder_cache(self, *args, **kwargs):
        return self.model._reorder_cache(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def forward(self, input_ids=None, **kwargs):
        if not self._enabled:
            return self.model(input_ids=input_ids, **kwargs)
        kwargs['output_hidden_states'] = True
        outputs = self.model(input_ids=input_ids, **kwargs)
        self._apply_intersr_logic(input_ids, outputs)
        return outputs

    def generate(self, input_ids=None, initial_mode=1, **kwargs):
        """Generate with InterSR. Automatically resets state before generation."""
        self.reset_intersr_state(initial_mode=initial_mode)

        original_forward = self.model.forward
        intersr_self = self

        def intersr_forward(input_ids=None, **fwd_kwargs):
            fwd_kwargs['output_hidden_states'] = True
            outputs = original_forward(input_ids=input_ids, **fwd_kwargs)
            intersr_self._apply_intersr_logic(input_ids, outputs)
            return outputs

        self.model.forward = intersr_forward
        try:
            if 'pad_token_id' not in kwargs:
                kwargs['pad_token_id'] = EOS_TOKEN_ID
            result = self.model.generate(input_ids=input_ids, **kwargs)
        finally:
            self.model.forward = original_forward

        return result

    def _apply_intersr_logic(self, input_ids, outputs):
        """Apply InterSR indicator-gated logit modification."""
        logits = outputs.logits

        if input_ids is not None and input_ids.numel() > 0:
            last_token = input_ids[0, -1].item()
            if last_token == self.think_token_id:
                self._mode = 1
                self._counter = 0
            elif last_token == self.end_think_token_id:
                self._mode = 0
                self._counter = 0
            else:
                self._counter += 1

        hidden_states = outputs.hidden_states[-1]
        last_hs = hidden_states[:, -1, :]

        with torch.no_grad():
            split_score = self.sentence_split_mlp(last_hs)
            signal_score = self.signal_predictor_mlp(last_hs)

        I_boundary = (split_score > 0).float()
        I_high_signal = (signal_score > self.think_threshold).float()
        I_low_signal = 1.0 - I_high_signal
        I_normal = float(self._mode == 0)
        I_thinking = float(self._mode == 1)
        I_min_normal = float(self._counter >= self.min_normal_tokens)
        I_min_think = float(self._counter >= self.min_think_tokens)

        enter_gate = I_normal * I_boundary * I_high_signal * I_min_normal
        exit_gate = I_thinking * I_boundary * I_low_signal * I_min_think

        batch_size = logits.shape[0]
        for b in range(batch_size):
            if enter_gate[b] > 0.5:
                if self.boost == float('inf'):
                    logits[b, -1, :].fill_(float('-inf'))
                    logits[b, -1, self.think_token_id] = 0.0
                else:
                    logits[b, -1, self.think_token_id] += self.boost
                self._mode = 1
                self._counter = 0
            elif exit_gate[b] > 0.5:
                if self.boost == float('inf'):
                    logits[b, -1, :].fill_(float('-inf'))
                    logits[b, -1, self.end_think_token_id] = 0.0
                else:
                    logits[b, -1, self.end_think_token_id] += self.boost
                self._mode = 0
                self._counter = 0

        outputs.logits = logits

    def save_intersr(self, save_dir):
        """Save InterSR model components."""
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(os.path.join(save_dir, "base_model"))
        torch.save(self.sentence_split_mlp.state_dict(),
                   os.path.join(save_dir, "sentence_split_mlp.pt"))
        torch.save(self.signal_predictor_mlp.state_dict(),
                   os.path.join(save_dir, "signal_predictor_mlp.pt"))
        config = {
            "think_threshold": self.think_threshold,
            "min_think_tokens": self.min_think_tokens,
            "min_normal_tokens": self.min_normal_tokens,
            "boost": self.boost if self.boost != float('inf') else "inf",
            "hidden_size": self.sentence_split_mlp.classifier[0].in_features,
            "think_token_id": self.think_token_id,
            "end_think_token_id": self.end_think_token_id,
        }
        with open(os.path.join(save_dir, "intersr_config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        print(f"InterSR model saved to {save_dir}")

    @classmethod
    def from_pretrained(cls, model_path, sentence_split_path, signal_predictor_path,
                        hidden_size=3584, think_threshold=0.4,
                        min_think_tokens=10, min_normal_tokens=10,
                        boost=float('inf'), device="cuda:0", **model_kwargs):
        """Load InterSR model from base model + separate MLP weights."""
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", **model_kwargs
        ).to(device)

        sentence_split_mlp = SentenceSplitMLP(hidden_size=hidden_size).to(device)
        signal_predictor_mlp = SignalPredictorMLP(hidden_size=hidden_size).to(device)
        sentence_split_mlp.load_state_dict(
            torch.load(sentence_split_path, map_location=device))
        signal_predictor_mlp.load_state_dict(
            torch.load(signal_predictor_path, map_location=device))
        sentence_split_mlp.eval()
        signal_predictor_mlp.eval()

        return cls(
            base_model=base_model,
            sentence_split_mlp=sentence_split_mlp,
            signal_predictor_mlp=signal_predictor_mlp,
            think_threshold=think_threshold,
            min_think_tokens=min_think_tokens,
            min_normal_tokens=min_normal_tokens,
            boost=boost,
        )

    @classmethod
    def load_intersr(cls, save_dir, device="cuda:0", **model_kwargs):
        """Load a previously saved InterSR model."""
        with open(os.path.join(save_dir, "intersr_config.json"), 'r') as f:
            config = json.load(f)
        boost = float('inf') if config["boost"] == "inf" else config["boost"]
        return cls.from_pretrained(
            model_path=os.path.join(save_dir, "base_model"),
            sentence_split_path=os.path.join(save_dir, "sentence_split_mlp.pt"),
            signal_predictor_path=os.path.join(save_dir, "signal_predictor_mlp.pt"),
            hidden_size=config["hidden_size"],
            think_threshold=config["think_threshold"],
            min_think_tokens=config["min_think_tokens"],
            min_normal_tokens=config["min_normal_tokens"],
            boost=boost,
            device=device,
            **model_kwargs,
        )


def demo():
    """Quick demo of InterSR model usage."""
    model = InterSRModel.from_pretrained(
        "model/Qwen/DeepSeek-R1-Distill-Qwen-7B",
        "intersr/signal_predictor/output/DeepSeek-R1-Distill-Qwen-7B/sentence_split_best.pt",
        "intersr/signal_predictor/output/DeepSeek-R1-Distill-Qwen-7B/signal_predictor_best.pt",
        hidden_size=3584,
        think_threshold=0.4,
    )
    tokenizer = AutoTokenizer.from_pretrained("model/Qwen/DeepSeek-R1-Distill-Qwen-7B")
    messages = [{"role": "user", "content": "What is 25 * 37?"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer([text], return_tensors="pt").input_ids.to(model.device)
    output = model.generate(
        input_ids, max_new_tokens=1024, temperature=0.6, top_p=0.95,
        do_sample=True, initial_mode=1,
    )
    print(tokenizer.decode(output[0], skip_special_tokens=False))


if __name__ == "__main__":
    demo()
