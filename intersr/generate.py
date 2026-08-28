"""
InterSR Generation Module

Implements the core InterSR (Interleaved System-1/2 Reasoning) generation logic
using HuggingFace Transformers. Dynamically switches between System-1 (fast) and
System-2 (slow) reasoning modes based on learned signal predictors.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import os
import torch
import sys
import random
import numpy as np
import time
import copy
import logging
from transformers.generation.logits_process import TemperatureLogitsWarper, TopPLogitsWarper
from transformers.generation.utils import LogitsProcessorList
from intersr.signal_predictor.architecture import SentenceSplitMLP, SignalPredictorMLP


def calculate_mlp_scores(last_hidden_state, sentence_split_mlp, signal_predictor_mlp):
    """Calculate MLP scores for mode switching decision."""
    sentence_split_score = sentence_split_mlp(last_hidden_state)
    signal_score = signal_predictor_mlp(last_hidden_state)
    return sentence_split_score, signal_score


def should_change_thinking_mode(split_score, signal_score, flag, think_counter,
                                think_token_count, think_threshold,
                                min_think_tokens, min_normal_tokens):
    """Determine whether to switch between System-1 and System-2 modes."""
    split_condition = split_score > 0
    signal_condition = signal_score > think_threshold

    if flag == 0:  # Currently in normal mode, check whether to enter thinking mode
        return (split_condition and signal_condition and think_counter >= min_normal_tokens)
    else:  # Currently in thinking mode, check whether to exit thinking mode
        return (split_condition and signal_score <= think_threshold and think_token_count >= min_think_tokens)


def process_think_tokens(generated_ids, think_ids, model, model_kwargs, use_cache):
    """Process tokens in thinking mode."""
    if think_ids is not None:
        think_ids = think_ids.to(model.device)
        for i in range(len(think_ids[0])):
            model_inputs = model.prepare_inputs_for_generation(generated_ids, **model_kwargs)
            outputs = model(**model_inputs, return_dict=True)
            if use_cache:
                model_kwargs = model._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False)
            next_think_token = think_ids[:, i:i+1]
            generated_ids = torch.cat([generated_ids, next_think_token], dim=1)
            del outputs
    return generated_ids, model_kwargs


def sample_next_token(logits_processor, generated_ids, next_token_logits):
    """Sample next token using logits processor."""
    next_token_scores = logits_processor(generated_ids, next_token_logits)
    probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def handle_eos_token(generated_ids, flag):
    """Handle end-of-sequence token."""
    if flag == 1:
        generated_ids = torch.cat([generated_ids, torch.tensor([[EXIT_THINK_TOKEN]], device=generated_ids.device)], dim=1)
    generated_ids = torch.cat([generated_ids, torch.tensor([[EOS_TOKEN]], device=generated_ids.device)], dim=1)
    return generated_ids


def custom_generate(input_ids, model, sentence_split_mlp, signal_predictor_mlp,
                    max_new_tokens=100, temperature=0.7, top_p=0.95, use_cache=True,
                    think_threshold=0.5, think_ids=None, min_think_tokens=10,
                    min_normal_tokens=10, flag=0):
    """
    Generate text with InterSR dynamic mode switching.

    Args:
        input_ids: Input token IDs.
        model: Base language model.
        sentence_split_mlp: Sentence boundary predictor.
        signal_predictor_mlp: Signal strength predictor for mode switching.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_p: Top-p sampling threshold.
        think_threshold: Signal threshold for triggering System-2 reasoning.
        think_ids: Optional prompt tokens to inject when entering thinking mode.
        min_think_tokens: Minimum tokens before exiting thinking mode.
        min_normal_tokens: Minimum tokens before entering thinking mode.
        flag: Initial mode (0=normal/System-1, 1=thinking/System-2).

    Returns:
        Generated token IDs tensor.
    """
    think_counter = 0
    think_token_count = 0

    device = model.device
    generated_ids = input_ids.to(device)

    model_kwargs = {
        "cache_position": torch.arange(len(generated_ids[0]), device=device),
        "past_key_values": None
    }

    logits_processor = LogitsProcessorList()
    if temperature != 1.0:
        logits_processor.append(TemperatureLogitsWarper(temperature))
    if top_p > 0:
        logits_processor.append(TopPLogitsWarper(top_p))

    with torch.no_grad():
        while len(generated_ids[0]) < max_new_tokens:
            model_inputs = model.prepare_inputs_for_generation(generated_ids, **model_kwargs)
            outputs = model(**model_inputs, return_dict=True, output_hidden_states=True)

            if use_cache:
                model_kwargs = model._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False)

            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=device)
            last_hidden_state = outputs.hidden_states[-1][:, -1, :]

            sentence_split_score, signal_score = calculate_mlp_scores(
                last_hidden_state, sentence_split_mlp, signal_predictor_mlp
            )

            next_token = sample_next_token(logits_processor, generated_ids, next_token_logits)
            if next_token.item() == ENTER_THINK_TOKEN:
                flag = 1
            elif next_token.item() == EXIT_THINK_TOKEN:
                think_counter = 0
                think_token_count = 0
                flag = 0
            elif next_token.item() == EOS_TOKEN:
                generated_ids = handle_eos_token(generated_ids, flag)
                break
            else:
                if should_change_thinking_mode(sentence_split_score, signal_score, flag,
                                               think_counter, think_token_count, think_threshold,
                                               min_think_tokens, min_normal_tokens):
                    logging.info("--> Trigger thinking mode switch.")
                    next_token = torch.tensor([[ENTER_THINK_TOKEN if flag == 0 else EXIT_THINK_TOKEN]], device=device)
                    if flag == 1:
                        think_counter = 0
                        think_token_count = 0
                    else:
                        if think_ids is not None:
                            generated_ids = torch.cat([generated_ids, torch.tensor([[ENTER_THINK_TOKEN]], device=device)], dim=1)
                            generated_ids, model_kwargs = process_think_tokens(generated_ids, think_ids, model, model_kwargs, use_cache)
                            flag = 1 - flag
                            continue
                    flag = 1 - flag
                    logging.info(f"<-- Thinking mode switch completed, new flag={flag}.")

            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if flag == 1:
                think_token_count += 1
            else:
                think_counter += 1

            del outputs

    return generated_ids


def load_models_and_mlps(model_path, sentence_split_path, signal_predictor_path,
                         device="cuda:0", hidden_size=1536):
    """
    Load base model and InterSR signal predictor modules.

    Args:
        model_path: Path to HuggingFace base model.
        sentence_split_path: Path to SentenceSplitMLP weights (.pt).
        signal_predictor_path: Path to SignalPredictorMLP weights (.pt).
        device: Target device.
        hidden_size: Hidden dimension of the base model.

    Returns:
        Tuple of (model, tokenizer, sentence_split_mlp, signal_predictor_mlp).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto"
    ).to(device)

    sentence_split_mlp = SentenceSplitMLP(hidden_size=hidden_size).to(device)
    signal_predictor_mlp = SignalPredictorMLP(hidden_size=hidden_size).to(device)
    sentence_split_mlp.load_state_dict(torch.load(sentence_split_path))
    signal_predictor_mlp.load_state_dict(torch.load(signal_predictor_path))

    return model, tokenizer, sentence_split_mlp, signal_predictor_mlp


# Special token IDs (DeepSeek-R1-Distill-Qwen series)
ENTER_THINK_TOKEN = 151648
EXIT_THINK_TOKEN = 151649
EOS_TOKEN = 151643


if __name__ == "__main__":
    model_name = "DeepSeek-R1-Distill-Qwen-14B"
    model_path = f"model/Qwen/{model_name}"
    sentence_split_path = f"intersr/signal_predictor/output/{model_name}/sentence_split_best.pt"
    signal_predictor_path = f"intersr/signal_predictor/output/{model_name}/signal_predictor_best.pt"
    mlp_device = "cuda:0"
    hidden_size = 5120

    model, tokenizer, sentence_split_mlp, signal_predictor_mlp = load_models_and_mlps(
        model_path, sentence_split_path, signal_predictor_path, hidden_size=hidden_size
    )

    prompt = "Find the number of ways to place a digit in each cell of a 2x3 grid so that the sum of the two numbers formed by reading left to right is $999$."
    messages = [{"role": "user", "content": prompt}]
    start_time = time.time()
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt")
    input_ids = model_inputs["input_ids"].to(model.device)

    generated_ids = custom_generate(
        input_ids, model, sentence_split_mlp, signal_predictor_mlp,
        max_new_tokens=16384, temperature=0.6, top_p=0.95,
        think_threshold=0.2, think_ids=None,
        min_think_tokens=10, min_normal_tokens=10, flag=1
    )
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("\nGenerated text:", generated_text)
    print("Token count:", len(generated_ids[0]))
    print(f"Generation time: {time.time() - start_time} seconds")
