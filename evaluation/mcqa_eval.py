"""
Unified evaluation script for non-math benchmarks.
Supports: MMLU, ARC-Challenge, HellaSwag (multiple-choice format)

Handles the multiple-choice format differently from math benchmarks:
- Formats questions with answer choices
- Evaluates by matching predicted letter (A/B/C/D) against ground truth
"""

import os
import sys
import json
import yaml
import time
import re
import torch
import argparse
from tqdm import tqdm
from datasets import load_from_disk, load_dataset
from jinja2 import Template

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from intersr.generate import (
    custom_generate, load_models_and_mlps,
    ENTER_THINK_TOKEN, EXIT_THINK_TOKEN, EOS_TOKEN
)


def format_mmlu_question(item):
    """Format MMLU question with choices"""
    question = item['question']
    choices = item['choices']
    labels = ['A', 'B', 'C', 'D']
    formatted = question + "\n"
    for label, choice in zip(labels, choices):
        formatted += f"\n{label}. {choice}"
    answer = labels[item['answer']]  # MMLU uses integer index
    return formatted, answer


def format_arc_question(item):
    """Format ARC-Challenge question with choices"""
    question = item['question']
    choices = item['choices']
    labels = choices['label']
    texts = choices['text']
    formatted = question + "\n"
    for label, text in zip(labels, texts):
        formatted += f"\n{label}. {text}"
    return formatted, item['answerKey']


def format_gpqa_question(item):
    """Format GPQA-Diamond question (already in the paper)"""
    question = item['Question']
    choices = [
        item['Correct Answer'],
        item['Incorrect Answer 1'],
        item['Incorrect Answer 2'],
        item['Incorrect Answer 3']
    ]
    import random
    random.seed(hash(question) % (2**32))
    indices = list(range(4))
    random.shuffle(indices)
    labels = ['A', 'B', 'C', 'D']
    correct_idx = indices.index(0)  # Correct answer was at index 0
    formatted = question + "\n"
    for i, label in enumerate(labels):
        formatted += f"\n{label}. {choices[indices[i]]}"
    return formatted, labels[correct_idx]


def extract_answer_letter(text):
    """Extract answer letter (A/B/C/D) from model output"""
    # Try boxed format first
    boxed_match = re.findall(r'\\boxed\{([A-Da-d])\}', text)
    if boxed_match:
        return boxed_match[-1].upper()

    # Try "answer is X" pattern
    answer_match = re.findall(r'(?:answer|Answer|ANSWER)\s*(?:is|:)\s*\(?([A-Da-d])\)?', text)
    if answer_match:
        return answer_match[-1].upper()

    # Try standalone letter at end
    end_match = re.findall(r'\b([A-D])\b', text[-100:])
    if end_match:
        return end_match[-1]

    return None


def evaluate_mcqa(config, format_fn, dataset, model, tokenizer,
                  sentence_split_mlp, signal_predictor_mlp, output_path):
    """Evaluate on multiple-choice QA benchmark"""
    results = []
    correct = 0
    total = 0
    total_tokens = 0

    output_format = config['prompt'].get('output_format', '')

    for idx, item in enumerate(tqdm(dataset, desc="Evaluating")):
        question, correct_answer = format_fn(item)
        full_question = question + output_format

        # Generate response
        messages = [{"role": "user", "content": full_question}]
        think_begin = config.get('think_begin', False)
        if not think_begin:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ) + '\n</think>'
        else:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        model_inputs = tokenizer([text], return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(model.device)

        think_ids = None
        if config['thinking']['think_prompt']:
            think_ids = tokenizer(
                [config['thinking']['think_prompt']], return_tensors="pt"
            )["input_ids"].to(model.device)

        start_time = time.time()

        if config['evaluation']['method'] == 'custom_generate':
            generated_ids = custom_generate(
                input_ids, model, sentence_split_mlp, signal_predictor_mlp,
                max_new_tokens=config['generation']['max_new_tokens'],
                temperature=config['generation']['temperature'],
                top_p=config['generation']['top_p'],
                think_threshold=config['thinking']['think_threshold'],
                think_ids=think_ids,
                min_think_tokens=config['thinking']['min_think_tokens'],
                min_normal_tokens=config['thinking']['min_normal_tokens'],
                flag=think_begin == True
            )
        else:
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=config['generation']['max_new_tokens'],
                temperature=config['generation']['temperature'],
                top_p=config['generation']['top_p'],
                do_sample=True,
            )

        gen_time = time.time() - start_time
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        num_tokens = len(generated_ids[0])

        # Extract and verify answer
        predicted = extract_answer_letter(generated_text)
        is_correct = (predicted == correct_answer)

        if is_correct:
            correct += 1
        total += 1
        total_tokens += num_tokens

        results.append({
            "id": idx,
            "question": question[:200],
            "correct_answer": correct_answer,
            "predicted_answer": predicted,
            "is_correct": is_correct,
            "tokens": num_tokens,
            "time": gen_time,
        })

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(dataset)}] Acc: {correct/total:.1%}, "
                  f"Avg tokens: {total_tokens/total:.0f}")

        torch.cuda.empty_cache()

    # Save results
    summary = {
        "accuracy": correct / total if total > 0 else 0,
        "total": total,
        "correct": correct,
        "avg_tokens": total_tokens / total if total > 0 else 0,
        "method": config['evaluation']['method'],
        "threshold": config['thinking']['think_threshold'],
    }

    output = {"summary": summary, "results": results}
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nAccuracy: {summary['accuracy']:.1%} ({correct}/{total})")
    print(f"Avg tokens: {summary['avg_tokens']:.0f}")
    print(f"Results saved to {output_path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["mmlu", "arc", "gpqa"])
    parser.add_argument("--subset", type=str, default=None,
                        help="MMLU subset name, e.g., 'abstract_algebra'")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    # Load config
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f.read())

    model_name = config['model_name']

    # Load model
    model, tokenizer, sentence_split_mlp, signal_predictor_mlp = load_models_and_mlps(
        f"model/Qwen/{model_name}",
        f"intersr/signal_predictor/output/{model_name}/sentence_split_best.pt",
        f"intersr/signal_predictor/output/{model_name}/signal_predictor_best.pt",
        device="cuda:0",
        hidden_size=config['embedding_dim']
    )

    # Load dataset and set format function
    if args.benchmark == "mmlu":
        dataset = load_dataset("cais/mmlu", args.subset or "all", split="test")
        format_fn = format_mmlu_question
    elif args.benchmark == "arc":
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        format_fn = format_arc_question
    elif args.benchmark == "gpqa":
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        format_fn = format_gpqa_question

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print(f"Benchmark: {args.benchmark}, Samples: {len(dataset)}")

    output_dir = f"./results/{args.benchmark}/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    method = config['evaluation']['method']
    threshold = config['thinking']['think_threshold']
    output_path = os.path.join(output_dir, f"{method}_t{threshold}.json")

    evaluate_mcqa(config, format_fn, dataset, model, tokenizer,
                  sentence_split_mlp, signal_predictor_mlp, output_path)


if __name__ == "__main__":
    main()
