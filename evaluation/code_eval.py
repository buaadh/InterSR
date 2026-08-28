"""
HumanEval evaluation for InterSR framework.
Compares code generation efficiency (token usage) between InterSR and standard generation.
"""

import os
import sys
import json
import time
import re
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from intersr.generate import (
    custom_generate, load_models_and_mlps,
    ENTER_THINK_TOKEN, EXIT_THINK_TOKEN, EOS_TOKEN
)


def extract_code(text, entry_point):
    """Extract the function code from model output"""
    # Try to find the function definition
    pattern = rf'(def {entry_point}\(.*?\n(?:(?!def ).*\n)*)'
    match = re.search(pattern, text, re.MULTILINE)
    if match:
        return match.group(1)

    # Fallback: extract everything after </think> or the last code block
    if '</think>' in text:
        code_part = text.split('</think>')[-1]
    else:
        code_part = text

    # Find code block
    code_match = re.search(r'```python\n(.*?)```', code_part, re.DOTALL)
    if code_match:
        return code_match.group(1)

    return code_part


def evaluate_humaneval(model, tokenizer, sentence_split_mlp, signal_predictor_mlp,
                       dataset, method='custom_generate', think_threshold=0.4,
                       max_new_tokens=2048, output_path='./results_humaneval.json'):
    """Evaluate on HumanEval"""
    results = []
    total_tokens = 0

    for idx, item in enumerate(tqdm(dataset, desc=f"HumanEval ({method})")):
        prompt = item['prompt']
        entry_point = item['entry_point']
        test_code = item['test']
        canonical = item['canonical_solution']

        # Format as instruction
        instruction = f"Complete the following Python function:\n\n{prompt}"
        messages = [{"role": "user", "content": instruction}]

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(model.device)

        start_time = time.time()

        if method == 'custom_generate':
            generated_ids = custom_generate(
                input_ids, model, sentence_split_mlp, signal_predictor_mlp,
                max_new_tokens=max_new_tokens,
                temperature=0.6, top_p=0.95,
                think_threshold=think_threshold,
                think_ids=None,
                min_think_tokens=10,
                min_normal_tokens=10,
                flag=True  # start in think mode
            )
        else:
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.6,
                top_p=0.95,
                do_sample=True,
            )

        gen_time = time.time() - start_time
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        num_tokens = len(generated_ids[0]) - len(input_ids[0])
        total_tokens += num_tokens

        # Extract code
        code = extract_code(generated_text, entry_point)

        # Try to execute (basic check)
        is_correct = False
        try:
            exec_globals = {}
            full_code = code + "\n" + test_code + f"\ncheck({entry_point})"
            exec(full_code, exec_globals)
            is_correct = True
        except Exception:
            pass

        results.append({
            "task_id": item['task_id'],
            "tokens": num_tokens,
            "time": gen_time,
            "is_correct": is_correct,
        })

        if (idx + 1) % 20 == 0:
            correct = sum(1 for r in results if r['is_correct'])
            print(f"  [{idx+1}/{len(dataset)}] Pass@1: {correct/len(results):.1%}, "
                  f"Avg tokens: {total_tokens/len(results):.0f}")

        torch.cuda.empty_cache()

    # Summary
    correct = sum(1 for r in results if r['is_correct'])
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    avg_tokens = total_tokens / total if total > 0 else 0

    print(f"\nHumanEval Results ({method}):")
    print(f"  Pass@1: {accuracy:.1%} ({correct}/{total})")
    print(f"  Avg tokens: {avg_tokens:.0f}")

    output = {
        "method": method,
        "threshold": think_threshold,
        "pass_at_1": accuracy,
        "total": total,
        "correct": correct,
        "avg_tokens": avg_tokens,
        "results": results,
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model/Qwen/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--method", type=str, default="custom_generate",
                        choices=["custom_generate", "standard"])
    parser.add_argument("--think_threshold", type=float, default=0.4)
    parser.add_argument("--output_dir", type=str, default="./results/humaneval")
    parser.add_argument("--hidden_size", type=int, default=3584)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_name = os.path.basename(args.model_path)

    # Load model and MLPs
    model, tokenizer, sentence_split_mlp, signal_predictor_mlp = load_models_and_mlps(
        args.model_path,
        f"intersr/signal_predictor/output/{model_name}/sentence_split_best.pt",
        f"intersr/signal_predictor/output/{model_name}/signal_predictor_best.pt",
        device=args.device,
        hidden_size=args.hidden_size
    )

    # Load dataset
    dataset = load_dataset("openai/openai_humaneval", split="test")
    print(f"HumanEval: {len(dataset)} problems")

    output_path = os.path.join(
        args.output_dir, model_name,
        f"{args.method}_t{args.think_threshold}.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    evaluate_humaneval(
        model, tokenizer, sentence_split_mlp, signal_predictor_mlp,
        dataset, method=args.method, think_threshold=args.think_threshold,
        output_path=output_path
    )


if __name__ == "__main__":
    main()
