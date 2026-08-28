import sys
import time
import json
import random
import yaml
import logging
import os
from datasets import load_from_disk, load_dataset
from tqdm import tqdm
from math_verify import parse, verify
import torch
import threading
import queue
from filelock import FileLock
from jinja2 import Template
from intersr.generate import custom_generate, load_models_and_mlps, ENTER_THINK_TOKEN, EXIT_THINK_TOKEN, EOS_TOKEN

# Configure logging
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '../evaluation', "evaluation_errors.log")
logging.basicConfig(
    filename=log_file_path,
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_config(config_path="eval_config.yaml"):
    """Load configuration file with variable reference support"""
    with open(config_path, 'r', encoding='utf-8') as f:
        raw = f.read()
    # First render with Jinja2
    template = Template(raw)
    rendered = template.render()
    # Then parse with yaml
    config = yaml.safe_load(rendered)
    return config

def verify_answer(question, answer):
    """Verify answer"""
    try:
        return verify(parse(question), parse(answer))
    except Exception:
        logging.exception(f"Error verifying answer: Question: {question}")
        return False

def evaluate_custom_method(question, config, model, tokenizer, sentence_split_mlp, signal_predictor_mlp):
    """Evaluate custom generate method"""
    messages = [{"role": "user", "content": question}]
    think_begin = config.get('think_begin',False)
    if not think_begin:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)+'\n</think>'
    else:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt")
    input_ids = model_inputs["input_ids"].to(model.device)

    if config['thinking']['think_prompt'] != "":
        # Prepare thinking tokens
        think_ids = tokenizer([config['thinking']['think_prompt']], return_tensors="pt")["input_ids"].to(model.device)
    else:
        think_ids = None

    start_time = time.time()
    generated_ids = custom_generate(
        input_ids,
        model,
        sentence_split_mlp,
        signal_predictor_mlp,
        max_new_tokens=config['generation']['max_new_tokens'],
        temperature=config['generation']['temperature'],
        top_p=config['generation']['top_p'],
        think_threshold=config['thinking']['think_threshold'],
        think_ids=think_ids,
        min_think_tokens=config['thinking']['min_think_tokens'],
        min_normal_tokens=config['thinking']['min_normal_tokens'],
        flag = config.get('think_begin', False)==True
    )
    generation_time = time.time() - start_time
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)


    return generated_text, generation_time, len(generated_ids[0])

def evaluate_official_method(questions, config, model, tokenizer):
    sample_n = config.get('sample_n', 1)
    messages_list = [[{"role": "user", "content": q}] for q in questions]
    texts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_list]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    input_batch_size = model_inputs['input_ids'].shape[0]

    start_time = time.time()
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=config['generation']['max_new_tokens'],
        temperature=config['generation']['temperature'],
        top_p=config['generation']['top_p'],
        do_sample=config['generation']['do_sample'],
        num_return_sequences=sample_n
    )
    generation_time = time.time() - start_time

    generated_ids = generated_ids.view(input_batch_size, sample_n, -1)
    generated_texts = []
    generated_tokens_lengths = []
    for i in range(input_batch_size):
        texts_i = [tokenizer.decode(generated_ids[i, j], skip_special_tokens=True) for j in range(sample_n)]
        tokens_i = [len(generated_ids[i, j]) for j in range(sample_n)]
        generated_texts.append(texts_i)
        generated_tokens_lengths.append(tokens_i)

    return generated_texts, generation_time, generated_tokens_lengths



def evaluate_parallel(config_path="evaluation_config.yaml",config=None):
    """
    Use multi-GPU to evaluate the dataset.
    """
    if config is None:
        config = load_config(config_path)

    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        logging.error("No available GPU, cannot perform parallel evaluation.")
        return

    try:
        gpu_num = int(config['evaluation']['gpu_num'])
    except (KeyError, TypeError, ValueError):
        logging.warning("'gpu_num' not configured or in incorrect format in config file. Will use all available GPUs.")
        gpu_num = available_gpus

    if gpu_num > available_gpus:
        logging.warning(
            f"Configured GPU count ({gpu_num}) is greater than available GPU count ({available_gpus})."
            f"Will use all available GPUs ({available_gpus})."
        )
        gpu_num = available_gpus

    if gpu_num <= 0:
        logging.error(f"Configured GPU count ({gpu_num}) is invalid, cannot perform parallel evaluation.")
        return
    model_name = config['model_name']
    method = config['evaluation']['method']
    batch_size = 1
    if 'official' in method:
        batch_size = config['evaluation'].get('batch_size', 1)

    output_dir = f"./results/{config['dataset']['name']}/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    if config['evaluation']['method'] == 'official_generate':
        output_filename = f'official_generate.json'
    else:
        output_filename = 'custom_generate'+str(config['thinking']['think_threshold'])+config['thinking']['think_prompt']+str(config.get('think_begin', False))+'.json'

    # Add method name to output filename
    base_name, ext = os.path.splitext(output_filename)
    output_filename_with_method = f"{base_name}_{method}{ext if ext else '.jsonl'}"
    parallel_output_file = os.path.join(output_dir, f"parallel_{output_filename_with_method}")

    file_lock = FileLock(parallel_output_file + ".lock")

    # Load dataset: try local first, then HuggingFace Hub
    dataset_config = config['dataset']
    local_path = f"dataset/{dataset_config['name']}"
    if os.path.exists(local_path):
        dataset = load_from_disk(local_path)
        test_data = dataset[dataset_config['split']]
    elif 'hf_path' in dataset_config:
        hf_subset = dataset_config.get('hf_subset', None)
        if hf_subset:
            test_data = load_dataset(dataset_config['hf_path'], hf_subset, split=dataset_config['split'])
        else:
            test_data = load_dataset(dataset_config['hf_path'], split=dataset_config['split'])
    else:
        dataset = load_from_disk(local_path)
        test_data = dataset[dataset_config['split']]

    processed_ids = set()
    if os.path.exists(parallel_output_file):
        with file_lock:
            with open(parallel_output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if 'id' in data:
                             processed_ids.add(data['id'])
                    except (json.JSONDecodeError, KeyError):
                        logging.error(f"Error parsing processed result line: {line.strip()}")
                        continue

    task_queue = queue.Queue()
    all_tasks = list(test_data)

    for i, sample in enumerate(all_tasks):
        if i not in processed_ids:
            task_queue.put((i, sample))

    if task_queue.empty() and len(processed_ids) > 0:
        return

    pbar = tqdm(total=task_queue.qsize(), desc=f"Progress {config['thinking']['think_threshold']} {str(config.get('think_begin', False))}", position=config.get('tqdm_position', 0))

    loading_semaphore = threading.Semaphore(1)
    loading_events = [threading.Event() for _ in range(gpu_num)]

    def worker(gpu_id):
        if gpu_id > 0:
            loading_events[gpu_id - 1].wait()

        with loading_semaphore:
            device = f"cuda:{gpu_id}"
            model, tokenizer, sentence_split_mlp, signal_predictor_mlp = load_models_and_mlps(
                f"model/Qwen/{model_name}",
                f"intersr/signal_predictor/output/{model_name}/sentence_split_best.pt",
                f"intersr/signal_predictor/output/{model_name}/signal_predictor_best.pt",
                device=device,
                hidden_size=config['embedding_dim']
            )

            if gpu_id < gpu_num - 1:
                loading_events[gpu_id].set()

        while not task_queue.empty():
            batch_tasks = []
            results_to_write = []

            try:
                # 1. Collect and process tasks based on method
                if method == 'custom_generate':
                    # Take one at a time
                    try:
                        task = task_queue.get_nowait()
                        batch_tasks.append(task)
                    except queue.Empty:
                        continue # Queue is empty

                    # Process single task
                    sample_id, sample = batch_tasks[0]
                    question = sample[config['dataset']['question_key']] + config['prompt']['output_format']

                    generated_texts = []
                    times = []
                    tokens_list = []

                    for _ in range(config['sample_n']):
                        text, time_taken, tokens = evaluate_custom_method(
                            question, config, model, tokenizer, sentence_split_mlp, signal_predictor_mlp
                        )
                        generated_texts.append(text)
                        times.append(time_taken)
                        tokens_list.append(tokens)

                    results_to_write.append({
                        'id': sample_id,
                        'question': sample[config['dataset']['question_key']],
                        'correct_answer': sample[config['dataset']['answer_key']],
                        'generated_text': generated_texts,
                        'time': times,
                        'tokens': tokens_list,
                        'is_correct': None
                    })

                elif 'official' in method:
                    # Take one batch at a time
                    try:
                        for _ in range(batch_size):
                            batch_tasks.append(task_queue.get_nowait())
                    except queue.Empty:
                        pass # Queue is empty or not enough for a batch

                    if not batch_tasks:
                        continue

                    # Process batch tasks
                    questions = [s[config['dataset']['question_key']] + config['prompt']['output_format'] for _, s in batch_tasks]

                    generated_texts, time_taken, tokens_list = evaluate_official_method(
                        questions, config, model, tokenizer
                    )
                    for i, (sample_id, sample) in enumerate(batch_tasks):
                        results_to_write.append({
                            'id': sample_id,
                            'question': sample[config['dataset']['question_key']],
                            'correct_answer': sample[config['dataset']['answer_key']],
                            'generated_text': generated_texts[i],  # Already a list of n samples
                            'time': [time_taken] * config['sample_n'],  # Time can be the same
                            'tokens': tokens_list[i],                   # Token count for n samples
                            'is_correct': None
                        })

            except Exception as e:
                logging.exception(f"Error during worker processing. Batch start ID: {batch_tasks[0][0] if batch_tasks else 'N/A'}")
                # Record error for each task in the batch
                for sample_id, sample in batch_tasks:
                    results_to_write.append({'id': sample_id, 'question': sample[config['dataset']['question_key']], 'error': str(e)})

            # 2. Write results
            if results_to_write:
                with file_lock:
                    with open(parallel_output_file, 'a', encoding='utf-8') as f:
                        for res in results_to_write:
                            json.dump(res, f, ensure_ascii=False)
                            f.write('\n')

            # 3. Update progress
            if batch_tasks:
                pbar.update(len(batch_tasks))
                for _ in batch_tasks:
                    task_queue.task_done()

            # 4. Clear cache - clear immediately after each batch
            del batch_tasks, results_to_write
            torch.cuda.empty_cache()

        del model, tokenizer, sentence_split_mlp, signal_predictor_mlp
        torch.cuda.empty_cache()

    threads = []
    for gpu_id in range(gpu_num):
        thread = threading.Thread(target=worker, args=(gpu_id,), name=str(gpu_id))
        thread.daemon = True
        thread.start()
        threads.append(thread)

    try:
        task_queue.join()
    except KeyboardInterrupt:
        print("\nCaught Ctrl+C, terminating program...")

    pbar.close()

    print("\nParallel evaluation completed. Starting to summarize results...")
    all_results = []
    with open(parallel_output_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                all_results.append(json.loads(line))
            except json.JSONDecodeError:
                logging.error(f"Error parsing result line: {line.strip()}")
                continue

    print(f"\nDetailed results saved to: {parallel_output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="./evaluation_config.yaml")
    parser.add_argument("--config", type=dict, default=None)
    args = parser.parse_args()
    evaluate_parallel(args.config_file,args.config)
