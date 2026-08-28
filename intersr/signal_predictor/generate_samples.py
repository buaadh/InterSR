from transformers import AutoModelForCausalLM, AutoTokenizer
import stanza
import logging
from datasets import load_from_disk
import json
from tqdm import tqdm
import os
import random
import torch


def process_dataset(dataset, output_file, gpu_num=1, batch_size=1):
    """
    Multi-GPU parallel processing of the dataset, supporting checkpointing and batch processing
    """
    import threading
    import queue
    from filelock import FileLock
    from tqdm.auto import tqdm

    # Create a semaphore to control model loading
    loading_semaphore = threading.Semaphore(1)
    # Create a list of events to notify the next thread to start loading
    loading_events = [threading.Event() for _ in range(gpu_num)]

    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Get all questions
    questions = dataset['query'].tolist()
    total_samples = len(questions)

    # Check for existing files and processing progress
    processed_ids = set()
    file_lock = FileLock(output_file + ".lock")

    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print("Found existing output file, reading processed data...")
        with file_lock:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        processed_ids.add(data['id'])
                    except:
                        continue

    # Create task queue
    task_queue = queue.Queue()
    total_tasks = 0
    for idx, question in enumerate(questions):
        if idx not in processed_ids:
            task_queue.put((idx, question))
            total_tasks += 1
    if total_tasks == 0:
        print("No data to process")
        return output_file
    # Create progress bar
    pbar = tqdm(total=total_tasks, desc="Processing Progress")

    def worker(gpu_id):
        """
        Worker function for each thread, each thread is responsible for one GPU, supporting batch processing
        """
        # Wait for the model loading of the previous thread to complete
        if gpu_id > 0:
            loading_events[gpu_id - 1].wait()

        # Get the semaphore, start loading the model
        with loading_semaphore:
            print(f"\nStarting to load model for GPU {gpu_id}...")
            device = f"cuda:{gpu_id}"
            local_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).to(device)
            local_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                padding_side="left",
                trust_remote_code=True
            )
            print(f"Model loaded for GPU {gpu_id}")

            # Notify the next thread to start loading
            if gpu_id < gpu_num - 1:
                loading_events[gpu_id].set()

        while True:
            # Get a batch of tasks
            batch_data = []
            for _ in range(batch_size):
                try:
                    idx, question = task_queue.get_nowait()
                    batch_data.append((idx, question))
                except queue.Empty:
                    break

            if not batch_data:  # If no data is obtained, exit the loop
                break

            try:
                # Prepare batch input
                batch_messages = [{"role": "user", "content": question} for _, question in batch_data]
                batch_texts = [
                    local_tokenizer.apply_chat_template(
                        [message],
                        tokenize=False,
                        add_generation_prompt=True
                    ) + "\n</think>" for message in batch_messages
                ]

                # Batch tokenize
                model_inputs = local_tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True
                ).to(device)

                # Batch generation
                generated_ids = local_model.generate(
                    **model_inputs,
                    max_new_tokens=16384,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.9,
                    pad_token_id=local_tokenizer.pad_token_id
                )

                # Process batch output
                batch_results = []
                for batch_idx, (idx, question) in enumerate(batch_data):
                    # Extract the newly generated tokens
                    output_ids = generated_ids[batch_idx][len(model_inputs.input_ids[batch_idx]):].tolist()

                    try:
                        index = len(output_ids) - output_ids[::-1].index(151668)
                    except ValueError:
                        index = 0

                    content = local_tokenizer.decode(
                        output_ids[index:],
                        skip_special_tokens=True
                    ).strip("\n")

                    # Sentence split
                    sentences = nlp_split(content)

                    result = {
                        'id': idx,
                        'question': question,
                        'split_sentences': [sent.text for sent in sentences.sentences]
                    }
                    batch_results.append(result)

                # Batch write results
                with file_lock:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        for result in batch_results:
                            json_line = json.dumps(result, ensure_ascii=False)
                            f.write(json_line + '\n')
                        f.flush()

            except Exception as e:
                print(f"Error processing batch on GPU {gpu_id}: {str(e)}")
                # Write error results
                with file_lock:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        for idx, question in batch_data:
                            error_result = {
                                'id': idx,
                                'question': question,
                                'split_sentences': [],
                                'error': str(e)
                            }
                            json_line = json.dumps(error_result, ensure_ascii=False)
                            f.write(json_line + '\n')
                        f.flush()

            finally:
                # Update progress bar
                pbar.update(len(batch_data))
                # Clear cache
                torch.cuda.empty_cache()
                # Mark tasks as done
                for _ in range(len(batch_data)):
                    task_queue.task_done()

    # Create and start worker threads
    threads = []
    for gpu_id in range(gpu_num):
        thread = threading.Thread(
            target=worker,
            args=(gpu_id,),
            name=f"GPU-Worker-{gpu_id}"
        )
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Close progress bar
    pbar.close()

    print(f"\nProcessing complete! Results saved to: {output_file}")
    return output_file

if __name__ == "__main__":
    # Set stanza log level to WARNING/load sentence splitting model
    stanza.logging.getLogger().setLevel(logging.WARNING)
    nlp_split = stanza.Pipeline(
        lang='zh',
        processors='tokenize',
        download_method=None,
        logging_level='WARNING',
        verbose=False
    )
    model_folder = "Qwen"
    model_name = "DeepSeek-R1-Distill-Qwen-7B"
    # Load model
    model_path = f"../../model/{model_folder}/{model_name}"
    output_path = f"./output/{model_name}"
    os.makedirs(output_path, exist_ok=True)

    # Set random seed
    random.seed(42)
    gpu_num = 2  # Get available GPU count
    batch_size = 64  # Batch size per GPU

    # Load dataset
    dataset = load_from_disk("../../dataset/dart-math-pool-math")
    train_df = dataset['train'].to_pandas()

    # Parse query_metadata field to get level
    train_df['level'] = train_df['query_metadata'].apply(lambda x: x['level'] if isinstance(x, dict) else None)

    # Process each level separately
    for level in range(1, 6):
        print(f"\nProcessing Level {level} data...")
        level_df = train_df[train_df['level'] == level]

        if len(level_df) > 1000:
            level_df = level_df.sample(n=1000, random_state=42)

        print(f"Level {level} data count: {len(level_df)}")

        output_file = process_dataset(
            dataset=level_df,
            output_file=f"{output_path}/dart_math_level_{level}_response_split.jsonl",
            gpu_num=gpu_num,
            batch_size=batch_size
        )
