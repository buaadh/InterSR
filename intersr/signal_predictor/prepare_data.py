import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
from intersr.signal_predictor.architecture import SentenceSplitDataset, SignalDataset
import json
import glob
import sys
import os

def prepare_sentence_split_and_signal_data(files, tokenizer, model, device):
    token_ids_list = []
    labels_list = []
    sentence_split_hidden_states_list = []
    sentence_split_labels_list = []
    signal_hidden_states_list = []
    signal_labels_list = []

    model.eval()
    for file_path in files:
        # Count file lines first
        with open(file_path, "r") as f:
            lines = f.readlines()
        for line in tqdm(lines, desc=f"Processing {os.path.basename(file_path)}", file=sys.stdout, dynamic_ncols=True):
            data = json.loads(line)
            # tokenize question
            messages = [{"role": "user", "content": data["question"]}]
            question_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            ) + '\n</think>'
            question_tokens = tokenizer(question_text, return_tensors="pt", add_special_tokens=False)
            question_ids = question_tokens.input_ids[0]
            q_len = len(question_ids)

            # tokenize partial response
            all_response_ids = []
            all_response_labels = []
            for i, response in enumerate(data['partial_response']):
                response_tokens = tokenizer(response, return_tensors="pt", add_special_tokens=False)
                response_ids = response_tokens.input_ids[0]
                all_response_ids.append(response_ids)
                # sentence split label: last one is 1, others are 0
                labels = [0] * (len(response_ids) - 1) + [1]
                all_response_labels.append(labels)

            # Concatenate all tokens and labels
            input_ids = torch.cat([question_ids] + all_response_ids)
            labels = [-1] * (q_len)
            for l in all_response_labels:
                labels += l

            # Get hidden state using model
            input_ids_tensor = input_ids.unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_ids_tensor, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1][0].cpu()  # [seq_len, hidden_dim]

            # sentence split: only keep hidden states and labels where label is not -1
            labels_tensor = torch.tensor(labels)
            mask_split = labels_tensor != -1
            if mask_split.sum() > 0:
                for idx in torch.where(mask_split)[0]:
                    sentence_split_labels_list.append(labels_tensor[idx].unsqueeze(0))
                    sentence_split_hidden_states_list.append(hidden_states[idx].unsqueeze(0))
            # signal: only keep the last token's hidden state and signal label
            signal_hidden_states_list.append(hidden_states[-1].unsqueeze(0))
            signal_labels_list.append(torch.tensor([data['uncertainty']]))


    sentence_split_hidden_states_all = torch.cat(sentence_split_hidden_states_list, dim=0)
    sentence_split_labels_all = torch.cat(sentence_split_labels_list, dim=0)
    signal_hidden_states_all = torch.cat(signal_hidden_states_list, dim=0)
    signal_labels_all = torch.cat(signal_labels_list, dim=0)


    return sentence_split_hidden_states_all, sentence_split_labels_all, signal_hidden_states_all, signal_labels_all

if __name__ == "__main__":
    model_folder = "Qwen"
    model_names = ["DeepSeek-R1-Distill-Qwen-14B", "DeepSeek-R1-Distill-Qwen-7B", "DeepSeek-R1-Distill-Qwen-1.5B"]
    for model_name in model_names:
        model_path = f"../../model/{model_folder}/{model_name}"
        input_dir = f"../signal_predictor/output/{model_name}"
        output_dir = f"./output/{model_name}"
        os.makedirs(output_dir, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")

        # Create training set
        files = glob.glob(os.path.join(input_dir, f"*_train_*.jsonl"))

        sentence_split_hidden_states_all, sentence_split_labels_all, signal_hidden_states_all, signal_labels_all = prepare_sentence_split_and_signal_data(
            files, tokenizer, model, device
        )

        # Save SentenceSplitDataset
        split_dataset = SentenceSplitDataset(sentence_split_hidden_states_all, sentence_split_labels_all)
        with open(os.path.join(output_dir, f"sentence_split_dataset_train.pkl"), "wb") as f:
            pickle.dump(split_dataset, f)

        # Save SignalDataset
        signal_dataset = SignalDataset(signal_hidden_states_all, signal_labels_all)
        with open(os.path.join(output_dir, f"signal_dataset_train.pkl"), "wb") as f:
            pickle.dump(signal_dataset, f)

        print(f"SentenceSplitDataset Train size: {len(split_dataset)}")
        print(f"SignalDataset Train size: {len(signal_dataset)}")

        # Create test set
        files = glob.glob(os.path.join(input_dir, f"*_test_*.jsonl"))

        sentence_split_hidden_states_all, sentence_split_labels_all, signal_hidden_states_all, signal_labels_all = prepare_sentence_split_and_signal_data(
            files, tokenizer, model, device
        )

        # Save SentenceSplitDataset
        split_dataset = SentenceSplitDataset(sentence_split_hidden_states_all, sentence_split_labels_all)
        with open(os.path.join(output_dir, f"sentence_split_dataset_test.pkl"), "wb") as f:
            pickle.dump(split_dataset, f)

        # Save SignalDataset
        signal_dataset = SignalDataset(signal_hidden_states_all, signal_labels_all)
        with open(os.path.join(output_dir, f"signal_dataset_test.pkl"), "wb") as f:
            pickle.dump(signal_dataset, f)

        print(f"SentenceSplitDataset Test size: {len(split_dataset)}")
        print(f"SignalDataset Test size: {len(signal_dataset)}")
