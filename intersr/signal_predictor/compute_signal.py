"""
Signal Computation Module

Computes semantic uncertainty for model responses using embedding-based clustering.
The signal is used as the training target for the SignalPredictorMLP module.
"""

import json
import numpy as np
from tqdm import tqdm
import os
import random
import argparse
from sentence_transformers import SentenceTransformer


def cal_embedding(texts, embedding_model):
    """Compute embeddings for a list of texts."""
    valid_texts = [text for text in texts if text != '']
    if not valid_texts:
        return None
    return embedding_model.encode(valid_texts)


def cal_uncertainty(texts, embedding_model, k=0.5, sample_num=8):
    """
    Calculate semantic uncertainty via embedding-based clustering.

    Args:
        texts: List of response texts to compute uncertainty over.
        embedding_model: SentenceTransformer model for encoding texts.
        k: Distance threshold for cluster merging.
        sample_num: Number of samples to use for uncertainty estimation.

    Returns:
        Normalized entropy of the cluster distribution, or None if insufficient data.
    """
    embedding = cal_embedding(texts, embedding_model)
    if embedding is None:
        return None
    if len(embedding) < sample_num:
        return None
    embedding = embedding[:sample_num]
    clusters = []
    cluster_embeddings = []
    if len(embedding) > 0:
        clusters.append(embedding[0])
        cluster_embeddings.append([embedding[0]])
    for emb in embedding[1:]:
        min_dist = float('inf')
        min_cluster_idx = -1
        for i, cluster_center in enumerate(clusters):
            if len(cluster_embeddings[i]) > 1:
                center = np.mean(cluster_embeddings[i], axis=0)
            else:
                center = cluster_center
            dist = np.linalg.norm(emb - center)
            if dist < min_dist:
                min_dist = dist
                min_cluster_idx = i
        if min_dist < k:
            cluster_embeddings[min_cluster_idx].append(emb)
            clusters[min_cluster_idx] = np.mean(cluster_embeddings[min_cluster_idx], axis=0)
        else:
            clusters.append(emb)
            cluster_embeddings.append([emb])
    total_samples = len(embedding)
    frequencies = [len(cluster) / total_samples for cluster in cluster_embeddings]
    entropy = -sum(f * np.log2(f) for f in frequencies)
    normalize_factor = np.log2(sample_num)
    return entropy / normalize_factor


def process_uncertainty(model_name, embedding_model, input_base, output_base):
    """
    Process sampled responses and compute uncertainty labels for training.

    Args:
        model_name: Name of the target model.
        embedding_model: SentenceTransformer model for embeddings.
        input_base: Directory containing sampled response files.
        output_base: Directory to write train/test splits with uncertainty labels.
    """
    os.makedirs(output_base, exist_ok=True)
    for level in range(1, 6):
        input_path = os.path.join(input_base, f'dart_math_level_{level}_response_split_with_samples.jsonl')
        output_train = os.path.join(output_base, f'dart_math_level_{level}_train_with_uncertainty.jsonl')
        output_test = os.path.join(output_base, f'dart_math_level_{level}_test_with_uncertainty.jsonl')
        with open(input_path, 'r') as f:
            data = [json.loads(line) for line in tqdm(f, desc=f"Level {level} load")]
        for item in tqdm(data, desc=f"Level {level} uncertainty"):
            uncertainty = cal_uncertainty(
                item.get('samples', item.get('split_sentences', [])),
                embedding_model
            )
            item['uncertainty'] = float(uncertainty) if uncertainty is not None else None
        data = [item for item in data if item['uncertainty'] is not None and np.isfinite(item['uncertainty'])]
        random.shuffle(data)
        split_idx = int(len(data) * 0.9)
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        with open(output_train, 'w') as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        with open(output_test, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Level {level}: {len(data)} samples ({len(train_data)} train / {len(test_data)} test)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate uncertainty labels from sampled responses.")
    parser.add_argument("--model-name", type=str, default="DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--embedding-model", type=str, default="Qwen/Qwen3-Embedding-0.6B",
                        help="SentenceTransformer model name or path for embeddings.")
    parser.add_argument("--input-dir", type=str, default="../samples/output",
                        help="Base directory containing sampled response files.")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Base directory for output files with uncertainty labels.")
    args = parser.parse_args()

    embedding_model = SentenceTransformer(
        args.embedding_model,
        model_kwargs={"device_map": "auto"},
        tokenizer_kwargs={"padding_side": "left"},
    )
    process_uncertainty(
        args.model_name,
        embedding_model,
        os.path.join(args.input_dir, args.model_name),
        os.path.join(args.output_dir, args.model_name),
    )
