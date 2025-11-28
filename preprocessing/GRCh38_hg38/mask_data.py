"""
Script to mask data using the nucleotide transformer tokenizer 
following approach in paper 
"""

# A.5.2:
#   Within a nucleotide sequence s of interest, we masked tokens using one of
#   two strategies (i.e. we replaced the tokens at these positions with the mask token [MASK]). We either
#   masked the central token of the sequence only, or we masked randomly 15% of the tokens within
#   the sequence.

# The Nucleotide Transformer model learned to reconstruct human genetic variants:
#   Specifically, we divided the human reference genome into non-overlapping 6kb sequences,
#   tokenized each sequence into 6-mers, randomly masked a certain number of tokens, and then calculated
#   the proportion of tokens that were accurately reconstructed

from transformers import AutoTokenizer
import torch
import math
import random

from datasets import Dataset, load_dataset, concatenate_datasets
import os
from tqdm import tqdm

def tokenize_6kb_batch(seq, tokenizer):
    """
    Tokenize substring of 6 consecutive nucleotides to evaluate 
    reconstruction of masked tokens
    return tokenized 6-mer sequence
    """
    encoded_seq = tokenizer(
        seq,
        add_special_tokens=False, # pure 6-mer sequences only
        padding=False,
        truncation=True,
        return_tensors=None,
    )
    return encoded_seq["input_ids"]


def mask_central_token(tokens, mask_id):
    """
    Given tokenized 6-mer sequence mask central token
    return masked seq and position 
    """
    center = len(tokens) // 2
    masked_seq = tokens.copy()
    masked_seq[center] = mask_id
    position_masked = [center]    # return [] for consitency with func below
    return masked_seq, position_masked


def mask_random_tokens(tokens, mask_id):
    """
    Given tokenized 6-mer sequence mask 15% randomly
    return masked seq and positions
    """ 
    indices = list(range(len(tokens)))
    n_mask = max(1, round(0.15 * len(tokens)))
    positions_masked = random.sample(indices, n_mask)

    masked_seq = tokens.copy()
    for i in positions_masked:
        masked_seq[i] = mask_id
    return masked_seq, positions_masked


def mask_human_reference_genome():
    """
    load data and mask in chunks to avoid needing a lot of memory
    """
    print("Loading data...")
    data_path = "artifacts/datasets/human_reference_genome/6kbp/0.0.0/1dda7e5c35a9246c397dab1c8343852df92847b32619e1a72261d5cee3b2f919"

    splits = {
        'train': os.path.join(data_path, "human_reference_genome-train.arrow"),
        'val':   os.path.join(data_path, "human_reference_genome-validation.arrow"),
        'test':  os.path.join(data_path, "human_reference_genome-test.arrow")
    }
    datasets = {}

    for split_name, split_path in splits.items():
        datasets[split_name] = Dataset.from_file(split_path)
        print(f"  Loaded {split_name}: {len(datasets[split_name])} sequences")

    print("Data loaded.")
    print("Loading tokenizer...")

    # be sure to download tokenizer using download_nt.sh in /nucleotide-transformer 
    model_path = "../../nucleotide-transformer/artifacts/nucleotide-transformer-500m-human-ref/"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    mask_id = tokenizer.mask_token_id
    print("Tokenizer loaded.")

    chunk_size = 50000
    print("Masking human reference genome...")
    for scheme_name, mask_func in [("central", mask_central_token), ("random", mask_random_tokens)]:
        print(f"\nMasking {scheme_name} token(s)...")

        for split_name, data in datasets.items():
            total_examples = len(data)
            all_chunks = []

            for start_idx in tqdm(range(0, total_examples, chunk_size), desc=f"Masking chunks"):
                end_idx = min(start_idx + chunk_size, total_examples)
                chunk_data = []

                for i in range(start_idx, end_idx):
                    seq = data[i]["sequence"]
                    tokens = tokenize_6kb_batch(seq, tokenizer)
                    masked_seq, positions_masked = mask_func(tokens, mask_id)

                    chunk_data.append({
                        "original_tokens": tokens,
                        "masked_tokens": masked_seq,
                        "positions_masked": positions_masked
                    }) # Each entry will be a dict with keys: original_tokens, masked_tokens, positions_masked

                chunk_ds = Dataset.from_list(chunk_data)
                all_chunks.append(chunk_ds)
                del chunk_data # Clear memory

            print(f"Concatenating {split_name}, {scheme_name} chunks...")
            final_ds = concatenate_datasets(all_chunks)
            output_path = f"artifacts/datasets/human_reference_genome/6kbp/masked/masked_{split_name}_{scheme_name}_ds"
            print(f"Saving to {output_path}...")

            final_ds.save_to_disk(output_path)
            del all_chunks, final_ds

    print("Done.")


if __name__ == "__main__":
    mask_human_reference_genome()
