#!/usr/bin/env python3
"""
Preprocess DeePromoter human promoter sequences for Nucleotide Transformer fine-tuning.

This script:
1. Loads positive TATA and nonTATA promoter sequences (300bp each)
2. Generates negative sequences using NT paper's 20-part shuffling strategy
3. Creates two datasets:
   - promoter vs non-promoter (binary classification)
   - TATA vs nonTATA (promoter subtype classification)
"""

from pathlib import Path
import random
from typing import List
import argparse


def load_txt_sequences(path: Path) -> List[str]:
    """Load sequences from a text file (one sequence per line)."""
    with path.open() as f:
        sequences = [line.strip().upper() for line in f if line.strip()]
    return sequences


def make_nt_negative(seq: str, n_parts: int = 20, n_shuffle: int = 12, seed: int = None) -> str:
    """
    Generate a negative (non-promoter) sequence using NT paper's shuffling strategy.
    
    Split sequence into n_parts subsequences, randomly select n_shuffle of them
    to shuffle among themselves, leaving the rest in place.
    
    Args:
        seq: Input DNA sequence (must be divisible by n_parts)
        n_parts: Number of parts to split sequence into (default: 20)
        n_shuffle: Number of parts to shuffle (default: 12)
        seed: Random seed for reproducibility (optional)
        
    Returns:
        Shuffled sequence of the same length
    """
    if seed is not None:
        random.seed(seed)
    
    assert len(seq) % n_parts == 0, f"Sequence length {len(seq)} must be divisible by {n_parts}"
    
    part_len = len(seq) // n_parts
    
    # Split into parts
    parts = [seq[i * part_len:(i + 1) * part_len] for i in range(n_parts)]
    
    # Randomly select which parts to shuffle
    idx = list(range(n_parts))
    random.shuffle(idx)
    
    shuffle_idx = idx[:n_shuffle]
    keep_idx = idx[n_shuffle:]
    
    # Get the segments to shuffle and shuffle them
    segments = [parts[i] for i in shuffle_idx]
    random.shuffle(segments)
    
    # Build new sequence: copy original, then overwrite shuffled positions
    new_parts = parts[:]
    for i, seg in zip(shuffle_idx, segments):
        new_parts[i] = seg
    
    # keep_idx parts remain in their original positions (no change needed)
    
    return "".join(new_parts)


def main():
    parser = argparse.ArgumentParser(description="Preprocess DeePromoter datasets for NT fine-tuning")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data",
                        help="Directory containing input data files")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--n-parts", type=int, default=20,
                        help="Number of parts to split sequences into")
    parser.add_argument("--n-shuffle", type=int, default=12,
                        help="Number of parts to shuffle")
    args = parser.parse_args()
    
    # Set global random seed
    random.seed(args.seed)
    
    DATA_DIR = args.data_dir
    
    # Input files
    tata_pos_path = DATA_DIR / "hs_pos_TATA.txt"
    non_tata_pos_path = DATA_DIR / "hs_pos_nonTATA.txt"
    
    print("=" * 70)
    print("DeePromoter Dataset Preprocessing")
    print("=" * 70)
    print(f"Data directory: {DATA_DIR}")
    print(f"Random seed: {args.seed}")
    print(f"Negative generation: {args.n_parts} parts, shuffle {args.n_shuffle}")
    print()
    
    # Check input files exist
    if not tata_pos_path.exists():
        raise FileNotFoundError(f"TATA promoter file not found: {tata_pos_path}")
    if not non_tata_pos_path.exists():
        raise FileNotFoundError(f"nonTATA promoter file not found: {non_tata_pos_path}")
    
    # Load positive sequences
    print("Loading positive sequences...")
    tata_pos = load_txt_sequences(tata_pos_path)
    non_tata_pos = load_txt_sequences(non_tata_pos_path)
    
    print(f"  TATA promoters: {len(tata_pos)}")
    print(f"  nonTATA promoters: {len(non_tata_pos)}")
    print(f"  Total positives: {len(tata_pos) + len(non_tata_pos)}")
    print()
    
    # Sanity checks
    print("Validating sequence lengths...")
    assert all(len(s) == 300 for s in tata_pos), "Not all TATA sequences are 300bp"
    assert all(len(s) == 300 for s in non_tata_pos), "Not all nonTATA sequences are 300bp"
    print("  ✓ All sequences are 300bp")
    print()
    
    # Validate sequences contain only valid nucleotides
    valid_nucleotides = set("ACGTN")
    for i, seq in enumerate(tata_pos):
        if not set(seq).issubset(valid_nucleotides):
            print(f"  Warning: TATA sequence {i} contains invalid nucleotides: {set(seq) - valid_nucleotides}")
    for i, seq in enumerate(non_tata_pos):
        if not set(seq).issubset(valid_nucleotides):
            print(f"  Warning: nonTATA sequence {i} contains invalid nucleotides: {set(seq) - valid_nucleotides}")
    
    # Generate negative sequences
    print("Generating negative sequences using NT-style shuffling...")
    tata_neg = [make_nt_negative(s, n_parts=args.n_parts, n_shuffle=args.n_shuffle) 
                for s in tata_pos]
    non_tata_neg = [make_nt_negative(s, n_parts=args.n_parts, n_shuffle=args.n_shuffle) 
                    for s in non_tata_pos]
    
    print(f"  TATA negatives: {len(tata_neg)}")
    print(f"  nonTATA negatives: {len(non_tata_neg)}")
    print(f"  Total negatives: {len(tata_neg) + len(non_tata_neg)}")
    print()
    
    # Sanity check negatives
    assert all(len(s) == 300 for s in tata_neg), "Not all TATA negatives are 300bp"
    assert all(len(s) == 300 for s in non_tata_neg), "Not all nonTATA negatives are 300bp"
    print("  ✓ All negative sequences are 300bp")
    print()
    
    # Build promoter vs non-promoter dataset
    print("Building promoter vs non-promoter dataset...")
    all_sequences = tata_pos + non_tata_pos + tata_neg + non_tata_neg
    all_labels = ([1] * (len(tata_pos) + len(non_tata_pos)) + 
                  [0] * (len(tata_neg) + len(non_tata_neg)))
    
    print(f"  Total sequences: {len(all_sequences)}")
    print(f"  Positive examples: {sum(all_labels)}")
    print(f"  Negative examples: {len(all_labels) - sum(all_labels)}")
    print()
    
    # Write promoter vs non-promoter dataset
    out_path = DATA_DIR / "nt_promoter_dataset.tsv"
    print(f"Writing to {out_path}...")
    with out_path.open("w") as f:
        f.write("sequence\tlabel\n")  # Header
        for seq, label in zip(all_sequences, all_labels):
            f.write(f"{seq}\t{label}\n")
    print(f"  ✓ Wrote {len(all_sequences)} sequences")
    print()
    
    # Build TATA vs nonTATA subtype dataset (positives only)
    print("Building TATA vs nonTATA subtype dataset...")
    subtype_sequences = tata_pos + non_tata_pos
    subtype_labels = [1] * len(tata_pos) + [0] * len(non_tata_pos)  # 1=TATA, 0=nonTATA
    
    print(f"  Total sequences: {len(subtype_sequences)}")
    print(f"  TATA examples (label=1): {sum(subtype_labels)}")
    print(f"  nonTATA examples (label=0): {len(subtype_labels) - sum(subtype_labels)}")
    print()
    
    # Write TATA vs nonTATA dataset
    subtype_path = DATA_DIR / "nt_promoter_subtype_tata_vs_non.tsv"
    print(f"Writing to {subtype_path}...")
    with subtype_path.open("w") as f:
        f.write("sequence\tlabel\n")  # Header
        for seq, label in zip(subtype_sequences, subtype_labels):
            f.write(f"{seq}\t{label}\n")
    print(f"  ✓ Wrote {len(subtype_sequences)} sequences")
    print()
    
    # Summary
    print("=" * 70)
    print("Preprocessing complete!")
    print("=" * 70)
    print(f"Output files:")
    print(f"  1. {out_path.name}")
    print(f"     - Task: Promoter vs non-promoter classification")
    print(f"     - Sequences: {len(all_sequences)}")
    print(f"     - Balance: {sum(all_labels)}/{len(all_labels) - sum(all_labels)} (pos/neg)")
    print()
    print(f"  2. {subtype_path.name}")
    print(f"     - Task: TATA vs nonTATA subtype classification")
    print(f"     - Sequences: {len(subtype_sequences)}")
    print(f"     - Balance: {sum(subtype_labels)}/{len(subtype_labels) - sum(subtype_labels)} (TATA/nonTATA)")
    print("=" * 70)


if __name__ == "__main__":
    main()