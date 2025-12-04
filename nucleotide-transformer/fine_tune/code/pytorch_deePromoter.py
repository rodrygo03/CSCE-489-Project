#!/usr/bin/env python3
"""
PyTorch Dataset for promoter vs non-promoter classification with Nucleotide Transformer.

This module provides dataset classes for loading, tokenizing, and batching
promoter sequences for fine-tuning the Nucleotide Transformer model.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from sklearn.model_selection import train_test_split
import numpy as np


class PromoterDataset(Dataset):
    """
    Dataset for promoter vs non-promoter sequences.
    
    Loads sequences from TSV file and tokenizes them using the Nucleotide Transformer
    tokenizer (6-mer tokenization).
    """
    
    def __init__(
        self,
        sequences: List[str],
        labels: List[int],
        tokenizer_name: str = "InstaDeepAI/nucleotide-transformer-500m-human-ref",
        max_length: int = 512,
    ):
        """
        Args:
            sequences: List of DNA sequences (300bp each)
            labels: List of binary labels (1=promoter, 0=non-promoter)
            tokenizer_name: HuggingFace model name for tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
        self.max_length = max_length
        
        assert len(sequences) == len(labels), "Sequences and labels must have same length"
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single tokenized sequence with its label.
        
        Returns:
            Dictionary with keys:
                - input_ids: Token IDs
                - attention_mask: Attention mask
                - labels: Binary label (0 or 1)
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Tokenize the sequence
        # NT tokenizer expects sequences in the format that they were pretrained on
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Squeeze to remove batch dimension added by return_tensors='pt'
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_promoter_data(
    data_path: Path,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_seed: int = 42,
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """
    Load promoter dataset from TSV and split into train/val/test sets.
    
    Args:
        data_path: Path to TSV file with columns 'sequence' and 'label'
        test_size: Proportion of data for test set (default: 0.2)
        val_size: Proportion of remaining data for validation set (default: 0.1)
                  Set to 0.0 to skip validation split (for cross-validation)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_seqs, train_labels, val_seqs, val_labels, test_seqs, test_labels)
        If val_size=0.0, val_seqs and val_labels will be empty lists.
    """
    # Load data
    df = pd.read_csv(data_path, sep='\t')
    
    print(f"Loaded {len(df)} sequences from {data_path}")
    print(f"  Positives: {df['label'].sum()} ({df['label'].sum()/len(df)*100:.1f}%)")
    print(f"  Negatives: {(df['label']==0).sum()} ({(df['label']==0).sum()/len(df)*100:.1f}%)")
    
    sequences = df['sequence'].tolist()
    labels = df['label'].tolist()
    
    # First split: separate test set
    train_val_seqs, test_seqs, train_val_labels, test_labels = train_test_split(
        sequences, labels,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels  # Maintain class balance
    )
    
    # Second split: separate validation from training (if val_size > 0)
    if val_size > 0:
        # val_size is relative to the remaining data
        val_size_adjusted = val_size / (1 - test_size)
        train_seqs, val_seqs, train_labels, val_labels = train_test_split(
            train_val_seqs, train_val_labels,
            test_size=val_size_adjusted,
            random_state=random_seed,
            stratify=train_val_labels
        )
    else:
        # No validation split - use all remaining data for training
        train_seqs = train_val_seqs
        train_labels = train_val_labels
        val_seqs = []
        val_labels = []
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_seqs)} ({len(train_seqs)/len(df)*100:.1f}%)")
    print(f"    Positives: {sum(train_labels)} ({sum(train_labels)/len(train_labels)*100:.1f}%)")
    if val_size > 0:
        print(f"  Val:   {len(val_seqs)} ({len(val_seqs)/len(df)*100:.1f}%)")
        print(f"    Positives: {sum(val_labels)} ({sum(val_labels)/len(val_labels)*100:.1f}%)")
    else:
        print(f"  Val:   0 (skipped for cross-validation)")
    print(f"  Test:  {len(test_seqs)} ({len(test_seqs)/len(df)*100:.1f}%)")
    print(f"    Positives: {sum(test_labels)} ({sum(test_labels)/len(test_labels)*100:.1f}%)")
    
    return train_seqs, train_labels, val_seqs, val_labels, test_seqs, test_labels


def create_dataloaders(
    data_path: Path,
    tokenizer_name: str = "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    batch_size: int = 16,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_seed: int = 42,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders for promoter classification.
    
    Args:
        data_path: Path to TSV file with promoter data
        tokenizer_name: HuggingFace model name for tokenizer
        batch_size: Batch size for DataLoaders
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining data)
        random_seed: Random seed for reproducibility
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load and split data
    train_seqs, train_labels, val_seqs, val_labels, test_seqs, test_labels = load_promoter_data(
        data_path=data_path,
        test_size=test_size,
        val_size=val_size,
        random_seed=random_seed,
    )
    
    # Create datasets
    train_dataset = PromoterDataset(train_seqs, train_labels, tokenizer_name=tokenizer_name)
    val_dataset = PromoterDataset(val_seqs, val_labels, tokenizer_name=tokenizer_name)
    test_dataset = PromoterDataset(test_seqs, test_labels, tokenizer_name=tokenizer_name)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


def compute_class_weights(labels: List[int]) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        labels: List of binary labels
        
    Returns:
        Tensor of class weights [weight_for_0, weight_for_1]
    """
    labels_array = np.array(labels)
    n_samples = len(labels_array)
    n_classes = 2
    
    # Count samples per class
    class_counts = np.bincount(labels_array)
    
    # Compute weights: n_samples / (n_classes * class_count)
    weights = n_samples / (n_classes * class_counts)
    
    return torch.tensor(weights, dtype=torch.float32)


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test PromoterDataset")
    parser.add_argument("--data-path", type=Path, required=True,
                        help="Path to promoter TSV file")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for testing")
    args = parser.parse_args()
    
    print("=" * 70)
    print("Testing PromoterDataset")
    print("=" * 70)
    print()
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
    )
    
    print("\n" + "=" * 70)
    print("DataLoader Statistics")
    print("=" * 70)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print()
    
    # Test loading a batch
    print("Testing batch loading...")
    batch = next(iter(train_loader))
    print(f"\nBatch contents:")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
    print(f"  labels shape: {batch['labels'].shape}")
    print(f"  labels: {batch['labels'].tolist()}")
    print()
    
    # Compute class weights
    all_labels = []
    for batch in train_loader:
        all_labels.extend(batch['labels'].tolist())
    
    weights = compute_class_weights(all_labels)
    print(f"Class weights for training set:")
    print(f"  Class 0 (non-promoter): {weights[0]:.4f}")
    print(f"  Class 1 (promoter): {weights[1]:.4f}")
    print()
    
    print("=" * 70)
    print("Dataset test complete!")
    print("=" * 70)