"""
Evaluate the best Hybrid Infini-gram + Neural LM configuration on the test set.
Uses the best hyperparameters found during validation tuning.
"""

import numpy as np
from datasets import load_from_disk
from tqdm import tqdm
import json
import argparse
from pathlib import Path
import math
from typing import Optional

from hybrid_model import HybridInfiniGramNeuralLM


def avg_loss_per_sequence(model, unmasked_ids, masked_ids, positions_masked):
    """
    Compute average loss for a single sequence using hybrid model.
    """
    if len(positions_masked) == 0:
        return None
    
    total_nll = 0.0
    for pos in positions_masked:
        true_token = unmasked_ids[pos]
        
        # Get hybrid model's negative log-likelihood
        nll = model.hybrid_token_nll(
            unmasked_ids=unmasked_ids,
            masked_ids=masked_ids,
            pos=pos,
            gold_tid=true_token,
            eps=1e-12
        )
        
        total_nll += nll
    
    # Average over masked positions
    return total_nll / len(positions_masked)


def avg_accuracy_per_sequence(model, unmasked_ids, masked_ids, positions_masked):
    """
    Compute accuracy for a single sequence using hybrid model.
    Following paper
    """
    if len(positions_masked) == 0:
        return None
    
    correct = 0
    for pos in positions_masked:
        true_token = unmasked_ids[pos]
        predicted_token = model.predict_hybrid_token(
            unmasked_ids=unmasked_ids,
            masked_ids=masked_ids,
            pos=pos
        )
        
        if predicted_token == true_token:
            correct += 1
    
    return correct / len(positions_masked)


def compute_perplexity_rives(avg_loss_value):
    """
    Compute perplexity following Rives et al.
    """
    return 2 ** avg_loss_value


def evaluate_on_dataset(model, dataset, max_sequences=None):
    """
    Evaluate hybrid model on masked dataset.
    """
    losses = []
    accuracies = []
    perplexities = []
    
    num_sequences = len(dataset) if max_sequences is None else min(max_sequences, len(dataset))
    
    for idx in tqdm(range(num_sequences), desc="Evaluating sequences"):
        seq = dataset[idx]
        
        unmasked_ids = seq["original_tokens"]
        masked_ids = seq["masked_tokens"]
        positions_masked = seq["positions_masked"]
        
        if len(positions_masked) == 0:
            continue

        seq_loss = avg_loss_per_sequence(
            model, unmasked_ids, masked_ids, positions_masked
        )
        seq_accuracy = avg_accuracy_per_sequence(
            model, unmasked_ids, masked_ids, positions_masked
        )
        
        if seq_loss is None or seq_accuracy is None:
            continue
        
        seq_perplexity = compute_perplexity_rives(seq_loss)
        
        losses.append(seq_loss)
        accuracies.append(seq_accuracy)
        perplexities.append(seq_perplexity)
    
    losses = np.array(losses)
    accuracies = np.array(accuracies)
    perplexities = np.array(perplexities)
    
    results = {
        'num_sequences': len(losses),
        'losses': losses,
        'accuracies': accuracies,
        'perplexities': perplexities,
        'mean_loss': float(losses.mean()),
        'std_loss': float(losses.std()),
        'mean_accuracy': float(accuracies.mean()),
        'std_accuracy': float(accuracies.std()),
        'median_accuracy': float(np.median(accuracies)),
        'mean_perplexity': float(perplexities.mean()),
        'std_perplexity': float(perplexities.std()),
        'median_perplexity': float(np.median(perplexities)),
    }
    
    return results


def evaluate_test_set(
    tok_path: str,
    nt_model_path: str,
    fwd_idx_path: str,
    bwd_idx_path: str,
    test_dataset_path: str,
    max_sequences: Optional[int] = None,
    output_dir: str = "./test_results",
    device: str = "cuda"
):

    max_support_fwd = 8
    max_support_bwd = 8
    lam_nt_sparse = 0.7
    lam_nt_dense = 0.5
    cnt_thresh = 7
    
    print(f"Loading test dataset from {test_dataset_path}...")
    test_dataset = load_from_disk(test_dataset_path)
    print(f"Test dataset loaded: {len(test_dataset)} sequences")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("EVALUATING BEST MODEL ON TEST SET")
    print(f"{'='*80}")
    print(f"\nBest hyperparameters:")
    print(f"  lam_nt_sparse: {lam_nt_sparse:.2f}")
    print(f"  lam_nt_dense: {lam_nt_dense:.2f}")
    print(f"  cnt_thresh: {cnt_thresh}")
    print(f"  max_support_fwd: {max_support_fwd}")
    print(f"  max_support_bwd: {max_support_bwd}")
    print()
    
    print("Initializing hybrid model with best hyperparameters...")
    model = HybridInfiniGramNeuralLM(
        tok_path=tok_path,
        nt_model_path=nt_model_path,
        fwd_idx_path=fwd_idx_path,
        bwd_idx_path=bwd_idx_path,
        max_support_fwd=max_support_fwd,
        max_support_bwd=max_support_bwd,
        lam_nt_sparse=lam_nt_sparse,
        lam_nt_dense=lam_nt_dense,
        cnt_thresh=cnt_thresh,
        device=device
    )
    print("Model initialized.")
    
    print("\nEvaluating on test set...")
    results = evaluate_on_dataset(
        model=model,
        dataset=test_dataset,
        max_sequences=max_sequences
    )
    
    print(f"\n{'='*80}")
    print("TEST SET RESULTS")
    print(f"{'='*80}")
    print(f"\nNumber of sequences: {results['num_sequences']}")
    print(f"\nLoss:")
    print(f"  Mean: {results['mean_loss']:.4f} ± {results['std_loss']:.4f}")
    print(f"\nAccuracy:")
    print(f"  Mean: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    print(f"  Median: {results['median_accuracy']:.4f}")
    print(f"\nPerplexity (Rives):")
    print(f"  Mean: {results['mean_perplexity']:.4f} ± {results['std_perplexity']:.4f}")
    print(f"  Median: {results['median_perplexity']:.4f}")
    
    print(f"\nSaving results to {output_dir}...")
    
    np.savez(
        output_path / "test_detailed_metrics.npz",
        losses=results['losses'],
        accuracies=results['accuracies'],
        perplexities=results['perplexities']
    )

    summary = {
        'hyperparameters': {
            'lam_nt_sparse': lam_nt_sparse,
            'lam_nt_dense': lam_nt_dense,
            'cnt_thresh': cnt_thresh,
            'max_support_fwd': max_support_fwd,
            'max_support_bwd': max_support_bwd,
        },
        'test_results': {
            'num_sequences': results['num_sequences'],
            'mean_loss': results['mean_loss'],
            'std_loss': results['std_loss'],
            'mean_accuracy': results['mean_accuracy'],
            'std_accuracy': results['std_accuracy'],
            'median_accuracy': results['median_accuracy'],
            'mean_perplexity': results['mean_perplexity'],
            'std_perplexity': results['std_perplexity'],
            'median_perplexity': results['median_perplexity'],
        }
    }
    
    summary_path = output_path / "test_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved:")
    print(f"  Detailed metrics: {output_path / 'test_detailed_metrics.npz'}")
    print(f"  Summary: {summary_path}")
    print(f"\n{'='*80}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate best Hybrid Infini-gram + Neural LM on test set"
    )
    
    parser.add_argument(
        "--tok_path",
        type=str,
        required=True,
        help="Path to tokenizer"
    )
    parser.add_argument(
        "--nt_model_path",
        type=str,
        required=True,
        help="Path to Nucleotide Transformer model"
    )
    parser.add_argument(
        "--fwd_idx_path",
        type=str,
        required=True,
        help="Path to forward Infini-gram index"
    )
    parser.add_argument(
        "--bwd_idx_path",
        type=str,
        required=True,
        help="Path to backward Infini-gram index"
    )
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        required=True,
        help="Path to test dataset"
    )
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=None,
        help="Maximum number of sequences to evaluate (default: all)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_results",
        help="Directory to save results (default: ./test_results)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run models on (default: cuda)"
    )
    
    args = parser.parse_args()
    
    results = evaluate_test_set(
        tok_path=args.tok_path,
        nt_model_path=args.nt_model_path,
        fwd_idx_path=args.fwd_idx_path,
        bwd_idx_path=args.bwd_idx_path,
        test_dataset_path=args.test_dataset_path,
        max_sequences=args.max_sequences,
        output_dir=args.output_dir,
        device=args.device
    )

