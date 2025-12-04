"""
Hyperparameter tuning script for lambda in Hybrid Infini-gram + Neural LM.
Tunes the interpolation weights (lambda_sparse, lambda_dense) and count threshold
"""

import numpy as np
from datasets import load_from_disk
from tqdm import tqdm
import json
import argparse
from pathlib import Path
import math
import itertools
from typing import Dict, List, Optional, Tuple

from hybrid_model import HybridInfiniGramNeuralLM


def avg_loss_per_sequence(model, unmasked_ids, masked_ids, positions_masked):
    """
    Compute average loss for a single sequence using hybrid model.
    Following Rives et al.: l(θ,s) = (1/|P_masked|) * Σ_{i∈P_masked} -log p(θ, i, s(i))
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
    
    return total_nll / len(positions_masked)


def avg_accuracy_per_sequence(model, unmasked_ids, masked_ids, positions_masked):
    """
    Compute accuracy for a single sequence using hybrid model.
    Following paper: acc(θ,s) = (1/|P_masked|) * Σ_{i∈P_masked} 1(argmax log p(θ, i, tok) = s(i))
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
    Compute perplexity following Rives et al. definition used in Nucleotide Transformer:
    perplexity(θ,s) = 2^l(θ,s)
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
        'mean_accuracy': float(accuracies.mean()),
        'median_accuracy': float(np.median(accuracies)),
        'mean_perplexity': float(perplexities.mean()),
        'median_perplexity': float(np.median(perplexities)),
    }
    
    return results


def tune_lambda(
    tok_path: str,
    nt_model_path: str,
    fwd_idx_path: str,
    bwd_idx_path: str,
    dataset_path: str,
    max_support_fwd: int,
    max_support_bwd: int,
    max_sequences: Optional[int] = None,
    output_dir: str = "./lambda_tuning_results",
    device: str = "cuda"
):
    """
    Grid search over lambda hyperparameters for hybrid model with dynamic lambda selection.
    """

    lam_nt_sparse_values = [0.7, 0.8]  # Higher weight for NT in sparse regions
    lam_nt_dense_values = [0.2, 0.5]   # Lower weight for NT in dense regions
    cnt_thresh_values = [2, 7]                # Count threshold for sparsity

    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    print(f"Dataset loaded: {len(dataset)} sequences")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = []
    
    # Grid search
    total_combinations = (len(lam_nt_sparse_values) * 
                         len(lam_nt_dense_values) * 
                         len(cnt_thresh_values))
    
    print(f"\nTuning over {total_combinations} hyperparameter combinations...")
    print(f"lam_nt_sparse values: {lam_nt_sparse_values}")
    print(f"lam_nt_dense values: {lam_nt_dense_values}")
    print(f"cnt_thresh values: {cnt_thresh_values}")
    print(f"Fixed hyperparameters:")
    print(f"  max_support_fwd: {max_support_fwd}")
    print(f"  max_support_bwd: {max_support_bwd}")
    print()
    
    for lam_sparse, lam_dense, cnt_thresh in itertools.product(
        lam_nt_sparse_values, lam_nt_dense_values, cnt_thresh_values
    ):
        print(f"\n{'='*80}")
        print(f"Testing: lam_nt_sparse={lam_sparse:.2f}, lam_nt_dense={lam_dense:.2f}, cnt_thresh={cnt_thresh}")
        print(f"{'='*80}")
        
        model = HybridInfiniGramNeuralLM(
            tok_path=tok_path,
            nt_model_path=nt_model_path,
            fwd_idx_path=fwd_idx_path,
            bwd_idx_path=bwd_idx_path,
            max_support_fwd=max_support_fwd,
            max_support_bwd=max_support_bwd,
            lam_nt_sparse=lam_sparse,
            lam_nt_dense=lam_dense,
            cnt_thresh=cnt_thresh,
            device=device
        )
        
        results = evaluate_on_dataset(
            model=model,
            dataset=dataset,
            max_sequences=max_sequences
        )

        results['lam_nt_sparse'] = lam_sparse
        results['lam_nt_dense'] = lam_dense
        results['cnt_thresh'] = cnt_thresh
        results['max_support_fwd'] = max_support_fwd
        results['max_support_bwd'] = max_support_bwd

        print(f"\nResults:")
        print(f"  Number of sequences: {results['num_sequences']}")
        print(f"  Mean Loss: {results['mean_loss']:.4f}")
        print(f"  Mean Accuracy: {results['mean_accuracy']:.4f}")
        print(f"  Median Accuracy: {results['median_accuracy']:.4f}")
        print(f"  Mean Perplexity (Rives): {results['mean_perplexity']:.4f}")
        print(f"  Median Perplexity (Rives): {results['median_perplexity']:.4f}")
        
        config_name = f"sparse{lam_sparse:.2f}_dense{lam_dense:.2f}_thresh{cnt_thresh}".replace('.', 'p')
        np.savez(
            output_path / f"{config_name}_detailed.npz",
            losses=results['losses'],
            accuracies=results['accuracies'],
            perplexities=results['perplexities']
        )

        summary = {
            'lam_nt_sparse': lam_sparse,
            'lam_nt_dense': lam_dense,
            'cnt_thresh': cnt_thresh,
            'max_support_fwd': max_support_fwd,
            'max_support_bwd': max_support_bwd,
            'num_sequences': results['num_sequences'],
            'mean_loss': results['mean_loss'],
            'mean_accuracy': results['mean_accuracy'],
            'median_accuracy': results['median_accuracy'],
            'mean_perplexity': results['mean_perplexity'],
            'median_perplexity': results['median_perplexity'],
        }
        all_results.append(summary)
    
    summary_path = output_path / "lambda_tuning_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("LAMBDA TUNING COMPLETE")
    print(f"{'='*80}")
    
    best_by_accuracy = max(all_results, key=lambda x: x['median_accuracy'])
    best_by_perplexity = min(all_results, key=lambda x: x['mean_perplexity'])
    
    print("\nBest configuration by MEDIAN ACCURACY:")
    print(f"  lam_nt_sparse={best_by_accuracy['lam_nt_sparse']:.2f}, "
          f"lam_nt_dense={best_by_accuracy['lam_nt_dense']:.2f}, "
          f"cnt_thresh={best_by_accuracy['cnt_thresh']}")
    print(f"  Median Accuracy: {best_by_accuracy['median_accuracy']:.4f}")
    print(f"  Mean Perplexity: {best_by_accuracy['mean_perplexity']:.4f}")
    
    print("\nBest configuration by MEAN PERPLEXITY:")
    print(f"  lam_nt_sparse={best_by_perplexity['lam_nt_sparse']:.2f}, "
          f"lam_nt_dense={best_by_perplexity['lam_nt_dense']:.2f}, "
          f"cnt_thresh={best_by_perplexity['cnt_thresh']}")
    print(f"  Median Accuracy: {best_by_perplexity['median_accuracy']:.4f}")
    print(f"  Mean Perplexity: {best_by_perplexity['mean_perplexity']:.4f}")
    
    print(f"\nAll results saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune lambda hyperparameters for Hybrid Infini-gram + Neural LM with dynamic lambda selection"
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
        "--dataset_path",
        type=str,
        required=True,
        help="Path to masked dataset"
    )
    parser.add_argument(
        "--max_support_fwd",
        type=int,
        required=True,
        help="Max support for forward context (use best value from previous tuning)"
    )
    parser.add_argument(
        "--max_support_bwd",
        type=int,
        required=True,
        help="Max support for backward context (use best value from previous tuning)"
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
        default="./lambda_tuning_results",
        help="Directory to save results (default: ./lambda_tuning_results)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run models on (default: cuda)"
    )
    
    args = parser.parse_args()
    

    results = tune_lambda(
        tok_path=args.tok_path,
        nt_model_path=args.nt_model_path,
        fwd_idx_path=args.fwd_idx_path,
        bwd_idx_path=args.bwd_idx_path,
        dataset_path=args.dataset_path,
        max_support_fwd=args.max_support_fwd,
        max_support_bwd=args.max_support_bwd,
        max_sequences=args.max_sequences,
        output_dir=args.output_dir,
        device=args.device
    )


