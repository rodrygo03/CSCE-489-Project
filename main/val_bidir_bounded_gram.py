"""
Tune hyperparameters to validation set for infini-gram bidirectional bounded-gram MLM.
"""

import numpy as np
from datasets import load_from_disk
from tqdm import tqdm
import json
import argparse
from pathlib import Path
import itertools
import math
from typing import Dict, List, Tuple, Optional

def avg_loss_per_sequence(model, unmasked_ids, masked_ids, positions_masked, mask_token_id):
    """
    Compute average loss for a single sequence.
    Following Rives et al.: l(θ,s) = (1/|P_masked|) * Σ_{i∈P_masked} -log p(θ, i, s(i))
    """
    if len(positions_masked) == 0:
        return None
    
    total_nll = 0.0
    for pos in positions_masked:
        true_token = unmasked_ids[pos]
        probs = model.bidirectional_mlm_probabilities(
            unmasked_ids, masked_ids, pos, mask_token_id
        )
        
        if not probs or true_token not in probs:
            # If model can't predict (no context), assign high penalty
            nll = 10.0  # or use -log(1/vocab_size)
        else:
            prob = probs[true_token]
            # Clip to avoid log(0)
            prob = max(prob, 1e-12)
            nll = -math.log(prob)
        
        total_nll += nll
    
    # Average over masked positions
    return total_nll / len(positions_masked)


def avg_accuracy_per_sequence(model, unmasked_ids, masked_ids, positions_masked, mask_token_id):
    """
    Compute accuracy for a single sequence.
    Following paper: acc(θ,s) = (1/|P_masked|) * Σ_{i∈P_masked} 1(argmax log p(θ, i, tok) = s(i))
    """
    if len(positions_masked) == 0:
        return None
    
    correct = 0
    for pos in positions_masked:
        true_token = unmasked_ids[pos]
        predicted_token = model.predict_mlm_token(
            unmasked_ids, masked_ids, pos, mask_token_id
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


def evaluate_on_dataset(model, dataset, mask_token_id, max_sequences=None):
    """
    Evaluate bidirectional bounded-gram model on masked dataset.
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
            model, unmasked_ids, masked_ids, positions_masked, mask_token_id
        )
        seq_accuracy = avg_accuracy_per_sequence(
            model, unmasked_ids, masked_ids, positions_masked, mask_token_id
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


def tune_hyperparameters(
    tok_path: str,
    fwd_idx_path: str,
    bwd_idx_path: str,
    dataset_path: str,
    mask_token_id: int,
    max_sequences: Optional[int] = None,
    output_dir: str = "./tuning_results"
):
    """
    Grid search over max_support_fwd and max_support_bwd hyperparameters.
    """
    from bidir_bounded_gram import BidirectionalBoundedGramLM
    import numpy as np
    import itertools
    from pathlib import Path
    
    # Define hyperparameter grid to search
    max_support_fwd_values = [8, 16]
    max_support_bwd_values = [8, 16]
    
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    print(f"Dataset loaded: {len(dataset)} sequences")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # Grid search
    total_combinations = len(max_support_fwd_values) * len(max_support_bwd_values)
    print(f"\nTuning over {total_combinations} hyperparameter combinations...")
    print(f"max_support_fwd: {max_support_fwd_values}")
    print(f"max_support_bwd: {max_support_bwd_values}")
    print()
    
    for max_support_fwd, max_support_bwd in itertools.product(
        max_support_fwd_values, max_support_bwd_values
    ):
        print(f"\n{'='*80}")
        print(f"Testing: max_support_fwd={max_support_fwd}, max_support_bwd={max_support_bwd}")
        print(f"{'='*80}")

        model = BidirectionalBoundedGramLM(
            tok_path=tok_path,
            fwd_idx_path=fwd_idx_path,
            bwd_idx_path=bwd_idx_path,
            max_support_fwd=max_support_fwd,
            max_support_bwd=max_support_bwd
        )
        
        results = evaluate_on_dataset(
            model=model,
            dataset=dataset,
            mask_token_id=mask_token_id,
            max_sequences=max_sequences
        )
        

        results['max_support_fwd'] = max_support_fwd
        results['max_support_bwd'] = max_support_bwd
        
        print(f"\nResults:")
        print(f"  Number of sequences: {results['num_sequences']}")
        print(f"  Mean Loss: {results['mean_loss']:.4f}")
        print(f"  Mean Accuracy: {results['mean_accuracy']:.4f}")
        print(f"  Median Accuracy: {results['median_accuracy']:.4f}")
        print(f"  Mean Perplexity (Rives): {results['mean_perplexity']:.4f}")
        print(f"  Median Perplexity (Rives): {results['median_perplexity']:.4f}")
        
        config_name = f"fwd{max_support_fwd}_bwd{max_support_bwd}"
        np.savez(
            output_path / f"{config_name}_detailed.npz",
            losses=results['losses'],
            accuracies=results['accuracies'],
            perplexities=results['perplexities']
        )

        summary = {
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
    
    summary_path = output_path / "tuning_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("TUNING COMPLETE")
    print(f"{'='*80}")
    
    best_by_accuracy = max(all_results, key=lambda x: x['median_accuracy'])
    best_by_perplexity = min(all_results, key=lambda x: x['mean_perplexity'])
    
    print("\nBest configuration by MEDIAN ACCURACY:")
    print(f"  max_support_fwd={best_by_accuracy['max_support_fwd']}, "
          f"max_support_bwd={best_by_accuracy['max_support_bwd']}")
    print(f"  Median Accuracy: {best_by_accuracy['median_accuracy']:.4f}")
    print(f"  Mean Perplexity: {best_by_accuracy['mean_perplexity']:.4f}")
    
    print("\nBest configuration by MEAN PERPLEXITY:")
    print(f"  max_support_fwd={best_by_perplexity['max_support_fwd']}, "
          f"max_support_bwd={best_by_perplexity['max_support_bwd']}")
    print(f"  Median Accuracy: {best_by_perplexity['median_accuracy']:.4f}")
    print(f"  Mean Perplexity: {best_by_perplexity['mean_perplexity']:.4f}")
    
    print(f"\nAll results saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune hyperparameters for Bidirectional Bounded-Gram MLM"
    )
    
    parser.add_argument(
        "--tok_path",
        type=str,
        required=True,
        help="Path to tokenizer"
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
        nargs="+",
        default=[5, 10, 15, 20],
        help="Values to try for max_support_fwd (default: 5 10 15 20)"
    )
    parser.add_argument(
        "--max_support_bwd",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20],
        help="Values to try for max_support_bwd (default: 5 10 15 20)"
    )
    parser.add_argument(
        "--mask_token_id",
        type=int,
        required=True,
        help="Token ID used for masking"
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
        default="./tuning_results",
        help="Directory to save results (default: ./tuning_results)"
    )
    
    args = parser.parse_args()
    
    results = tune_hyperparameters(
        tok_path=args.tok_path,
        fwd_idx_path=args.fwd_idx_path,
        bwd_idx_path=args.bwd_idx_path,
        dataset_path=args.dataset_path,
        mask_token_id=args.mask_token_id,
        max_sequences=args.max_sequences,
        output_dir=args.output_dir
    )

