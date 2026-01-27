"""
Parallel evaluation of hyperparameters on test set using multiprocessing
"""

import numpy as np
from datasets import load_from_disk
from tqdm import tqdm
import json
import argparse
from pathlib import Path
import math
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool, cpu_count
from functools import partial


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
            nll = 10.0
        else:
            prob = probs[true_token]
            prob = max(prob, 1e-12)
            nll = -math.log(prob)
        
        total_nll += nll
    
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


def process_single_sequence(args):
    """
    Worker function to process a single sequence.
    This will be called in parallel by multiple processes.
    """
    seq, tok_path, fwd_idx_path, bwd_idx_path, max_support_fwd, max_support_bwd, mask_token_id = args
    
    # Import here to ensure each worker has its own model instance
    from bidir_bounded_gram import BidirectionalBoundedGramLM
    
    # Each worker creates its own model instance
    model = BidirectionalBoundedGramLM(
        tok_path=tok_path,
        fwd_idx_path=fwd_idx_path,
        bwd_idx_path=bwd_idx_path,
        max_support_fwd=max_support_fwd,
        max_support_bwd=max_support_bwd
    )
    
    unmasked_ids = seq["original_tokens"]
    masked_ids = seq["masked_tokens"]
    positions_masked = seq["positions_masked"]
    
    if len(positions_masked) == 0:
        return None
    
    # Compute metrics for this sequence
    seq_loss = avg_loss_per_sequence(
        model, unmasked_ids, masked_ids, positions_masked, mask_token_id
    )
    seq_accuracy = avg_accuracy_per_sequence(
        model, unmasked_ids, masked_ids, positions_masked, mask_token_id
    )
    
    if seq_loss is None or seq_accuracy is None:
        return None
    
    # Compute perplexity using Rives et al. definition
    seq_perplexity = compute_perplexity_rives(seq_loss)
    
    return {
        'loss': seq_loss,
        'accuracy': seq_accuracy,
        'perplexity': seq_perplexity
    }


def evaluate_on_dataset_parallel(
    tok_path, fwd_idx_path, bwd_idx_path, 
    max_support_fwd, max_support_bwd,
    dataset, mask_token_id, 
    max_sequences=None, 
    num_workers=None
):
    """
    Evaluate bidirectional bounded-gram model on masked dataset using multiprocessing.
    """
    if num_workers is None:
        num_workers = cpu_count()
    
    num_sequences = len(dataset) if max_sequences is None else min(max_sequences, len(dataset))
    
    print(f"Using {num_workers} workers for parallel evaluation")
    
    # Prepare arguments for each sequence
    args_list = [
        (
            dataset[idx],
            tok_path,
            fwd_idx_path,
            bwd_idx_path,
            max_support_fwd,
            max_support_bwd,
            mask_token_id
        )
        for idx in range(num_sequences)
    ]
    
    # Process sequences in parallel
    losses = []
    accuracies = []
    perplexities = []
    
    with Pool(processes=num_workers) as pool:
        # Use imap for progress bar
        results = list(tqdm(
            pool.imap(process_single_sequence, args_list),
            total=num_sequences,
            desc="Evaluating sequences"
        ))
    
    # Filter out None results and collect metrics
    for result in results:
        if result is not None:
            losses.append(result['loss'])
            accuracies.append(result['accuracy'])
            perplexities.append(result['perplexity'])
    
    # Convert to numpy arrays
    losses = np.array(losses)
    accuracies = np.array(accuracies)
    perplexities = np.array(perplexities)
    
    # Compute summary statistics
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


def evaluate_test_set(
    tok_path: str,
    fwd_idx_path: str,
    bwd_idx_path: str,
    dataset_path: str,
    mask_token_id: int,
    max_sequences: Optional[int] = None,
    output_dir: str = "./test_results",
    num_workers: Optional[int] = None
    ):

    from pathlib import Path
    
    # Define selected hyperparameters (from validation tuning)
    max_support_fwd = 8
    max_support_bwd = 8

    print(f"Loading test dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    print(f"Dataset loaded: {len(dataset)} sequences")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Evaluating on TEST SET (PARALLEL)")
    print(f"{'='*80}")
    print(f"Selected hyperparameters:")
    print(f"  max_support_fwd = {max_support_fwd}")
    print(f"  max_support_bwd = {max_support_bwd}")
    print(f"{'='*80}\n")
    
    results = evaluate_on_dataset_parallel(
        tok_path=tok_path,
        fwd_idx_path=fwd_idx_path,
        bwd_idx_path=bwd_idx_path,
        max_support_fwd=max_support_fwd,
        max_support_bwd=max_support_bwd,
        dataset=dataset,
        mask_token_id=mask_token_id,
        max_sequences=max_sequences,
        num_workers=num_workers
    )
    
    results['max_support_fwd'] = max_support_fwd
    results['max_support_bwd'] = max_support_bwd
    
    print(f"\n{'='*80}")
    print("TEST SET RESULTS")
    print(f"{'='*80}")
    print(f"Number of sequences: {results['num_sequences']}")
    print(f"Mean Loss: {results['mean_loss']:.4f}")
    print(f"Mean Accuracy: {results['mean_accuracy']:.4f}")
    print(f"Median Accuracy: {results['median_accuracy']:.4f}")
    print(f"Mean Perplexity (Rives): {results['mean_perplexity']:.4f}")
    print(f"Median Perplexity (Rives): {results['median_perplexity']:.4f}")
    print(f"{'='*80}\n")
    
    output_file = output_path / "test_results_detailed.npz"
    np.savez(
        output_file,
        losses=results['losses'],
        accuracies=results['accuracies'],
        perplexities=results['perplexities']
    )
    print(f"Detailed results saved to: {output_file}")
    
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
    
    summary_path = output_path / "test_results_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate selected hyperparameters on test set (parallel version)"
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
        help="Path to masked test dataset"
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
        default="./test_results",
        help="Directory to save results (default: ./test_results)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: cpu_count)"
    )
    
    args = parser.parse_args()
    
    results = evaluate_test_set(
        tok_path=args.tok_path,
        fwd_idx_path=args.fwd_idx_path,
        bwd_idx_path=args.bwd_idx_path,
        dataset_path=args.dataset_path,
        mask_token_id=args.mask_token_id,
        max_sequences=args.max_sequences,
        output_dir=args.output_dir,
        num_workers=args.num_workers
    )
