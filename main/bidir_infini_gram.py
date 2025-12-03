"""
Bidirectional infini-gram built on human reference genome
using infini-gram documentation
"""

from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer

import math
from typing import Sequence, List, Dict, Tuple, Optional

import argparse
from datasets import load_from_disk
import json

class BidirectionalInfiniGramLM:

    def __init__(self, tok_path, fwd_idx_path, bwd_idx_path, max_left, max_right, max_support_fwd, max_support_bwd, temp_fwd, temp_bwd, alpha):
        self.tokenizer =  AutoTokenizer.from_pretrained(tok_path, use_fast=True, local_files_only=True)
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            eos_id = self.tokenizer.pad_token_id
        if eos_id is None:
            # fallback just in case
            eos_id = 0

        self.fwd_engine = InfiniGramEngine(index_dir=fwd_idx_path, eos_token_id=eos_id)
        self.bwd_engine =  InfiniGramEngine(index_dir=bwd_idx_path, eos_token_id=eos_id)
        
        # maximum len of context
        self.max_left = max_left
        self.max_right = max_right

        #
        self.max_support_fwd = max_support_fwd
        self.max_support_bwd = max_support_bwd

        # 
        self.temp_fwd = temp_fwd
        self.temp_bwd = temp_bwd

        #
        self.alpha = alpha
        self.beta  = 1.0 - alpha

    def get_context(self, token_ids, pos):
        """
        Return left and right context around the masked bp
        """
        if (pos < 0 or pos >= len(token_ids)):
            raise IndexError(f"pos={pos} out of range for sequence of length {len(token_ids)}")

        start_left = max(0, pos - self.max_left)
        left_context = list(token_ids[start_left:pos])

        end_right = min(len(token_ids), pos + 1 + self.max_right)
        right_context = list(token_ids[pos + 1:end_right])

        return left_context, right_context
    

    def directional_nt_distribution(self, engine, context, max_support, temperature):
        """
        Computes next token distribution conditioned on the context with temp scailing
        return normalized distribution 
        """
        # computes the ∞-gram LM next-token distribution conditioning on a preceding prompt.
        res = engine.infgram_ntd(prompt_ids=list(context), max_support=max_support) # returns {'prompt_cnt', 'result_by_token_id: {candidate_token_info....}}
        res_by_token_id = res["result_by_token_id"]

        distribution: Dict[int, float] = {}
        for candidate_token, info in res_by_token_id.items():
            token_id = int(candidate_token)
            prob = info["prob"]
            if prob <= 0:
                continue
            
            log_prob = math.log(prob) # prob in log space
            log_prob = log_prob / temperature # temp scaling
            distribution[token_id] = math.exp(log_prob+ 1e-12) # prevent e^{0}
        
        # normalize distribution:
        denom = sum(distribution.values())
        if (denom == 0):
            return {}
        else:
            return {tid: val / denom for tid, val in distribution.items()}
    

    def bidirectional_probabilities(self, token_ids, pos):
        """
        Computes masked token distribution conditioned on the left and right context
        return prob distribution over foward and backward distribution candidates
        The bi-directional model is a geometric mixture of the forward and backward infini grams
        """
        left_context, right_context = self.get_context(token_ids, pos)
        right_context = list(reversed(right_context))   # reversed context for reversed index

        foward_dist = self.directional_nt_distribution(self.fwd_engine, left_context, self.max_support_fwd, self.temp_fwd)
        backward_dist = self.directional_nt_distribution(self.bwd_engine, right_context, self.max_support_bwd, self.temp_bwd)

        candidates = set(foward_dist.keys()) | set(backward_dist.keys())
        if not candidates:
            return {}
        
        # weighted average of log-prob in log space
        sample_scores: Dict[int, float] = {}
        for tid in candidates:
            foward_prob = foward_dist.get(tid, 1e-12) # prevent log(0)
            backward_prob = backward_dist.get(tid, 1e-12)

            foward_log_prob = math.log(foward_prob)
            backward_log_prob = math.log(backward_prob)

            sample_score = self.alpha*foward_log_prob + self.beta*backward_log_prob
            sample_scores[tid] = sample_score

        # prob distribution over candidates using softmax 
        max_score = max(sample_scores.values())
        scores = {tid: math.exp(score - max_score) for tid, score in sample_scores.items()} # normalize to prevent overflow
        denom = sum(scores.values())
        return {tid: val / denom for tid, val in scores.items()}
    

    def predict_token(self, token_ids, pos):
        """
        Argmax for masked token at pos using bidirectional masked token probabilities
        """
        probs = self.bidirectional_probabilities(token_ids, pos)
        return max(probs.items(), key=lambda kv: kv[1])[0] # select max prob in tuple pair


def load_val_sequences(model, masked_val_path, mask_token_id=None):
    """
    Load validation sequences from arrow dataset.
    Expected format (like your benchmark):
    - original_tokens: list of token IDs (unmasked)
    - masked_tokens: list of token IDs (with masks)
    - positions_masked: list of positions that were masked
    """
    from datasets import load_from_disk
    
    print(f"Loading masked validation dataset from: {masked_val_path}")
    dataset = load_from_disk(masked_val_path)
    print(f"Loaded {len(dataset)} sequences")
    
    # Get mask token ID (from argument, tokenizer, or infer from data)
    if mask_token_id is not None:
        mask_id = mask_token_id
        print(f"Using provided mask token ID: {mask_id}")
    else:
        mask_id = model.tokenizer.mask_token_id
    
        # If tokenizer doesn't have a mask token, try to infer it
        if mask_id is None:
            print("WARNING: Tokenizer doesn't have a mask_token_id. Attempting to infer from data...")
            # Check what token appears at masked positions
            from collections import Counter
            mask_token_candidates = Counter()
            
            for i, example in enumerate(dataset):
                if i >= 100:  # Sample first 100
                    break
                masked_tokens = example["masked_tokens"]
                positions_masked = example["positions_masked"]
                
                for pos in positions_masked:
                    mask_token_candidates[masked_tokens[pos]] += 1
            
            if mask_token_candidates:
                mask_id = mask_token_candidates.most_common(1)[0][0]
                print(f"  Inferred mask token ID: {mask_id} (appeared {mask_token_candidates[mask_id]} times)")
            else:
                raise ValueError("Could not determine mask token ID. Please specify it manually with --mask_token_id")
        else:
            print(f"Using tokenizer's mask token ID: {mask_id}")
    
    # Convert dataset to eval_examples format
    eval_examples = []
    skipped = 0
    
    for idx, example in enumerate(dataset):
        original_tokens = example["original_tokens"]
        positions_masked = example["positions_masked"]
        
        if not positions_masked or len(positions_masked) == 0:
            skipped += 1
            continue
        
        eval_examples.append({
            "unmasked_ids": original_tokens,  # Use original tokens for context
            "masked_positions": positions_masked,
        })
    
    print(f"Created {len(eval_examples)} evaluation examples ({skipped} skipped - no masked positions)")
    return eval_examples


def _evaluate_on_examples(model, eval_examples: List[Dict]) -> Tuple[float, float, float]:
    """
    Evaluate the model on validation examples.
    
    Returns:
        accuracy: proportion of correct predictions
        mean_nll: mean negative log-likelihood
        perplexity: exp(mean_nll)
    """
    total_positions = 0
    correct_predictions = 0
    total_nll = 0.0
    
    for example in eval_examples:
        unmasked_ids = example["unmasked_ids"]
        masked_positions = example["masked_positions"]
        
        for pos in masked_positions:
            true_token_id = unmasked_ids[pos]
            
            # Get probability distribution
            probs = model.bidirectional_probabilities(unmasked_ids, pos)
            
            if not probs:
                # If no predictions possible, skip this position
                continue
            
            # Check accuracy
            predicted_token_id = max(probs.items(), key=lambda kv: kv[1])[0]
            if predicted_token_id == true_token_id:
                correct_predictions += 1
            
            # Calculate negative log-likelihood
            true_prob = probs.get(true_token_id, 1e-12)
            nll = -math.log(true_prob)
            total_nll += nll
            
            total_positions += 1
    
    if total_positions == 0:
        return 0.0, float('inf'), float('inf')
    
    accuracy = correct_predictions / total_positions
    mean_nll = total_nll / total_positions
    perplexity = math.exp(mean_nll)
    
    return accuracy, mean_nll, perplexity


def tune_hyperparameters(model, masked_val_path, mask_token_id=None):
    """
    Tune max_support_fwd/bwd, temp_fwd/bwd, and alpha on a validation set.
    
    Args:
        model: BidirectionalInfiniGramLM instance
        masked_val_path: path to arrow dataset with masked sequences
        mask_token_id: optional mask token ID (will auto-detect if None)
    
    Returns:
        best_config: dict of best hyperparameters
        best_metrics: dict of metrics at best config
    """
    # Load validation examples
    print("Loading validation sequences...")
    eval_examples = load_val_sequences(model, masked_val_path, mask_token_id)
    print(f"Loaded {len(eval_examples)} validation examples")
    
    # Define hyperparameter search space
    max_support_candidates = [32, 64, 128]
    temp_candidates = [0.25, 0.5, 0.75, 1.0, 1.25]
    alpha_candidates = [0.1, 0.25, 0.5, 0.75, 0.9]
    
    best_config = None
    best_perplexity = float("inf")
    best_metrics = None
    
    total_configs = (len(max_support_candidates) ** 2 * 
                     len(temp_candidates) ** 2 * 
                     len(alpha_candidates))
    config_num = 0
    
    print(f"Starting grid search over {total_configs} configurations...\n")
    
    for ms_fwd in max_support_candidates:
        for ms_bwd in max_support_candidates:
            for Tf in temp_candidates:
                for Tb in temp_candidates:
                    for alpha in alpha_candidates:
                        config_num += 1
                        
                        # Set parameters
                        model.max_support_fwd = ms_fwd
                        model.max_support_bwd = ms_bwd
                        model.temp_fwd = Tf
                        model.temp_bwd = Tb
                        model.alpha = alpha
                        model.beta = 1.0 - alpha
                        
                        # Evaluate on validation
                        accuracy, mean_nll, perplexity = _evaluate_on_examples(model, eval_examples)
                        
                        # Print progress
                        if config_num % 50 == 0 or perplexity < best_perplexity:
                            print(f"[{config_num}/{total_configs}] "
                                  f"ms_fwd={ms_fwd}, ms_bwd={ms_bwd}, "
                                  f"Tf={Tf}, Tb={Tb}, α={alpha} → "
                                  f"PPL={perplexity:.4f}, Acc={accuracy:.4f}")
                        
                        # Track best configuration
                        if perplexity < best_perplexity:
                            best_perplexity = perplexity
                            best_config = {
                                "max_support_fwd": ms_fwd,
                                "max_support_bwd": ms_bwd,
                                "temp_fwd": Tf,
                                "temp_bwd": Tb,
                                "alpha": alpha,
                            }
                            best_metrics = {
                                "accuracy": accuracy,
                                "mean_nll": mean_nll,
                                "perplexity": perplexity,
                            }
    
    # Set model to best configuration
    print(f"\n{'='*60}")
    print("Best configuration found:")
    print(f"  max_support_fwd: {best_config['max_support_fwd']}")
    print(f"  max_support_bwd: {best_config['max_support_bwd']}")
    print(f"  temp_fwd: {best_config['temp_fwd']}")
    print(f"  temp_bwd: {best_config['temp_bwd']}")
    print(f"  alpha: {best_config['alpha']}")
    print(f"\nBest metrics:")
    print(f"  Perplexity: {best_metrics['perplexity']:.4f}")
    print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"  Mean NLL: {best_metrics['mean_nll']:.4f}")
    print(f"{'='*60}\n")
    
    model.max_support_fwd = best_config["max_support_fwd"]
    model.max_support_bwd = best_config["max_support_bwd"]
    model.temp_fwd = best_config["temp_fwd"]
    model.temp_bwd = best_config["temp_bwd"]
    model.alpha = best_config["alpha"]
    model.beta = 1.0 - model.alpha
    
    return best_config, best_metrics


def main():
    print("debug 1")
    parser = argparse.ArgumentParser(
        description="Tune hyperparameters for bidirectional infini-gram model"
    )
    
    # Required arguments
    parser.add_argument(
        "--tok_path",
        type=str,
        required=True,
        help="Path to tokenizer (e.g., nucleotide-transformer-500m-human-ref)"
    )
    parser.add_argument(
        "--fwd_idx_path",
        type=str,
        required=True,
        help="Path to forward infini-gram index directory"
    )
    parser.add_argument(
        "--bwd_idx_path",
        type=str,
        required=True,
        help="Path to backward infini-gram index directory"
    )
    parser.add_argument(
        "--masked_val_path",
        type=str,
        required=True,
        help="Path to arrow dataset with masked validation sequences (contains original_tokens, masked_tokens, positions_masked)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--max_left",
        type=int,
        default=1000,
        help="Maximum left context length (default: 1000)"
    )
    parser.add_argument(
        "--max_right",
        type=int,
        default=1000,
        help="Maximum right context length (default: 1000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="best_hyperparameters.json",
        help="Output file for best hyperparameters (default: best_hyperparameters.json)"
    )
    parser.add_argument(
        "--mask_token_id",
        type=int,
        default=None,
        help="Mask token ID (will auto-detect if not specified)"
    )
    
    args = parser.parse_args()
    
    # Initialize model with default hyperparameters (will be tuned)
    print("Initializing BidirectionalInfiniGramLM...")
    print(f"  Tokenizer: {args.tok_path}")
    print(f"  Forward index: {args.fwd_idx_path}")
    print(f"  Backward index: {args.bwd_idx_path}")
    print(f"  Max context: left={args.max_left}, right={args.max_right}")
    print()
    
    model = BidirectionalInfiniGramLM(
        tok_path=args.tok_path,
        fwd_idx_path=args.fwd_idx_path,
        bwd_idx_path=args.bwd_idx_path,
        max_left=args.max_left,
        max_right=args.max_right,
        max_support_fwd=64,  # Initial values (will be tuned)
        max_support_bwd=64,
        temp_fwd=1.0,
        temp_bwd=1.0,
        alpha=0.5
    )
    print("Model initialized successfully!\n")
    
    # Tune hyperparameters
    best_config, best_metrics = tune_hyperparameters(
        model,
        masked_val_path=args.masked_val_path,
        mask_token_id=args.mask_token_id
    )
    
    # Save results
    results = {
        "config": best_config,
        "metrics": best_metrics,
        "model_params": {
            "tok_path": args.tok_path,
            "fwd_idx_path": args.fwd_idx_path,
            "bwd_idx_path": args.bwd_idx_path,
            "max_left": args.max_left,
            "max_right": args.max_right
        }
    }
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
