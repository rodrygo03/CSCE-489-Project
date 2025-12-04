"""
Hybrid model following linear interpolation 
as seen in infini-gram paper, with dynamic lambda selection
based on prompt count (context sparsity)
"""

from bidir_bounded_gram import BidirectionalBoundedGramLM

import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import Dict, List, Optional, Tuple


class HybridInfiniGramNeuralLM:
    """
    Hybrid model that interpolates between Nucleotide Transformer and Infini-gram
    using dynamic lambda selection based on context sparsity.
    
    When prompt_cnt < cnt_thresh: Use lam_nt_sparse (higher weight for NT)
    When prompt_cnt >= cnt_thresh: Use lam_nt_dense (trust Infini-gram more)
    """

    def __init__(
        self,
        tok_path: str,
        nt_model_path: str,
        fwd_idx_path: str,
        bwd_idx_path: str,
        max_support_fwd: int,
        max_support_bwd: int,
        lam_nt_sparse: float = 0.8,
        lam_nt_dense: float = 0.3,
        cnt_thresh: int = 10,
        device: str = "cuda",
    ):

        self.device = device
        
        # Lambda parameters for dynamic selection
        self.lam_nt_sparse = lam_nt_sparse
        self.lam_nt_dense = lam_nt_dense
        self.cnt_thresh = cnt_thresh
    
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True, local_files_only=True)
        self.mask_token_id = self.tokenizer.mask_token_id

        # Nucleotide Transformer
        self.nt_model = AutoModelForMaskedLM.from_pretrained(nt_model_path, local_files_only=True).to(self.device)
        self.nt_model.eval()

        # Bidirectional bounded Infini-gram LM
        self.infgram_lm = BidirectionalBoundedGramLM(tok_path=tok_path, fwd_idx_path=fwd_idx_path, bwd_idx_path=bwd_idx_path, max_support_fwd=max_support_fwd, max_support_bwd=max_support_bwd)


    def nt_mlm_probabilities(self, masked_ids: List[int], pos: int):
        """
        Get MLM probability distribution from Nucleotide Transformer.
        """
        input_ids = torch.tensor(
            [masked_ids], dtype=torch.long, device=self.device
        )

        with torch.no_grad():
            outputs = self.nt_model(input_ids=input_ids)
            logits = outputs.logits

        logits_pos = logits[0, pos]
        probs_pos = F.softmax(logits_pos, dim=-1)

        vocab_size = probs_pos.shape[0]
        return {tid: float(probs_pos[tid].item()) for tid in range(vocab_size)}


    def ig_mlm_probabilities(self, unmasked_ids: List[int], masked_ids: List[int], pos: int):
        """
        Get MLM probability distribution from Infini-gram.
        """
        return self.infgram_lm.bidirectional_mlm_probabilities(
            unmasked_ids=unmasked_ids,
            masked_ids=masked_ids,
            pos=pos,
            mask_token_id=self.mask_token_id,
        )


    def hybrid_mlm_probabilities(self, unmasked_ids: List[int], masked_ids: List[int], pos: int):
        """
        Hybrid MLM distribution with dynamic lambda selection based on prompt count.
        
        Linear interpolation: p_hybrid = lam_nt * p_NT + (1 - lam_nt) * p_IG
        where lam_nt is chosen based on context sparsity (prompt_cnt)
        """
        # Get ∞-gram bi-directional distribution *and* prompt_cnt
        ig_probs, prompt_cnt = self.infgram_lm.bidirectional_mlm_probabilities_with_promptcnt(
            unmasked_ids=unmasked_ids,
            masked_ids=masked_ids,
            pos=pos,
            mask_token_id=self.mask_token_id,
        )

        # If IG has no support, fall back to pure NT
        if not ig_probs:
            return self.nt_mlm_probabilities(masked_ids, pos)

        nt_probs = self.nt_mlm_probabilities(masked_ids, pos)

        if prompt_cnt < self.cnt_thresh:
            lam_nt = self.lam_nt_sparse   # mostly NT (sparse context)
        else:
            lam_nt = self.lam_nt_dense    # trust ∞-gram more (dense context)

        lam_ig = 1.0 - lam_nt

        # Interpolate distributions
        all_tokens = set(nt_probs.keys()) | set(ig_probs.keys())
        hybrid = {}
        for tid in all_tokens:
            p_nt = nt_probs.get(tid, 0.0)
            p_ig = ig_probs.get(tid, 0.0)
            hybrid[tid] = lam_nt * p_nt + lam_ig * p_ig

        # renormalization just in case
        total = sum(hybrid.values())
        if total > 0.0:
            for tid in hybrid:
                hybrid[tid] /= total

        return hybrid


    def predict_hybrid_token(self, unmasked_ids: List[int], masked_ids: List[int], pos: int):
        """
        Predict the most likely token id at `pos` under the hybrid model.
        """
        probs = self.hybrid_mlm_probabilities(unmasked_ids, masked_ids, pos)
        if not probs:
            return None
        return max(probs.items(), key=lambda kv: kv[1])[0]


    def hybrid_token_nll(self, unmasked_ids: List[int], masked_ids: List[int], pos: int, gold_tid: int, eps: float = 1e-12):
        """
        Compute negative log-likelihood of gold token under hybrid model.
        """
        probs = self.hybrid_mlm_probabilities(unmasked_ids, masked_ids, pos)
        p = probs.get(gold_tid, 0.0)
        p = max(p, eps)
        return -math.log(p)