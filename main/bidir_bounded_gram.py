"""
Modified Bidirectional infini-gram built on human reference genome
to better approximate MLM
using infini-gram documentation
"""

from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer

import math
import random
from typing import Sequence, List, Dict, Tuple, Optional

import argparse
from datasets import load_from_disk
from tqdm import tqdm
import json

class BidirectionalBoundedGramLM:

    def __init__(self, tok_path, fwd_idx_path, bwd_idx_path, max_support_fwd, max_support_bwd):
        self.tokenizer =  AutoTokenizer.from_pretrained(tok_path, use_fast=True, local_files_only=True)
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            eos_id = self.tokenizer.pad_token_id
        if eos_id is None:
            # fallback just in case
            eos_id = 0

        self.fwd_engine = InfiniGramEngine(index_dir=fwd_idx_path, eos_token_id=eos_id)
        self.bwd_engine =  InfiniGramEngine(index_dir=bwd_idx_path, eos_token_id=eos_id)

        # max support for Infini-gram
        self.max_support_fwd = max_support_fwd
        self.max_support_bwd = max_support_bwd
        
    
    def get_mlm_context(self, unmasked_ids, masked_ids, pos, mask_token_id):
        """
        Return left and right context up until another mask is encountered
        """
        L = len(unmasked_ids)
        assert len(masked_ids) == L, "unmasked and masked sequences must be same length"

        if pos < 0 or pos >= L:
            raise IndexError(f"pos={pos} out of range for sequence of length {L}")

        # walk left until you hit another mask
        left_tokens = []
        j = pos - 1
        while j >= 0:
            if masked_ids[j] == mask_token_id:
                break  # stop at first mask on the left
            left_tokens.append(unmasked_ids[j])
            j -= 1
        left_context = list(reversed(left_tokens))  # restore original order

        # walk right until you hit another mask
        right_context = []
        j = pos + 1
        while j < L:
            if masked_ids[j] == mask_token_id:
                break  # stop at first mask on the right
            right_context.append(unmasked_ids[j])
            j += 1

        return left_context, right_context

    
    def directional_nt_distribution(self, engine, context, max_support):
        """
        Computes next token distribution conditioned on the context
        """
        # computes the ∞-gram LM next-token distribution conditioning on a preceding prompt.
        res = engine.infgram_ntd(prompt_ids=list(context), max_support=max_support) # returns {'prompt_cnt', 'result_by_token_id: {candidate_token_info....}}
        res_by_token_id = res["result_by_token_id"]

        distribution: Dict[int, float] = {}
        for tid, info in res_by_token_id.items():
            prob = float(info["prob"])
            if prob <= 0.0:
                continue
            tid = int(tid)
            distribution[tid] = prob

        # Normalize to ensure sum of probs is 1
        total = sum(distribution.values())
        if total > 0.0:
            for tid in distribution:
                distribution[tid] /= total

        return distribution


    def bidirectional_mlm_probabilities(self, unmasked_ids, masked_ids, pos, mask_token_id):
        """
        MLM-style masked-token distribution:
        - Context is truncated at nearest masks to avoid leaking other masked tokens.
        """
        left_context, right_context = self.get_mlm_context(unmasked_ids, masked_ids, pos, mask_token_id)
        right_context_reversed = list(reversed(right_context))

        fwd_dist = self.directional_nt_distribution(self.fwd_engine, left_context, self.max_support_fwd)
        bwd_dist = self.directional_nt_distribution(self.bwd_engine, right_context_reversed, self.max_support_bwd)
        candidates = set(fwd_dist.keys()) | set(bwd_dist.keys())
        if not candidates:
            return {}

        # weight the forward and backward Infini-gram distributions according to the fraction of visible context on each side of the masked token
        L = len(left_context)
        R = len(right_context)
        if (L + R) == 0:
            alpha = 0.5
        else:
            alpha = L / (L + R)
        beta = 1.0 - alpha

        sample_scores = {}
        for tid in candidates:
            fwd_prob = fwd_dist.get(tid, 1e-12)
            bwd_prob = bwd_dist.get(tid, 1e-12)

            fwd_log = math.log(fwd_prob)
            bwd_log = math.log(bwd_prob)

            score = alpha * fwd_log + beta * bwd_log
            sample_scores[tid] = score

        # Softmax over scores
        max_score = max(sample_scores.values())
        exps = {tid: math.exp(s - max_score) for tid, s in sample_scores.items()}
        denom = sum(exps.values())
        return {tid: v / denom for tid, v in exps.items()}


    def predict_mlm_token(self, unmasked_ids, masked_ids, pos, mask_token_id):
        probs = self.bidirectional_mlm_probabilities(unmasked_ids, masked_ids, pos, mask_token_id)
        if not probs:
            return None
        return max(probs.items(), key=lambda kv: kv[1])[0]


    def directional_nt_distribution_with_meta(self, engine, context, max_support):
        """
        Like directional_nt_distribution but also returns prompt_cnt.
        """
        res = engine.infgram_ntd(prompt_ids=list(context), max_support=max_support)
        res_by_token_id = res["result_by_token_id"]
        prompt_cnt = res["prompt_cnt"]

        distribution = {}
        for tid, info in res_by_token_id.items():
            prob = float(info["prob"])
            if prob <= 0.0:
                continue
            tid = int(tid)
            distribution[tid] = prob

        total = sum(distribution.values())
        if total > 0.0:
            for tid in distribution:
                distribution[tid] /= total

        return distribution, prompt_cnt


    def bidirectional_mlm_probabilities_with_promptcnt(self, unmasked_ids, masked_ids, pos, mask_token_id):
        left_context, right_context = self.get_mlm_context(unmasked_ids, masked_ids, pos, mask_token_id)
        right_context_reversed = list(reversed(right_context))

        fwd_dist, fwd_cnt = self.directional_nt_distribution_with_meta(self.fwd_engine, left_context, self.max_support_fwd)
        bwd_dist, bwd_cnt = self.directional_nt_distribution_with_meta(self.bwd_engine, right_context_reversed, self.max_support_bwd)

        candidates = set(fwd_dist.keys()) | set(bwd_dist.keys())
        if not candidates:
            return {}, 0

        L = len(left_context)
        R = len(right_context)
        if (L + R) == 0:
            alpha = 0.5
        else:
            alpha = L / (L + R)
        beta = 1.0 - alpha

        sample_scores = {}
        for tid in candidates:
            fwd_prob = fwd_dist.get(tid, 1e-12)
            bwd_prob = bwd_dist.get(tid, 1e-12)
            fwd_log = math.log(fwd_prob)
            bwd_log = math.log(bwd_prob)
            score = alpha * fwd_log + beta * bwd_log
            sample_scores[tid] = score

        max_score = max(sample_scores.values())
        exps = {tid: math.exp(s - max_score) for tid, s in sample_scores.items()}
        denom = sum(exps.values())
        bi_dist = {tid: v / denom for tid, v in exps.items()}

        # Combine fwd/bwd prompt counts into a single sparsity signal
        prompt_cnt = max(fwd_cnt, bwd_cnt)

        return bi_dist, prompt_cnt

