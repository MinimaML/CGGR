"""
CGGR - Confidence-Gated Gradient Routing
=========================================
Selective loss computation with multiple strategies.
All operations accelerated with fused Triton kernels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, List

from triton_kernels import (
    fused_difficulty_score,
    compute_dynamic_threshold,
    select_tokens_topk,
    select_tokens_stratified,
    ensure_sequence_coverage,
    apply_mask_to_loss,
)

# Enable TF32 for Ampere+ GPUs (free 10-15% speedup)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class CGGRLoss(nn.Module):
    """
    Advanced selective loss with multiple strategies.
    
    Args:
        scoring: Difficulty scoring method
            - 'entropy': Pure entropy-based
            - 'margin': Margin between top-2 predictions  
            - 'loss': Use per-token loss directly
            - 'combined': Entropy + margin + loss (default)
        
        selection: Token selection strategy
            - 'topk': Top-k hardest tokens
            - 'stratified': Sample from difficulty buckets
            - 'sequence_aware': Ensure coverage per sequence
        
        dynamic_threshold: Adjust ratio based on batch confidence
        threshold_sensitivity: How much to adjust (0-1)
        
        min_tokens_ratio: Target fraction of tokens to keep
        warmup_steps: Steps to reach target sparsity
        
        num_strata: Buckets for stratified sampling
        min_tokens_per_sequence: Minimum coverage per sequence
    """
    
    def __init__(
        self,
        scoring: Literal['entropy', 'margin', 'loss', 'combined'] = 'combined',
        selection: Literal['topk', 'stratified', 'sequence_aware', 'fixed_quota'] = 'topk',
        dynamic_threshold: bool = True,
        threshold_sensitivity: float = 0.5,
        min_tokens_ratio: float = 0.25,
        warmup_steps: int = 1000,
        num_strata: int = 4,
        min_tokens_per_sequence: int = 1,
        base_loss: nn.Module = None,
    ):
        super().__init__()
        
        self.scoring = scoring
        self.selection = selection
        self.dynamic_threshold = dynamic_threshold
        self.threshold_sensitivity = threshold_sensitivity
        self.min_tokens_ratio = min_tokens_ratio
        self.warmup_steps = warmup_steps
        self.num_strata = num_strata
        self.min_tokens_per_sequence = min_tokens_per_sequence
        self.base_loss = base_loss or nn.CrossEntropyLoss(reduction='none')
        
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))
        self.metrics = {}
    
    def step(self):
        """Call after optimizer.step()"""
        self.step_count += 1
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute selective loss.
        
        Args:
            logits: (batch, seq, vocab) or (N, vocab)
            targets: (batch, seq) or (N,)
        """
        # Handle shapes
        if logits.dim() == 3:
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = targets.view(-1)
        else:
            batch_size, seq_len = 1, logits.shape[0]
            vocab_size = logits.shape[-1]
            logits_flat = logits
            targets_flat = targets
        
        num_tokens = logits_flat.shape[0]
        
        # STEP 1: Compute difficulty scores to select tokens (no grad needed)
        with torch.no_grad():
            # Compute difficulty from logits (fast - just softmax + entropy)
            difficulty, confidence, entropy = fused_difficulty_score(
                logits_flat.unsqueeze(0) if logits_flat.dim() == 2 else logits_flat,
                targets=None,  # Don't use targets for selection to avoid computing loss
                mode=self.scoring,
            )
            difficulty = difficulty.view(-1)
            confidence = confidence.view(-1)
            entropy = entropy.view(-1)
            
            # Compute current ratio with curriculum
            if self.warmup_steps <= 0:
                progress = 1.0
            else:
                progress = min(1.0, self.step_count.item() / self.warmup_steps)
            base_ratio = 1.0 - progress * (1.0 - self.min_tokens_ratio)
            
            # Dynamic threshold adjustment
            if self.selection == 'fixed_quota':
                # FIXED QUOTA: Force fixed ratio, ignore dynamic_threshold
                current_ratio = base_ratio
            elif self.dynamic_threshold:
                current_ratio = compute_dynamic_threshold(
                    confidence, base_ratio, self.threshold_sensitivity
                )
            else:
                current_ratio = base_ratio
            
            # Token selection based on strategy
            if self.selection == 'stratified':
                mask = select_tokens_stratified(
                    difficulty, current_ratio, self.num_strata
                )
            elif self.selection == 'sequence_aware':
                mask = select_tokens_topk(difficulty, current_ratio)
                mask = ensure_sequence_coverage(
                    difficulty, mask, batch_size, seq_len, 
                    self.min_tokens_per_sequence
                )
            else:  # topk OR fixed_quota
                mask = select_tokens_topk(difficulty, current_ratio)
            
            mask = mask.view(-1)
            selected_indices = torch.nonzero(mask, as_tuple=True)[0]
            tokens_selected = selected_indices.numel()
        
        # STEP 2: Compute loss ONLY for selected tokens (this is where savings come from!)
        if tokens_selected > 0:
            selected_logits = logits_flat[selected_indices]
            selected_targets = targets_flat[selected_indices]
            loss = self.base_loss(selected_logits, selected_targets).mean()
        else:
            # Fallback: if no tokens selected, use full loss
            loss = self.base_loss(logits_flat, targets_flat).mean()
            tokens_selected = num_tokens
        
        # Metrics
        self.metrics = {
            'step': self.step_count.item(),
            'token_ratio': current_ratio,
            'tokens_selected': int(tokens_selected),
            'tokens_total': num_tokens,
            'avg_confidence': confidence.mean().item(),
            'avg_entropy': entropy.mean().item(),
            'avg_difficulty': difficulty.mean().item(),
            'selection': self.selection,
            'scoring': self.scoring,
        }
        
        return loss
    
    def get_metrics(self) -> dict:
        return self.metrics.copy()


class TruncatedRouter(nn.Module):
    """
    Lightweight proxy model for difficulty scoring.
    Constructed by slicing a full model (sharing weights).
    """
    def __init__(self, model: nn.Module, num_layers: int = 2):
        super().__init__()
        import copy
        
        # Heuristics for common architectures
        if hasattr(model, 'model'):
            # Llama/Mistral/Qwen style
            base_model = model.model
            self.style = 'llama'
        elif hasattr(model, 'transformer'):
            # GPT-2/GPT-J/Falcon style
            base_model = model.transformer
            self.style = 'gpt'
        else:
            raise ValueError("Unsupported model architecture for automatic truncation. Please provide a custom router.")

        # Clone config and truncate layers
        config = copy.deepcopy(model.config)
        if hasattr(config, 'num_hidden_layers'):
            config.num_hidden_layers = num_layers
        elif hasattr(config, 'n_layer'):
            config.n_layer = num_layers
        
        # Instantiate mini-model (random weights initially)
        cls = base_model.__class__
        self.mini_model = cls(config)
        
        # Share weights (Overwrite with references)
        # 1. Embeddings
        if hasattr(base_model, 'embed_tokens'):
            self.mini_model.embed_tokens = base_model.embed_tokens
        elif hasattr(base_model, 'wte'):
            self.mini_model.wte = base_model.wte
            self.mini_model.wpe = getattr(base_model, 'wpe', None)
            
        # 2. Layers
        if hasattr(base_model, 'layers'):
            for i in range(num_layers):
                self.mini_model.layers[i] = base_model.layers[i]
        elif hasattr(base_model, 'h'):
            for i in range(num_layers):
                self.mini_model.h[i] = base_model.h[i]
        
        # 3. Norm
        if hasattr(base_model, 'norm'):
            self.mini_model.norm = base_model.norm
        elif hasattr(base_model, 'ln_f'):
            self.mini_model.ln_f = base_model.ln_f
        
        # 4. Rotary Embeddings (if present, share reference)
        if hasattr(base_model, 'rotary_emb'):
            # For Models that store rotary_emb in the base model
            self.mini_model.rotary_emb = base_model.rotary_emb
            
        # 5. Head (not part of base model usually)
        self.head = model.lm_head if hasattr(model, 'lm_head') else None
            
    def forward(self, input_ids: torch.Tensor, **kwargs):
        # Forward through mini-base-model
        # This handles all position_ids, masking, and rotary embeddings correctly!
        outputs = self.mini_model(input_ids, **kwargs)
        
        # Get hidden states
        hidden_states = outputs[0]
        
        # Project to logits using head
        if self.head is not None:
            logits = self.head(hidden_states)
        else:
            logits = hidden_states
            
        return logits


def create_truncated_router(model: nn.Module, num_layers: int = 2) -> nn.Module:
    """Create a lightweight router sharing weights with the main model."""
    return TruncatedRouter(model, num_layers)



class CGGRScorer(nn.Module):
    """
    Standalone difficulty scorer for native architecture integration.
    
    Usage:
        scorer = CGGRScorer(router, min_tokens_ratio=0.5)
        scores, mask = scorer(input_ids)
    """
    def __init__(
        self,
        router: nn.Module,
        min_tokens_ratio: float = 0.25,
        warmup_steps: int = 1000,
        dynamic_threshold: bool = True,
        threshold_sensitivity: float = 0.5,
        selection: Literal['topk', 'stratified', 'sequence_aware', 'fixed_quota'] = 'topk',
    ):
        super().__init__()
        self.router = router
        self.min_tokens_ratio = min_tokens_ratio
        self.warmup_steps = warmup_steps
        self.dynamic_threshold = dynamic_threshold
        self.threshold_sensitivity = threshold_sensitivity
        self.selection = selection
        
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))

    def step(self):
        """Call after optimizer.step()"""
        self.step_count += 1

    def forward(self, input_ids: torch.Tensor, **kwargs):
        """
        Returns difficulty scores and boolean mask of hard tokens.
        
        Returns:
            difficulty: (batch*seq,) float scores
            mask: (batch*seq,) boolean mask (True=Hard/Keep)
            info: dict with metrics
        """
        with torch.no_grad():
            outputs = self.router(input_ids, **kwargs)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Compute difficulty
            difficulty, confidence, entropy = fused_difficulty_score(
                logits,
                mode=self.scoring if hasattr(self, 'scoring') else 'combined'
            )
            
            # Curriculum
            if self.warmup_steps <= 0:
                progress = 1.0
            else:
                progress = min(1.0, self.step_count.item() / self.warmup_steps)
            base_ratio = 1.0 - progress * (1.0 - self.min_tokens_ratio)
            
            # Dynamic threshold
            if hasattr(self, 'selection') and self.selection == 'fixed_quota':
                 current_ratio = base_ratio
            elif self.dynamic_threshold:
                current_ratio = compute_dynamic_threshold(
                    confidence.view(-1), base_ratio, self.threshold_sensitivity
                )
            else:
                current_ratio = base_ratio
            
            # Select hard tokens
            mask = select_tokens_topk(difficulty, current_ratio)
            
            info = {
                'confidence': confidence,
                'entropy': entropy,
                'current_ratio': current_ratio
            }
            return difficulty, mask, info


class CGGRModel(nn.Module):
    """
    Model wrapper with batch splitting for real backward speedup.
    
    Uses two-pass forward:
    1. First forward (Router): Lightweight difficulty scoring (fast)
    2. Second forward (Main): Only hard tokens → loss → backward (grad)
    """
    
    def __init__(
        self,
        model: nn.Module,
        router: Optional[nn.Module] = None,
        min_tokens_ratio: float = 0.25,
        warmup_steps: int = 1000,
        dynamic_threshold: bool = True,
        threshold_sensitivity: float = 0.5,
        selection: Literal['topk', 'stratified', 'sequence_aware', 'fixed_quota'] = 'topk',
    ):
        super().__init__()
        self.model = model
        router_model = router if router is not None else model
        
        # Use common scorer
        self.scorer = CGGRScorer(
            router=router_model,
            min_tokens_ratio=min_tokens_ratio,
            warmup_steps=warmup_steps,
            dynamic_threshold=dynamic_threshold,
            threshold_sensitivity=threshold_sensitivity,
            selection=selection
        )
        self.metrics = {}
    
    def step(self):
        self.scorer.step()
    
    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None, **kwargs):
        if labels is None:
            return self.model(input_ids, **kwargs)
        
        batch_size, seq_len = input_ids.shape
        
        # PASS 1: Get scores from scorer
        difficulty, mask, info = self.scorer(input_ids, **kwargs)
        
        current_ratio = info['current_ratio']
        confidence = info['confidence']
        entropy = info['entropy']
        
        # Get hard token indices
        hard_mask = mask.view(batch_size, seq_len) > 0.5
        tokens_total = batch_size * seq_len
        tokens_selected = hard_mask.sum().item()
        
        # PASS 2: Main model forward
        if tokens_selected > 0:
            # Batch Splitting Logic (same as before)
            seq_difficulty = difficulty.view(batch_size, seq_len).mean(dim=-1)
            k = max(1, int(batch_size * current_ratio))
            _, hard_seq_indices = torch.topk(seq_difficulty, k)
            
            hard_input_ids = input_ids[hard_seq_indices]
            hard_labels = labels[hard_seq_indices]
            
            hard_outputs = self.model(hard_input_ids, **kwargs)
            if hasattr(hard_outputs, 'logits'):
                hard_logits = hard_outputs.logits
            else:
                hard_logits = hard_outputs
            
            hard_logits_flat = hard_logits[:, :-1, :].contiguous().view(-1, hard_logits.shape[-1])
            hard_labels_flat = hard_labels[:, 1:].contiguous().view(-1)
            
            loss = F.cross_entropy(hard_logits_flat, hard_labels_flat)
            
            tokens_selected = hard_seq_indices.numel() * (seq_len - 1)
        else:
            # Fallback
            outputs = self.model(input_ids, **kwargs)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            logits_flat = logits[:, :-1, :].contiguous().view(-1, logits.shape[-1])
            labels_flat = labels[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(logits_flat, labels_flat)
            tokens_selected = tokens_total
        
        self.metrics = {
            'step': self.scorer.step_count.item(),
            'token_ratio': current_ratio,
            'tokens_selected': int(tokens_selected),
            'tokens_total': tokens_total,
            'avg_confidence': confidence.mean().item(),
            'avg_entropy': entropy.mean().item(),
        }
        
        return loss
    
    def get_metrics(self) -> dict:
        return self.metrics.copy()


# Export key components
__all__ = [
    'CGGRLoss', 
    'CGGRModel', 
    'CGGRScorer',
    'create_truncated_router', 
    'TruncatedRouter',
]

# Tier 1 Optimizations (import separately)
# from cggr_checkpointing import CGGRCheckpointedModel, SelectiveCheckpointWrapper
# from cggr_async import AsyncCGGRScorer, AsyncCGGRModel
# from cggr_dataloader import NanoRouter, DifficultyFilteredDataLoader


