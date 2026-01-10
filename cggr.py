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
# Only set these if CUDA is available to avoid errors on other platforms
if torch.cuda.is_available():
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
    
    Supported Architectures:
    - Llama/Mistral/Qwen/Gemma/Phi-3 (model.layers, embed_tokens, norm)
    - GPT-2/GPT-J/Falcon/GPT-NeoX (transformer.h, wte, ln_f)
    - BERT/RoBERTa (encoder.layer, embeddings)
    - Mamba/SSM (backbone.layers)
    - Passthrough (any model - uses model directly as router)
    """
    
    # Architecture detection patterns
    ARCH_PATTERNS = {
        'llama': {'base': 'model', 'layers': 'layers', 'embed': 'embed_tokens', 'norm': 'norm'},
        'gpt': {'base': 'transformer', 'layers': 'h', 'embed': 'wte', 'norm': 'ln_f'},
        'bert': {'base': 'encoder', 'layers': 'layer', 'embed': None, 'norm': None},
        'mamba': {'base': 'backbone', 'layers': 'layers', 'embed': 'embedding', 'norm': 'norm_f'},
    }
    
    def __init__(self, model: nn.Module, num_layers: int = 2, architecture: str = 'auto'):
        super().__init__()
        import copy
        
        self.num_layers = num_layers
        self.style = architecture
        self.passthrough = False
        
        # Auto-detect architecture
        if architecture == 'auto':
            self.style = self._detect_architecture(model)
        
        # Special case: passthrough mode (use model directly)
        if self.style == 'passthrough':
            self.passthrough = True
            self.model = model
            self.head = model.lm_head if hasattr(model, 'lm_head') else None
            return
        
        # Get architecture pattern
        pattern = self.ARCH_PATTERNS.get(self.style)
        if pattern is None:
            raise ValueError(
                f"Unknown architecture '{self.style}'. Supported: {list(self.ARCH_PATTERNS.keys())} or 'passthrough'. "
                f"Use architecture='passthrough' to use the full model as router."
            )
        
        # Get base model
        base_model = getattr(model, pattern['base'], None)
        if base_model is None:
            raise ValueError(
                f"Model does not have '{pattern['base']}' attribute expected for {self.style} architecture. "
                f"Detected attributes: {[a for a in dir(model) if not a.startswith('_')]}. "
                f"Try architecture='passthrough' or provide a custom router."
            )
        
        # Clone config and truncate layers
        config = copy.deepcopy(model.config)
        self._truncate_config(config, num_layers)
        
        # Instantiate mini-model (random weights initially)
        cls = base_model.__class__
        try:
            self.mini_model = cls(config)
        except Exception as e:
            raise ValueError(
                f"Failed to create truncated model: {e}. "
                f"Try architecture='passthrough' or provide a custom router."
            )
        
        # Share weights
        self._share_weights(base_model, model, pattern, num_layers)
            
    def _detect_architecture(self, model: nn.Module) -> str:
        """Auto-detect model architecture."""
        # Check for each known pattern
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return 'llama'
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            return 'gpt'
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            return 'bert'
        elif hasattr(model, 'backbone') and hasattr(model.backbone, 'layers'):
            return 'mamba'
        else:
            # Fallback to passthrough
            import warnings
            warnings.warn(
                f"Could not auto-detect architecture for {type(model).__name__}. "
                f"Using passthrough mode (full model as router). "
                f"For better performance, provide a custom router or specify architecture explicitly."
            )
            return 'passthrough'
    
    def _truncate_config(self, config, num_layers: int):
        """Truncate layer count in config."""
        if hasattr(config, 'num_hidden_layers'):
            config.num_hidden_layers = num_layers
        elif hasattr(config, 'n_layer'):
            config.n_layer = num_layers
        elif hasattr(config, 'num_layers'):
            config.num_layers = num_layers
        elif hasattr(config, 'n_layers'):
            config.n_layers = num_layers
    
    def _share_weights(self, base_model: nn.Module, full_model: nn.Module, pattern: dict, num_layers: int):
        """Share weights between full model and mini model."""
        # 1. Embeddings
        embed_attr = pattern.get('embed')
        if embed_attr and hasattr(base_model, embed_attr):
            setattr(self.mini_model, embed_attr, getattr(base_model, embed_attr))
        # GPT-style also has positional embeddings
        if self.style == 'gpt' and hasattr(base_model, 'wpe'):
            self.mini_model.wpe = base_model.wpe
        
        # 2. Layers
        layers_attr = pattern.get('layers')
        if layers_attr:
            src_layers = getattr(base_model, layers_attr, None)
            dst_layers = getattr(self.mini_model, layers_attr, None)
            if src_layers is not None and dst_layers is not None:
                for i in range(min(num_layers, len(src_layers), len(dst_layers))):
                    dst_layers[i] = src_layers[i]
        
        # 3. Norm
        norm_attr = pattern.get('norm')
        if norm_attr and hasattr(base_model, norm_attr):
            setattr(self.mini_model, norm_attr, getattr(base_model, norm_attr))
        
        # 4. Rotary Embeddings (if present)
        if hasattr(base_model, 'rotary_emb'):
            self.mini_model.rotary_emb = base_model.rotary_emb
        
        # 5. Head
        self.head = full_model.lm_head if hasattr(full_model, 'lm_head') else None
            
    def forward(self, input_ids: torch.Tensor, **kwargs):
        # Passthrough mode: use full model directly
        if self.passthrough:
            outputs = self.model(input_ids, **kwargs)
            if hasattr(outputs, 'logits'):
                return outputs.logits
            return outputs[0] if isinstance(outputs, tuple) else outputs
        
        # Forward through mini-base-model
        outputs = self.mini_model(input_ids, **kwargs)
        
        # Get hidden states
        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        
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


