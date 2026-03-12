"""
CGGR - Confidence-Gated Gradient Routing
=========================================
Selective loss computation with multiple strategies.
All operations accelerated with fused Triton kernels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Literal, Optional

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


VALID_SCORING = {'entropy', 'margin', 'loss', 'combined'}
VALID_SELECTION = {'topk', 'stratified', 'sequence_aware', 'fixed_quota'}


def _validate_routing_config(
    *,
    scoring: str,
    selection: str,
    min_tokens_ratio: float,
    warmup_steps: int,
    threshold_sensitivity: float,
    num_strata: int,
    min_tokens_per_sequence: int,
):
    if scoring not in VALID_SCORING:
        raise ValueError(f"Unsupported scoring mode '{scoring}'. Expected one of {sorted(VALID_SCORING)}.")
    if selection not in VALID_SELECTION:
        raise ValueError(f"Unsupported selection mode '{selection}'. Expected one of {sorted(VALID_SELECTION)}.")
    if not (0.0 < min_tokens_ratio <= 1.0):
        raise ValueError(f"min_tokens_ratio must be in (0, 1], got {min_tokens_ratio}.")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}.")
    if not (0.0 <= threshold_sensitivity <= 1.0):
        raise ValueError(f"threshold_sensitivity must be in [0, 1], got {threshold_sensitivity}.")
    if num_strata < 1:
        raise ValueError(f"num_strata must be >= 1, got {num_strata}.")
    if min_tokens_per_sequence < 1:
        raise ValueError(f"min_tokens_per_sequence must be >= 1, got {min_tokens_per_sequence}.")


def _validate_logits_targets_shapes(logits: torch.Tensor, targets: torch.Tensor):
    if logits.dim() == 3:
        if targets.dim() != 2:
            raise ValueError(
                f"For 3D logits (batch, seq, vocab), targets must be 2D (batch, seq). Got targets dim={targets.dim()}."
            )
        if logits.shape[:2] != targets.shape:
            raise ValueError(
                f"Logits batch/seq shape {tuple(logits.shape[:2])} does not match targets shape {tuple(targets.shape)}."
            )
    elif logits.dim() == 2:
        if targets.dim() != 1:
            raise ValueError(
                f"For 2D logits (N, vocab), targets must be 1D (N,). Got targets dim={targets.dim()}."
            )
        if logits.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Logits first dimension {logits.shape[0]} does not match targets length {targets.shape[0]}."
            )
    else:
        raise ValueError(
            f"Unsupported logits shape {tuple(logits.shape)}. Expected 2D (N, vocab) or 3D (batch, seq, vocab)."
        )


def _compute_base_ratio(
    step_count: torch.Tensor,
    warmup_steps: int,
    min_tokens_ratio: float,
) -> float:
    """Linear curriculum from 1.0 to min_tokens_ratio."""
    if warmup_steps <= 0:
        progress = 1.0
    else:
        progress = min(1.0, step_count.item() / warmup_steps)
    return 1.0 - progress * (1.0 - min_tokens_ratio)


def _select_mask(
    difficulty: torch.Tensor,
    confidence: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    step_count: torch.Tensor,
    min_tokens_ratio: float,
    warmup_steps: int,
    dynamic_threshold: bool,
    threshold_sensitivity: float,
    selection: str,
    num_strata: int,
    min_tokens_per_sequence: int,
) -> tuple:
    """Shared token-selection logic for loss and scorer wrappers."""
    base_ratio = _compute_base_ratio(step_count, warmup_steps, min_tokens_ratio)

    if selection == 'fixed_quota':
        current_ratio = base_ratio
    elif dynamic_threshold:
        current_ratio = compute_dynamic_threshold(
            confidence.view(-1), base_ratio, threshold_sensitivity
        )
    else:
        current_ratio = base_ratio

    if selection == 'stratified':
        mask = select_tokens_stratified(difficulty, current_ratio, num_strata)
    else:
        mask = select_tokens_topk(difficulty, current_ratio)
        if selection == 'sequence_aware':
            mask = ensure_sequence_coverage(
                difficulty, mask, batch_size, seq_len, min_tokens_per_sequence
            )

    return mask, current_ratio


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

        _validate_routing_config(
            scoring=scoring,
            selection=selection,
            min_tokens_ratio=min_tokens_ratio,
            warmup_steps=warmup_steps,
            threshold_sensitivity=threshold_sensitivity,
            num_strata=num_strata,
            min_tokens_per_sequence=min_tokens_per_sequence,
        )

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
        _validate_logits_targets_shapes(logits, targets)

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
            use_targets = self.scoring in {'loss', 'combined'}
            difficulty, confidence, entropy = fused_difficulty_score(
                logits_flat.unsqueeze(0) if logits_flat.dim() == 2 else logits_flat,
                targets=targets_flat if use_targets else None,
                mode=self.scoring,
            )
            difficulty = difficulty.view(-1)
            confidence = confidence.view(-1)
            entropy = entropy.view(-1)

            mask, current_ratio = _select_mask(
                difficulty,
                confidence,
                batch_size=batch_size,
                seq_len=seq_len,
                step_count=self.step_count,
                min_tokens_ratio=self.min_tokens_ratio,
                warmup_steps=self.warmup_steps,
                dynamic_threshold=self.dynamic_threshold,
                threshold_sensitivity=self.threshold_sensitivity,
                selection=self.selection,
                num_strata=self.num_strata,
                min_tokens_per_sequence=self.min_tokens_per_sequence,
            )
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
        'llama': {
            'base_candidates': ['model', ''],
            'layers': 'layers',
            'embed': 'embed_tokens',
            'norm': 'norm',
        },
        'gpt': {
            'base_candidates': ['transformer', ''],
            'layers': 'h',
            'embed': 'wte',
            'norm': 'ln_f',
        },
        'bert': {
            'base_candidates': ['bert', 'roberta', 'deberta', 'deberta_v2', 'electra', ''],
            'layers': 'encoder.layer',
            'embed': 'embeddings',
            'norm': None,
        },
        'mamba': {
            'base_candidates': ['backbone', ''],
            'layers': 'layers',
            'embed': 'embedding',
            'norm': 'norm_f',
        },
    }
    
    def __init__(self, model: nn.Module, num_layers: int = 2, architecture: str = 'auto'):
        super().__init__()
        import copy
        import warnings
        
        self.num_layers = num_layers
        self.style = architecture
        self.passthrough = False
        self._auto_architecture = architecture == 'auto'
        
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
        base_model = self._resolve_base_model(model, pattern)
        if base_model is None:
            self._fallback_or_raise(
                model,
                warnings,
                ValueError(
                    f"Model does not expose the expected base module for '{self.style}'. "
                    f"Try architecture='passthrough' or provide a custom router."
                ),
            )
            return
        
        # Clone config and truncate layers
        config = copy.deepcopy(model.config)
        self._truncate_config(config, num_layers)
        
        # Instantiate mini-model (random weights initially)
        cls = base_model.__class__
        try:
            self.mini_model = cls(config)
        except Exception as e:
            self._fallback_or_raise(
                model,
                warnings,
                ValueError(
                    f"Failed to create truncated model: {e}. "
                    f"Try architecture='passthrough' or provide a custom router."
                ),
            )
            return
        
        # Share weights
        self._share_weights(base_model, model, pattern, num_layers)

        if self.head is None:
            self._fallback_or_raise(
                model,
                warnings,
                ValueError(
                    f"Could not resolve an output head for '{type(model).__name__}'. "
                    f"Provide a custom router or use a model wrapper with logits."
                ),
            )
            return

    @staticmethod
    def _get_nested_attr(obj: nn.Module, path: str):
        if path in (None, ''):
            return obj
        current = obj
        for part in path.split('.'):
            current = getattr(current, part, None)
            if current is None:
                return None
        return current

    @staticmethod
    def _set_nested_attr(obj: nn.Module, path: str, value):
        if path in (None, ''):
            raise ValueError("Cannot set empty attribute path")
        parts = path.split('.')
        parent = obj
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], value)

    def _resolve_base_model(self, model: nn.Module, pattern: dict):
        for candidate in pattern.get('base_candidates', []):
            base_model = self._get_nested_attr(model, candidate)
            if base_model is None:
                continue
            layers = self._get_nested_attr(base_model, pattern.get('layers'))
            if layers is not None:
                return base_model
        return None

    def _resolve_output_head(self, model: nn.Module):
        if hasattr(model, 'lm_head'):
            return model.lm_head
        if hasattr(model, 'get_output_embeddings'):
            head = model.get_output_embeddings()
            if head is not None:
                return head
        if hasattr(model, 'cls') and hasattr(model.cls, 'predictions') and hasattr(model.cls.predictions, 'decoder'):
            return model.cls.predictions.decoder
        return None

    def _fallback_or_raise(self, model: nn.Module, warnings, error: Exception):
        if self._auto_architecture:
            warnings.warn(f"{error} Falling back to passthrough router.")
            self.passthrough = True
            self.model = model
            self.head = self._resolve_output_head(model)
            return
        raise error
            
    def _detect_architecture(self, model: nn.Module) -> str:
        """Auto-detect model architecture."""
        import warnings

        # Check for each known pattern
        if (
            (hasattr(model, 'model') and hasattr(model.model, 'layers'))
            or (hasattr(model, 'layers') and hasattr(model, 'embed_tokens'))
        ):
            return 'llama'
        elif (
            (hasattr(model, 'transformer') and hasattr(model.transformer, 'h'))
            or (hasattr(model, 'h') and hasattr(model, 'wte'))
        ):
            return 'gpt'
        elif (
            (hasattr(model, 'bert') and hasattr(model.bert, 'encoder') and hasattr(model.bert.encoder, 'layer'))
            or (hasattr(model, 'roberta') and hasattr(model.roberta, 'encoder') and hasattr(model.roberta.encoder, 'layer'))
            or (hasattr(model, 'deberta') and hasattr(model.deberta, 'encoder') and hasattr(model.deberta.encoder, 'layer'))
            or (hasattr(model, 'deberta_v2') and hasattr(model.deberta_v2, 'encoder') and hasattr(model.deberta_v2.encoder, 'layer'))
            or (hasattr(model, 'electra') and hasattr(model.electra, 'encoder') and hasattr(model.electra.encoder, 'layer'))
            or (hasattr(model, 'encoder') and hasattr(model.encoder, 'layer') and hasattr(model, 'embeddings'))
        ):
            return 'bert'
        elif (
            (hasattr(model, 'backbone') and hasattr(model.backbone, 'layers'))
            or (hasattr(model, 'layers') and hasattr(model, 'embedding'))
        ):
            return 'mamba'
        else:
            # Fallback to passthrough
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
        src_embed = self._get_nested_attr(base_model, embed_attr)
        dst_embed = self._get_nested_attr(self.mini_model, embed_attr)
        if embed_attr and src_embed is not None and dst_embed is not None:
            self._set_nested_attr(self.mini_model, embed_attr, src_embed)
        # GPT-style also has positional embeddings
        if self.style == 'gpt' and hasattr(base_model, 'wpe'):
            self.mini_model.wpe = base_model.wpe
        
        # 2. Layers
        layers_attr = pattern.get('layers')
        if layers_attr:
            src_layers = self._get_nested_attr(base_model, layers_attr)
            dst_layers = self._get_nested_attr(self.mini_model, layers_attr)
            if src_layers is not None and dst_layers is not None:
                for i in range(min(num_layers, len(src_layers), len(dst_layers))):
                    dst_layers[i] = src_layers[i]
        
        # 3. Norm
        norm_attr = pattern.get('norm')
        src_norm = self._get_nested_attr(base_model, norm_attr)
        dst_norm = self._get_nested_attr(self.mini_model, norm_attr)
        if norm_attr and src_norm is not None and dst_norm is not None:
            self._set_nested_attr(self.mini_model, norm_attr, src_norm)
        
        # 4. Rotary Embeddings (if present)
        if hasattr(base_model, 'rotary_emb'):
            self.mini_model.rotary_emb = base_model.rotary_emb
        
        # 5. Head
        self.head = self._resolve_output_head(full_model)
            
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
        scoring: Literal['entropy', 'margin', 'loss', 'combined'] = 'combined',
        min_tokens_ratio: float = 0.25,
        warmup_steps: int = 1000,
        dynamic_threshold: bool = True,
        threshold_sensitivity: float = 0.5,
        selection: Literal['topk', 'stratified', 'sequence_aware', 'fixed_quota'] = 'topk',
        num_strata: int = 4,
        min_tokens_per_sequence: int = 1,
    ):
        super().__init__()
        _validate_routing_config(
            scoring=scoring,
            selection=selection,
            min_tokens_ratio=min_tokens_ratio,
            warmup_steps=warmup_steps,
            threshold_sensitivity=threshold_sensitivity,
            num_strata=num_strata,
            min_tokens_per_sequence=min_tokens_per_sequence,
        )
        self.router = router
        self.scoring = scoring
        self.min_tokens_ratio = min_tokens_ratio
        self.warmup_steps = warmup_steps
        self.dynamic_threshold = dynamic_threshold
        self.threshold_sensitivity = threshold_sensitivity
        self.selection = selection
        self.num_strata = num_strata
        self.min_tokens_per_sequence = min_tokens_per_sequence
        
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))

    def step(self):
        """Call after optimizer.step()"""
        self.step_count += 1

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Returns difficulty scores and boolean mask of hard tokens.
        
        Returns:
            difficulty: (batch, seq) float scores
            mask: (batch, seq) boolean-like mask (True/1=Hard/Keep)
            info: dict with metrics
        """
        if input_ids.dim() != 2:
            raise ValueError(
                f"CGGRScorer expects input_ids with shape (batch, seq). Got shape {tuple(input_ids.shape)}."
            )
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError(
                f"targets must match input_ids shape {tuple(input_ids.shape)}, got {tuple(targets.shape)}."
            )

        with torch.no_grad():
            outputs = self.router(input_ids, **kwargs)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Compute difficulty
            difficulty, confidence, entropy = fused_difficulty_score(
                logits,
                targets=targets if self.scoring in {'loss', 'combined'} else None,
                mode=self.scoring,
            )

            batch_size, seq_len = input_ids.shape
            mask, current_ratio = _select_mask(
                difficulty,
                confidence,
                batch_size=batch_size,
                seq_len=seq_len,
                step_count=self.step_count,
                min_tokens_ratio=self.min_tokens_ratio,
                warmup_steps=self.warmup_steps,
                dynamic_threshold=self.dynamic_threshold,
                threshold_sensitivity=self.threshold_sensitivity,
                selection=self.selection,
                num_strata=self.num_strata,
                min_tokens_per_sequence=self.min_tokens_per_sequence,
            )

            info = {
                'confidence': confidence,
                'entropy': entropy,
                'current_ratio': current_ratio,
                'tokens_selected': int(mask.view(-1).sum().item()),
                'tokens_total': int(mask.numel()),
                'selection': self.selection,
                'scoring': self.scoring,
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
        scoring: Literal['entropy', 'margin', 'loss', 'combined'] = 'combined',
        min_tokens_ratio: float = 0.25,
        warmup_steps: int = 1000,
        dynamic_threshold: bool = True,
        threshold_sensitivity: float = 0.5,
        selection: Literal['topk', 'stratified', 'sequence_aware', 'fixed_quota'] = 'topk',
        num_strata: int = 4,
        min_tokens_per_sequence: int = 1,
    ):
        super().__init__()
        if model is None:
            raise ValueError("CGGRModel requires a non-null model.")
        if router is not None and not isinstance(router, nn.Module):
            raise ValueError("router must be an nn.Module when provided.")
        self.model = model
        router_model = router if router is not None else model
        self.router = router_model
        
        # Use common scorer
        self.scorer = CGGRScorer(
            router=router_model,
            scoring=scoring,
            min_tokens_ratio=min_tokens_ratio,
            warmup_steps=warmup_steps,
            dynamic_threshold=dynamic_threshold,
            threshold_sensitivity=threshold_sensitivity,
            selection=selection,
            num_strata=num_strata,
            min_tokens_per_sequence=min_tokens_per_sequence,
        )
        self.metrics = {}
    
    def step(self):
        self.scorer.step()
    
    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None, **kwargs):
        if input_ids.dim() != 2:
            raise ValueError(
                f"CGGRModel expects input_ids with shape (batch, seq). Got shape {tuple(input_ids.shape)}."
            )
        if labels is None:
            return self.model(input_ids, **kwargs)
        if labels.shape != input_ids.shape:
            raise ValueError(
                f"labels must match input_ids shape {tuple(input_ids.shape)}, got {tuple(labels.shape)}."
            )

        batch_size, seq_len = input_ids.shape
        
        # PASS 1: Get scores from scorer
        difficulty, mask, info = self.scorer(input_ids, targets=labels, **kwargs)
        
        current_ratio = info['current_ratio']
        confidence = info['confidence']
        entropy = info['entropy']
        
        # Get hard token indices
        hard_mask = mask.view(batch_size, seq_len) > 0.5
        tokens_total = batch_size * seq_len
        router_tokens_selected = hard_mask.sum().item()
        hard_sequence_mask = hard_mask.any(dim=-1)
        hard_seq_indices = torch.nonzero(hard_sequence_mask, as_tuple=True)[0]
        sequences_selected = hard_seq_indices.numel()
        sequence_selection_mode = 'mask_any'

        # With long sequences, "any hard token" can collapse to selecting every sequence
        # even for small token ratios. In that case, fall back to ratio-based sequence quota.
        if sequences_selected == batch_size and current_ratio < 1.0:
            k_seq = max(1, math.ceil(batch_size * current_ratio))
            seq_scores = difficulty.float().mean(dim=-1)
            hard_seq_indices = torch.topk(seq_scores, k=k_seq).indices
            sequences_selected = hard_seq_indices.numel()
            sequence_selection_mode = 'quota_fallback'

        # PASS 2: Main model forward
        if sequences_selected > 0:
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
            
            tokens_selected = sequences_selected * max(seq_len - 1, 0)
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
            sequences_selected = batch_size
        
        self.metrics = {
            'step': self.scorer.step_count.item(),
            'token_ratio': current_ratio,
            'tokens_selected': int(tokens_selected),
            'tokens_total': tokens_total,
            'router_tokens_selected': int(router_tokens_selected),
            'sequences_selected': int(sequences_selected),
            'avg_confidence': confidence.mean().item(),
            'avg_entropy': entropy.mean().item(),
            'sequence_selection_mode': sequence_selection_mode,
            'selection': self.scorer.selection,
            'scoring': self.scorer.scoring,
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

# Persistent CGGR Kernels (optional - for Token-Routed MLP optimization)
try:
    from persistent_cggr_kernels import (
        PersistentTRMLP,
        PersistentKernelConfig,
        create_persistent_tr_mlp,
        persistent_grouped_gemm,
    )
    __all__.extend([
        'PersistentTRMLP',
        'PersistentKernelConfig', 
        'create_persistent_tr_mlp',
        'persistent_grouped_gemm',
    ])
except ImportError:
    pass
