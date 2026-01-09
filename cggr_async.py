"""
CGGR Async Prefetching
======================
Use background CUDA streams to score Batch N+1 while Batch N is training.
Hides the routing latency entirely.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class PrefetchedMask:
    """Container for prefetched scoring results."""
    difficulty: torch.Tensor
    mask: torch.Tensor
    info: Dict[str, Any]
    ready: bool = False


class AsyncCGGRScorer(nn.Module):
    """
    Asynchronous CGGR scorer using CUDA streams.
    
    Scores the next batch on a background stream while the main model trains.
    
    Usage:
        async_scorer = AsyncCGGRScorer(router, min_tokens_ratio=0.25)
        
        for i, batch in enumerate(dataloader):
            # Prefetch scoring for NEXT batch (non-blocking)
            if i + 1 < len(dataloader):
                async_scorer.prefetch(next_batch['input_ids'])
            
            # Get mask for CURRENT batch (blocking if not ready)
            difficulty, mask, info = async_scorer.get_mask(batch['input_ids'])
            
            # Train on current batch
            loss = model(batch['input_ids'][mask], ...)
    """
    
    def __init__(
        self,
        router: nn.Module,
        min_tokens_ratio: float = 0.25,
        warmup_steps: int = 1000,
        selection: str = 'fixed_quota',
    ):
        super().__init__()
        from cggr import CGGRScorer
        
        self.scorer = CGGRScorer(
            router=router,
            min_tokens_ratio=min_tokens_ratio,
            warmup_steps=warmup_steps,
            selection=selection,
        )
        
        # Background stream for async scoring
        self._scoring_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self._prefetched: Optional[PrefetchedMask] = None
        self._prefetch_input_ids: Optional[torch.Tensor] = None
        self._event: Optional[torch.cuda.Event] = None
        
    def step(self):
        """Call after optimizer.step()"""
        self.scorer.step()
    
    def prefetch(self, input_ids: torch.Tensor, **kwargs):
        """
        Non-blocking: Start scoring on background stream.
        
        Call this with the NEXT batch's input_ids while the current batch is training.
        """
        if self._scoring_stream is None:
            # CPU fallback: just compute synchronously
            self._prefetched = self._score_sync(input_ids, **kwargs)
            return
        
        # Store input for verification during get_mask
        self._prefetch_input_ids = input_ids
        
        # Create event to track completion
        self._event = torch.cuda.Event()
        
        # Launch scoring on background stream
        with torch.cuda.stream(self._scoring_stream):
            difficulty, mask, info = self.scorer(input_ids, **kwargs)
            self._prefetched = PrefetchedMask(
                difficulty=difficulty,
                mask=mask,
                info=info,
                ready=True
            )
        
        # Record event for synchronization
        self._event.record(self._scoring_stream)
    
    def get_mask(self, input_ids: torch.Tensor, **kwargs) -> tuple:
        """
        Get the mask for the given input_ids.
        
        If this matches the prefetched batch, returns immediately (after sync).
        Otherwise, computes synchronously.
        """
        # Check if we have a prefetched result for this input
        if (
            self._prefetched is not None 
            and self._prefetch_input_ids is not None
            and input_ids.shape == self._prefetch_input_ids.shape
            and torch.equal(input_ids, self._prefetch_input_ids)
        ):
            # Wait for prefetch to complete
            if self._event is not None:
                self._event.synchronize()
            
            result = self._prefetched
            self._prefetched = None
            self._prefetch_input_ids = None
            return result.difficulty, result.mask, result.info
        
        # Cache miss: compute synchronously
        return self._score_sync(input_ids, **kwargs)
    
    def _score_sync(self, input_ids: torch.Tensor, **kwargs):
        """Synchronous scoring fallback."""
        difficulty, mask, info = self.scorer(input_ids, **kwargs)
        return difficulty, mask, info
    
    def forward(self, input_ids: torch.Tensor, **kwargs):
        """Alias for get_mask for nn.Module compatibility."""
        return self.get_mask(input_ids, **kwargs)


class AsyncCGGRModel(nn.Module):
    """
    CGGR Model with integrated async prefetching.
    
    Automatically prefetches scoring for the next batch during training.
    
    Usage:
        model = AsyncCGGRModel(base_model)
        
        for batch, next_batch in zip(dataloader, dataloader[1:] + [None]):
            loss = model(batch['input_ids'], labels=batch['labels'], 
                        next_input_ids=next_batch['input_ids'] if next_batch else None)
    """
    
    def __init__(
        self,
        model: nn.Module,
        router: Optional[nn.Module] = None,
        min_tokens_ratio: float = 0.25,
        warmup_steps: int = 1000,
    ):
        super().__init__()
        self.model = model
        router_model = router if router is not None else model
        
        self.async_scorer = AsyncCGGRScorer(
            router=router_model,
            min_tokens_ratio=min_tokens_ratio,
            warmup_steps=warmup_steps,
        )
        self.metrics = {}
    
    def step(self):
        self.async_scorer.step()
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        labels: torch.Tensor = None,
        next_input_ids: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass with async prefetching.
        
        Args:
            input_ids: Current batch input
            labels: Current batch labels
            next_input_ids: NEXT batch input (for prefetching). Optional.
        """
        if labels is None:
            return self.model(input_ids, **kwargs)
        
        batch_size, seq_len = input_ids.shape
        
        # Get mask for current batch (may use prefetched result)
        difficulty, mask, info = self.async_scorer.get_mask(input_ids, **kwargs)
        current_ratio = info['current_ratio']
        
        # Prefetch for next batch (non-blocking)
        if next_input_ids is not None:
            self.async_scorer.prefetch(next_input_ids, **kwargs)
        
        # Select hard sequences
        seq_difficulty = difficulty.view(batch_size, seq_len).mean(dim=-1)
        k = max(1, int(batch_size * current_ratio))
        _, hard_seq_indices = torch.topk(seq_difficulty, k)
        
        # Forward on hard sequences only
        hard_input_ids = input_ids[hard_seq_indices]
        hard_labels = labels[hard_seq_indices]
        
        hard_outputs = self.model(hard_input_ids, **kwargs)
        
        if hasattr(hard_outputs, 'logits'):
            hard_logits = hard_outputs.logits
        else:
            hard_logits = hard_outputs
        
        shift_logits = hard_logits[:, :-1, :].contiguous().view(-1, hard_logits.shape[-1])
        shift_labels = hard_labels[:, 1:].contiguous().view(-1)
        loss = nn.functional.cross_entropy(shift_logits, shift_labels)
        
        self.metrics = {
            'step': self.async_scorer.scorer.step_count.item(),
            'token_ratio': current_ratio,
            'tokens_selected': k * (seq_len - 1),
            'tokens_total': batch_size * seq_len,
            'prefetch_active': next_input_ids is not None,
        }
        
        return loss
    
    def get_metrics(self) -> dict:
        return self.metrics.copy()


def create_prefetching_iterator(dataloader, async_scorer: AsyncCGGRScorer):
    """
    Generator that automatically prefetches scoring for the next batch.
    
    Usage:
        async_scorer = AsyncCGGRScorer(router)
        for batch, mask_info in create_prefetching_iterator(dataloader, async_scorer):
            difficulty, mask, info = mask_info
            # Train...
    """
    iterator = iter(dataloader)
    
    try:
        current_batch = next(iterator)
    except StopIteration:
        return
    
    # Prefetch first batch
    input_key = 'input_ids' if 'input_ids' in current_batch else 0
    async_scorer.prefetch(current_batch[input_key])
    
    for next_batch in iterator:
        # Get mask for current batch
        mask_info = async_scorer.get_mask(current_batch[input_key])
        
        # Prefetch next batch
        async_scorer.prefetch(next_batch[input_key])
        
        yield current_batch, mask_info
        current_batch = next_batch
    
    # Last batch
    mask_info = async_scorer.get_mask(current_batch[input_key])
    yield current_batch, mask_info
