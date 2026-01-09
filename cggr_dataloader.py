"""
CGGR Difficulty-Aware Data Loader
=================================
CPU-side pre-filtering of easy batches using a tiny "nano-router".
Skips batches that are predicted to be mostly easy BEFORE they hit the GPU.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from typing import Iterator, Optional, Callable, Any, Dict
import threading
from queue import Queue


class NanoRouter(nn.Module):
    """
    Ultra-lightweight difficulty scorer for CPU-side pre-filtering.
    
    A single embedding + linear layer that approximates the full router's
    difficulty scores at 100x less compute.
    """
    
    def __init__(
        self, 
        vocab_size: int = 32000,
        hidden_dim: int = 64,
        pretrained_embeddings: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        if pretrained_embeddings is not None:
            # Use compressed version of real embeddings
            self.embed = nn.Embedding.from_pretrained(
                pretrained_embeddings[:, :hidden_dim].contiguous(),
                freeze=True
            )
        else:
            self.embed = nn.Embedding(vocab_size, hidden_dim)
        
        # Simple difficulty predictor
        self.difficulty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute difficulty scores.
        
        Args:
            input_ids: (batch, seq) token IDs
            
        Returns:
            difficulty: (batch,) mean difficulty per sequence
        """
        # (batch, seq, hidden)
        embeddings = self.embed(input_ids)
        
        # (batch, seq, 1) -> (batch, seq)
        token_difficulty = self.difficulty_head(embeddings).squeeze(-1)
        
        # Mean difficulty per sequence
        return token_difficulty.mean(dim=-1)
    
    @classmethod
    def from_model(cls, model: nn.Module, hidden_dim: int = 64) -> 'NanoRouter':
        """
        Create a NanoRouter initialized from a full model's embeddings.
        
        This gives the nano-router a head start by using real token representations.
        """
        # Find embeddings in various architectures
        embeddings = None
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            embeddings = model.model.embed_tokens.weight.data
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
            embeddings = model.transformer.wte.weight.data
        elif hasattr(model, 'embed_tokens'):
            embeddings = model.embed_tokens.weight.data
        
        if embeddings is not None:
            vocab_size = embeddings.shape[0]
            # Compress to hidden_dim via truncation (simple but effective)
            compressed = embeddings[:, :hidden_dim].cpu()
            return cls(vocab_size, hidden_dim, pretrained_embeddings=compressed)
        
        # Fallback: random init
        vocab_size = getattr(model.config, 'vocab_size', 32000)
        return cls(vocab_size, hidden_dim)
    
    def calibrate(
        self, 
        dataloader: DataLoader, 
        full_scorer: Callable[[torch.Tensor], torch.Tensor],
        num_batches: int = 100,
        lr: float = 1e-3,
    ):
        """
        Calibrate the nano-router against a full scorer.
        
        Trains the difficulty_head to match the full router's difficulty predictions.
        """
        optimizer = torch.optim.Adam(self.difficulty_head.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        self.train()
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            input_ids = batch['input_ids'] if isinstance(batch, dict) else batch[0]
            
            # Get target from full scorer
            with torch.no_grad():
                difficulty, _, _ = full_scorer(input_ids)
                target_difficulty = difficulty.view(input_ids.shape[0], -1).mean(dim=-1)
            
            # Predict
            pred_difficulty = self(input_ids.cpu())
            
            # Loss
            loss = criterion(pred_difficulty, target_difficulty.cpu())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 20 == 0:
                print(f"Calibration batch {i+1}/{num_batches}, Loss: {loss.item():.4f}")
        
        self.eval()
        print("NanoRouter calibration complete.")


class DifficultyFilteredDataLoader:
    """
    DataLoader wrapper that pre-filters easy batches on CPU.
    
    Uses a NanoRouter to estimate batch difficulty and skips batches
    that are predicted to be mostly easy (below threshold).
    
    Usage:
        nano_router = NanoRouter.from_model(model)
        filtered_loader = DifficultyFilteredDataLoader(
            dataloader,
            nano_router,
            threshold=0.3,  # Skip batches with mean difficulty < 0.3
        )
        
        for batch in filtered_loader:
            # Only hard batches reach here
            loss = model(batch)
    """
    
    def __init__(
        self,
        dataloader: DataLoader,
        nano_router: NanoRouter,
        threshold: float = 0.3,
        input_key: str = 'input_ids',
        skip_ratio: float = 0.5,  # Max fraction of batches to skip
    ):
        self.dataloader = dataloader
        self.nano_router = nano_router.eval()
        self.threshold = threshold
        self.input_key = input_key
        self.skip_ratio = skip_ratio
        
        # Stats
        self.total_batches = 0
        self.skipped_batches = 0
        
    def __iter__(self) -> Iterator[Any]:
        self.total_batches = 0
        self.skipped_batches = 0
        max_consecutive_skips = int(1.0 / (1.0 - self.skip_ratio)) if self.skip_ratio < 1.0 else float('inf')
        consecutive_skips = 0
        
        for batch in self.dataloader:
            self.total_batches += 1
            
            # Extract input_ids
            if isinstance(batch, dict):
                input_ids = batch[self.input_key]
            elif isinstance(batch, (tuple, list)):
                input_ids = batch[0]
            else:
                input_ids = batch
            
            # Score on CPU
            with torch.no_grad():
                difficulty = self.nano_router(input_ids.cpu())
                mean_difficulty = difficulty.mean().item()
            
            # Skip if too easy (but respect max skip ratio)
            if mean_difficulty < self.threshold and consecutive_skips < max_consecutive_skips:
                self.skipped_batches += 1
                consecutive_skips += 1
                continue
            
            consecutive_skips = 0
            yield batch
    
    def __len__(self):
        # Estimate based on skip ratio
        return max(1, int(len(self.dataloader) * (1 - self.skip_ratio * 0.5)))
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_batches': self.total_batches,
            'skipped_batches': self.skipped_batches,
            'skip_rate': self.skipped_batches / max(1, self.total_batches),
            'threshold': self.threshold,
        }


class AsyncDifficultyFilteredDataLoader:
    """
    Async version that pre-filters in a background thread.
    
    Maintains a queue of "hard" batches so the main training loop
    never waits for filtering.
    """
    
    def __init__(
        self,
        dataloader: DataLoader,
        nano_router: NanoRouter,
        threshold: float = 0.3,
        queue_size: int = 4,
        input_key: str = 'input_ids',
    ):
        self.dataloader = dataloader
        self.nano_router = nano_router.eval()
        self.threshold = threshold
        self.queue_size = queue_size
        self.input_key = input_key
        
        self._queue: Queue = Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        
        # Stats
        self.total_batches = 0
        self.skipped_batches = 0
        
    def _producer(self):
        """Background thread that filters batches."""
        for batch in self.dataloader:
            if self._stop_event.is_set():
                break
                
            self.total_batches += 1
            
            # Extract and score
            if isinstance(batch, dict):
                input_ids = batch[self.input_key]
            else:
                input_ids = batch[0] if isinstance(batch, (tuple, list)) else batch
            
            with torch.no_grad():
                difficulty = self.nano_router(input_ids.cpu())
                mean_difficulty = difficulty.mean().item()
            
            if mean_difficulty < self.threshold:
                self.skipped_batches += 1
                continue
            
            self._queue.put(batch)
        
        # Signal end
        self._queue.put(None)
    
    def __iter__(self):
        self.total_batches = 0
        self.skipped_batches = 0
        self._stop_event.clear()
        
        # Start producer thread
        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._thread.start()
        
        while True:
            batch = self._queue.get()
            if batch is None:
                break
            yield batch
        
        self._thread.join()
    
    def stop(self):
        """Stop the background thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_batches': self.total_batches,
            'skipped_batches': self.skipped_batches,
            'skip_rate': self.skipped_batches / max(1, self.total_batches),
            'threshold': self.threshold,
        }
