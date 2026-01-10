"""
Verification script for Tier 1 CGGR optimizations.
Tests all three components: Checkpointing, Async, DataLoader.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ===== Mock Model for Testing =====
class MockLlamaModel(nn.Module):
    """Minimal model that mimics Llama structure for testing."""
    def __init__(self, vocab_size=100, hidden_size=32, num_layers=2):
        super().__init__()
        self.config = type('Config', (), {
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'num_hidden_layers': num_layers,
        })()
        
        self.model = nn.ModuleDict({
            'embed_tokens': nn.Embedding(vocab_size, hidden_size),
            'layers': nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]),
            'norm': nn.LayerNorm(hidden_size),
        })
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        self._gradient_checkpointing = False
        
    def gradient_checkpointing_enable(self):
        self._gradient_checkpointing = True
        
    def gradient_checkpointing_disable(self):
        self._gradient_checkpointing = False
    
    def forward(self, input_ids, **kwargs):
        x = self.model['embed_tokens'](input_ids)
        for layer in self.model['layers']:
            x = layer(x)
        x = self.model['norm'](x)
        logits = self.lm_head(x)
        return type('Output', (), {'logits': logits})()


def test_checkpointing():
    print("\n=== Testing Selective Gradient Checkpointing ===")
    from cggr_checkpointing import CGGRCheckpointedModel
    
    model = MockLlamaModel()
    cggr = CGGRCheckpointedModel(model, min_tokens_ratio=0.5, warmup_steps=0, use_checkpointing=True)
    
    input_ids = torch.randint(0, 100, (4, 8))
    labels = torch.randint(0, 100, (4, 8))
    
    loss = cggr(input_ids, labels=labels)
    loss.backward()
    
    metrics = cggr.get_metrics()
    print(f"Metrics: {metrics}")
    assert metrics['checkpointing_enabled'] == True
    assert metrics['token_ratio'] == 0.5
    print("✓ Checkpointing test passed!")


def test_async_prefetch():
    print("\n=== Testing Async Prefetching ===")
    from cggr_async import AsyncCGGRScorer
    
    model = MockLlamaModel()
    scorer = AsyncCGGRScorer(router=model, min_tokens_ratio=0.4, warmup_steps=0)
    
    batch1 = torch.randint(0, 100, (4, 8))
    batch2 = torch.randint(0, 100, (4, 8))
    
    # Prefetch batch2 while we "process" batch1
    scorer.prefetch(batch2)
    
    # Get mask for batch1 (should compute synchronously since not prefetched)
    diff1, mask1, info1 = scorer.get_mask(batch1)
    print(f"Batch 1 info: {info1}")
    
    # Get mask for batch2 (should use prefetched result)
    diff2, mask2, info2 = scorer.get_mask(batch2)
    print(f"Batch 2 info: {info2}")
    
    assert diff1.shape == (4, 8)
    assert diff2.shape == (4, 8)
    print("✓ Async prefetch test passed!")


def test_dataloader():
    print("\n=== Testing Difficulty-Aware DataLoader ===")
    from cggr_dataloader import NanoRouter, DifficultyFilteredDataLoader
    
    # Create mock data
    data = TensorDataset(
        torch.randint(0, 100, (100, 16)),  # input_ids
        torch.randint(0, 100, (100, 16)),  # labels
    )
    loader = DataLoader(data, batch_size=10)
    
    # Create nano-router
    nano_router = NanoRouter(vocab_size=100, hidden_dim=32)
    
    # Wrap with filtering
    filtered_loader = DifficultyFilteredDataLoader(
        loader,
        nano_router,
        threshold=0.5,  # Skip batches with mean difficulty < 0.5
        input_key=0,  # TensorDataset uses integer keys
    )
    
    # Iterate
    batch_count = 0
    for batch in filtered_loader:
        batch_count += 1
    
    stats = filtered_loader.get_stats()
    print(f"Stats: {stats}")
    print(f"Processed {batch_count} batches out of {stats['total_batches']} total")
    assert batch_count <= stats['total_batches']
    print("✓ DataLoader test passed!")


def test_integration():
    print("\n=== Testing Full Integration ===")
    from cggr_checkpointing import CGGRCheckpointedModel
    from cggr_dataloader import NanoRouter, DifficultyFilteredDataLoader
    
    model = MockLlamaModel()
    cggr = CGGRCheckpointedModel(model, min_tokens_ratio=0.5, warmup_steps=0)
    
    # Create filtered dataloader
    data = TensorDataset(
        torch.randint(0, 100, (50, 16)),
        torch.randint(0, 100, (50, 16)),
    )
    loader = DataLoader(data, batch_size=10)
    nano_router = NanoRouter(vocab_size=100, hidden_dim=32)
    filtered_loader = DifficultyFilteredDataLoader(loader, nano_router, threshold=0.3, input_key=0)
    
    # Training loop
    total_loss = 0
    for batch in filtered_loader:
        input_ids, labels = batch
        loss = cggr(input_ids, labels=labels)
        loss.backward()
        total_loss += loss.item()
        cggr.step()
    
    print(f"Total loss: {total_loss:.4f}")
    print(f"DataLoader stats: {filtered_loader.get_stats()}")
    print("✓ Integration test passed!")


if __name__ == "__main__":
    test_checkpointing()
    test_async_prefetch()
    test_dataloader()
    test_integration()
    print("\n" + "="*50)
    print("ALL TIER 1 OPTIMIZATION TESTS PASSED!")
    print("="*50)
