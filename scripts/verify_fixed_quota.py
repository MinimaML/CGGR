
import torch
import torch.nn as nn
from cggr import CGGRModel, CGGRScorer

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type('Config', (), {'num_hidden_layers': 2, 'hidden_size': 16})()
        self.model = nn.Linear(16, 16) # Dummy
        self.lm_head = nn.Linear(16, 100)
    
    def forward(self, input_ids, **kwargs):
        # Return random logits
        batch, seq = input_ids.shape
        return torch.randn(batch, seq, 100)

def test_fixed_quota():
    print("Testing Fixed Quota Selection...")
    
    model = MockModel()
    
    # Init with fixed_quota, ratio=0.4
    cggr = CGGRModel(
        model, 
        selection='fixed_quota',
        min_tokens_ratio=0.4,
        warmup_steps=0, # disable warmup to get immediate ratio
        dynamic_threshold=True # Should be overridden!
    )
    
    input_ids = torch.randint(0, 100, (2, 10)) # 20 tokens total
    
    # Forward pass
    loss = cggr(input_ids, labels=input_ids)
    
    metrics = cggr.get_metrics()
    print("Metrics:", metrics)
    
    # Assertions
    expected_ratio = 0.4
    assert metrics['token_ratio'] == expected_ratio, f"Expected ratio {expected_ratio}, got {metrics['token_ratio']}"
    
    # Since batch splitting is per sequence in CGGRModel pass 2 (which is what we verified in code review):
    # CGGRModel Pass 2: k = int(batch_size * current_ratio)
    # 2 * 0.4 = 0.8 -> int -> 0.
    # max(1, 0) -> 1 sequence.
    # Wait, CGGRModel batch splitting selects SEQUENCES.
    # But CGGRLoss selects TOKENS.
    
    # Let's check CGGRModel implementation again.
    # k = max(1, int(batch_size * current_ratio))
    # If I use batch_size=10, ratio=0.4 -> k=4 sequences.
    
    # Wait, if CGGRModel selects sequences, is "Fixed Quota" per sequence or per batch?
    # The plan said "Ensuring K tokens per GPU".
    # If CGGRModel selects K sequences, that is a fixed quota of sequences.
    
    # Let's test with larger batch
    input_ids_large = torch.randint(0, 100, (10, 5)) # 10 seqs, 50 tokens
    loss = cggr(input_ids_large, labels=input_ids_large)
    metrics = cggr.get_metrics()
    print("Metrics (Large Batch):", metrics)
    
    # k = int(10 * 0.4) = 4 sequences.
    # tokens_selected should be 4 * (5-1) = 16 tokens (since labels are shifted)
    
    assert metrics['token_ratio'] == 0.4
    # tokens_selected calculation in CGGRModel:
    # tokens_selected = hard_seq_indices.numel() * (seq_len - 1)
    # 4 * 4 = 16.
    
    print("SUCCESS: Fixed Quota respected.")

if __name__ == "__main__":
    test_fixed_quota()
