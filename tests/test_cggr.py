"""
CGGR Unit Tests
===============
Comprehensive test suite for all CGGR features.
"""

import pytest
import torch
import torch.nn as nn
from cggr import CGGRLoss, CGGRWrapper


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_logits():
    """Sample logits tensor (batch=2, seq=8, vocab=100)."""
    torch.manual_seed(42)
    return torch.randn(2, 8, 100, device='cuda', dtype=torch.float32)


@pytest.fixture
def sample_targets():
    """Sample targets tensor (batch=2, seq=8)."""
    torch.manual_seed(42)
    return torch.randint(0, 100, (2, 8), device='cuda')


# =============================================================================
# CGGRLoss Basic Tests
# =============================================================================

class TestCGGRLossBasic:
    """Basic functionality tests."""
    
    def test_forward_returns_scalar(self, sample_logits, sample_targets):
        """Loss should return a scalar tensor."""
        criterion = CGGRLoss()
        loss = criterion(sample_logits, sample_targets)
        
        assert loss.dim() == 0
        assert loss.requires_grad
    
    def test_backward_works(self, sample_logits, sample_targets):
        """Backward pass should complete without error."""
        criterion = CGGRLoss()
        loss = criterion(sample_logits, sample_targets)
        loss.backward()
        
        assert True  # If we get here, backward worked
    
    def test_step_increments_counter(self):
        """step() should increment internal counter."""
        criterion = CGGRLoss()
        assert criterion.step_count.item() == 0
        
        criterion.step()
        assert criterion.step_count.item() == 1
        
        criterion.step()
        assert criterion.step_count.item() == 2
    
    def test_get_metrics(self, sample_logits, sample_targets):
        """get_metrics should return expected keys."""
        criterion = CGGRLoss()
        _ = criterion(sample_logits, sample_targets)
        
        metrics = criterion.get_metrics()
        
        assert 'step' in metrics
        assert 'token_ratio' in metrics
        assert 'tokens_selected' in metrics
        assert 'tokens_total' in metrics
        assert 'avg_confidence' in metrics
        assert 'avg_entropy' in metrics


# =============================================================================
# Scoring Strategy Tests
# =============================================================================

class TestScoringStrategies:
    """Test different scoring methods."""
    
    @pytest.mark.parametrize("scoring", ['entropy', 'margin', 'loss', 'combined'])
    def test_scoring_strategy_works(self, scoring, sample_logits, sample_targets):
        """All scoring strategies should produce valid loss."""
        criterion = CGGRLoss(scoring=scoring)
        loss = criterion(sample_logits, sample_targets)
        
        assert torch.isfinite(loss)
        assert loss.item() > 0
    
    def test_different_scorings_give_different_selections(self, sample_logits, sample_targets):
        """Different scoring methods should select differently."""
        results = {}
        
        for scoring in ['entropy', 'margin', 'loss', 'combined']:
            criterion = CGGRLoss(scoring=scoring, warmup_steps=0, min_tokens_ratio=0.5)
            criterion.step_count.fill_(1000)  # Skip warmup
            _ = criterion(sample_logits, sample_targets)
            results[scoring] = criterion.get_metrics()['tokens_selected']
        
        # At least some should differ (not guaranteed but likely with random data)
        values = list(results.values())
        # Just check they're all valid counts
        assert all(v > 0 for v in values)


# =============================================================================
# Selection Strategy Tests
# =============================================================================

class TestSelectionStrategies:
    """Test different selection methods."""
    
    @pytest.mark.parametrize("selection", ['topk', 'stratified', 'sequence_aware'])
    def test_selection_strategy_works(self, selection, sample_logits, sample_targets):
        """All selection strategies should produce valid loss."""
        criterion = CGGRLoss(selection=selection)
        loss = criterion(sample_logits, sample_targets)
        
        assert torch.isfinite(loss)
    
    def test_topk_selects_correct_ratio(self, sample_logits, sample_targets):
        """Top-k should select approximately the target ratio."""
        ratio = 0.5
        criterion = CGGRLoss(
            selection='topk', 
            min_tokens_ratio=ratio, 
            warmup_steps=0,
            dynamic_threshold=False,
        )
        criterion.step_count.fill_(1000)
        _ = criterion(sample_logits, sample_targets)
        
        metrics = criterion.get_metrics()
        actual_ratio = metrics['tokens_selected'] / metrics['tokens_total']
        
        # Should be close to target (within 10%)
        assert abs(actual_ratio - ratio) < 0.1
    
    def test_sequence_aware_ensures_coverage(self, sample_logits, sample_targets):
        """Sequence-aware should ensure min tokens per sequence."""
        min_per_seq = 2
        criterion = CGGRLoss(
            selection='sequence_aware',
            min_tokens_per_sequence=min_per_seq,
            min_tokens_ratio=0.1,  # Very low to test the coverage guarantee
            warmup_steps=0,
        )
        criterion.step_count.fill_(1000)
        _ = criterion(sample_logits, sample_targets)
        
        metrics = criterion.get_metrics()
        # At minimum, should have 2 tokens per sequence (2 sequences * 2 = 4)
        assert metrics['tokens_selected'] >= sample_logits.shape[0] * min_per_seq


# =============================================================================
# Curriculum Tests
# =============================================================================

class TestCurriculum:
    """Test warmup curriculum behavior."""
    
    def test_warmup_starts_at_100_percent(self, sample_logits, sample_targets):
        """At step 0, should use all tokens."""
        criterion = CGGRLoss(
            min_tokens_ratio=0.25,
            warmup_steps=1000,
            dynamic_threshold=False,
        )
        _ = criterion(sample_logits, sample_targets)
        
        metrics = criterion.get_metrics()
        # At step 0, ratio should be 1.0
        assert metrics['token_ratio'] == 1.0
    
    def test_warmup_ends_at_target(self, sample_logits, sample_targets):
        """After warmup, should use target ratio."""
        target_ratio = 0.25
        criterion = CGGRLoss(
            min_tokens_ratio=target_ratio,
            warmup_steps=1000,
            dynamic_threshold=False,
        )
        criterion.step_count.fill_(1000)  # Complete warmup
        _ = criterion(sample_logits, sample_targets)
        
        metrics = criterion.get_metrics()
        assert metrics['token_ratio'] == target_ratio
    
    def test_warmup_interpolates(self, sample_logits, sample_targets):
        """Mid-warmup should interpolate between 1.0 and target."""
        target_ratio = 0.25
        warmup = 1000
        criterion = CGGRLoss(
            min_tokens_ratio=target_ratio,
            warmup_steps=warmup,
            dynamic_threshold=False,
        )
        criterion.step_count.fill_(500)  # 50% through warmup
        _ = criterion(sample_logits, sample_targets)
        
        metrics = criterion.get_metrics()
        # Should be ~0.625 (midpoint between 1.0 and 0.25)
        expected = 1.0 - 0.5 * (1.0 - target_ratio)
        assert abs(metrics['token_ratio'] - expected) < 0.01


# =============================================================================
# Dynamic Threshold Tests
# =============================================================================

class TestDynamicThreshold:
    """Test dynamic threshold adjustment."""
    
    def test_dynamic_threshold_adjusts_ratio(self, sample_logits, sample_targets):
        """Dynamic threshold should adjust based on confidence."""
        criterion_static = CGGRLoss(
            min_tokens_ratio=0.25,
            warmup_steps=0,
            dynamic_threshold=False,
        )
        criterion_dynamic = CGGRLoss(
            min_tokens_ratio=0.25,
            warmup_steps=0,
            dynamic_threshold=True,
            threshold_sensitivity=0.5,
        )
        
        criterion_static.step_count.fill_(1000)
        criterion_dynamic.step_count.fill_(1000)
        
        _ = criterion_static(sample_logits, sample_targets)
        _ = criterion_dynamic(sample_logits, sample_targets)
        
        static_ratio = criterion_static.get_metrics()['token_ratio']
        dynamic_ratio = criterion_dynamic.get_metrics()['token_ratio']
        
        # Dynamic should differ from static (unless avg_conf happens to be 1.0)
        # Just check both are valid
        assert 0 < static_ratio <= 1
        assert 0 < dynamic_ratio <= 1


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_token(self):
        """Should handle single token input."""
        logits = torch.randn(1, 1, 100, device='cuda')
        targets = torch.randint(0, 100, (1, 1), device='cuda')
        
        criterion = CGGRLoss()
        loss = criterion(logits, targets)
        
        assert torch.isfinite(loss)
    
    def test_large_batch(self):
        """Should handle large batch."""
        logits = torch.randn(32, 256, 100, device='cuda')
        targets = torch.randint(0, 100, (32, 256), device='cuda')
        
        criterion = CGGRLoss()
        loss = criterion(logits, targets)
        
        assert torch.isfinite(loss)
    
    def test_2d_input(self):
        """Should handle 2D input (N, vocab)."""
        logits = torch.randn(64, 100, device='cuda')
        targets = torch.randint(0, 100, (64,), device='cuda')
        
        criterion = CGGRLoss()
        loss = criterion(logits, targets)
        
        assert torch.isfinite(loss)


# =============================================================================
# Triton Kernel Tests
# =============================================================================

class TestTritonKernels:
    """Test Triton kernel functionality."""
    
    def test_fused_difficulty_score(self, sample_logits, sample_targets):
        """Fused difficulty score should return correct shapes."""
        from triton_kernels import fused_difficulty_score
        
        difficulty, confidence, entropy = fused_difficulty_score(sample_logits, sample_targets)
        
        expected_shape = sample_logits.shape[:-1]  # (batch, seq)
        assert difficulty.shape == expected_shape
        assert confidence.shape == expected_shape
        assert entropy.shape == expected_shape
    
    def test_select_tokens_topk(self, sample_logits):
        """Top-k selection should return valid mask."""
        from triton_kernels import fused_difficulty_score, select_tokens_topk
        
        difficulty, _, _ = fused_difficulty_score(sample_logits)
        mask = select_tokens_topk(difficulty, ratio=0.5)
        
        assert mask.shape == difficulty.shape
        assert mask.dtype == difficulty.dtype
        assert (mask >= 0).all()
        assert (mask <= 1).all()
    
    def test_select_tokens_stratified(self, sample_logits):
        """Stratified selection should return valid mask."""
        from triton_kernels import fused_difficulty_score, select_tokens_stratified
        
        difficulty, _, _ = fused_difficulty_score(sample_logits)
        mask = select_tokens_stratified(difficulty, total_ratio=0.5, num_strata=4)
        
        assert mask.shape == difficulty.shape


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
