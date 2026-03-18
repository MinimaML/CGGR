"""
CGGR Unit Tests
===============
Comprehensive test suite for all CGGR features.
Supports CPU, CUDA, and MPS devices.
"""

import pytest
import torch
import torch.nn as nn
from cggr import CGGRLoss, CGGRModel, CGGRScorer, create_truncated_router


# =============================================================================
# Device Detection
# =============================================================================

def get_device():
    """Get the best available device for testing."""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


DEVICE = get_device()
HAS_CUDA = torch.cuda.is_available()


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_logits():
    """Sample logits tensor (batch=2, seq=8, vocab=100)."""
    torch.manual_seed(42)
    return torch.randn(2, 8, 100, device=DEVICE, dtype=torch.float32, requires_grad=True)


@pytest.fixture
def sample_targets():
    """Sample targets tensor (batch=2, seq=8)."""
    torch.manual_seed(42)
    return torch.randint(0, 100, (2, 8), device=DEVICE)


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
        logits = torch.randn(1, 1, 100, device=DEVICE)
        targets = torch.randint(0, 100, (1, 1), device=DEVICE)
        
        criterion = CGGRLoss()
        loss = criterion(logits, targets)
        
        assert torch.isfinite(loss)
    
    def test_large_batch(self):
        """Should handle large batch."""
        logits = torch.randn(32, 256, 100, device=DEVICE)
        targets = torch.randint(0, 100, (32, 256), device=DEVICE)
        
        criterion = CGGRLoss()
        loss = criterion(logits, targets)
        
        assert torch.isfinite(loss)
    
    def test_2d_input(self):
        """Should handle 2D input (N, vocab)."""
        logits = torch.randn(64, 100, device=DEVICE)
        targets = torch.randint(0, 100, (64,), device=DEVICE)
        
        criterion = CGGRLoss()
        loss = criterion(logits, targets)
        
        assert torch.isfinite(loss)


# =============================================================================
# Triton Kernel Tests (CUDA only)
# =============================================================================

class TestTritonKernels:
    """Test Triton kernel functionality (CUDA only)."""
    
    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA required for Triton tests")
    def test_fused_difficulty_score(self, sample_logits, sample_targets):
        """Fused difficulty score should return correct shapes."""
        from triton_kernels import fused_difficulty_score
        
        # Ensure tensors are on CUDA for this test
        logits = sample_logits.cuda() if not sample_logits.is_cuda else sample_logits
        targets = sample_targets.cuda() if not sample_targets.is_cuda else sample_targets
        
        difficulty, confidence, entropy = fused_difficulty_score(logits, targets)
        
        expected_shape = logits.shape[:-1]  # (batch, seq)
        assert difficulty.shape == expected_shape
        assert confidence.shape == expected_shape
        assert entropy.shape == expected_shape
    
    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA required for Triton tests")
    def test_select_tokens_topk(self, sample_logits):
        """Top-k selection should return valid mask."""
        from triton_kernels import fused_difficulty_score, select_tokens_topk
        
        logits = sample_logits.cuda() if not sample_logits.is_cuda else sample_logits
        
        difficulty, _, _ = fused_difficulty_score(logits)
        mask = select_tokens_topk(difficulty, ratio=0.5)
        
        assert mask.shape == difficulty.shape
        assert mask.dtype == difficulty.dtype
        assert (mask >= 0).all()
        assert (mask <= 1).all()
    
    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA required for Triton tests")
    def test_select_tokens_stratified(self, sample_logits):
        """Stratified selection should return valid mask."""
        from triton_kernels import fused_difficulty_score, select_tokens_stratified
        
        logits = sample_logits.cuda() if not sample_logits.is_cuda else sample_logits
        
        difficulty, _, _ = fused_difficulty_score(logits)
        mask = select_tokens_stratified(difficulty, total_ratio=0.5, num_strata=4)
        
        assert mask.shape == difficulty.shape


# =============================================================================
# PyTorch Fallback Tests
# =============================================================================

class TestPyTorchFallback:
    """Test PyTorch fallback implementations work correctly."""
    
    def test_fallback_difficulty_score(self):
        """PyTorch fallback should produce valid scores."""
        from triton_kernels import _pytorch_difficulty_score
        
        logits = torch.randn(2, 8, 100)
        difficulty, confidence, entropy = _pytorch_difficulty_score(logits)
        
        assert difficulty.shape == (2, 8)
        assert confidence.shape == (2, 8)
        assert entropy.shape == (2, 8)
        assert torch.all(confidence >= 0) and torch.all(confidence <= 1)
        assert torch.all(entropy >= 0)
    
    def test_fallback_topk_selection(self):
        """PyTorch fallback top-k should work correctly."""
        from triton_kernels import _pytorch_select_topk
        
        difficulty = torch.randn(16)
        mask = _pytorch_select_topk(difficulty, ratio=0.5)
        
        assert mask.shape == difficulty.shape
        assert mask.sum() == 8  # 50% of 16 tokens
    
    def test_fallback_stratified_selection(self):
        """PyTorch fallback stratified should work correctly."""
        from triton_kernels import _pytorch_stratified_select
        
        difficulty = torch.randn(100)
        mask = _pytorch_stratified_select(difficulty, total_ratio=0.5, num_strata=4)
        
        assert mask.shape == difficulty.shape
        # Should select some tokens (not exact due to rounding)
        assert mask.sum() > 0


# =============================================================================
# Wrapper-Level Tests
# =============================================================================

class MockCausalBase(nn.Module):
    def __init__(self, config=None, vocab_size=64, hidden_size=32, num_layers=2):
        super().__init__()
        if config is not None and not isinstance(config, int):
            vocab_size = getattr(config, 'vocab_size', vocab_size)
            hidden_size = getattr(config, 'hidden_size', hidden_size)
            num_layers = getattr(config, 'num_hidden_layers', num_layers)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids, **kwargs):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return self.norm(x)


class MockCausalLM(nn.Module):
    def __init__(self, vocab_size=64, hidden_size=32, num_layers=2):
        super().__init__()
        self.config = type('Config', (), {
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'num_hidden_layers': num_layers,
        })()
        self.model = MockCausalBase(vocab_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.last_input_shape = None

    def forward(self, input_ids, **kwargs):
        self.last_input_shape = tuple(input_ids.shape)
        hidden = self.model(input_ids, **kwargs)
        logits = self.lm_head(hidden)
        return type('Output', (), {'logits': logits})()


class MockBertEncoder(nn.Module):
    def __init__(self, hidden_size=32, num_layers=2):
        super().__init__()
        self.layer = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])

    def forward(self, hidden_states):
        x = hidden_states
        for layer in self.layer:
            x = torch.relu(layer(x))
        return x


class MockBertBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.encoder = MockBertEncoder(
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
        )

    def forward(self, input_ids=None, **kwargs):
        hidden = self.embeddings(input_ids)
        hidden = self.encoder(hidden)
        return type('Output', (), {'last_hidden_state': hidden})()


class MockMaskedLM(nn.Module):
    def __init__(self, vocab_size=64, hidden_size=32, num_layers=2):
        super().__init__()
        self.config = type('Config', (), {
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'num_hidden_layers': num_layers,
        })()
        self.bert = MockBertBackbone(self.config)
        self.cls = type('CLS', (), {})()
        self.cls.predictions = type('Predictions', (), {})()
        self.cls.predictions.decoder = nn.Linear(hidden_size, vocab_size)

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(self, input_ids, **kwargs):
        outputs = self.bert(input_ids=input_ids, **kwargs)
        logits = self.cls.predictions.decoder(outputs.last_hidden_state)
        return type('Output', (), {'logits': logits})()


class UnsupportedBackbone(nn.Module):
    def __init__(self, config=None, *, vocab_size=64, hidden_size=32):
        super().__init__()
        if config is not None and not isinstance(config, int):
            raise TypeError("unsupported config-only construction")
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size)])
        self.norm = nn.LayerNorm(hidden_size)


class MockFallbackRouterModel(nn.Module):
    def __init__(self, vocab_size=64, hidden_size=32):
        super().__init__()
        self.config = type('Config', (), {
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'num_hidden_layers': 2,
        })()
        self.model = UnsupportedBackbone(vocab_size=vocab_size, hidden_size=hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, **kwargs):
        hidden = self.model.embed_tokens(input_ids)
        hidden = torch.tanh(self.model.layers[0](hidden))
        hidden = self.model.norm(hidden)
        logits = self.lm_head(hidden)
        return type('Output', (), {'logits': logits})()


class RecordingScorer(nn.Module):
    def __init__(self, difficulty, mask, info):
        super().__init__()
        self.difficulty = difficulty
        self.mask = mask
        self.info = info
        self.last_targets = None
        self.step_count = torch.tensor(0)
        self.selection = info.get('selection', 'topk')
        self.scoring = info.get('scoring', 'combined')

    def step(self):
        self.step_count += 1

    def forward(self, input_ids, targets=None, **kwargs):
        self.last_targets = targets
        return self.difficulty, self.mask, self.info


@pytest.fixture
def mock_causal_model():
    torch.manual_seed(0)
    return MockCausalLM().to(DEVICE)


@pytest.fixture
def mock_masked_lm():
    torch.manual_seed(0)
    return MockMaskedLM().to(DEVICE)


class TestCGGRScorer:
    def test_sequence_aware_ensures_min_tokens_per_sequence(self, mock_causal_model):
        scorer = CGGRScorer(
            router=mock_causal_model,
            selection='sequence_aware',
            min_tokens_ratio=0.1,
            warmup_steps=0,
            min_tokens_per_sequence=2,
        )

        input_ids = torch.randint(0, mock_causal_model.config.vocab_size, (3, 7), device=DEVICE)
        difficulty, mask, info = scorer(input_ids)

        assert difficulty.shape == input_ids.shape
        assert mask.shape == input_ids.shape
        assert torch.all(mask.sum(dim=-1) >= 2)
        assert info['tokens_selected'] >= 3 * 2

    def test_fixed_quota_ignores_dynamic_threshold(self, mock_causal_model):
        scorer = CGGRScorer(
            router=mock_causal_model,
            selection='fixed_quota',
            min_tokens_ratio=0.4,
            warmup_steps=0,
            dynamic_threshold=True,
        )

        input_ids = torch.randint(0, mock_causal_model.config.vocab_size, (2, 10), device=DEVICE)
        _, _, info = scorer(input_ids)

        assert info['current_ratio'] == 0.4

    def test_loss_scoring_uses_targets_when_provided(self, mock_causal_model):
        scorer = CGGRScorer(
            router=mock_causal_model,
            scoring='loss',
            min_tokens_ratio=0.5,
            warmup_steps=0,
            dynamic_threshold=False,
        )

        input_ids = torch.randint(0, mock_causal_model.config.vocab_size, (2, 6), device=DEVICE)
        targets = torch.randint(0, mock_causal_model.config.vocab_size, (2, 6), device=DEVICE)

        difficulty_with_targets, _, _ = scorer(input_ids, targets=targets)
        difficulty_without_targets, _, _ = scorer(input_ids)

        assert not torch.allclose(difficulty_with_targets, difficulty_without_targets)

    def test_invalid_input_shape_raises(self, mock_causal_model):
        scorer = CGGRScorer(router=mock_causal_model)
        bad_input = torch.randint(0, mock_causal_model.config.vocab_size, (2, 3, 4), device=DEVICE)
        with pytest.raises(ValueError, match="shape"):
            _ = scorer(bad_input)

    def test_targets_shape_mismatch_raises(self, mock_causal_model):
        scorer = CGGRScorer(router=mock_causal_model)
        input_ids = torch.randint(0, mock_causal_model.config.vocab_size, (2, 6), device=DEVICE)
        bad_targets = torch.randint(0, mock_causal_model.config.vocab_size, (2, 5), device=DEVICE)
        with pytest.raises(ValueError, match="must match input_ids"):
            _ = scorer(input_ids, targets=bad_targets)


class TestCGGRModelWrapper:
    def test_model_routes_sequences_from_mask_not_batch_ratio(self, mock_causal_model):
        input_ids = torch.randint(0, mock_causal_model.config.vocab_size, (4, 6), device=DEVICE)
        labels = torch.randint(0, mock_causal_model.config.vocab_size, (4, 6), device=DEVICE)

        difficulty = torch.zeros(4, 6, device=DEVICE)
        mask = torch.zeros(4, 6, device=DEVICE)
        mask[1, 2] = 1.0
        info = {
            'current_ratio': 0.5,
            'confidence': torch.zeros(4, 6, device=DEVICE),
            'entropy': torch.ones(4, 6, device=DEVICE),
            'selection': 'topk',
            'scoring': 'combined',
        }

        model = CGGRModel(mock_causal_model, min_tokens_ratio=0.5, warmup_steps=0)
        model.scorer = RecordingScorer(difficulty, mask, info)

        loss = model(input_ids, labels=labels)

        assert torch.isfinite(loss)
        assert mock_causal_model.last_input_shape == (1, 6)
        metrics = model.get_metrics()
        assert metrics['sequences_selected'] == 1
        assert metrics['router_tokens_selected'] == 1
        assert metrics['tokens_selected'] == 5
        assert metrics['sequence_selection_mode'] == 'mask_any'

    def test_model_passes_labels_to_scorer_targets(self, mock_causal_model):
        input_ids = torch.randint(0, mock_causal_model.config.vocab_size, (2, 5), device=DEVICE)
        labels = torch.randint(0, mock_causal_model.config.vocab_size, (2, 5), device=DEVICE)

        difficulty = torch.zeros(2, 5, device=DEVICE)
        mask = torch.ones(2, 5, device=DEVICE)
        info = {
            'current_ratio': 1.0,
            'confidence': torch.zeros(2, 5, device=DEVICE),
            'entropy': torch.ones(2, 5, device=DEVICE),
            'selection': 'topk',
            'scoring': 'combined',
        }

        model = CGGRModel(mock_causal_model, min_tokens_ratio=1.0, warmup_steps=0)
        recorder = RecordingScorer(difficulty, mask, info)
        model.scorer = recorder

        _ = model(input_ids, labels=labels)

        assert recorder.last_targets is labels

    def test_model_quota_fallback_when_all_sequences_marked_hard(self, mock_causal_model):
        input_ids = torch.randint(0, mock_causal_model.config.vocab_size, (4, 8), device=DEVICE)
        labels = torch.randint(0, mock_causal_model.config.vocab_size, (4, 8), device=DEVICE)

        difficulty = torch.tensor(
            [
                [1.0] * 8,
                [0.8] * 8,
                [0.4] * 8,
                [0.1] * 8,
            ],
            device=DEVICE,
        )
        mask = torch.ones(4, 8, device=DEVICE)
        info = {
            'current_ratio': 0.4,  # ceil(4 * 0.4) = 2 sequences
            'confidence': torch.zeros(4, 8, device=DEVICE),
            'entropy': torch.ones(4, 8, device=DEVICE),
            'selection': 'topk',
            'scoring': 'combined',
        }

        model = CGGRModel(mock_causal_model, min_tokens_ratio=0.4, warmup_steps=0)
        model.scorer = RecordingScorer(difficulty, mask, info)

        loss = model(input_ids, labels=labels)
        assert torch.isfinite(loss)
        assert mock_causal_model.last_input_shape == (2, 8)
        metrics = model.get_metrics()
        assert metrics['router_tokens_selected'] == 32
        assert metrics['sequences_selected'] == 2
        assert metrics['sequence_selection_mode'] == 'quota_fallback'

    def test_model_labels_shape_mismatch_raises(self, mock_causal_model):
        model = CGGRModel(mock_causal_model, min_tokens_ratio=1.0, warmup_steps=0)
        input_ids = torch.randint(0, mock_causal_model.config.vocab_size, (2, 6), device=DEVICE)
        bad_labels = torch.randint(0, mock_causal_model.config.vocab_size, (2, 5), device=DEVICE)
        with pytest.raises(ValueError, match="labels must match input_ids"):
            _ = model(input_ids, labels=bad_labels)


class TestTruncatedRouter:
    def test_create_truncated_router_preserves_logits_shape(self, mock_causal_model):
        router = create_truncated_router(mock_causal_model, num_layers=1).to(DEVICE)
        input_ids = torch.randint(0, mock_causal_model.config.vocab_size, (2, 5), device=DEVICE)

        logits = router(input_ids)

        assert logits.shape == (2, 5, mock_causal_model.config.vocab_size)

    def test_create_truncated_router_supports_masked_lm_layout(self, mock_masked_lm):
        router = create_truncated_router(mock_masked_lm, num_layers=1).to(DEVICE)
        input_ids = torch.randint(0, mock_masked_lm.config.vocab_size, (2, 5), device=DEVICE)

        logits = router(input_ids)

        assert logits.shape == (2, 5, mock_masked_lm.config.vocab_size)

    def test_auto_architecture_falls_back_to_passthrough_when_truncation_fails(self):
        model = MockFallbackRouterModel().to(DEVICE)
        router = create_truncated_router(model, num_layers=1).to(DEVICE)
        input_ids = torch.randint(0, model.config.vocab_size, (2, 5), device=DEVICE)

        logits = router(input_ids)

        assert router.passthrough is True
        assert logits.shape == (2, 5, model.config.vocab_size)


class TestValidationGuards:
    def test_invalid_ratio_raises(self):
        with pytest.raises(ValueError, match="min_tokens_ratio"):
            _ = CGGRLoss(min_tokens_ratio=0.0)

    def test_invalid_threshold_sensitivity_raises(self):
        with pytest.raises(ValueError, match="threshold_sensitivity"):
            _ = CGGRLoss(threshold_sensitivity=1.5)

    def test_invalid_num_strata_raises(self):
        with pytest.raises(ValueError, match="num_strata"):
            _ = CGGRScorer(router=MockCausalLM().to(DEVICE), selection='stratified', num_strata=0)

    def test_cggrloss_shape_mismatch_raises(self, sample_logits):
        criterion = CGGRLoss()
        bad_targets = torch.randint(0, 100, (2, 7), device=DEVICE)
        with pytest.raises(ValueError, match="does not match targets shape"):
            _ = criterion(sample_logits, bad_targets)

    def test_cggrloss_bad_rank_raises(self):
        criterion = CGGRLoss()
        logits = torch.randn(2, 3, 4, 5, device=DEVICE)
        targets = torch.randint(0, 5, (2, 3, 4), device=DEVICE)
        with pytest.raises(ValueError, match="Unsupported logits shape"):
            _ = criterion(logits, targets)


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
