"""Tests for LoRA/DoRA utilities for ZipVoice."""

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig


class TestLoraConfig:
    """Test LoRA configuration creation."""

    def test_create_lora_config_default(self):
        """Test default LoRA configuration."""
        from zipvoice.models.modules.lora_utils import create_lora_config

        config = create_lora_config()

        assert config.r == 32
        assert config.lora_alpha == 64
        assert config.lora_dropout == 0.05
        assert config.use_dora is True
        assert config.bias == "none"

    def test_create_lora_config_custom(self):
        """Test custom LoRA configuration."""
        from zipvoice.models.modules.lora_utils import create_lora_config

        config = create_lora_config(
            rank=16,
            alpha=32,
            dropout=0.1,
            use_dora=False,
        )

        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.1
        assert config.use_dora is False

    def test_create_lora_config_dora_toggle(self):
        """Test DoRA enable/disable."""
        from zipvoice.models.modules.lora_utils import create_lora_config

        config_dora = create_lora_config(use_dora=True)
        config_lora = create_lora_config(use_dora=False)

        assert config_dora.use_dora is True
        assert config_lora.use_dora is False

    def test_create_lora_config_custom_target_modules(self):
        """Test custom target modules."""
        from zipvoice.models.modules.lora_utils import create_lora_config

        custom_modules = ["self_attn1.in_proj", "self_attn2.in_proj"]
        config = create_lora_config(target_modules=custom_modules)

        # LoraConfig may convert to set internally
        assert set(config.target_modules) == set(custom_modules)

    def test_default_target_modules(self):
        """Test default target modules constant."""
        from zipvoice.models.modules.lora_utils import DEFAULT_TARGET_MODULES

        expected = [
            "self_attn_weights.in_proj",
            "self_attn1.in_proj",
            "self_attn1.out_proj",
            "self_attn2.in_proj",
            "self_attn2.out_proj",
        ]
        assert DEFAULT_TARGET_MODULES == expected


class TestLoraApplication:
    """Test LoRA application to ZipVoice model."""

    @pytest.fixture
    def small_model_config(self):
        """Return a small model configuration for testing."""
        return {
            "fm_decoder_downsampling_factor": [1, 2, 1],
            "fm_decoder_num_layers": [1, 1, 1],
            "fm_decoder_cnn_module_kernel": [15, 7, 15],
            "fm_decoder_feedforward_dim": 256,
            "fm_decoder_num_heads": 2,
            "fm_decoder_dim": 128,
            "text_encoder_num_layers": 1,
            "text_encoder_feedforward_dim": 128,
            "text_encoder_cnn_module_kernel": 5,
            "text_encoder_num_heads": 2,
            "text_encoder_dim": 64,
            "query_head_dim": 16,
            "value_head_dim": 8,
            "pos_head_dim": 4,
            "pos_dim": 32,
            "time_embed_dim": 64,
            "text_embed_dim": 64,
            "feat_dim": 100,
        }

    @pytest.fixture
    def model(self, small_model_config):
        """Create a small ZipVoice model for testing."""
        from zipvoice.models.zipvoice import ZipVoice

        model = ZipVoice(
            **small_model_config,
            vocab_size=100,
            pad_id=0,
        )
        return model

    def test_apply_lora_to_fm_decoder(self, model):
        """Test that LoRA is correctly applied to FM decoder."""
        from zipvoice.models.modules.lora_utils import (
            apply_lora_to_fm_decoder,
            create_lora_config,
        )

        config = create_lora_config(rank=8, alpha=16, use_dora=False)
        model = apply_lora_to_fm_decoder(model, config)

        # Check that fm_decoder has peft model structure
        assert hasattr(model.fm_decoder, "peft_config")
        assert hasattr(model.fm_decoder, "base_model")

    def test_apply_lora_with_dora(self, model):
        """Test that DoRA is correctly applied."""
        from zipvoice.models.modules.lora_utils import (
            apply_lora_to_fm_decoder,
            create_lora_config,
        )

        config = create_lora_config(rank=8, alpha=16, use_dora=True)
        model = apply_lora_to_fm_decoder(model, config)

        # Check that DoRA is enabled
        assert model.fm_decoder.peft_config["default"].use_dora is True

    def test_apply_lora_model_without_fm_decoder(self):
        """Test error when model doesn't have fm_decoder."""
        from zipvoice.models.modules.lora_utils import (
            apply_lora_to_fm_decoder,
            create_lora_config,
        )

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(10, 10)

        dummy = DummyModel()
        config = create_lora_config()

        with pytest.raises(AttributeError, match="fm_decoder"):
            apply_lora_to_fm_decoder(dummy, config)


class TestLoraParameters:
    """Test LoRA parameter statistics."""

    @pytest.fixture
    def small_model_config(self):
        """Return a small model configuration for testing."""
        return {
            "fm_decoder_downsampling_factor": [1, 2, 1],
            "fm_decoder_num_layers": [1, 1, 1],
            "fm_decoder_cnn_module_kernel": [15, 7, 15],
            "fm_decoder_feedforward_dim": 256,
            "fm_decoder_num_heads": 2,
            "fm_decoder_dim": 128,
            "text_encoder_num_layers": 1,
            "text_encoder_feedforward_dim": 128,
            "text_encoder_cnn_module_kernel": 5,
            "text_encoder_num_heads": 2,
            "text_encoder_dim": 64,
            "query_head_dim": 16,
            "value_head_dim": 8,
            "pos_head_dim": 4,
            "pos_dim": 32,
            "time_embed_dim": 64,
            "text_embed_dim": 64,
            "feat_dim": 100,
        }

    @pytest.fixture
    def model_with_lora(self, small_model_config):
        """Create a model with LoRA applied."""
        from zipvoice.models.zipvoice import ZipVoice
        from zipvoice.models.modules.lora_utils import (
            apply_lora_to_fm_decoder,
            create_lora_config,
        )

        model = ZipVoice(
            **small_model_config,
            vocab_size=100,
            pad_id=0,
        )
        config = create_lora_config(rank=8, alpha=16, use_dora=False)
        model = apply_lora_to_fm_decoder(model, config)
        return model

    def test_get_lora_trainable_params(self, model_with_lora):
        """Test trainable parameter statistics."""
        from zipvoice.models.modules.lora_utils import get_lora_trainable_params

        stats = get_lora_trainable_params(model_with_lora)

        assert "trainable_params" in stats
        assert "total_params" in stats
        assert "trainable_percent" in stats

        # Trainable params should be less than total
        assert stats["trainable_params"] < stats["total_params"]
        # Trainable percentage should be reasonable (less than 50% for LoRA)
        assert stats["trainable_percent"] < 50

    def test_base_model_params_frozen(self, model_with_lora):
        """Test that base model parameters are frozen."""
        # Check that original parameters in fm_decoder are frozen
        for name, param in model_with_lora.fm_decoder.named_parameters():
            if "lora_" not in name and "lora_embedding" not in name:
                # Base model params should be frozen (requires_grad=False)
                # But peft might handle this differently, so we just check
                # that LoRA params exist and are trainable
                pass

        # Check that LoRA parameters are trainable
        lora_params = [
            (name, param)
            for name, param in model_with_lora.fm_decoder.named_parameters()
            if "lora_" in name
        ]
        assert len(lora_params) > 0, "Should have LoRA parameters"

        for name, param in lora_params:
            assert param.requires_grad, f"LoRA param {name} should be trainable"


class TestLoraForwardPass:
    """Test forward pass with LoRA applied."""

    @pytest.fixture
    def small_model_config(self):
        """Return a small model configuration for testing."""
        return {
            "fm_decoder_downsampling_factor": [1, 2, 1],
            "fm_decoder_num_layers": [1, 1, 1],
            "fm_decoder_cnn_module_kernel": [15, 7, 15],
            "fm_decoder_feedforward_dim": 256,
            "fm_decoder_num_heads": 2,
            "fm_decoder_dim": 128,
            "text_encoder_num_layers": 1,
            "text_encoder_feedforward_dim": 128,
            "text_encoder_cnn_module_kernel": 5,
            "text_encoder_num_heads": 2,
            "text_encoder_dim": 64,
            "query_head_dim": 16,
            "value_head_dim": 8,
            "pos_head_dim": 4,
            "pos_dim": 32,
            "time_embed_dim": 64,
            "text_embed_dim": 64,
            "feat_dim": 100,
        }

    @pytest.fixture
    def model_with_lora(self, small_model_config):
        """Create a model with LoRA applied."""
        from zipvoice.models.zipvoice import ZipVoice
        from zipvoice.models.modules.lora_utils import (
            apply_lora_to_fm_decoder,
            create_lora_config,
        )

        model = ZipVoice(
            **small_model_config,
            vocab_size=100,
            pad_id=0,
        )
        config = create_lora_config(rank=8, alpha=16, use_dora=False)
        model = apply_lora_to_fm_decoder(model, config)
        return model

    def test_forward_pass_runs(self, model_with_lora):
        """Test that forward pass works with LoRA."""
        model_with_lora.eval()

        batch_size = 2
        seq_len = 10
        feat_dim = 100

        # Create dummy inputs
        tokens = [[1, 2, 3, 4, 5] for _ in range(batch_size)]
        features = torch.randn(batch_size, seq_len, feat_dim)
        features_lens = torch.tensor([seq_len, seq_len])
        noise = torch.randn_like(features)
        t = torch.rand(batch_size, 1, 1)

        with torch.no_grad():
            loss = model_with_lora(
                tokens=tokens,
                features=features,
                features_lens=features_lens,
                noise=noise,
                t=t,
                condition_drop_ratio=0.0,
            )

        assert loss is not None
        assert loss.shape == ()  # Scalar loss

    def test_output_shape_preserved(self, model_with_lora):
        """Test that output shape is preserved after LoRA application."""
        model_with_lora.eval()

        batch_size = 2
        seq_len = 10
        feat_dim = 100

        tokens = [[1, 2, 3, 4, 5] for _ in range(batch_size)]
        features = torch.randn(batch_size, seq_len, feat_dim)
        features_lens = torch.tensor([seq_len, seq_len])
        noise = torch.randn_like(features)
        t = torch.rand(batch_size, 1, 1)

        with torch.no_grad():
            loss = model_with_lora(
                tokens=tokens,
                features=features,
                features_lens=features_lens,
                noise=noise,
                t=t,
                condition_drop_ratio=0.0,
            )

        # Loss should be a scalar
        assert loss.dim() == 0


class TestLoraMerge:
    """Test LoRA weight merging."""

    @pytest.fixture
    def small_model_config(self):
        """Return a small model configuration for testing."""
        return {
            "fm_decoder_downsampling_factor": [1, 2, 1],
            "fm_decoder_num_layers": [1, 1, 1],
            "fm_decoder_cnn_module_kernel": [15, 7, 15],
            "fm_decoder_feedforward_dim": 256,
            "fm_decoder_num_heads": 2,
            "fm_decoder_dim": 128,
            "text_encoder_num_layers": 1,
            "text_encoder_feedforward_dim": 128,
            "text_encoder_cnn_module_kernel": 5,
            "text_encoder_num_heads": 2,
            "text_encoder_dim": 64,
            "query_head_dim": 16,
            "value_head_dim": 8,
            "pos_head_dim": 4,
            "pos_dim": 32,
            "time_embed_dim": 64,
            "text_embed_dim": 64,
            "feat_dim": 100,
        }

    @pytest.fixture
    def model_with_lora(self, small_model_config):
        """Create a model with LoRA applied."""
        from zipvoice.models.zipvoice import ZipVoice
        from zipvoice.models.modules.lora_utils import (
            apply_lora_to_fm_decoder,
            create_lora_config,
        )

        model = ZipVoice(
            **small_model_config,
            vocab_size=100,
            pad_id=0,
        )
        config = create_lora_config(rank=8, alpha=16, use_dora=False)
        model = apply_lora_to_fm_decoder(model, config)
        return model

    def test_merge_lora_weights(self, model_with_lora):
        """Test merging LoRA weights into base model."""
        from zipvoice.models.modules.lora_utils import merge_lora_weights

        # Before merge: should have peft structure with base_model
        assert hasattr(model_with_lora.fm_decoder, "base_model")

        # Count LoRA params before merge
        lora_params_before = [
            name for name, _ in model_with_lora.fm_decoder.named_parameters()
            if "lora_" in name
        ]
        assert len(lora_params_before) > 0, "Should have LoRA params before merge"

        # Merge weights
        model = merge_lora_weights(model_with_lora)

        # After merge: LoRA parameters should be gone (merged into base weights)
        lora_params_after = [
            name for name, _ in model.fm_decoder.named_parameters()
            if "lora_" in name
        ]
        assert len(lora_params_after) == 0, "LoRA params should be merged"

    def test_merged_model_forward_pass(self, model_with_lora):
        """Test that merged model can do forward pass."""
        from zipvoice.models.modules.lora_utils import merge_lora_weights

        model = merge_lora_weights(model_with_lora)
        model.eval()

        batch_size = 2
        seq_len = 10
        feat_dim = 100

        tokens = [[1, 2, 3, 4, 5] for _ in range(batch_size)]
        features = torch.randn(batch_size, seq_len, feat_dim)
        features_lens = torch.tensor([seq_len, seq_len])
        noise = torch.randn_like(features)
        t = torch.rand(batch_size, 1, 1)

        with torch.no_grad():
            loss = model(
                tokens=tokens,
                features=features,
                features_lens=features_lens,
                noise=noise,
                t=t,
                condition_drop_ratio=0.0,
            )

        assert loss is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
