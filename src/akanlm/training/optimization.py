"""Startup checks for the Mesh T5 optimization recipe, plus its schedule, z-loss and Adafactor."""

import math


def validate_optimization(config):
    """Reject out-of-range optimizer, loss, schedule and batch settings before loading weights."""
    if config.loss_normalization not in ("target_capacity", "valid_tokens"):
        raise ValueError("loss_normalization must be target_capacity or valid_tokens")
    if config.dropout_rate is not None and not 0 <= config.dropout_rate < 1:
        raise ValueError("dropout_rate must be in [0, 1), or null to inherit")
    if not math.isfinite(config.z_loss) or config.z_loss < 0:
        raise ValueError("z_loss must be finite and nonnegative")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be nonnegative")
    if config.adafactor_min_dim_size_to_factor < 0:
        raise ValueError("adafactor_min_dim_size_to_factor must be nonnegative")
    if not math.isfinite(config.adafactor_decay_rate) or config.adafactor_decay_rate >= 0:
        raise ValueError("adafactor_decay_rate must be a finite negative exponent")
    if config.adafactor_beta1 is not None and not 0 <= config.adafactor_beta1 < 1:
        raise ValueError("adafactor_beta1 must be in [0, 1), or null")
    if (not math.isfinite(config.adafactor_clip_threshold)
            or config.adafactor_clip_threshold < 1):
        raise ValueError("adafactor_clip_threshold must be finite and at least 1")
    if (not math.isfinite(config.adafactor_epsilon1) or config.adafactor_epsilon1 <= 0
            or not math.isfinite(config.adafactor_epsilon2)
            or config.adafactor_epsilon2 <= 0):
        raise ValueError("Adafactor epsilon values must be finite and positive")
    if config.per_device_train_batch_size < 1 or config.gradient_accumulation_steps < 1:
        raise ValueError("Training batch size and gradient accumulation must be positive")
    if config.lr_scheduler_type == "rsqrt_no_ramp_down" and config.warmup_steps < 1:
        raise ValueError("rsqrt_no_ramp_down requires warmup_steps >= 1 for its plateau")


def validate_model_setup(model, tokenizer, config):
    """Fail early if the PyTorch port cannot preserve Mesh T5 semantics."""
    import torch

    expected = {
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        # T5 has no BOS: shifting labels inserts <pad> at decoder position 0.
        "decoder_start_token_id": tokenizer.pad_token_id,
    }
    mismatches = {
        key: (getattr(model.config, key, None), value)
        for key, value in expected.items()
        if value is None or getattr(model.config, key, None) != value
    }
    if mismatches:
        raise ValueError(f"Model/tokenizer T5 special-token mismatch: {mismatches}")
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise ValueError("T5 requires distinct pad and EOS token IDs")

    embedding_rows = model.get_input_embeddings().weight.shape[0]
    head = model.get_output_embeddings()
    head_rows = head.weight.shape[0] if head is not None else None
    if not (len(tokenizer) <= embedding_rows == model.config.vocab_size == head_rows):
        raise ValueError(
            "Tokenizer/model vocabulary mismatch: "
            f"tokenizer={len(tokenizer)}, embeddings={embedding_rows}, "
            f"config={model.config.vocab_size}, lm_head={head_rows}"
        )
    if config.dropout_rate is not None and model.config.dropout_rate != config.dropout_rate:
        raise ValueError(
            f"Model dropout_rate={model.config.dropout_rate}, expected {config.dropout_rate}"
        )
    non_fp32 = sorted({str(parameter.dtype) for parameter in model.parameters()
                       if parameter.dtype != torch.float32})
    if non_fp32:
        raise ValueError(
            "CPT requires float32 master parameters with BF16 autocast; found "
            + ", ".join(non_fp32)
        )
    high_rank = [name for name, parameter in model.named_parameters() if parameter.ndim > 2]
    if high_rank:
        raise ValueError(
            "Mesh/HF Adafactor factoring parity is only established for this T5's "
            f"1D/2D parameters; found higher-rank tensors: {high_rank[:5]}"
        )


def rsqrt_multiplier(step, plateau_steps):
    """Plateau at the configured LR, then inverse-square-root decay.

    learning_rate=0.01 and plateau_steps=10000 reproduce Google's scalar
    1/sqrt(max(step, 10000)).
    """
    return math.sqrt(plateau_steps / max(step, plateau_steps))


def adafactor_class(min_dim_size_to_factor=128):
    """Honor Mesh's minimum factored dimension for HF's matrix parameters.

    HF otherwise factors even small relative-attention bias matrices. Mesh
    defaults to an unfactored accumulator unless both dimensions are >=128.
    A threshold of zero recovers HF's original behavior for controls.
    """
    from transformers import Adafactor

    class CPTAdafactor(Adafactor):
        @staticmethod
        def _get_options(param_group, param_shape):
            factored, first_moment = Adafactor._get_options(param_group, param_shape)
            factored = factored and min(param_shape[-2:]) >= min_dim_size_to_factor
            return factored, first_moment

    return CPTAdafactor


def training_z_loss(logits, labels, coefficient):
    """``coefficient`` × mean squared log-partition over targets other than ``-100``, in float32."""
    valid = labels.ne(-100)
    if not valid.any():
        raise ValueError("z-loss requires at least one nonpadding target")
    log_z = logits.float().logsumexp(dim=-1)
    return coefficient * log_z[valid].square().mean()
