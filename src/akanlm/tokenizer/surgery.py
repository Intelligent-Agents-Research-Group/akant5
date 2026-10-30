"""Replace vocabulary-dependent mT5 rows while preserving the backbone.

Encoder/decoder inputs share ``shared.weight``; ``lm_head`` stays separate.
Transformers 5.16 misreports tying: inspect tensors and swap modules directly,
not via ``resize_token_embeddings``. Preserve positional config."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from akanlm.tokenizer.train_spm import pad_vocab_size, verify_layout
from akanlm.tokenizer.transfer import TransferReport, build_matrix
from akanlm.utils.logging import get_logger
from akanlm.utils.paths import ensure_dir

logger = get_logger(__name__)

#: Config fields whose values must survive surgery untouched. The relative
#: attention fields are load-bearing for the Graph-LM port.
PINNED_CONFIG_FIELDS = (
    "d_model",
    "d_ff",
    "d_kv",
    "num_layers",
    "num_decoder_layers",
    "num_heads",
    "relative_attention_num_buckets",
    "relative_attention_max_distance",
    "feed_forward_proj",
    "tie_word_embeddings",
    "layer_norm_epsilon",
)

#: The only tensors permitted to change shape.
VOCAB_TENSORS = (
    "shared.weight",
    "encoder.embed_tokens.weight",
    "decoder.embed_tokens.weight",
    "lm_head.weight",
)


@dataclass
class SurgeryReport:
    base_model: str
    tokenizer_dir: str
    old_vocab_size: int
    new_vocab_size: int
    tokenizer_len: int
    params_before: int
    params_after: int
    embedding_reports: list[TransferReport]

    @property
    def params_removed(self) -> int:
        return self.params_before - self.params_after

    def to_dict(self) -> dict:
        return {
            "base_model": self.base_model,
            "tokenizer_dir": self.tokenizer_dir,
            "old_vocab_size": self.old_vocab_size,
            "new_vocab_size": self.new_vocab_size,
            "tokenizer_len": self.tokenizer_len,
            "params_before": self.params_before,
            "params_after": self.params_after,
            "params_removed": self.params_removed,
            "params_removed_pct": round(100 * self.params_removed / self.params_before, 2),
            "embeddings": [r.to_dict() for r in self.embedding_reports],
        }


#: mT5 sizes surgery supports; the model ladder is small/base/large only.
MT5_SIZES = ("small", "base", "large")


def vocabulary_label(tokenizer_dir) -> str:
    """Path-safe vocabulary name: ``akant5_spm_48k_chars_v3`` -> ``48k-chars-v3``."""
    return Path(tokenizer_dir).name.removeprefix("akant5_spm_").replace("_", "-")


def default_output_dir(base_model: str, tokenizer_dir: Path, root: Path = Path("runs/init")) -> Path:
    """Name initializations by mT5 size and tokenizer; reject unsupported base IDs."""
    name = str(base_model).rstrip("/").split("/")[-1]
    size = name.removeprefix("mt5-")
    if not name.startswith("mt5-") or size not in MT5_SIZES:
        raise ValueError(f"Cannot name an initialization for {base_model}; pass --output explicitly")
    return Path(root) / f"akant5-{size}-{vocabulary_label(tokenizer_dir)}-sentinels-v1"


def assert_architecture_preserved(before: dict, after) -> None:
    """Raise if surgery changed anything but the vocabulary size.

    Raises:
        AssertionError: On any drift in :data:`PINNED_CONFIG_FIELDS`.
    """
    drift = {
        field: (before.get(field), getattr(after, field, None))
        for field in PINNED_CONFIG_FIELDS
        if before.get(field) != getattr(after, field, None)
    }
    assert not drift, f"surgery changed architecture fields: {drift}"


def apply(
    base_model: str,
    tokenizer_dir: Path,
    output_dir: Path,
    dtype: str = "float32",
) -> SurgeryReport:
    """Save a vocabulary-replaced model/tokenizer and return its ``SurgeryReport``."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Surgery requires an empty output directory; preserve existing checkpoint {output_dir}")
    ensure_dir(output_dir)

    new_tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    verify_layout(new_tokenizer)
    old_tokenizer = load_tokenizer(base_model)

    logger.info("loading %s", base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model, dtype=getattr(torch, dtype))

    # Establish tying from the weights, before the config gets a chance to lie.
    normalize_tying(model)

    config_before = model.config.to_dict()
    old_vocab_size = model.config.vocab_size
    tokenizer_len = len(new_tokenizer)
    new_vocab_size = pad_vocab_size(tokenizer_len)
    params_before = sum(p.numel() for p in model.parameters())

    logger.info(
        "vocabulary %d -> %d (tokenizer len %d, %d padding rows)",
        old_vocab_size,
        new_vocab_size,
        tokenizer_len,
        new_vocab_size - tokenizer_len,
    )

    embed_weight, embed_report = build_matrix(
        model.get_input_embeddings().weight,
        old_tokenizer,
        new_tokenizer,
        new_vocab_size,
        "input_embeddings",
        # Rescaling averaged input rows raised initial loss; see transfer.py.
        rescale_derived=False,
    )
    head_weight, head_report = build_matrix(
        model.lm_head.weight,
        old_tokenizer,
        new_tokenizer,
        new_vocab_size,
        "lm_head",
        rescale_derived=True,
    )

    # Swap the modules directly. resize_token_embeddings is avoided because it
    # consults config.tie_word_embeddings and would collapse lm_head onto the
    # embeddings — see the module docstring.
    new_embeddings = torch.nn.Embedding(new_vocab_size, model.config.d_model)
    new_embeddings.weight.data = embed_weight
    model.set_input_embeddings(new_embeddings)

    new_head = torch.nn.Linear(model.config.d_model, new_vocab_size, bias=False)
    new_head.weight.data = head_weight
    model.lm_head = new_head

    model.config.vocab_size = new_vocab_size
    model.vocab_size = new_vocab_size

    assert_architecture_preserved(config_before, model.config)
    assert_embeddings_shared(model)

    params_after = sum(p.numel() for p in model.parameters())
    model.save_pretrained(str(output_dir))
    new_tokenizer.save_pretrained(str(output_dir))

    report = SurgeryReport(
        base_model=base_model,
        tokenizer_dir=str(tokenizer_dir),
        old_vocab_size=old_vocab_size,
        new_vocab_size=new_vocab_size,
        tokenizer_len=tokenizer_len,
        params_before=params_before,
        params_after=params_after,
        embedding_reports=[embed_report, head_report],
    )
    (output_dir / "surgery_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "saved %s: %.1fM -> %.1fM params (-%.1f%%)",
        output_dir,
        params_before / 1e6,
        params_after / 1e6,
        100 * report.params_removed / params_before,
    )
    return report


def load_tokenizer(name_or_path):
    """Load mT5 SentencePiece without the incompatible Mistral regex patch."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(name_or_path), fix_mistral_regex=False)


def normalize_tying(model) -> bool:
    """After loading, set config tying from actual input/output weight sharing.

    Return whether weights are tied; call before training or saving."""
    tied = detect_tied(model)
    if model.config.tie_word_embeddings != tied:
        logger.warning(
            "config says tie_word_embeddings=%s but the weights are %s; "
            "correcting the config to %s",
            model.config.tie_word_embeddings,
            "tied" if tied else "untied",
            tied,
        )
        model.config.tie_word_embeddings = tied
    return tied


def detect_tied(model) -> bool:
    """Check actual input/output tensor storage; mT5 config flags can be wrong."""
    head = model.get_output_embeddings()
    if head is None:
        return True
    return head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()


def assert_embeddings_shared(model) -> None:
    """Require shared encoder/decoder input weights and a separate LM head."""
    shared = model.get_input_embeddings().weight
    assert model.encoder.embed_tokens.weight is shared, "encoder embeddings unshared"
    assert model.decoder.embed_tokens.weight is shared, "decoder embeddings unshared"
    assert model.lm_head.weight is not shared, (
        "lm_head must stay untied from the embeddings (mT5 sets tie_word_embeddings=false)"
    )


#: Fixed Twi text for :func:`check_forward`, repeated to fill the task.
CHECK_TEXT = (
    "Anadwo biara ɔfrɛ no. Ɔkɔɔ sukuu no mu. Me din de Kwaku. Ɛyɛ me dɛ paa. "
    "Yɛn nyinaa bɛkɔ fie. Ɔbaa no rekɔ adwuma. Wɔn mma no nyinaa wɔ sukuu. "
    "Nnipa no kɔɔ asɔre mu nnɛ. Ɔhene no tenaa ase wɔ ahenfie."
)


def check_forward(model_dir: Path, text: str = CHECK_TEXT, input_length: int = 128) -> float:
    """Return fixed-seed span-corruption loss; require finite loss below ``ln(vocab_size)``."""
    import math

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from akanlm.data.collator import DataCollatorForT5MLM, compute_input_and_target_lengths

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir))
    model.eval()

    raw_length, target_length = compute_input_and_target_lengths(input_length)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    ids = (ids * (raw_length // len(ids) + 1))[:raw_length]
    collator = DataCollatorForT5MLM(
        tokenizer,
        input_length=input_length,
        target_length=target_length,
        decoder_start_token_id=model.config.decoder_start_token_id,
        seed=13,
    )
    batch = collator([{"input_ids": ids}])
    with torch.no_grad():
        loss = model(**batch).loss

    value = float(loss)
    uniform = math.log(model.config.vocab_size)
    assert torch.isfinite(loss), f"non-finite loss after surgery: {value}"
    assert value < uniform, (
        f"span-corruption loss {value:.2f} is not below uniform ln({model.config.vocab_size}) = "
        f"{uniform:.2f}; embedding transfer is broken"
    )
    logger.info("forward check: span-corruption loss=%.4f (uniform %.4f)", value, uniform)
    return value
