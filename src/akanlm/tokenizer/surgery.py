"""Swap an mT5 checkpoint's vocabulary for the AkanT5 one.

Only four tensors change shape:

* ``shared.weight``
* ``encoder.embed_tokens.weight``
* ``decoder.embed_tokens.weight``
* ``lm_head.weight``

mT5 checkpoints ship ``tie_word_embeddings: false``, so ``lm_head`` is
genuinely separate and is transferred independently. The first three are the
*same* tensor object in a T5 model — ``shared`` — and must stay that way.

.. warning::

   ``MT5Config`` in transformers 5.16 **discards** the ``tie_word_embeddings:
   false`` recorded in ``google/mt5-{small,base,large}/config.json`` and
   reports ``True`` instead. The loaded weights are still untied — the two
   tensors have distinct storage — but ``resize_token_embeddings`` reads the
   config, concludes the model is tied, and collapses ``lm_head`` onto
   ``shared``. That silently halves the vocabulary parameters, discards the
   pretrained output head, and produces a model architecturally different from
   the one Google trained.

   This module therefore (a) determines tying from the *weights* rather than
   the config, via :func:`detect_tied`, (b) writes the corrected value back
   onto the config so a later ``from_pretrained`` does not re-tie on load, and
   (c) swaps the modules directly instead of calling
   ``resize_token_embeddings``.

Everything else loads unchanged. :func:`assert_architecture_preserved` pins the
config fields the later Graph-LM port depends on: the relative-position bucket
count and max distance define T5's positional mechanism, and Plenz & Frank's
graph-relative attention is a modification of exactly that. If surgery silently
changed them, the port would be building on different foundations than the
paper's.

Parameter arithmetic, from the verified configs:

======== ======= ========== ================ ==================
model    d_model original    vocab-bound      after 32,128 vocab
======== ======= ========== ================ ==================
small     512     300.2M      256.1M  (85%)    77.0M   (-74.4%)
base      768     582.4M      384.2M  (66%)   247.6M   (-57.5%)
large    1024    1229.6M      512.2M  (42%)   783.2M   (-36.3%)
======== ======= ========== ================ ==================
"""

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
    """Initialization path naming both the mT5 size and the vocabulary.

    ``google/mt5-base`` with ``runs/tokenizer/akant5_spm_48k_chars_v3`` gives
    ``runs/init/akant5-base-48k-chars-v3-sentinels-v1``. The vocabulary part is
    the tokenizer directory name without ``akant5_spm_``, so initializations
    from different tokenizers never share a path.

    Raises:
        ValueError: If *base_model* is not ``google/mt5-{small,base,large}``.
    """
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
    """Build an AkanT5 initialization checkpoint from an mT5 checkpoint.

    Args:
        base_model: Hub id or local path, e.g. ``google/mt5-base``.
        tokenizer_dir: Directory produced by :mod:`akanlm.tokenizer.train_spm`.
        output_dir: Where to save the resulting model and tokenizer.
        dtype: Storage dtype for the saved checkpoint.
    Returns:
        A :class:`SurgeryReport`, also written to ``surgery_report.json``.
    """
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
    """Load an mT5-family tokenizer without transformers' Mistral regex patch.

    transformers 5.16 treats any tokenizer whose ``config.json`` lacks
    ``transformers_version`` as a possible Mistral tokenizer when the Hub is
    offline, and warns that ``fix_mistral_regex=True`` is needed. The cached
    ``google/mt5-small`` and ``google/mt5-large`` configs lack that field.
    mT5 uses SentencePiece, not the regex, and ``fix_mistral_regex=True``
    would replace its Metaspace pre-tokenizer and change the pieces;
    ``False`` keeps the tokenizer unchanged and suppresses the warning.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(name_or_path), fix_mistral_regex=False)


def normalize_tying(model) -> bool:
    """Force ``config.tie_word_embeddings`` to match the actual weights.

    ``MT5Config`` re-asserts ``tie_word_embeddings=True`` **every time** a
    checkpoint is loaded, even one whose ``config.json`` says otherwise. Left
    alone the lie propagates: a checkpoint saved after training records
    ``True``, and the next loader may act on it and collapse the output head.

    Call this immediately after every ``from_pretrained`` of an mT5-family
    model, before training or saving.

    Returns:
        The true tying state, as read from the weights.
    """
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
    """Whether ``lm_head`` and the input embeddings share storage.

    Reads the tensors rather than ``config.tie_word_embeddings``, which is
    unreliable for mT5 under transformers 5.x. Distinct ``data_ptr`` values
    mean the checkpoint carries a genuinely separate output head.
    """
    head = model.get_output_embeddings()
    if head is None:
        return True
    return head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()


def assert_embeddings_shared(model) -> None:
    """Encoder and decoder must still read the same ``shared`` tensor.

    T5 ties encoder and decoder input embeddings by object identity, not by a
    config flag. A resize that broke that would train three separate matrices
    and silently triple the vocabulary parameter cost.

    Raises:
        AssertionError: If the three tensors are not the same object.
    """
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
    """Return a span-corruption loss and require it to beat a uniform prediction.

    A finite loss does not show that transfer worked: the first AkanT5
    surgery gave a finite 83.6 nats on a copy task and 27.7 nats on span
    corruption, far above ln(32128) = 10.4, because random fallback rows in
    ``lm_head`` took all probability mass. This check scores the objective CPT
    trains on, built by the training collator with a fixed seed.

    Raises:
        AssertionError: If the loss is not finite or not below
            ``ln(config.vocab_size)``.
    """
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
