"""Train an mT5-layout Twi/English SentencePiece tokenizer.

Preserve special/byte pieces, descending sentinels, and 128-row model padding.
The generic fast wrapper restores native byte fallback and whitespace behavior."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

from akanlm.utils.logging import get_logger
from akanlm.utils.paths import ensure_dir

logger = get_logger(__name__)

NUM_SENTINELS = 100
PAD_MULTIPLE = 128

PAD_ID, EOS_ID, UNK_ID = 0, 1, 2
NUM_SPECIAL_TOKENS = 3
"""``<pad>``, ``</s>``, ``<unk>``. T5 has no BOS."""

BYTE_FALLBACK_PIECES = 256
"""``<0x00>`` … ``<0xFF>``, occupying real vocabulary slots."""


def pad_vocab_size(size: int, multiple: int = PAD_MULTIPLE) -> int:
    """Round *size* up to the next multiple of *multiple*."""
    return ((size + multiple - 1) // multiple) * multiple


def minimum_vocab_size(required_chars: int, byte_fallback: bool = True) -> int:
    """Count required characters, byte pieces, and special tokens before fitting."""
    return required_chars + (BYTE_FALLBACK_PIECES if byte_fallback else 0) + NUM_SPECIAL_TOKENS


def sentinel_token(i: int) -> str:
    return f"<extra_id_{i}>"


def sentinel_id(index: int, tokenizer_vocab_size: int) -> int:
    """Id of ``<extra_id_{index}>`` under T5's descending convention.

    ``<extra_id_0>`` is the *last* id in the tokenizer vocabulary.
    """
    if not 0 <= index < NUM_SENTINELS:
        raise ValueError(f"sentinel index out of range: {index}")
    return tokenizer_vocab_size - 1 - index


@dataclass
class TokenizerConfig:
    """Declarative SentencePiece training config."""

    name: str
    vocab_size: int
    """SentencePiece piece count, *excluding* the 100 sentinels."""

    language_config: str = "configs/languages/akan.json"
    model_type: str = "unigram"
    character_coverage: float = 1.0
    byte_fallback: bool = True

    #: Corpus source names, resolved against configs/corpora/.
    twi_sources: list[str] = field(default_factory=lambda: [
        "pristine_twi_english", "ghok_parallel_en_twi", "ghana_farmer_qa_twi",
        "true_asr_ipa_twi", "kumawood_twi", "twi_wikipedia", "ghana_speech_akuapem",
        "ghana_speech_asante",
    ])
    english_sources: list[str] = field(default_factory=lambda: ["english_replay_mc4"])
    english_ratio: float = 0.0567
    """Exact English share of normalized training characters, never lines."""
    total_characters: int = 200_000_000
    source_weights: dict[str, float] = field(default_factory=lambda: {
        "pristine_twi_english": 1_375_556_554,
        "ghok_parallel_en_twi": 1_973_383_000,
        "ghana_farmer_qa_twi": 34_366_000,
        "true_asr_ipa_twi": 23_422_000,
        "kumawood_twi": 9_072_000,
        "twi_wikipedia": 7_116_629,
        "ghana_speech_akuapem": 3_081_078,
        "ghana_speech_asante": 11_454_868,
        "english_replay_mc4": 1,
    })
    """Relative character weights WITHIN each language; frozen experiment recipe."""

    max_sentence_length: int = 8192
    split_digits: bool = True
    allow_whitespace_only_pieces: bool = False
    remove_extra_whitespaces: bool = False
    train_extremely_large_corpus: bool = False
    seed: int = 13
    num_threads: int = 16
    """SentencePiece worker count; its upstream default is 16."""
    require_split_manifest: bool = False
    """Require frozen train-only fitting; false preserves historical recipes."""

    @property
    def tokenizer_vocab_size(self) -> int:
        """Piece count plus sentinels — what ``len(tokenizer)`` will report."""
        return self.vocab_size + NUM_SENTINELS

    @property
    def model_vocab_size(self) -> int:
        """Padded size for the model config and embedding matrices."""
        return pad_vocab_size(self.tokenizer_vocab_size)

    @classmethod
    def from_yaml(cls, path: Path) -> "TokenizerConfig":
        return cls(**yaml.safe_load(path.read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        from dataclasses import asdict

        data = asdict(self)
        data["tokenizer_vocab_size"] = self.tokenizer_vocab_size
        data["model_vocab_size"] = self.model_vocab_size
        return data


def akan_required_chars(language_config: Path) -> list[str]:
    """Keep configured letters, digraph characters, and combining tone marks."""
    spec = json.loads(language_config.read_text(encoding="utf-8"))
    chars: list[str] = []
    seen: set[str] = set()

    def add(item: str) -> None:
        for ch in item:
            if ch and not ch.isspace() and ch not in seen:
                seen.add(ch)
                chars.append(ch)

    for letter in spec.get("alphabet", []):
        add(letter)
        add(letter.upper())
    for value in spec.get("all_special_characters", []):
        add(value)
    for value in spec.get("uppercase_forms", {}):
        add(value)
    for value in spec.get("loanword_letters", {}).get("letters", []):
        add(value)
        add(value.upper())
    return chars


def build_sentence_iterator(
    config: TokenizerConfig,
    corpora_dir: Path,
    split_manifest: Path | None = None,
) -> Iterable[str]:
    """Stream normalized text under exact language and source character quotas."""
    if split_manifest is not None:
        from akanlm.tokenizer.training_partitions import build_training_partition_iterator

        return build_training_partition_iterator(config, corpora_dir, split_manifest)
    if config.require_split_manifest:
        raise ValueError(f"{config.name} requires a frozen CPT split manifest for tokenizer fitting")
    from akanlm.audit.sources import assert_trainable, load_sources
    from akanlm.corpora.loaders import load_texts
    from akanlm.tokenizer.sampling import CharacterBudgetIterator, character_quotas, text_pieces

    budgets = character_quotas(config)
    sources = load_sources(corpora_dir)
    selected = [sources[n] for n in (*config.twi_sources, *config.english_sources)]
    assert_trainable(selected)

    def stream(name):
        # hf_files sources are a reproducible local extraction, never a Hub
        # stream. This keeps a future tokenizer rebuild using mC4 offline.
        for record in load_texts(
            sources[name], streaming=name in config.english_sources and sources[name].hf_files is None
        ):
            yield from text_pieces(record.text, config.max_sentence_length)

    languages = {name: "twi" for name in config.twi_sources}
    languages.update({name: "english" for name in config.english_sources})
    return CharacterBudgetIterator({name: stream(name) for name in budgets}, budgets,
                                   languages, config.seed, config.max_sentence_length)


def train(
    config: TokenizerConfig,
    output_dir: Path,
    sentence_iterator: Iterable[str] | None = None,
    corpora_dir: Path = Path("configs/corpora"),
    split_manifest: Path | None = None,
) -> Path:
    """Fit SentencePiece and save its HF tokenizer; return the output directory.

    A split manifest restricts fitting to protected training records."""
    import sentencepiece as spm
    from akanlm.tokenizer.sampling import CharacterBudgetIterator, character_quotas

    if sentence_iterator is not None and (split_manifest is not None or config.require_split_manifest):
        raise ValueError("A supplied iterator cannot bypass frozen training-partition protection")
    if any((output_dir / name).exists() for name in ("spm.model", "tokenizer.json", "akanlm_tokenizer.json")):
        raise FileExistsError(f"Tokenizer already exists at {output_dir}; choose a fresh output directory")
    ensure_dir(output_dir)
    prefix = output_dir / "spm"

    if sentence_iterator is None:
        sentence_iterator = build_sentence_iterator(config, corpora_dir, split_manifest=split_manifest)

    required = akan_required_chars(Path(config.language_config))
    floor = minimum_vocab_size(len(required), config.byte_fallback)
    if config.vocab_size < floor:
        raise ValueError(
            f"vocab_size={config.vocab_size} is below the floor of {floor} for this "
            f"configuration: {len(required)} required Akan characters + "
            f"{BYTE_FALLBACK_PIECES if config.byte_fallback else 0} byte-fallback pieces + "
            f"{NUM_SPECIAL_TOKENS} special tokens. Raise vocab_size, or drop byte_fallback "
            "and accept <unk> on unseen characters."
        )

    logger.info(
        "training %s: %d pieces, %d required chars, coverage=%.4f, byte_fallback=%s",
        config.name,
        config.vocab_size,
        len(required),
        config.character_coverage,
        config.byte_fallback,
    )

    spm.set_random_generator_seed(config.seed)
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(sentence_iterator),
        model_prefix=str(prefix),
        vocab_size=config.vocab_size,
        model_type=config.model_type,
        num_threads=config.num_threads,
        character_coverage=config.character_coverage,
        byte_fallback=config.byte_fallback,
        required_chars=required,
        # T5's special-token layout. bos_id=-1 because T5 has no BOS.
        pad_id=PAD_ID,
        eos_id=EOS_ID,
        unk_id=UNK_ID,
        bos_id=-1,
        pad_piece="<pad>",
        eos_piece="</s>",
        unk_piece="<unk>",
        # byte_fallback requires unk_surface to stay addressable
        unk_surface="<unk>",
        # Consume the entire budgeted stream: line subsampling breaks char ratios.
        input_sentence_size=0,
        shuffle_input_sentence=False,
        max_sentence_length=config.max_sentence_length,
        split_digits=config.split_digits,
        allow_whitespace_only_pieces=config.allow_whitespace_only_pieces,
        remove_extra_whitespaces=config.remove_extra_whitespaces,
        train_extremely_large_corpus=config.train_extremely_large_corpus,
        normalization_rule_name="identity",
        add_dummy_prefix=True,
    )

    sampling = {"protocol": "caller_supplied_iterator", "language_budget_verified": False}
    if isinstance(sentence_iterator, CharacterBudgetIterator):
        sampling = sentence_iterator.report()
        if (not sampling["completed"] or sampling["characters"] != config.total_characters
                or sampling["source_budgets"] != character_quotas(config)):
            raise RuntimeError("SentencePiece did not consume the complete character budget")
        sampling["language_budget_verified"] = True
        (output_dir / "sampling.json").write_text(
            json.dumps(sampling, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    spm_model = prefix.with_suffix(".model")
    append_sentinels(spm_model)
    tokenizer = build_hf_tokenizer(spm_model)
    assert_matches_sentencepiece(tokenizer, spm_model)
    tokenizer.save_pretrained(str(output_dir))

    observed = verify_layout(tokenizer)
    from akanlm.corpora.normalize import DEFAULT_RULES, GENERAL_RULES, normalization_spec

    (output_dir / "akanlm_tokenizer.json").write_text(
        json.dumps({"config": config.to_dict(), "observed": observed,
                    "sampling": sampling,
                    "expected_input_normalization": {
                        "twi": normalization_spec(DEFAULT_RULES),
                        "english": normalization_spec(GENERAL_RULES),
                    }}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "saved tokenizer to %s (len=%d, model vocab_size=%d)",
        output_dir,
        len(tokenizer),
        pad_vocab_size(len(tokenizer)),
    )
    return output_dir


def append_sentinels(spm_model: Path, count: int = NUM_SENTINELS) -> int:
    """Append descending ``<extra_id_N>`` pieces in place; return the new piece count.

    ``<extra_id_0>`` receives the highest ID."""
    from sentencepiece import sentencepiece_model_pb2 as sp_pb2

    proto = sp_pb2.ModelProto()
    proto.ParseFromString(spm_model.read_bytes())

    existing = {piece.piece for piece in proto.pieces}
    for index in range(count - 1, -1, -1):
        name = sentinel_token(index)
        if name in existing:
            raise ValueError(f"{spm_model} already contains {name}; refusing to double-append")
        piece = proto.pieces.add()
        piece.piece = name
        piece.score = 0.0
        piece.type = sp_pb2.ModelProto.SentencePiece.USER_DEFINED

    spm_model.write_bytes(proto.SerializeToString())
    logger.info("appended %d sentinels to %s (now %d pieces)", count, spm_model, len(proto.pieces))
    return len(proto.pieces)


def build_hf_tokenizer(spm_model: Path, model_max_length: int = 1024):
    """Wrap SentencePiece as a generic fast tokenizer preserving native behavior.

    Restore byte fallback, sentinel names, and ``normalizer_spec`` whitespace rules.
    Metaspace ``first`` avoids inserting word boundaries after sentinels."""
    from sentencepiece import sentencepiece_model_pb2 as sp_pb2
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast, T5Tokenizer
    from transformers.convert_slow_tokenizer import T5Converter

    slow = T5Tokenizer(
        vocab_file=str(spm_model),
        extra_ids=0,  # already in the SentencePiece model
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
        legacy=False,
    )
    spec = json.loads(T5Converter(slow).converted().to_str())

    # Restore byte fallback, which T5Converter drops.
    spec["model"]["byte_fallback"] = True
    decoder = spec.get("decoder")
    if decoder and decoder.get("type") == "Sequence":
        steps = list(decoder["decoders"])
    elif decoder:
        steps = [decoder]
    else:
        steps = []
    if not any(step.get("type") == "ByteFallback" for step in steps):
        steps.insert(0, {"type": "ByteFallback"})
    spec["decoder"] = {"type": "Sequence", "decoders": steps}

    # Restore SentencePiece's own whitespace handling.
    proto = sp_pb2.ModelProto()
    proto.ParseFromString(Path(spm_model).read_bytes())
    normalizer_spec = proto.normalizer_spec
    if normalizer_spec.name != "identity" or normalizer_spec.precompiled_charsmap:
        raise ValueError(f"{spm_model}: only identity SentencePiece normalization is supported")
    prepend_scheme = "first" if normalizer_spec.add_dummy_prefix else "never"
    metaspace = [spec["pre_tokenizer"], *steps]
    metaspace = [step for step in metaspace if step and step.get("type") == "Metaspace"]
    if len(metaspace) != 2:
        raise ValueError(f"expected Metaspace pre-tokenizer and decoder, got {spec['pre_tokenizer']}, {steps}")
    for step in metaspace:
        step["prepend_scheme"] = prepend_scheme
    if not normalizer_spec.remove_extra_whitespaces:
        spec["normalizer"] = None

    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_str(json.dumps(spec)),
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
        additional_special_tokens=[sentinel_token(i) for i in range(NUM_SENTINELS)],
        model_max_length=model_max_length,
    )


#: Check letters, tones, bytes, digits, and whitespace against SentencePiece.
#: Exclude leading whitespace, which corpus normalization strips.
PARITY_TEXTS = (
    "Anadwo biara ɔfrɛ no.",
    "Ɔbaa no rekɔ adwuma. Ɛyɛ me dɛ paa.",
    "tone: ɛ́ ɔ̀ á",
    "rare: ẍ 𝔮 ⚡ 中文",
    "Kwaku's e-mail — 2024, 3.5%",
    "two  spaces and   three",
    "trailing space ",
    "line one\nline two\n\nafter a blank line",
    "tab\tseparated",
    "no break",
)


def assert_matches_sentencepiece(tokenizer, spm_model: Path, texts: Iterable[str] = PARITY_TEXTS) -> int:
    """Require identical SentencePiece pieces on every input; return the input count."""
    import sentencepiece as spm

    processor = spm.SentencePieceProcessor(model_file=str(spm_model))
    count = 0
    for text in texts:
        expected = processor.encode(text, out_type=str)
        actual = tokenizer.tokenize(text)
        assert actual == expected, (
            f"HF tokenizer diverges from {spm_model} on {text!r}: {actual} != {expected}"
        )
        count += 1
    return count


def load(tokenizer_dir: Path):
    """Load a tokenizer previously saved by :func:`train`."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(tokenizer_dir))


def verify_layout(tokenizer) -> dict:
    """Validate special/byte/sentinel IDs for positional row transfer; return the layout."""
    size = len(tokenizer)
    observed = {
        "len": size,
        "pad_id": tokenizer.pad_token_id,
        "eos_id": tokenizer.eos_token_id,
        "unk_id": tokenizer.unk_token_id,
        "extra_id_0": tokenizer.convert_tokens_to_ids(sentinel_token(0)),
        "extra_id_99": tokenizer.convert_tokens_to_ids(sentinel_token(99)),
        "model_vocab_size": pad_vocab_size(size),
    }
    assert observed["pad_id"] == PAD_ID, f"pad must be {PAD_ID}, got {observed['pad_id']}"
    assert observed["eos_id"] == EOS_ID, f"eos must be {EOS_ID}, got {observed['eos_id']}"
    assert observed["unk_id"] == UNK_ID, f"unk must be {UNK_ID}, got {observed['unk_id']}"
    assert observed["extra_id_0"] == size - 1, (
        f"<extra_id_0> must be the last id ({size - 1}), got {observed['extra_id_0']}"
    )
    assert observed["extra_id_99"] == size - NUM_SENTINELS, (
        f"<extra_id_99> must be {size - NUM_SENTINELS}, got {observed['extra_id_99']}"
    )
    return observed
