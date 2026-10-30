"""Assemble CPT mixtures with source caps, deduplication, and English replay.

Budget replay by retained normalized training characters before tokenization."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from decimal import ROUND_CEILING, Decimal
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, Iterator, NamedTuple

import yaml

from akanlm.audit.sources import CorpusSource, Modality, load_sources
from akanlm.utils.logging import get_logger

logger = get_logger(__name__)

#: Prompt prefixes for the translation examples added in M3.
TRANSLATE_TO_EN = "translate Twi to English: "
TRANSLATE_TO_TWI = "translate English to Twi: "


class SourceCap(NamedTuple):
    """How much of a source the mixtures actually consume.

    ``None`` on an axis means uncapped on that axis. Both may be set, in which
    case whichever is reached first stops the read.
    """

    rows: int | None = None
    chars: int | None = None


@dataclass
class MixtureComponent:
    """One source's contribution to a mixture."""

    source: str
    repeat: float = 1.0
    """Oversampling factor. 4.0 means the source is seen four times per epoch."""

    limit: int | None = None
    """Row cap. A proxy for a volume budget — prefer limit_chars for that."""

    limit_chars: int | None = None
    """Character budget on the normalized target side.

    Prefer this field when the specification intends a text-volume budget.
    A row cap is only an approximate proxy for volume and can under- or
    overestimate total text due to row-length variance. In practice, this
    budget is applied after normalization and should be used to control
    replay or corpus consumption at the character level rather than by the
    number of rows.
    """

    streaming: bool = False
    as_pairs: bool = False
    """Consume a parallel source as translation examples, not monolingual text."""

    both_directions: bool = True
    """For pair sources, emit Twi->English and English->Twi."""


@dataclass
class ReplayConfig:
    """English share of retained normalized training text, before tokenization."""

    source: str
    training_char_fraction: float


@dataclass
class MixtureConfig:
    """A named continued-pretraining mixture."""

    name: str
    description: str = ""
    components: list[MixtureComponent] = field(default_factory=list)
    seed: int = 13
    blocked_on: str | None = None
    """Set when the mixture cannot yet be built, with the reason."""

    replay: ReplayConfig | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> "MixtureConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        components = [MixtureComponent(**c) for c in raw.pop("components", [])]
        if raw.get("replay") is not None:
            raw["replay"] = ReplayConfig(**raw["replay"])
        return cls(components=components, **raw)

    def source_names(self) -> list[str]:
        return [c.source for c in self.components]



def load_all(mixtures_dir: Path = Path("configs/mixtures")) -> list["MixtureConfig"]:
    """Every mixture that declares components, in name order."""
    configs = []
    for path in sorted(mixtures_dir.glob("*.yaml")):
        config = MixtureConfig.from_yaml(path)
        if config.components:
            configs.append(config)
    return configs


def derive_source_caps(configs: Iterable["MixtureConfig"]) -> dict[str, int | None]:
    """Merge row/character caps across all mixtures, including blocked ones.

    Any uncapped use leaves that axis uncapped; otherwise keep the largest cap.
    Omit sources unused by every mixture."""
    caps: dict[str, SourceCap] = {}
    for config in configs:
        for component in config.components:
            name = component.source
            cap = SourceCap(component.limit, component.limit_chars)
            caps[name] = cap if name not in caps else _widen(caps[name], cap)
    return caps


def _widen(a: "SourceCap", b: "SourceCap") -> "SourceCap":
    """Merge each cap axis by maximum; an uncapped input keeps that axis uncapped."""
    return SourceCap(
        None if a.rows is None or b.rows is None else max(a.rows, b.rows),
        None if a.chars is None or b.chars is None else max(a.chars, b.chars),
    )


class ExactDuplicates:
    """Keep the first (source, index) per NFC/casefolded/whitespace-collapsed content key.

    Count later duplicates, preserving deterministic source ownership."""

    def __init__(self) -> None:
        self._owner: dict[int, int] = {}
        self._sources: list[str] = []
        self.removed: dict[str, Counter] = defaultdict(Counter)

    def accept(self, source: str, index: int, text: str) -> bool:
        return self.claim(source, index, self.key(text))

    @staticmethod
    def key(text: str) -> int:
        """First 64 bits of the text's ``content_key``; pure, so workers can compute it."""
        from akanlm.data.cpt_splits import content_key

        return int(content_key(text)[:16], 16)

    def claim(self, source: str, index: int, key: int) -> bool:
        """Keep ``(source, index)`` unless an earlier record already owns *key*."""
        if source not in self._sources:
            self._sources.append(source)
        identity = self._sources.index(source) << 40 | index
        owner = self._owner.setdefault(key, identity)
        if owner == identity:
            return True
        self.removed[source][self._sources[owner >> 40]] += 1
        return False

    def report(self) -> dict:
        return {"rule": "first (source, index) per content_key in component order",
                "removed_by_source_and_owner": {s: dict(c) for s, c in self.removed.items()}}


def _pair_examples(record, both_directions: bool) -> Iterator[str]:
    """Render a translation pair as prefixed text for span corruption."""
    yield f"{TRANSLATE_TO_EN}{record.tgt_text}\n{record.src_text}"
    if both_directions:
        yield f"{TRANSLATE_TO_TWI}{record.src_text}\n{record.tgt_text}"


def _decide_text(shared, record, *, other_column: str | None) -> tuple:
    """Worker half of a monolingual record: filter outcome, duplicate key and documents.

    *other_column* names the other-language field of a parallel source, which
    the filter also checks. The caller counts the outcome and claims the key.
    """
    document_filter, transform = shared
    outcome = None
    if document_filter is not None:
        from akanlm.corpora.normalize import GENERAL_RULES, normalize_text

        # Exclude the same parallel record in its monolingual and bilingual
        # uses, even if only its English side hits.
        texts = [record.text]
        other = record.meta.get(other_column) if other_column else None
        if isinstance(other, str):
            texts.append(normalize_text(other, GENERAL_RULES))
        outcome = document_filter.check(*texts)
        if not outcome:
            return record.index, outcome, None, [], 0
    document = transform(record.text) if transform else record.text
    return record.index, outcome, ExactDuplicates.key(record.text), [document], len(record.text)


def _decide_pair(shared, record, *, both_directions: bool) -> tuple:
    """Worker half of a translation pair; see :func:`_decide_text`."""
    document_filter, transform = shared
    outcome = None
    if document_filter is not None:
        outcome = document_filter.check(record.tgt_text, record.src_text)
        if not outcome:
            return record.index, outcome, None, [], 0
    texts = list(_pair_examples(record, both_directions))
    documents = [transform(text) if transform else text for text in texts]
    return record.index, outcome, ExactDuplicates.key(record.tgt_text), documents, sum(map(len, texts))


def iter_documents(
    config: MixtureConfig,
    corpora_dir: Path = Path("configs/corpora"),
    sources: dict[str, CorpusSource] | None = None,
    document_filter=None,
    duplicates: ExactDuplicates | None = None,
    transform: Callable[[str], object] | None = None,
    workers: int = 1,
    exposure: dict | None = None,
) -> Iterator[tuple[object, str, str]]:
    """Yield ``(text, source, provenance)`` in component order with configured repeats.

    Filter every parallel side before deduplication. Update budgets/counters and
    first-seen ownership in the parent, preserving output across worker counts.
    Transforms replace text; worker inputs must be picklable.
    Replay requires TrainingFilter; exposure counts text before transforms and
    is written only after successful iteration. Reject blocked/nontraining sources."""
    if config.blocked_on:
        raise ValueError(f"mixture {config.name!r} is blocked: {config.blocked_on}")

    from akanlm.audit.sources import assert_trainable
    from akanlm.corpora.loaders import load_pairs, load_texts
    from akanlm.utils.parallel import OrderedPool

    sources = sources if sources is not None else load_sources(corpora_dir)

    missing = [c.source for c in config.components if c.source not in sources]
    if missing:
        raise ValueError(f"mixture {config.name!r} names unknown sources: {missing}")

    selected = [sources[c.source] for c in config.components]
    assert_trainable(selected)
    _validate_replay(config, sources)
    if config.replay is not None:
        from akanlm.data.cpt_splits import TrainingFilter

        if not isinstance(document_filter, TrainingFilter):
            raise ValueError("character-budget replay requires a CPT TrainingFilter")
    extracted_streams = [
        c.source for c in config.components if c.streaming and sources[c.source].hf_files is not None
    ]
    if extracted_streams:
        raise ValueError(
            "extracted sources cannot be streamed: " + ", ".join(sorted(set(extracted_streams)))
        )
    duplicates = duplicates if duplicates is not None else ExactDuplicates()
    characters, documents_seen = Counter(), Counter()
    replay_budget = None

    with OrderedPool((document_filter, transform), workers) as pool:
        for component in config.components:
            source = sources[component.source]
            provenance = source.provenance.value
            whole_repeats, fraction = divmod(component.repeat, 1.0)
            is_replay = config.replay is not None and source.name == config.replay.source
            if is_replay:
                if not sum(characters.values()):
                    raise ValueError("replay requires nonempty retained target-language training text")
                share = Decimal(str(config.replay.training_char_fraction))
                replay_budget = int((Decimal(sum(characters.values())) * share / (1 - share))
                                    .to_integral_value(rounding=ROUND_CEILING))

            for pass_index in range(int(whole_repeats) + (1 if fraction else 0)):
                # A fractional final pass takes a prefix of the source.
                limit = component.limit
                if fraction and pass_index == int(whole_repeats):
                    counted = _count(component, source)
                    limit = int(counted * fraction) if counted else limit

                budget = {"limit": limit, "streaming": component.streaming,
                          "max_chars": component.limit_chars, "pool": pool}
                if component.as_pairs:
                    if source.modality is not Modality.PARALLEL:
                        raise ValueError(
                            f"{source.name} is monolingual but the mixture asks for pairs"
                        )
                    records = load_pairs(source, **budget)
                    decide = partial(_decide_pair, both_directions=component.both_directions)
                else:
                    records = load_texts(source, **budget)
                    parallel = source.modality is Modality.PARALLEL
                    decide = partial(_decide_text, other_column=(source.source_column or "en") if parallel else None)

                for index, outcome, key, documents, n_chars in pool.map(decide, records):
                    if document_filter is not None and not document_filter.record(source.name, outcome):
                        continue
                    if not duplicates.claim(source.name, index, key):
                        continue
                    characters[source.name] += n_chars
                    documents_seen[source.name] += len(documents)
                    for document in documents:
                        yield (document, source.name, provenance)
                    if is_replay and characters[source.name] >= replay_budget:
                        break

            if is_replay and characters[source.name] < replay_budget:
                raise ValueError(
                    f"insufficient English replay: retained {characters[source.name]:,} characters, "
                    f"need {replay_budget:,}; expand the candidate extraction and prepare a fresh split"
                )

            logger.info("mixture %s: finished %s (repeat=%s)", config.name, source.name, component.repeat)

    if exposure is not None:
        exposure.clear()
        exposure.update({
            "protocol": "retained_training_characters_v1",
            "unit": "normalized Unicode code points, including whitespace, before tokenization",
            "characters_by_source": dict(characters),
            "documents_by_source": dict(documents_seen),
            "total_characters": sum(characters.values()),
        })
        if config.replay is not None:
            english = characters[config.replay.source]
            exposure.update({
                "replay": asdict(config.replay), "replay_budget_characters": replay_budget,
                "replay_overshoot_characters": english - replay_budget,
                "realized_replay_fraction": english / sum(characters.values()),
            })


def _count(component: MixtureComponent, source: CorpusSource) -> int | None:
    """Best available row count for resolving a fractional repeat."""
    return component.limit or source.expected_rows


def _validate_replay(config: MixtureConfig, sources: dict[str, CorpusSource]) -> None:
    """Keep the train-only budget unambiguous and the candidate read bounded."""
    replay = config.replay
    if replay is None:
        return
    if not 0 < replay.training_char_fraction < 1:
        raise ValueError("replay training_char_fraction must be strictly between zero and one")
    names = config.source_names()
    if (len(names) < 2 or names[-1] != replay.source or len(set(names)) != len(names)
            or any(c.repeat != 1 or c.as_pairs for c in config.components)):
        raise ValueError("replay needs unique monolingual components at repeat=1, with English last")
    component = config.components[-1]
    if component.limit_chars is None or component.limit_chars <= 0 or component.streaming:
        raise ValueError("replay needs a positive candidate character cap and an offline source")
    if sources[replay.source].language != "en" or any(sources[n].language == "en" for n in names[:-1]):
        raise ValueError("replay source must be English and follow only non-English components")


def validate(config: MixtureConfig, corpora_dir: Path = Path("configs/corpora")) -> dict:
    """Validate source existence, training roles, and pair modalities; return a run-card summary."""
    from akanlm.audit.sources import assert_trainable

    sources = load_sources(corpora_dir)
    missing = [c.source for c in config.components if c.source not in sources]
    if missing:
        raise ValueError(f"mixture {config.name!r} names unknown sources: {missing}")

    selected = [sources[c.source] for c in config.components]
    assert_trainable(selected)
    _validate_replay(config, sources)
    extracted_streams = [
        c.source for c in config.components if c.streaming and sources[c.source].hf_files is not None
    ]
    if extracted_streams:
        raise ValueError(
            "extracted sources cannot be streamed: " + ", ".join(sorted(set(extracted_streams)))
        )

    for component in config.components:
        source = sources[component.source]
        if component.as_pairs and source.modality is not Modality.PARALLEL:
            raise ValueError(f"{source.name} is monolingual but is configured as_pairs")

    provenances = {sources[c.source].provenance.value for c in config.components}
    noncommercial = sorted(s.name for s in selected if not s.commercial_use)

    return {
        "mixture": config.name,
        "components": [
            {
                "source": c.source,
                "repeat": c.repeat,
                "as_pairs": c.as_pairs,
                "provenance": sources[c.source].provenance.value,
                "licence": sources[c.source].licence,
            }
            for c in config.components
        ],
        "provenances": sorted(provenances),
        "noncommercial_sources": noncommercial,
        "blocked_on": config.blocked_on,
        "replay": asdict(config.replay) if config.replay is not None else None,
    }
