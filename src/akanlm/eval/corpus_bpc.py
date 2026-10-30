"""Tokenizer-independent conditional BPC on CPT document holdouts.

Google's token-span procedure (eval/denoising.py) masks tokens, so each
tokenizer receives a different task and native perplexity cannot rank AkanT5
against Google mT5. This protocol masks Unicode characters before tokenization
(eval/lm.py, char_span_v1), so every system reconstructs the same masked text
from the same held-out documents. Bits per masked character is then comparable
across tokenizers; per-token loss still is not.

It is conditional denoising cost, not full-text BPC, and it is not a translation
metric. FLORES+ and MAFAND remain downstream probes.
"""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from akanlm.eval.lm import LN2
from akanlm.eval.uncertainty import evaluate_repeated

PROTOCOL = "corpus_char_span_v1"


@dataclass(frozen=True)
class CharSpanConfig:
    # On the Twi held-out documents, 1,024 characters reach at most 677 mT5
    # encoder tokens, inside the 1,024-token limit, so no window is truncated. Longer windows remove the 128-character suffix artifact.
    max_chars: int = 1024
    noise_density: float = 0.15
    mean_span_chars: float = 12.0
    token_length_limit: int = 1024
    # Same cap as the token-span protocol's max_examples. A 10% Pristine split
    # holds ~134M characters; 10,000 windows score ~10M of them. Whole documents
    # are taken in manifest order, whose four Pristine styles alternate row by
    # row.
    max_windows: int = 10000
    # About 1.5M masked characters per capped split make mask-placement
    # variance small relative to document sampling, which the bootstrap covers.
    seeds: tuple[int, ...] = (20260910,)
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 13

    def __post_init__(self):
        object.__setattr__(self, "seeds", tuple(self.seeds))
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Character-span evaluation requires distinct mask seeds")
        if self.bootstrap_samples < 1 or self.max_windows < 1:
            raise ValueError("Character-span evaluation requires positive window and bootstrap limits")


DEFAULT_CHAR_SPAN_CONFIG = CharSpanConfig()


def implementation_sha256():
    from akanlm.training.checkpoints import file_sha256

    root = Path(__file__).parent
    return {name: file_sha256(root / name) for name in (
        "corpus_bpc.py", "lm.py", "uncertainty.py", "denoising.py")}


def select_documents(texts, config=DEFAULT_CHAR_SPAN_CONFIG):
    """Whole nonempty documents in manifest order, up to max_windows windows."""
    selected, windows = [], 0
    for text in texts:
        count = math.ceil(len(text) / config.max_chars)
        if not count:
            continue
        if windows + count > config.max_windows:
            break
        selected.append(text)
        windows += count
    if not selected:
        raise ValueError("No held-out document fits the character-span window cap")
    return selected


def score_corpora(model, tokenizer, corpora, name, *, config=DEFAULT_CHAR_SPAN_CONFIG, batch_size=16):
    """Score every declared source/mode on identical character tasks."""
    rows = []
    for (label, split), (texts, corpus) in corpora.items():
        documents = select_documents(texts, config)
        result = evaluate_repeated(
            model, tokenizer, documents, label, name, seeds=list(config.seeds),
            bootstrap_samples=config.bootstrap_samples, bootstrap_seed=config.bootstrap_seed,
            input_length=config.token_length_limit, max_chars=config.max_chars,
            noise_density=config.noise_density, mean_span_chars=config.mean_span_chars,
            batch_size=batch_size, provenance=corpus["source"]["provenance"])
        rows.append({"split": split, "dataset": label, "corpus": corpus,
                     "n_texts_available": len(texts), "n_texts_scored": len(documents),
                     "protocol": PROTOCOL, "config": asdict(config), "result": result.to_dict()})
    return rows


def compare(candidate, baseline, *, samples=2000, seed=13):
    """Paired document bootstrap of the BPC reduction, candidate against baseline.

    Both rows must carry the same masked text; only then is a per-character
    difference meaningful between tokenizers. The interval is conditional on
    the mask seeds and on one training run per system.
    """
    for key in ("protocol", "config", "split", "dataset", "corpus", "split_manifest_sha256"):
        if candidate[key] != baseline[key]:
            raise ValueError(f"Character BPC comparison requires identical {key}")
    a, b = candidate["result"], baseline["result"]
    for key in ("protocol", "task_sha256", "n_chars", "n_masked_chars"):
        if a[key] != b[key]:
            raise ValueError(f"Character BPC comparison requires identical {key}")
    docs_a = {d["document_index"]: d for d in a["document_scores"]}
    docs_b = {d["document_index"]: d for d in b["document_scores"]}
    if docs_a.keys() != docs_b.keys() or any(
            docs_a[i]["n_masked_chars"] != docs_b[i]["n_masked_chars"] for i in docs_a):
        raise ValueError("Character BPC comparison requires identical documents and masks")
    order = sorted(docs_a)
    nats_a = np.array([docs_a[i]["total_nats"] for i in order])
    nats_b = np.array([docs_b[i]["total_nats"] for i in order])
    chars = np.array([docs_a[i]["n_masked_chars"] for i in order], dtype=float)
    bpc_a = nats_a.sum() / (LN2 * chars.sum())
    bpc_b = nats_b.sum() / (LN2 * chars.sum())
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples)
    for i in range(samples):
        index = rng.integers(0, len(order), size=len(order))
        deltas[i] = (nats_b[index].sum() - nats_a[index].sum()) / (LN2 * chars[index].sum())
    lower, upper = np.quantile(deltas, [0.025, 0.975]).tolist()
    return {"candidate": candidate["model"], "baseline": baseline["model"],
            "split": candidate["split"], "dataset": candidate["dataset"],
            "candidate_bpc": bpc_a, "baseline_bpc": bpc_b,
            "bpc_reduction": bpc_b - bpc_a,
            "bpc_reduction_pct": 100 * (bpc_b - bpc_a) / bpc_b if bpc_b > 0 else math.nan,
            "bpc_reduction_ci95": {"lower": lower, "upper": upper, "samples": samples, "seed": seed,
                                   "method": "paired_document_bootstrap_percentile",
                                   "n_documents": len(order)}}
