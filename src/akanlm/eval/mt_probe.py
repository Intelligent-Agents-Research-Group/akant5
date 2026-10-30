"""MT probes with shared data, budgets, stopping rules, and decoding settings.

Select weights by development loss only; score stop-step or best-dev weights.
Report chrF++ (primary), BLEU, and FLORES+ varieties separately: Akuapem is
primary; Asante measures transfer. Learned metrics are not computed.
"""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from akanlm.utils.logging import get_logger
from akanlm.utils.paths import ensure_dir

logger = get_logger(__name__)


#: Task-prefix names; preregistered languages preserve the frozen procedure.
LANGUAGE_NAMES = {"en": "English", "twi": "Twi", "kpe": "Kpelle", "vai": "Vai"}


def direction_languages(direction: str) -> tuple[str, str]:
    """``"en2twi"`` -> ``("en", "twi")``. One side must be English."""
    source, separator, target = direction.partition("2")
    if (
        not separator or source not in LANGUAGE_NAMES or target not in LANGUAGE_NAMES
        or source == target or "en" not in (source, target)
    ):
        raise ValueError(f"Unknown translation direction: {direction}")
    return source, target


def directions_for(language: str) -> tuple[str, str]:
    """Both probe directions between English and *language*."""
    forward = f"en2{language}"
    direction_languages(forward)
    return forward, f"{language}2en"


def task_prefix(direction: str) -> str:
    source, target = direction_languages(direction)
    return f"translate {LANGUAGE_NAMES[source]} to {LANGUAGE_NAMES[target]}: "


def orient_evaluation_texts(
    english: Sequence[str], twi: Sequence[str], direction: str
) -> tuple[Sequence[str], Sequence[str]]:
    """Return held-out sources and references in the declared MT direction."""
    source_language, _ = direction_languages(direction)
    if len(english) != len(twi):
        raise ValueError("Held-out English and Twi text lists have different lengths")
    return (english, twi) if source_language == "en" else (twi, english)


PREFIX_EN_TO_TWI = task_prefix("en2twi")
PREFIX_TWI_TO_EN = task_prefix("twi2en")


@dataclass
class ProbeConfig:
    """Fine-tuning and decoding budget, held identical across systems."""

    direction: str = "en2twi"
    max_steps: int = 3000
    num_train_epochs: float | None = None
    eval_steps: int = 100
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001
    learning_rate: float = 3e-4
    optim: str = "adafactor"
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_source_length: int = 128
    max_target_length: int = 128
    num_beams: int = 4
    generation_max_length: int = 128
    generation_batch_size: int = 16
    seed: int = 13
    gradient_checkpointing: bool = False
    bf16: bool = True
    # Pad to a multiple to reduce shape overhead on B200; mask added padding.
    pad_to_multiple_of: int = 32
    lr_scheduler_type: str = "linear"
    # Restore minimum-dev-loss weights; False retains stop-step weights.
    restore_best_dev_loss: bool = False
    # None preserves checkpoint dropout; apply overrides before construction
    # without changing source artifacts.
    dropout_rate: float | None = None

    @classmethod
    def flairs_six_epoch_setup(cls, direction: str = "en2twi") -> "ProbeConfig":
        """Six-epoch FLAIRS MT settings applied to mT5 CPT checkpoints."""
        return cls(
            direction=direction,
            max_steps=-1,
            num_train_epochs=6.0,
            eval_steps=500,
            learning_rate=5e-5,
            optim="adamw_torch_fused",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            max_source_length=1024,
            max_target_length=128,
            num_beams=10,
            generation_max_length=128,
            generation_batch_size=4,
            seed=42,
            bf16=False,
        )

    def __post_init__(self) -> None:
        direction_languages(self.direction)
        if self.dropout_rate is not None and (
            not math.isfinite(self.dropout_rate) or not 0 <= self.dropout_rate < 1
        ):
            raise ValueError("dropout_rate must be finite and satisfy 0 <= rate < 1")
        if self.generation_max_length < 2:
            raise ValueError("generation_max_length must be at least 2")
        if self.generation_batch_size < 1 or self.num_beams < 1:
            raise ValueError("generation_batch_size and num_beams must be positive")
        if self.max_steps == -1:
            if self.num_train_epochs is None or not math.isfinite(self.num_train_epochs) or self.num_train_epochs <= 0:
                raise ValueError("epoch-based training requires positive num_train_epochs")
        elif self.max_steps < 1 or self.num_train_epochs is not None:
            raise ValueError("specify positive max_steps or num_train_epochs with max_steps=-1")
        if self.eval_steps < 1:
            raise ValueError("eval_steps must be positive")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be nonnegative (0 disables it)")
        if not 0 <= self.early_stopping_threshold < float("inf"):
            raise ValueError("early_stopping_threshold must be finite and nonnegative")
        if self.pad_to_multiple_of < 1:
            raise ValueError("pad_to_multiple_of must be positive")

    @property
    def prefix(self) -> str:
        return task_prefix(self.direction)


@dataclass
class MTResult:
    """Scores for one system on one evaluation set."""

    system: str
    dataset: str
    direction: str
    variety: str | None
    n_examples: int
    chrf2pp: float
    bleu: float
    predictions: list[str] = field(default_factory=list)
    raw_predictions: list[str] = field(default_factory=list)
    normalization: dict = field(default_factory=dict)
    reference_sha256: str = ""
    signatures: dict = field(default_factory=dict)

    def to_dict(self, keep_predictions: int = 20) -> dict:
        return {
            "system": self.system,
            "dataset": self.dataset,
            "direction": self.direction,
            "variety": self.variety,
            "n_examples": self.n_examples,
            "chrf2pp": round(self.chrf2pp, 4),
            "bleu": round(self.bleu, 4),
            "primary_metric": "chrf2pp",
            "signatures": self.signatures,
            "sample_predictions": self.predictions[:keep_predictions],
            "sample_raw_predictions": self.raw_predictions[:keep_predictions],
            "normalization": self.normalization,
            "reference_sha256": self.reference_sha256,
        }

    @property
    def slug(self) -> str:
        """Filesystem-safe identifier, used to name this result's artifacts."""
        parts = [self.system, self.dataset, self.direction, self.variety or "all"]
        return "_".join(
            "".join(c if c.isalnum() else "-" for c in part).strip("-") for part in parts
        )


def score(
    hypotheses: Sequence[str],
    references: Sequence[str],
    system: str,
    dataset: str,
    direction: str,
    variety: str | None = None,
) -> MTResult:
    """Compute chrF++ and sacreBLEU; reject unequal prediction/reference counts."""
    import sacrebleu

    if len(hypotheses) != len(references):
        raise ValueError(
            f"{len(hypotheses)} hypotheses vs {len(references)} references for {dataset}"
        )

    from akanlm.corpora.normalize import normalize_for_language, normalization_spec, rules_for_language

    _, language = direction_languages(direction)
    raw_predictions = list(hypotheses)
    hypotheses = [normalize_for_language(text, language) for text in hypotheses]
    references = [normalize_for_language(text, language) for text in references]

    # word_order=2 is what makes chrF into chrF++.
    chrf_metric = sacrebleu.CHRF(word_order=2)
    bleu_metric = sacrebleu.BLEU()
    chrf = chrf_metric.corpus_score(list(hypotheses), [list(references)])
    bleu = bleu_metric.corpus_score(list(hypotheses), [list(references)])

    result = MTResult(
        system=system,
        dataset=dataset,
        direction=direction,
        variety=variety,
        n_examples=len(hypotheses),
        chrf2pp=float(chrf.score),
        bleu=float(bleu.score),
        predictions=list(hypotheses),
        raw_predictions=raw_predictions,
        normalization=normalization_spec(rules_for_language(language)),
        reference_sha256=hashlib.sha256(
            json.dumps(references, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        signatures={"chrf2pp": str(chrf_metric.get_signature()),
                    "bleu": str(bleu_metric.get_signature())},
    )
    logger.info(
        "%s on %s (%s, %s): chrF++=%.2f BLEU=%.2f over %d examples",
        system, dataset, direction, variety or "-", result.chrf2pp, result.bleu, result.n_examples,
    )
    return result


def load_flores_pairs(
    sources: dict, variety: str, limit: int | None = None
) -> tuple[list[str], list[str]]:
    """``(english, twi)`` from :func:`load_flores_items`, without the ids."""
    english, twi, _ = load_flores_items(sources, variety, limit)
    return english, twi


def load_flores_items(
    sources: dict, variety: str, limit: int | None = None
) -> tuple[list[str], list[str], list[str]]:
    """Return aligned ``(english, twi, ids)`` for Akuapem or Asante FLORES+.
    Join monolingual configs by row; reject unequal lengths or sentence IDs.
    ``sources`` maps names to CorpusSource objects; ``limit`` caps smoke runs.
    """
    from akanlm.corpora.loaders import _open

    twi_source = sources[f"flores_plus_{variety}"]
    english_source = sources["flores_plus_english"]

    twi_rows = list(_open(twi_source))
    english_rows = list(_open(english_source))

    if len(twi_rows) != len(english_rows):
        raise ValueError(
            f"FLORES+ {variety} has {len(twi_rows)} rows, eng_Latn has "
            f"{len(english_rows)}; they are meant to be parallel"
        )
    mismatched = [
        index
        for index, (t, e) in enumerate(zip(twi_rows, english_rows))
        if t.get("id") != e.get("id")
    ]
    if mismatched:
        raise ValueError(
            f"FLORES+ {variety} and eng_Latn disagree on ids at "
            f"{len(mismatched)} positions (first: row {mismatched[0]})"
        )

    if limit is not None:
        twi_rows, english_rows = twi_rows[:limit], english_rows[:limit]

    logger.info("FLORES+ %s: %d aligned pairs", variety, len(twi_rows))
    return (
        [row["text"] for row in english_rows],
        [row["text"] for row in twi_rows],
        [str(row["id"]) for row in twi_rows],
    )


def build_pairs(records, config: ProbeConfig) -> list[dict]:
    """Turn ``PairRecord`` objects into prefixed source/target strings."""
    from akanlm.corpora.normalize import normalize_for_language

    source_language, target_language = direction_languages(config.direction)
    rows = []
    for record in records:
        if source_language == "en":
            source, target = record.src_text, record.tgt_text
        else:
            source, target = record.tgt_text, record.src_text
        rows.append({"source": config.prefix + normalize_for_language(source, source_language),
                     "target": normalize_for_language(target, target_language)})
    return rows


def load_local_jsonl_pairs(path: Path, config: ProbeConfig) -> list[dict]:
    """Load ``translation.<code>`` JSONL using MAFAND probe normalization.

    Accept CompExpress's ``sw`` column as an alias for Twi only.
    """
    from akanlm.corpora.normalize import normalize_for_language

    source_language, target_language = direction_languages(config.direction)
    other = target_language if source_language == "en" else source_language
    aliases = (other, "sw") if other == "twi" else (other,)
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                translation = json.loads(line)["translation"]
                english = translation["en"]
                twi = next(
                    (translation[key] for key in aliases if key in translation), None
                )
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}: invalid translation at line {line_number}") from exc
            if not isinstance(english, str) or not isinstance(twi, str):
                raise ValueError(f"{path}: missing translation text at line {line_number}")
            raw_source, raw_target = (
                (english, twi) if source_language == "en" else (twi, english)
            )
            normalized_source = normalize_for_language(raw_source, source_language)
            normalized_target = normalize_for_language(raw_target, target_language)
            if not normalized_source or not normalized_target:
                raise ValueError(f"{path}: empty translation text at line {line_number}")
            rows.append({
                "source": config.prefix + normalized_source,
                "target": normalized_target,
            })
    if not rows:
        raise ValueError(f"{path}: no translation pairs")
    return rows


def _probe_manifest(
    checkpoint: str, train_rows: list[dict], eval_rows: list[dict], config: ProbeConfig
) -> dict:
    """Inputs that must match before an interrupted fine-tune resumes."""
    def rows_hash(rows):
        return hashlib.sha256(
            json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    settings = asdict(config)
    if config.dropout_rate is None:
        settings.pop("dropout_rate")  # Preserve legacy cache/resume identities.
    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "config": settings,
        "train_rows_sha256": rows_hash(train_rows),
        "dev_rows_sha256": rows_hash(eval_rows),
    }


def finetune(
    checkpoint: str,
    train_rows: list[dict],
    eval_rows: list[dict],
    config: ProbeConfig,
    output_dir: Path,
    resume: bool = False,
):
    """Train and reload stop-step or first minimum-dev-loss weights for scoring.
    Best-only saves supply the restore/resume checkpoint. Thresholded patience
    may reference a different step; both steps are recorded.
    """
    import torch
    from datasets import Dataset
    from transformers import (
        AutoConfig,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
        set_seed,
    )
    from transformers.trainer_callback import ExportableState
    from transformers.trainer_utils import get_last_checkpoint
    from akanlm.tokenizer.surgery import normalize_tying

    class StopOnDevLoss(TrainerCallback, ExportableState):
        """Count dev checks since the best loss without restoring best weights."""

        def __init__(self, patience: int, threshold: float):
            self.patience = patience
            self.threshold = threshold
            self.best_loss = None
            self.best_step = None
            self.bad_checks = 0
            self.stopped_early = False

        def on_evaluate(self, args, state, control, metrics, **kwargs):
            loss = metrics.get("eval_loss")
            if loss is None or not math.isfinite(loss):
                raise ValueError(
                    f"Missing or nonfinite dev loss at step {state.global_step}: {loss}"
                )
            if self.best_loss is None or self.best_loss - loss > self.threshold:
                self.best_loss = loss
                self.best_step = state.global_step
                self.bad_checks = 0
            else:
                self.bad_checks += 1
                if self.bad_checks >= self.patience:
                    self.stopped_early = True
                    control.should_training_stop = True
            return control

        def state(self):
            return {
                "args": {"patience": self.patience, "threshold": self.threshold},
                "attributes": {
                    "best_loss": self.best_loss,
                    "best_step": self.best_step,
                    "bad_checks": self.bad_checks,
                    "stopped_early": self.stopped_early,
                },
            }

    early_stopping = config.early_stopping_patience > 0
    restore = config.restore_best_dev_loss
    evaluate = early_stopping or restore
    if evaluate and not eval_rows:
        raise ValueError(
            "MT probe early stopping and best-weight restoration require "
            "nonempty validation rows"
        )
    eval_steps = (
        min(config.eval_steps, config.max_steps)
        if config.max_steps > 0 else config.eval_steps
    )
    manifest = _probe_manifest(checkpoint, train_rows, eval_rows, config)
    manifest_path = output_dir / "input_manifest.json"
    final_dir = output_dir / "final"
    summary_path = output_dir / "training_summary.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not resume:
            raise ValueError(
                f"{output_dir} is nonempty; use a fresh output directory or --resume"
            )
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(f"{output_dir}: input manifest does not match; cannot resume")
        if final_dir.exists() and summary_path.exists():
            logger.info("%s: loading completed fine-tune from %s", checkpoint, final_dir)
            restored = AutoModelForSeq2SeqLM.from_pretrained(final_dir)
            normalize_tying(restored)
            return (
                restored,
                AutoTokenizer.from_pretrained(final_dir),
            )
        resume_checkpoint = get_last_checkpoint(str(output_dir))
        if resume_checkpoint is None:
            # Matching manifest permits restart before the first save.
            logger.warning("%s: no Trainer checkpoint; restarting the fine-tune", output_dir)
    else:
        ensure_dir(output_dir)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        resume_checkpoint = None

    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # Keep float32 training weights; bf16 storage/loading changes training.
    model_options = {"dtype": torch.float32}
    if config.dropout_rate is not None:
        model_config = AutoConfig.from_pretrained(checkpoint)
        model_config.dropout_rate = config.dropout_rate
        model_options["config"] = model_config
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, **model_options)
    normalize_tying(model)
    # mT5 returns a microbatch mean despite accepting **kwargs.
    from akanlm.training.cpt import configure_loss_normalization
    configure_loss_normalization(model)

    def encode(batch):
        encoded = tokenizer(
            batch["source"], max_length=config.max_source_length,
            truncation=True, padding=False,
        )
        labels = tokenizer(
            text_target=batch["target"], max_length=config.max_target_length,
            truncation=True, padding=False,
        )
        encoded["labels"] = labels["input_ids"]
        return encoded

    train_ds = Dataset.from_list(train_rows).map(
        encode, batched=True, remove_columns=["source", "target"]
    )
    eval_ds = (
        Dataset.from_list(eval_rows).map(
            encode, batched=True, remove_columns=["source", "target"]
        )
        if eval_rows else None
    )
    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        max_steps=config.max_steps,
        num_train_epochs=config.num_train_epochs or 3.0,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=config.bf16 and torch.cuda.is_available(),
        optim=config.optim,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=100,
        eval_strategy="steps" if evaluate else "no",
        eval_steps=eval_steps,
        save_strategy="best" if restore else "steps",
        save_steps=eval_steps,
        save_total_limit=1 if restore else 2,
        save_only_model=False,
        load_best_model_at_end=restore,
        metric_for_best_model="eval_loss" if restore else None,
        greater_is_better=False if restore else None,
        restore_callback_states_from_checkpoint=True,
        seed=config.seed,
        report_to=[],
        predict_with_generate=False,
        prediction_loss_only=True,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, model=model, pad_to_multiple_of=config.pad_to_multiple_of
        ),
        callbacks=[
            StopOnDevLoss(
                config.early_stopping_patience, config.early_stopping_threshold
            )
        ] if early_stopping else [],
    )
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_state()
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    # Score a disk round-trip, not Trainer's in-memory model.
    scored_model = AutoModelForSeq2SeqLM.from_pretrained(final_dir, dtype=torch.float32)
    normalize_tying(scored_model)
    scored_tokenizer = AutoTokenizer.from_pretrained(final_dir)
    stop = next(
        (cb for cb in trainer.callback_handler.callbacks
         if isinstance(cb, StopOnDevLoss)),
        None,
    )
    reason = (
        "dev_loss_patience" if stop is not None and stop.stopped_early
        else "epoch_cap" if config.max_steps == -1 else "step_cap"
    )
    summary = {
        "checkpoint": manifest["checkpoint"],
        "completed_steps": trainer.state.global_step,
        "completed_epochs": trainer.state.epoch,
        "stopping_reason": reason,
        "scored_weights": "best_dev_loss" if restore else "stop_step",
        "scored_artifact": str(final_dir.resolve()),
        "dropout_rate": scored_model.config.dropout_rate,
    }
    if restore:
        summary.update(_verify_restored_weights(trainer, scored_model, stop))
    else:
        summary["best_dev_loss"] = None if stop is None else stop.best_loss
        summary["best_dev_step"] = None if stop is None else stop.best_step
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "%s: MT fine-tune stopped at step %d (%s); scoring %s weights from %s",
        checkpoint, trainer.state.global_step, reason, summary["scored_weights"], final_dir,
    )
    return scored_model, scored_tokenizer


#: Restored-vs-logged dev-loss tolerance for non-bit-exact bf16 evaluation.
RESTORED_LOSS_TOLERANCE = 1e-3


def _verify_restored_weights(trainer, scored_model, stop) -> dict:
    """Verify the first minimum-dev-loss checkpoint, saved tensors, and loss.

    Raise RuntimeError if checkpoint selection, round-trip weights, or loss differ.
    """
    import torch

    state = trainer.state
    history = [
        (entry["step"], float(entry["eval_loss"]))
        for entry in state.log_history if "eval_loss" in entry
    ]
    if not history or state.best_global_step is None or state.best_model_checkpoint is None:
        raise RuntimeError("No best development checkpoint was recorded")
    best_loss = min(loss for _, loss in history)
    first_best_step = next(step for step, loss in history if loss == best_loss)
    if (
        first_best_step != state.best_global_step
        or not math.isclose(state.best_metric, best_loss, rel_tol=0, abs_tol=1e-12)
        or Path(state.best_model_checkpoint).name != f"checkpoint-{first_best_step}"
    ):
        raise RuntimeError(
            f"Restored checkpoint {state.best_model_checkpoint} (step "
            f"{state.best_global_step}, loss {state.best_metric}) is not the first "
            f"development-loss minimum (step {first_best_step}, loss {best_loss})"
        )
    reloaded = scored_model.state_dict()
    for name, value in trainer.model.state_dict().items():
        if not torch.equal(value.detach().cpu(), reloaded[name].detach().cpu()):
            raise RuntimeError(f"Saved weights differ from the restored model at {name}")
    if stop is not None:
        trainer.remove_callback(stop)
    reevaluated = float(trainer.evaluate(metric_key_prefix="restored")["restored_loss"])
    if abs(reevaluated - best_loss) > RESTORED_LOSS_TOLERANCE:
        raise RuntimeError(
            f"Restored weights re-evaluate to {reevaluated:.6f}, not the best "
            f"development loss {best_loss:.6f}"
        )
    return {
        "restored_step": first_best_step,
        "restored_dev_loss": best_loss,
        "reevaluated_dev_loss": reevaluated,
        "restored_checkpoint": Path(state.best_model_checkpoint).name,
        "patience_reference_step": None if stop is None else stop.best_step,
        "patience_reference_loss": None if stop is None else stop.best_loss,
        "dev_loss_history": history,
    }


def generate(
    model,
    tokenizer,
    sources: Sequence[str],
    config: ProbeConfig,
    batch_size: int | None = None,
    device: str | None = None,
    normalize_outputs: bool = True,
) -> list[str]:
    """Decode with a hard length cap including the start token.

    The cap is independent of training truncation; raw output may be retained.
    """
    import torch
    from akanlm.corpora.normalize import normalize_for_language

    source_language, target_language = direction_languages(config.direction)
    sources = [normalize_for_language(text, source_language) for text in sources]
    batch_size = config.generation_batch_size if batch_size is None else batch_size
    if batch_size < 1:
        raise ValueError("generation batch_size must be positive")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    outputs: list[str] = []
    with torch.no_grad():
        for start in range(0, len(sources), batch_size):
            window = [f"{config.prefix}{s}" for s in sources[start : start + batch_size]]
            batch = tokenizer(
                window, return_tensors="pt", padding=True, truncation=True,
                max_length=config.max_source_length,
                pad_to_multiple_of=config.pad_to_multiple_of,
            ).to(device)
            generated = model.generate(
                **batch,
                max_length=config.generation_max_length,
    # max_new_tokens would override the requested max_length cap.
                max_new_tokens=None,
                num_beams=config.num_beams,
                early_stopping=config.num_beams > 1,
            )
            outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return ([normalize_for_language(text, target_language) for text in outputs]
            if normalize_outputs else outputs)


def report_path(output_dir: Path, direction: str) -> Path:
    """Direction-specific report path, preventing cross-direction overwrites."""
    return output_dir / f"mt_probe_report_{direction}.json"


def system_dir(output_dir: Path, system: str, direction: str) -> Path:
    """Scratch directory per system/direction, including saved weights and state.

    ``trainer_state.json`` records steps and validation history.
    """
    return output_dir / f"{system}_{direction}"


def write_report(
    results: list[MTResult], config: ProbeConfig, path: Path,
    training_data: dict | None = None,
    training_runs: list[dict] | None = None,
) -> Path:
    """Serialise probe results with the shared budget that produced them."""
    ensure_dir(path.parent)
    document = {
        "probe_config": asdict(config),
        "training_protocol": ("local_jsonl_stop_step_v3" if training_data else "mafand_stop_step_v3"),
        "primary_metric": "chrf2pp",
        "variety_note": (
            "Akuapem (twi_Latn_akua1239) is the primary variety. Asante "
            "(twi_Latn_asan1239) is cross-variety transfer. Never averaged."
        ),
        "metric_note": (
            "chrF++ is primary and reproducible from the predictions alone. "
            "BLEU is secondary."
        ),
        "results": [r.to_dict() for r in results],
    }
    if training_data is not None:
        document["training_data"] = training_data
    if training_runs is not None:
        document["training_runs"] = training_runs
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote MT probe report to %s", path)
    return path


def render_table(results: list[MTResult]) -> str:
    """Markdown table, chrF++ first."""
    header = (
        "| system | dataset | variety | direction | n | chrF++ | BLEU |\n"
        "| --- | --- | --- | --- | ---: | ---: | ---: |\n"
    )
    rows = [
        f"| {r.system} | {r.dataset} | {r.variety or '-'} | {r.direction} | "
        f"{r.n_examples} | {r.chrf2pp:.2f} | {r.bleu:.2f} |"
        for r in results
    ]
    return header + "\n".join(rows)
