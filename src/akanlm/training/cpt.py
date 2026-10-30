"""Single-process/DDP mT5 CPT using the resolved Mesh T5 recipe.

Each GPU holds a full replica. Run cards record provenance and token budgets;
numerical parity with Mesh is not claimed."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from akanlm.utils.logging import get_logger
from akanlm.utils.paths import digest, ensure_dir, file_sha256

logger = get_logger(__name__)


def configure_loss_normalization(model) -> None:
    """Disable Trainer token counting and world-size scaling for mT5's mean loss.

    Trainer handles accumulation; ValidationTrainer applies target-capacity normalization."""
    model.accepts_loss_kwargs = False


ACCEL_ROOT = Path(__file__).resolve().parents[3] / "configs" / "accel"

#: Keys in an accelerator profile that describe it rather than override a field.
_ACCEL_METADATA = frozenset({"description"})


def load_accel_profile(accel: Path | str, model_name: str) -> dict:
    """Read a profile path or ``configs/accel/<name>.yaml`` as config overrides.

    Values may be shared or keyed by model; reject missing or unknown fields."""
    path = Path(accel)
    if not path.suffix:
        path = ACCEL_ROOT / f"{accel}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in ACCEL_ROOT.glob("*.yaml"))
        raise FileNotFoundError(f"no accelerator profile {accel!r}; have {available}")

    profile = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    fields = {f.name for f in dataclasses.fields(TrainingConfig)}

    resolved: dict = {}
    for key, value in profile.items():
        if key in _ACCEL_METADATA:
            continue
        if key not in fields:
            raise KeyError(f"{path}: '{key}' is not a TrainingConfig field")
        if isinstance(value, dict):
            if model_name not in value:
                raise KeyError(
                    f"{path}: '{key}' has no row for {model_name!r} "
                    f"(has {sorted(value)})"
                )
            resolved[key] = value[model_name]
        else:
            resolved[key] = value

    logger.info("accel profile %s applied to %s: %s", path.stem, model_name, resolved)
    return resolved


@dataclass
class TrainingConfig:
    """Declarative continued-pretraining config."""

    name: str
    base_checkpoint: str
    """Directory produced by akanlm.tokenizer.surgery."""

    expected_tokenizer: str | None = None
    """Require tokenizer.json identity with this selected tokenizer directory."""

    mixture: str = "m1x_en"

    # 1,024 encoder slots require 1,137 raw tokens and 229 targets, padded to 256.
    # Validation/test use the same lengths (DenoisingConfig.from_training).
    input_length: int = 1024
    target_length: int = 256
    noise_density: float = 0.15
    mean_noise_span_length: float = 3.0
    dropout_rate: float | None = 0.0
    """Override before model construction; null preserves checkpoint dropout."""

    # Optimization
    append_document_eos: bool = False
    """Google concatenates tokenized documents before adding corrupted-sequence EOS."""
    learning_rate: float = 1e-2
    warmup_steps: int = 10000
    max_steps: int = 100_000
    lr_scheduler_type: str = "rsqrt_no_ramp_down"
    weight_decay: float = 0.0
    max_grad_norm: float = 0.0
    adafactor_scale_parameter: bool = True
    adafactor_min_dim_size_to_factor: int = 128
    adafactor_decay_rate: float = -0.8
    """HF's signed form of Mesh Adafactor's ``step**-0.8`` decay schedule."""
    adafactor_beta1: float | None = None
    """No first moment. Mesh spells this ``beta1=0.0``; HF spells it ``None``."""
    adafactor_clip_threshold: float = 1.0
    adafactor_epsilon1: float = 1e-30
    adafactor_epsilon2: float = 1e-3
    z_loss: float = 1e-4
    """Training-only log-partition penalty; validation remains cross-entropy."""
    loss_normalization: str = "target_capacity"
    """Mesh divides training loss by padded target capacity; valid_tokens is the HF per-token mean."""

    # Throughput knobs; the per-device value is what changes between GPUs.
    world_size: int = 1
    """Declared training processes; included in the global optimizer batch."""
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 128
    per_device_eval_batch_size: int = 8
    gradient_checkpointing: bool = True
    bf16: bool = True
    dataloader_num_workers: int = 4

    # Bookkeeping
    logging_steps: int = 50
    eval_steps: int = 2000
    save_steps: int = 2000
    save_total_limit: int = 3
    seed: int = 13

    validation_seed: int = 20260910
    """Mask seed for the frozen corpus-validation denoising tasks."""
    early_stopping_patience: int = 0
    """Opt-in loss-based stopping; zero keeps the planned CPT budget."""
    early_stopping_threshold: float = 0.001
    milestone_steps: list[int] | str = "doubling"
    """Checkpoints kept past ``save_total_limit`` rotation: an explicit list, or
    ``"doubling"`` for the first save at or after warmup, doubling until
    ``max_steps``, plus ``max_steps``. See :func:`resolve_milestone_steps`."""

    ignore_data_skip: bool = True
    """Restart the interrupted epoch on resume instead of skipping consumed batches.

    False resumes the exact data order. For a map-style dataset, Accelerate
    skips consumed batches in the batch sampler (``SkipBatchSampler``) without
    loading or collating them. ``mt5-small.yaml`` and the ``b200_2gpu``
    profile set false."""

    @classmethod
    def from_yaml(
        cls, path: Path, accel: Path | str | None = None, **overrides
    ) -> TrainingConfig:
        """Load model settings, then accelerator overrides, then explicit overrides."""
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if accel is not None:
            data.update(load_accel_profile(accel, data["name"]))
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps * self.world_size

    def resolved_lengths(self) -> tuple[int, int]:
        """``(tokens_length, target_length)`` for packing and collation."""
        from akanlm.data.collator import compute_input_and_target_lengths

        tokens_length, _ = compute_input_and_target_lengths(
            self.input_length, self.noise_density, self.mean_noise_span_length
        )
        return tokens_length, self.target_length


def resolve_milestone_steps(config: TrainingConfig) -> list[int]:
    """Resolve explicit or doubling milestones, always including ``max_steps``.

    Doubling starts at the first save-aligned step at/after warmup: s, 2s, 4s, ... .
    Explicit steps must align with ``save_steps``; longer resumes add later milestones."""
    if isinstance(config.milestone_steps, str):
        if config.milestone_steps != "doubling":
            raise ValueError(f"Unknown milestone schedule {config.milestone_steps!r}; use 'doubling' or a list")
        step = config.save_steps * max(1, -(-config.warmup_steps // config.save_steps))
        steps = []
        while step < config.max_steps:
            steps.append(step)
            step *= 2
        return steps + [config.max_steps]
    if any(step % config.save_steps for step in config.milestone_steps):
        raise ValueError("Milestones must align with save_steps")
    return sorted(set(config.milestone_steps))


@dataclass
class RunCard:
    """Provenance record written beside every checkpoint."""

    run_name: str
    config: dict
    framework: dict
    mixture_summary: dict
    tokenizer_len: int
    model_vocab_size: int
    params_total: int
    params_trainable: int
    tokens_length: int
    target_length: int
    effective_batch_size: int
    tokens_per_step: int
    raw_tokens_per_step: int
    target_capacity_per_step: int
    git_sha: str = ""
    manifest_sha256: str | None = None
    normalization: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    milestone_steps: list[int] = field(default_factory=list)
    """Resolved preserved-checkpoint steps for the final ``max_steps``."""

    def write(self, path: Path) -> Path:
        ensure_dir(path.parent)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("wrote run card to %s", path)
        return path


def git_sha() -> str:
    """Short SHA of the working tree, or ``"unknown"`` outside a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        sha = result.stdout.strip()
        if sha and result.returncode == 0:
            dirty = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=5, check=False,
            ).stdout.strip()
            return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


PACKING_CACHE_ROOT = Path("runs/cache/cpt")


class PackingCacheMissing(RuntimeError):
    """A caller required cached packed data, and packing would have started."""


def build_dataset(
    config: TrainingConfig,
    tokenizer,
    corpora_dir: Path,
    mixtures_dir: Path,
    protected=None,
    cache_dir: Path | None = None,
    fingerprint: str | None = None,
    require_cache: bool = False,
    workers: int = 1,
):
    """Return packed ``(dataset, mixture_summary, packing_stats)``.

    ``require_cache`` rejects missing caches. Worker count does not change cache identity."""
    from functools import partial

    from datasets import Dataset

    from akanlm.data.mixture import (
        ExactDuplicates,
        MixtureConfig,
        iter_documents,
        validate,
    )
    from akanlm.data.packing import PackingStats, encode, pack

    if require_cache and (cache_dir is None or fingerprint is None):
        raise ValueError("require_cache needs a cache_dir and fingerprint")
    mixture = MixtureConfig.from_yaml(mixtures_dir / f"{config.mixture}.yaml")
    summary = validate(mixture, corpora_dir)

    tokens_length, _ = config.resolved_lengths()
    stats = PackingStats()
    duplicates = ExactDuplicates()
    exposure = {}

    def generate():
        # datasets calls this only when the fingerprinted cache is absent.
        if require_cache:
            raise PackingCacheMissing(
                f"no packed training data under {cache_dir}; run prepare_training_data "
                "with this code, tokenizer and split first")
        documents = iter_documents(mixture, corpora_dir, duplicates=duplicates, document_filter=protected,
                                   transform=partial(encode, tokenizer), workers=workers, exposure=exposure)
        for chunk in pack(documents, tokenizer, tokens_length, stats, append_eos=config.append_document_eos):
            yield chunk.to_dict()
        if cache_dir is not None:
            (cache_dir / "preparation.json").write_text(json.dumps({
                "packing": stats.to_dict(),
                "exact_duplicates": duplicates.report(),
                "exclusion": protected.report() if protected else {},
                "training_text_exposure": exposure,
            }, indent=2))

    if cache_dir is not None:
        ensure_dir(cache_dir)
    try:
        dataset = Dataset.from_generator(generate, cache_dir=str(cache_dir) if cache_dir else None,
                                         fingerprint=fingerprint)
    except Exception as error:
        # datasets re-raises generator errors as a generic DatasetGenerationError.
        if isinstance(error.__cause__, PackingCacheMissing):
            raise error.__cause__ from None
        raise
    if cache_dir is not None:
        summary["preparation"] = json.loads((cache_dir / "preparation.json").read_text())
    return dataset, summary, stats


def align_dataset_to_full_updates(dataset, effective_batch_size: int, seed: int):
    """Seed-shuffle before dropping chunks to complete optimizer batches.

    Return ``(dataset, report)``; reject fewer than one full update."""
    if effective_batch_size < 1:
        raise ValueError("effective_batch_size must be positive")
    original = len(dataset)
    usable = original - original % effective_batch_size
    if not usable:
        raise ValueError(
            f"Training dataset has {original} chunks, fewer than one complete "
            f"optimizer update of {effective_batch_size} chunks"
        )
    dropped = original - usable
    if dropped:
        dataset = dataset.shuffle(seed=seed).select(range(usable))
        logger.warning(
            "dropping %d of %d packed chunks after a seeded shuffle so every "
            "optimizer update has exactly %d examples",
            dropped, original, effective_batch_size,
        )
    return dataset, {
        "original_chunks": original,
        "usable_chunks": usable,
        "dropped_chunks": dropped,
        "examples_per_optimizer_update": effective_batch_size,
        "complete_updates_per_epoch": usable // effective_batch_size,
        "shuffle_seed_if_trimmed": seed if dropped else None,
    }


def check_expected_tokenizer(config: TrainingConfig) -> None:
    """Refuse an initialization whose tokenizer is not the selected one."""
    if config.expected_tokenizer:
        expected = Path(config.expected_tokenizer) / "tokenizer.json"
        actual = Path(config.base_checkpoint) / "tokenizer.json"
        if file_sha256(expected) != file_sha256(actual):
            raise ValueError(f"Initialization tokenizer differs from selected {config.expected_tokenizer}; rerun surgery")


def training_dataset_key(config, tokenizer, validation_report, manifest_sha256, corpora_dir, mixtures_dir) -> str:
    """Identity of the packed training data, which also names its cache directory."""
    tokens_length, _ = config.resolved_lengths()
    return digest({
        "validation": validation_report["sha256"], "manifest": manifest_sha256,
        "tokenizer": tokenizer.backend_tokenizer.to_str(), "raw_length": tokens_length,
        "append_document_eos": config.append_document_eos,
        "mixture": (mixtures_dir / f"{config.mixture}.yaml").read_text(),
        "sources": {p.name: p.read_text() for p in sorted(corpora_dir.glob("*.yaml"))},
        "implementation": {name: (Path(__file__).parents[1] / name).read_text() for name in
                           ("training/cpt.py", "training/validation.py", "data/mixture.py",
                            "data/packing.py", "corpora/loaders.py", "corpora/extract.py", "data/cpt_splits.py",
                            "eval/denoising.py", "data/collator.py", "utils/parallel.py")},
    })


def prepare_training_data(
    config: TrainingConfig,
    corpora_dir: Path = Path("configs/corpora"),
    mixtures_dir: Path = Path("configs/mixtures"),
    manifest_sha256: str | None = None,
    *,
    split_manifest: Path,
    workers: int = 1,
) -> dict:
    """Build the training cache on CPU, without loading a model.

    Return ``{dataset_key, chunks, preparation}``; training reuses the same key."""
    from transformers import AutoTokenizer

    from akanlm.data.cpt_splits import build_validation

    check_expected_tokenizer(config)
    tokenizer = AutoTokenizer.from_pretrained(config.base_checkpoint)
    _, protected, validation_report = build_validation(config, tokenizer, corpora_dir, split_manifest)
    dataset_key = training_dataset_key(config, tokenizer, validation_report, manifest_sha256,
                                       corpora_dir, mixtures_dir)
    dataset, summary, _ = build_dataset(
        config, tokenizer, corpora_dir, mixtures_dir,
        protected=protected, cache_dir=PACKING_CACHE_ROOT / dataset_key, fingerprint=dataset_key,
        workers=workers,
    )
    return {"dataset_key": dataset_key, "chunks": len(dataset), "preparation": summary["preparation"]}


def train(
    config: TrainingConfig,
    output_dir: Path,
    corpora_dir: Path = Path("configs/corpora"),
    mixtures_dir: Path = Path("configs/mixtures"),
    manifest_sha256: str | None = None,
    resume_from_checkpoint: str | bool | None = None,
    *,
    split_manifest: Path,
    require_cache: bool = False,
) -> RunCard:
    """Train or explicitly resume CPT and persist ``run_card.json``.

    Use the frozen split for validation/exclusion; ``require_cache`` forbids on-demand packing."""
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        Seq2SeqTrainingArguments,
        set_seed,
    )

    from akanlm.data.collator import DataCollatorForT5MLM
    from akanlm.data.cpt_splits import build_validation
    from akanlm.tokenizer.surgery import assert_embeddings_shared, normalize_tying
    from akanlm.training.checkpoints import (
        bind_training_plan,
        preserve_milestones_callback,
        runtime_framework,
    )
    from akanlm.training.distributed import (
        describe_rank,
        on_main_process,
        require_identical_across_ranks,
        training_state,
    )
    from akanlm.training.optimization import validate_model_setup, validate_optimization
    from akanlm.training.validation import (
        TrainingAndFixedEvalCollator,
        trainer_classes,
    )

    validate_optimization(config)

    state = training_state(config)
    world_size = state.num_processes
    # Logged at startup so a pilot can confirm GPU and core binding in its
    # first minute, and a run that later fails still records its devices.
    rank_identity = describe_rank(state)
    logger.info("rank %d resources: %s", state.process_index, json.dumps(rank_identity, sort_keys=True))

    if config.eval_steps != config.save_steps:
        raise ValueError("CPT save_steps must equal eval_steps for attributable validation checkpoints")
    if config.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be nonnegative")
    milestone_steps = resolve_milestone_steps(config)
    logger.info("preserving milestone checkpoints at steps %s", milestone_steps)

    ensure_dir(output_dir)
    set_seed(config.seed)

    check_expected_tokenizer(config)
    tokenizer = AutoTokenizer.from_pretrained(config.base_checkpoint)
    model_config = AutoConfig.from_pretrained(config.base_checkpoint)
    if config.dropout_rate is not None:
        model_config.dropout_rate = config.dropout_rate
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_checkpoint, config=model_config)

    # Correct MT5Config's tying flag before saving the separate output head.
    if normalize_tying(model):
        raise RuntimeError(
            f"{config.base_checkpoint} loaded with lm_head tied to the embeddings. "
            "mT5 checkpoints are untied; see akanlm.tokenizer.surgery."
        )
    assert_embeddings_shared(model)
    validate_model_setup(model, tokenizer, config)
    configure_loss_normalization(model)

    tokens_length, target_length = config.resolved_lengths()
    logger.info(
        "packing to %d raw tokens -> %d inputs / %d targets",
        tokens_length, config.input_length, target_length,
    )

    eval_dataset, protected, validation_report = build_validation(
        config, tokenizer, corpora_dir, split_manifest)
    dataset_key = training_dataset_key(config, tokenizer, validation_report, manifest_sha256,
                                       corpora_dir, mixtures_dir)
    framework = {**runtime_framework(), "world_size": world_size,
                 "distributed_backend": state.backend}
    on_main_process(bind_training_plan, output_dir, config, dataset_key, validation_report,
                    resume_from_checkpoint, framework=framework)
    # Rank zero loads first, and its trimming shuffle writes the index cache
    # that the other ranks then read instead of writing it concurrently.
    with state.main_process_first():
        dataset, mixture_summary, _ = build_dataset(
            config, tokenizer, corpora_dir, mixtures_dir,
            protected=protected, cache_dir=PACKING_CACHE_ROOT / dataset_key,
            fingerprint=dataset_key, require_cache=require_cache,
        )
        if not len(dataset):
            raise ValueError("No training chunks remain after document exclusion")
        dataset, batching_report = align_dataset_to_full_updates(
            dataset, config.effective_batch_size, config.seed
        )
    # Sampler sharding and validation sharding both index each rank's own
    # copy, so the copies must be identical.
    require_identical_across_ranks("validation tasks and packed training data", {
        "validation_sha256": validation_report["sha256"], "dataset_key": dataset_key,
        "usable_chunks": len(dataset),
    })
    batching_report["planned_dataset_epochs"] = (
        config.max_steps / batching_report["complete_updates_per_epoch"]
    )
    if batching_report["planned_dataset_epochs"] > 10:
        logger.warning(
            "the configured %d updates equal %.1f passes over the packed dataset; "
            "monitor held-out loss for overfitting and report this exposure",
            config.max_steps, batching_report["planned_dataset_epochs"],
        )
    mixture_summary["preparation"]["optimizer_batching"] = batching_report
    validation_report["training_preparation"] = mixture_summary["preparation"]
    on_main_process((output_dir / "validation.json").write_text, json.dumps(validation_report, indent=2))

    collator = DataCollatorForT5MLM(
        tokenizer=tokenizer,
        noise_density=config.noise_density,
        mean_noise_span_length=config.mean_noise_span_length,
        input_length=config.input_length,
        target_length=target_length,
        pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=model.config.decoder_start_token_id,
        seed=config.seed + state.process_index,
    )

    # Extra workers add prefetch depth, not CPU allocation.
    available_cpus = len(os.sched_getaffinity(0))
    num_workers = config.dataloader_num_workers
    if num_workers > available_cpus:
        logger.warning("%d data-loader workers and the training process share %d CPU cores",
                       num_workers, available_cpus)
    else:
        logger.info("%d data-loader workers share %d CPU cores with this rank", num_workers, available_cpus)

    # transformers 5.x dropped `overwrite_output_dir` and `save_safetensors`
    # (safetensors is now the only serialization format), so neither is passed.
    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type=("constant" if config.lr_scheduler_type == "rsqrt_no_ramp_down"
                           else config.lr_scheduler_type),
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        optim="adafactor",
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        ddp_timeout=7200,
        bf16=config.bf16 and torch.cuda.is_available(),
        dataloader_num_workers=num_workers,
        # Forking loader workers after NCCL initialization is unsafe.
        dataloader_multiprocessing_context="spawn" if world_size > 1 and num_workers else None,
        # The dataset is aligned above; this remains enabled as a defensive
        # guard if a future sampler shards or filters it.
        dataloader_drop_last=True,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        seed=config.seed,
        report_to=[],
        remove_unused_columns=False,
        ignore_data_skip=config.ignore_data_skip,
        eval_on_start=not bool(resume_from_checkpoint),
        metric_for_best_model="eval_cpt_loss",
        greater_is_better=False,
        load_best_model_at_end=config.early_stopping_patience > 0,
        restore_callback_states_from_checkpoint=True,
        prediction_loss_only=True,
    )

    ValidationTrainer, CPTEarlyStopping = trainer_classes()
    callbacks = [preserve_milestones_callback(milestone_steps)]
    if config.early_stopping_patience:
        callbacks.append(CPTEarlyStopping(config.early_stopping_patience,
                                          config.early_stopping_threshold))
    trainer = ValidationTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=TrainingAndFixedEvalCollator(collator),
        callbacks=callbacks,
        cpt_config=config,
    )

    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(output_dir / "final"))
    on_main_process(tokenizer.save_pretrained, str(output_dir / "final"))
    # Explicitly score the exported model, including runs ending between evals.
    final_metrics = trainer.evaluate()
    trainer.save_state()

    from accelerate.utils import gather_object

    from akanlm.audit.sources import load_sources
    from akanlm.corpora.normalize import GENERAL_RULES, normalization_spec

    rank_resources = gather_object([{
        **rank_identity, "dataloader_workers": num_workers, "torch_threads": torch.get_num_threads(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(state.device) if state.device.type == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(state.device) if state.device.type == "cuda" else None,
    }])

    declared_sources = load_sources(corpora_dir)
    normalization = {
        component["source"]: {
            "target": normalization_spec(declared_sources[component["source"]].effective_rules()),
            "source": normalization_spec(GENERAL_RULES) if component.get("as_pairs") else None,
        }
        for component in mixture_summary["components"]
    }
    card = RunCard(
        run_name=config.name,
        config=asdict(config),
        framework={
            **framework,
            "parameter_dtype": "float32",
            "activation_dtype": "bfloat16" if args.bf16 else "float32",
            "dataloader_start_method": args.dataloader_multiprocessing_context,
            "corruption_seed_rule": ("collator seed = config.seed + global_rank; each loader worker draws from "
                                     "numpy SeedSequence([collator seed, torch worker seed])"),
            "exact_corruption_resume": False,
            "rank_resources": rank_resources,
        },
        mixture_summary=mixture_summary,
        tokenizer_len=len(tokenizer),
        model_vocab_size=model.config.vocab_size,
        params_total=sum(p.numel() for p in model.parameters()),
        params_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        tokens_length=tokens_length,
        target_length=target_length,
        effective_batch_size=config.effective_batch_size,
        # Mesh counts max(inputs, targets) slots; record raw/decoder budgets separately.
        tokens_per_step=config.effective_batch_size * config.input_length,
        raw_tokens_per_step=config.effective_batch_size * tokens_length,
        target_capacity_per_step=config.effective_batch_size * target_length,
        git_sha=git_sha(),
        manifest_sha256=manifest_sha256,
        normalization=normalization,
        validation=validation_report,
        metrics={**result.metrics, **final_metrics,
                 "completed_steps": trainer.state.global_step,
                 "best_loss_checkpoint": trainer.state.best_model_checkpoint,
                 "best_cpt_loss": trainer.state.best_metric,
                 "resolved_dropout_rate": model.config.dropout_rate,
                 "exported_model": "best_cpt_loss" if config.early_stopping_patience else "last_step"},
        milestone_steps=milestone_steps,
    )
    on_main_process(card.write, output_dir / "run_card.json")
    return card
