"""Command-line entry point for the AkanT5 pipeline.

Mirrors the ``kgmod`` CLI conventions of the parent repository: a Typer app,
sub-apps per stage, and all paths relative to the project root.

The pipeline runs in this order, and each step is checkable before the next:

.. code-block:: text

    audit manifest      corpus provenance, sizes, hashes
    audit decontaminate n-gram ledger against held-out sets
    tokenizer train     SentencePiece over Twi + English replay
    tokenizer analyze   fertility vs mT5 and afri-mT5
    model surgery       swap mT5's vocabulary for ours
    cpt-split           80/10/10 corpus document holdouts, reusable across sizes
    cpt                 pre-CPT scores, paired continued pretraining, reference evaluation
    eval cpt            regenerate the corpus validation/test report
    eval mt             downstream MT probe on MAFAND and FLORES+ (stop-step v3)
    eval mt-freeze      write the frozen best-dev MT protocol for a study
    eval mt-select      learning-rate selection on development data
    eval mt-test        seeded test runs at the selected learning rate
    eval mt-report      seed means and paired bootstrap over a study
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# AKANLM_ALLOW_NETWORK=1 lets a command reach the Hub even in an environment
# that already exports HF_HUB_OFFLINE=1 (for example, a job script that keeps
# compute nodes offline by default). Without this override, such a command
# would fail with an offline cache miss although downloading was requested.
#
# huggingface_hub.constants and datasets.config both read HF_HUB_OFFLINE once,
# at import. This runs before either is imported (every Hub-touching import in
# this module is deliberately lazy, inside the command functions), so setting
# it here is what makes the override effective.
if os.environ.get("AKANLM_ALLOW_NETWORK") == "1":
    os.environ["HF_HUB_OFFLINE"] = "0"

import typer

from akanlm.utils.logging import get_logger
from akanlm.utils.paths import MANIFEST_DIR, RUNS_ROOT, ensure_dir, run_label

logger = get_logger("akanlm")

app = typer.Typer(help="AkanT5: Twi continued pretraining of the mT5 family.", no_args_is_help=True)
audit_app = typer.Typer(help="Corpus provenance and decontamination.", no_args_is_help=True)
tokenizer_app = typer.Typer(help="Twi tokenizer training and analysis.", no_args_is_help=True)
model_app = typer.Typer(help="Vocabulary surgery on mT5 checkpoints.", no_args_is_help=True)
eval_app = typer.Typer(help="Evaluation harness.", no_args_is_help=True)

app.add_typer(audit_app, name="audit")
app.add_typer(tokenizer_app, name="tokenizer")
app.add_typer(model_app, name="model")
app.add_typer(eval_app, name="eval")

CORPORA_DIR = Path("configs/corpora")

MIXTURES_DIR = Path("configs/mixtures")
LANGUAGE_CONFIG = Path("configs/languages/akan.json")
EVAL_DIR = RUNS_ROOT / "eval"

def eval_report_path(kind: str, subjects: list[str]) -> Path:
    """Default destination for an eval report: ``runs/eval/<kind>_<run>_<stamp>``.

    The run name is in the filename because reports are produced per CPT
    mixture; a timestamp alone leaves a directory of reports in which nothing
    says which mixture produced which numbers.
    """
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return EVAL_DIR / f"{kind}_{run_label(subjects)}_{stamp}.json"


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


@audit_app.command("sources")
def audit_sources(
    corpora_dir: Path = typer.Option(CORPORA_DIR, help="Corpus declarations."),
) -> None:
    """Validate every corpus declaration and print the roster."""
    from akanlm.audit.sources import load_sources, unlicensed_for_release

    sources = load_sources(corpora_dir)
    typer.echo(f"{len(sources)} sources declared\n")
    typer.echo(f"{'name':<28} {'role':<10} {'provenance':<18} {'licence':<16} variety")
    typer.echo("-" * 100)
    for name, source in sorted(sources.items()):
        variety = source.claimed_variety or "-"
        flag = "" if source.variety_audited else " (unaudited)"
        typer.echo(
            f"{name:<28} {source.role.value:<10} {source.provenance.value:<18} "
            f"{source.licence:<16} {variety}{flag}"
        )

    blocking = unlicensed_for_release(sources.values())
    if blocking:
        typer.echo(f"\nNon-commercial sources: {', '.join(blocking)}")
        typer.echo("Any checkpoint trained on these is research-only.")


@audit_app.command("manifest")
def audit_manifest(
    corpora_dir: Path = typer.Option(CORPORA_DIR),
    output: Path = typer.Option(MANIFEST_DIR / "corpus_manifest.json"),
    limit: int | None = typer.Option(None, help="Cap rows per source, for a fast dry run."),
    training_only: bool = typer.Option(True, help="Measure only role=train sources."),
    mixtures: list[str] = typer.Option(
        ["m1x", "m1x_en"], help="Report provenance shares for these mixtures."
    ),
) -> None:
    """Measure the declared corpora and write the manifest."""
    from akanlm.audit import manifest as manifest_mod
    from akanlm.audit.sources import Role, load_sources
    from akanlm.data.mixture import MixtureConfig

    from akanlm.data.mixture import derive_source_caps, load_all

    sources = load_sources(corpora_dir)
    # Same rule as `audit decontaminate`: measure what the mixtures actually
    # consume. A role=train source no mixture uses is not measured at all.
    caps = derive_source_caps(load_all(MIXTURES_DIR))
    targets = {
        n: s for n, s in sources.items()
        if (not training_only or s.role is Role.TRAIN)
        and (s.role is not Role.TRAIN or n in caps)
    }
    for name, source in sorted(sources.items()):
        if source.role is Role.TRAIN and name not in caps:
            typer.echo(f"skipping {name}: role=train but used by no mixture")

    stats = {}
    for name, source in targets.items():
        # hf_files sources (including the mC4 English replay) are always read from
        # their local extraction. Wikipedia is the sole replay source that is
        # intentionally streamed from the Hub.
        streaming = source.hf_files is None and name == "english_replay_wikipedia"
        cap = caps.get(name)
        rows = limit if limit is not None else (cap.rows if cap else None)
        chars = None if limit is not None else (cap.chars if cap else None)
        stats[name] = manifest_mod.measure(
            source, limit=rows, streaming=streaming, max_chars=chars
        )

    document = manifest_mod.build(sources, stats)
    document["provenance_summary"] = manifest_mod.provenance_summary(sources.values(), stats)

    per_mixture = {}
    for name in mixtures:
        path = MIXTURES_DIR / f"{name}.yaml"
        if not path.exists():
            continue
        config = MixtureConfig.from_yaml(path)
        if not config.components:
            continue
        per_mixture[name] = manifest_mod.provenance_summary(sources.values(), stats, config)
    document["provenance_by_mixture"] = per_mixture
    manifest_mod.write(document, output)

    typer.echo(f"\nmanifest sha256: {document['manifest_sha256'][:16]}")
    if limit:
        typer.echo(f"(sampled at --limit {limit}; shares are indicative, not final)")
    typer.echo("\nprovenance share by mixture (character-weighted, repeats applied):")
    for name, summary in per_mixture.items():
        parts = ", ".join(f"{k} {v:.1%}" for k, v in sorted(summary["share"].items()))
        typer.echo(f"  {name:<4} human {summary['human_share']:>6.1%}   [{parts}]")


@audit_app.command("decontaminate")
def audit_decontaminate(
    corpora_dir: Path = typer.Option(CORPORA_DIR),
    output: Path = typer.Option(MANIFEST_DIR / "contamination_ledger.json"),
    mixtures_dir: Path = typer.Option(MIXTURES_DIR, help="Mixtures that define the row caps."),
    ngram: int = typer.Option(13, help="N-gram width for the overlap index."),
    limit: int | None = typer.Option(
        None,
        help="Override every per-source cap, for a dry run. Produces a ledger that is "
        "NOT publication-valid — see the warning it prints.",
    ),
) -> None:
    """Build the n-gram contamination ledger against held-out sets."""
    from akanlm.audit.decontaminate import EvalItem, build_index, decontaminate
    from akanlm.audit.sources import Modality, Role, load_sources
    from akanlm.corpora.loaders import load_pairs, load_texts
    from akanlm.data.mixture import derive_source_caps, load_all

    sources = load_sources(corpora_dir)

    # Sweep exactly what training will read: the mixtures already declare how
    # much of each source is consumed, so a source used by no mixture is skipped
    # and one capped by every mixture is capped here too. A single global
    # --limit cannot express this — it would truncate pristine_twi_english to
    # the same row count as the English replay buffer and silently under-report
    # contamination across the bulk of the Twi corpus.
    caps = derive_source_caps(load_all(mixtures_dir))
    typer.echo("training rows to sweep (derived from mixtures):")
    for name, source in sorted(sources.items()):
        if source.role is not Role.TRAIN:
            continue
        if name not in caps:
            typer.echo(f"  {name:<28} SKIPPED — role=train but used by no mixture")
        else:
            cap = caps[name]
            if cap.chars is not None:
                shown = f"{cap.chars:,} chars"
            elif cap.rows is not None:
                shown = f"{cap.rows:,} rows"
            else:
                shown = "all rows"
            typer.echo(f"  {name:<28} {shown}")
    if limit is not None:
        typer.secho(
            f"\nWARNING: --limit {limit} overrides every per-source cap. The resulting "
            "ledger samples\n         each training source and must not be frozen as the "
            "publication artifact.",
            fg="yellow",
        )

    items: list[EvalItem] = []
    for name, source in sources.items():
        if source.role is Role.TRAIN:
            continue
        if source.modality is Modality.MONOLINGUAL:
            texts = (r.text for r in load_texts(source))
        else:
            texts = (r.tgt_text for r in load_pairs(source))
        for index, text in enumerate(texts):
            items.append(EvalItem(source=name, index=index, text=text))

    index, counts = build_index(items, ngram)
    typer.echo(f"indexed {len(index)} {ngram}-grams from {sum(counts.values())} held-out items")

    def train_docs():
        for name, source in sources.items():
            if source.role is not Role.TRAIN or name not in caps:
                continue
            cap = caps[name]
            rows = limit if limit is not None else cap.rows
            chars = None if limit is not None else cap.chars
            streaming = source.hf_files is None and name == "english_replay_wikipedia"
            if source.modality is Modality.MONOLINGUAL:
                for record in load_texts(
                    source, limit=rows, streaming=streaming, max_chars=chars
                ):
                    yield (name, record.text)
            else:
                for record in load_pairs(
                    source, limit=rows, streaming=streaming, max_chars=chars
                ):
                    yield (name, record.tgt_text)

    _, ledger = decontaminate(train_docs(), index, counts)
    from akanlm.corpora.normalize import normalization_spec

    ledger.normalization = {
        name: normalization_spec(source.effective_rules()) for name, source in sources.items()
        if name in caps or source.role is not Role.TRAIN
    }
    ledger.write(output)

    typer.echo("\nremoval rate by training source:")
    for name in sorted(ledger.train_docs_seen):
        typer.echo(f"  {name:<28} {ledger.removal_rate(name):.4%}")
    typer.echo(f"\nheld-out items matched at least once: {len(ledger.eval_items_hit)}")


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------


@tokenizer_app.command("train")
def tokenizer_train(
    config: Path = typer.Option(..., "--config", "-c", help="configs/tokenizer/*.yaml"),
    output: Path | None = typer.Option(None, help="Defaults to runs/tokenizer/<name>."),
    corpora_dir: Path = typer.Option(CORPORA_DIR),
    split_manifest: Path | None = typer.Option(
        None, help="Frozen CPT document splits; fit candidates on training partitions only."),
) -> None:
    """Train the Twi/English SentencePiece tokenizer."""
    from akanlm.tokenizer.train_spm import TokenizerConfig, train

    spec = TokenizerConfig.from_yaml(config)
    destination = output or (RUNS_ROOT / "tokenizer" / spec.name)
    train(spec, destination, corpora_dir=corpora_dir, split_manifest=split_manifest)

    typer.echo(f"\ntokenizer: {destination}")
    typer.echo(f"pieces {spec.vocab_size} + 100 sentinels -> model vocab_size {spec.model_vocab_size}")


@tokenizer_app.command("extend")
def tokenizer_extend(
    candidates: Path = typer.Option(..., help="Training-only SentencePiece candidate directory."),
    output: Path = typer.Option(..., help="New extended tokenizer directory; must be empty."),
    base: str = typer.Option("google/mt5-small", help="Original tokenizer reference."),
) -> None:
    """Keep original mT5 IDs and append all missing ordinary candidate pieces."""
    from akanlm.tokenizer.extension import extend_tokenizer

    extend_tokenizer(base, candidates, output, require_training_partitions=True)
    typer.echo(f"Extended tokenizer: {output}")


@tokenizer_app.command("analyze")
def tokenizer_analyze(
    tokenizers: list[str] = typer.Argument(..., help="Paths or hub ids to compare."),
    corpora_dir: Path = typer.Option(CORPORA_DIR),
    sample: int = typer.Option(2000, help="Documents per language."),
    english_source: str = typer.Option(
        "english_replay_mc4", help="Declared English source for the fertility sample."
    ),
    output: Path | None = typer.Option(
        None, help="Defaults to runs/eval/tokenizer_<run>_<timestamp>.json."
    ),
) -> None:
    """Compare tokenizers on Twi and English fertility.

    Pass ``google/mt5-base`` and ``masakhane/afri-mt5-base`` alongside a local
    tokenizer to reproduce the baseline comparison.
    """
    from akanlm.audit.sources import load_sources
    from akanlm.corpora.loaders import load_texts
    from akanlm.tokenizer.analyze import compare, render_table
    from akanlm.tokenizer.surgery import load_tokenizer

    sources = load_sources(corpora_dir)
    if english_source not in sources:
        raise typer.BadParameter(f"unknown English source: {english_source}")
    twi = [r.text for r in load_texts(sources["twi_wikipedia"], limit=sample)]
    english = [
        r.text for r in load_texts(
            sources[english_source], limit=min(sample, 500),
            streaming=sources[english_source].hf_files is None,
        )
    ]
    # Twi Wikipedia is in the tokenizer training pool, which favours larger
    # vocabularies. FLORES+ dev is held out from tokenizer training and is not
    # the devtest split used for evaluation.
    heldout = {
        f"twi_{name.removeprefix('flores_plus_')}_flores_dev": [
            r.text for r in load_texts(sources[name].model_copy(update={"hf_split": "dev"}))
        ]
        for name in ("flores_plus_asante", "flores_plus_akuapem")
    }

    loaded = {}
    for reference in tokenizers:
        try:
            loaded[reference] = load_tokenizer(reference)
        except Exception as error:  # noqa: BLE001 - report and continue
            typer.echo(f"could not load {reference}: {type(error).__name__}: {error}")

    if not loaded:
        raise typer.Exit(code=1)

    report = compare(loaded, {"twi": twi, **heldout, "en": english}, LANGUAGE_CONFIG)
    typer.echo("\n" + render_table(report))

    destination = output or eval_report_path("tokenizer", tokenizers)
    ensure_dir(destination.parent)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(f"\nwrote {destination}")


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


@model_app.command("surgery")
def model_surgery(
    base: str = typer.Option(..., help="e.g. google/mt5-small."),
    tokenizer: Path = typer.Option(..., help="Directory from `tokenizer train`."),
    output: Path | None = typer.Option(
        None, help="Defaults to runs/init/akant5-<size>-<vocabulary>-sentinels-v1."),
) -> None:
    """Replace an mT5 checkpoint's vocabulary with the AkanT5 one."""
    from akanlm.tokenizer.surgery import apply, check_forward, default_output_dir

    output = output or default_output_dir(base, tokenizer, RUNS_ROOT / "init")
    report = apply(base, tokenizer, output)
    loss = check_forward(output)

    typer.echo(f"\n{base} -> {output}")
    typer.echo(f"  vocab      {report.old_vocab_size} -> {report.new_vocab_size}")
    typer.echo(
        f"  parameters {report.params_before / 1e6:.1f}M -> {report.params_after / 1e6:.1f}M "
        f"(-{100 * report.params_removed / report.params_before:.1f}%)"
    )
    for embedding in report.embedding_reports:
        typer.echo(
            f"  {embedding.matrix:<18} overlap {embedding.overlap_rate:.1%}, "
            f"fallback rows {embedding.strategies.get('fallback', 0)}"
        )
    typer.echo(f"  span-corruption loss {loss:.4f} (below uniform ln(V))")


# ---------------------------------------------------------------------------
# continued pretraining
# ---------------------------------------------------------------------------


@model_app.command("extend")
def model_extend(
    base: str = typer.Option(..., help="e.g. google/mt5-small, google/mt5-base, google/mt5-large."),
    tokenizer: Path = typer.Option(..., help="Directory from `tokenizer extend`."),
    output: Path = typer.Option(..., help="New initialization directory; must be empty."),
) -> None:
    """Preserve original token rows and initialize added input and output rows."""
    from akanlm.tokenizer.extension import apply, check_initialization

    report = apply(base, tokenizer, output)
    checks = check_initialization(output)
    typer.echo(json.dumps(report.to_dict(), indent=2))
    typer.echo(json.dumps(checks, indent=2))


@app.command("cpt-split")
def cpt_split(
    mixture: str = typer.Option("m1x_en"),
    output: Path = typer.Option(...),
    validation_fraction: float = typer.Option(0.1),
    test_fraction: float = typer.Option(0.1),
    corpora_dir: Path = typer.Option(CORPORA_DIR),
) -> None:
    """Freeze 80/10/10 corpus document holdouts before any CPT run or tokenization."""
    from akanlm.data.cpt_splits import SplitConfig, prepare
    from akanlm.data.mixture import MixtureConfig
    spec = MixtureConfig.from_yaml(Path("configs/mixtures") / f"{mixture}.yaml")
    result = prepare(spec, corpora_dir, output, SplitConfig(
        validation_fraction=validation_fraction, test_fraction=test_fraction))
    typer.echo(json.dumps({"sha256": result["sha256"], "sources": {
        name: record["heldout_documents"] for name, record in result["sources"].items()}}, indent=2))


@app.command("cpt")
def cpt_experiment(
    config: Path = typer.Option(..., "--config", "-c"),
    accel: str | None = typer.Option(None),
    mixture: str | None = typer.Option(None),
    output: Path | None = typer.Option(None),
    tokenizer: Path | None = typer.Option(
        None, help="Tokenizer directory; overrides expected_tokenizer. Replacement infers its surgery "
        "output; extension retains the matching configured initialization or requires --base-checkpoint."),
    base_checkpoint: str | None = typer.Option(None),
    max_steps: int | None = typer.Option(None),
    resume: bool = typer.Option(False),
    manifest: Path | None = typer.Option(None),
    split_manifest: Path | None = typer.Option(None, help="Reuse an immutable CPT document-split manifest."),
    validation_fraction: float = typer.Option(0.1),
    test_fraction: float = typer.Option(0.1),
    pre_cpt_only: bool = typer.Option(
        False, help="Score the starting systems on the corpus holdouts, then stop without freezing a plan."),
    corpora_dir: Path = typer.Option(CORPORA_DIR),
    single_arm: bool = typer.Option(False, help="Train the selected vocabulary only, validating every mixture source."),
    prepare_only: bool = typer.Option(False, help="Freeze full-source holdouts and pack the single-arm cache on CPU."),
    workers: int = typer.Option(4, min=1, help="CPU workers for --prepare-only."),
) -> None:
    """CPT with full-mixture holdouts: single-vocabulary or paired-reference training.

    --single-arm --prepare-only freezes all-source holdouts and packs data on
    CPU. --single-arm trains from that cache and supports --resume. Omitting
    --single-arm retains the paired original-tokenizer reference experiment.
    """
    from dataclasses import replace

    from akanlm.data.cpt_splits import SplitConfig
    from akanlm.tokenizer.surgery import MT5_SIZES, default_output_dir, vocabulary_label
    from akanlm.training.cpt import TrainingConfig
    from akanlm.training.distributed import on_main_process, training_state
    from akanlm.training.experiment import run

    spec = TrainingConfig.from_yaml(config, accel=accel, mixture=mixture, base_checkpoint=base_checkpoint,
                                   expected_tokenizer=str(tokenizer) if tokenizer else None,
                                   max_steps=max_steps)
    if tokenizer is not None and base_checkpoint is None:
        if (tokenizer / "extension_report.json").is_file():
            configured = TrainingConfig.from_yaml(config)
            if (not configured.expected_tokenizer
                    or Path(configured.expected_tokenizer).resolve() != tokenizer.resolve()):
                raise typer.BadParameter(
                    "Pass --base-checkpoint for an extended tokenizer that differs from the model config",
                    param_hint="--base-checkpoint")
            spec = replace(spec, base_checkpoint=configured.base_checkpoint)
        else:
            size = spec.name.removeprefix("akant5-")
            if size not in MT5_SIZES:
                raise typer.BadParameter(f"cannot infer the mT5 size from config name {spec.name}; "
                                         "pass --base-checkpoint", param_hint="--tokenizer")
            spec = replace(spec, base_checkpoint=str(default_output_dir(f"google/mt5-{size}", tokenizer,
                                                                       RUNS_ROOT / "init")))
    selected_tokenizer = Path(spec.expected_tokenizer or spec.base_checkpoint)
    if (selected_tokenizer / "extension_report.json").is_file():
        from akanlm.tokenizer.extension import verify_training_split

        verify_training_split(selected_tokenizer, split_manifest)
    manifest_sha = None
    if manifest:
        from akanlm.audit.manifest import assert_normalization_current
        from akanlm.audit.sources import load_sources
        document = json.loads(manifest.read_text())
        assert_normalization_current(document, load_sources(corpora_dir))
        manifest_sha = document["manifest_sha256"]
    # Name the vocabulary so runs of one config with different tokenizers never share a path.
    vocabulary = f"-{vocabulary_label(spec.expected_tokenizer)}" if spec.expected_tokenizer else ""
    suffix = "full-v1" if single_arm else "reference-v1"
    destination = output or RUNS_ROOT / "cpt" / f"{spec.name}{vocabulary}-{spec.mixture}-{suffix}"
    if prepare_only and not single_arm:
        raise typer.BadParameter("--prepare-only requires --single-arm")
    import os

    launched_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if prepare_only and launched_world_size != 1:
        raise typer.BadParameter("CPU preparation must run in one process")
    if not single_arm and (spec.world_size != 1 or launched_world_size != 1):
        raise typer.BadParameter("Distributed CPT requires --single-arm; launch each vocabulary arm separately")
    if single_arm:
        if pre_cpt_only:
            raise typer.BadParameter("--pre-cpt-only belongs to the paired reference workflow")
        from akanlm.training.experiment import prepare_single, run_single, setup_single
        setup_kwargs = dict(corpora_dir=corpora_dir, manifest_sha256=manifest_sha,
                            split_manifest=split_manifest,
                            split_config=SplitConfig(validation_fraction=validation_fraction, test_fraction=test_fraction))
        if prepare_only:
            setup_single(spec, destination, **setup_kwargs)
            prepare_single(destination, workers=workers)
        else:
            training_state(spec)
            on_main_process(setup_single, spec, destination, **setup_kwargs)
            run_single(destination, resume=resume)
        return
    run(spec, destination, corpora_dir=corpora_dir, manifest_sha256=manifest_sha,
        resume=resume, split_manifest=split_manifest, pre_cpt_only=pre_cpt_only,
        split_config=SplitConfig(validation_fraction=validation_fraction, test_fraction=test_fraction))
    if pre_cpt_only:
        typer.echo(f"Pre-CPT evaluation: {destination / 'evaluation/pre_cpt.json'}")
        return
    typer.echo(f"Completed paired CPT experiment: {destination}")


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


@eval_app.command("cpt")
def eval_cpt(
    run_dir: Path = typer.Argument(..., help="Completed paired CPT experiment directory."),
    batch_size: int = typer.Option(4, min=1),
) -> None:
    """Repeat the frozen Google-procedure evaluation on CPT validation and test."""
    from akanlm.training.experiment import evaluate_experiment
    evaluate_experiment(run_dir, batch_size=batch_size)
    typer.echo(f"wrote {run_dir / 'evaluation/cpt_reference.json'}")


@eval_app.command("test-bpc")
def eval_test_bpc(
    run_dir: Path = typer.Argument(..., help="Completed single-arm CPT run, in the form CPT recorded it."),
    checkpoint: str = typer.Option("final", help="'initialization' or 'final'."),
    recheck: bool = typer.Option(False, help="Rescore an existing test result and require equal values."),
    preflight: bool = typer.Option(False, help="Run the identity checks only; needs no GPU."),
    ignore_scorer_version: bool = typer.Option(
        False, help="Accept a validation record written by another version of the scorer; "
                    "both versions are recorded."),
    output: Path | None = typer.Option(None, help="Write the result here instead of the run directory."),
    batch_size: int = typer.Option(4, min=1),
) -> None:
    """Character-span BPC on the run's frozen test split, with its validation settings."""
    from akanlm.eval.test_bpc import score_test_split

    path = score_test_split(run_dir, checkpoint, recheck=recheck, preflight=preflight,
                            ignore_scorer_version=ignore_scorer_version, output=output,
                            batch_size=batch_size)
    typer.echo(f"{'preflight passed; would write' if preflight else 'wrote'} {path}")


@eval_app.command("mt")
def eval_mt(
    checkpoints: list[str] = typer.Argument(..., help="Model paths or hub ids."),
    baselines: bool = typer.Option(True, help="Include recorded CPT init, matching mT5 and Afri-mT5 controls."),
    corpora_dir: Path = typer.Option(CORPORA_DIR),
    output_dir: Path = typer.Option(Path("runs/eval/mt_probe")),
    direction: str = typer.Option("en2twi"),
    flairs_six_epoch_setup: bool = typer.Option(
        False, help="Apply the six-epoch FLAIRS MT setup to the supplied mT5 checkpoint."
    ),
    resume: bool = typer.Option(False, help="Resume matching interrupted MT fine-tunes."),
    train_data_dir: Path | None = typer.Option(
        None, help="Local JSONL directory with train.json and dev.json for MT fine-tuning."
    ),
    max_length: int = typer.Option(
        128, "--max-length", "--max_length", min=2,
        help="Maximum decoder tokens including the start token; independent of training target length.",
    ),
    num_beams: int = typer.Option(4, min=1, help="Translation decoding beam count."),
    batch_size: int = typer.Option(16, min=1, help="Translation generation batch size."),
    max_steps: int = typer.Option(3000, min=1, help="Fine-tuning step cap, shared by all systems."),
    eval_steps: int = typer.Option(100, min=1, help="Validation interval for MT early stopping."),
    early_stopping_patience: int = typer.Option(
        5, min=0, help="Validation checks without improvement before stopping; 0 runs the fixed budget.",
    ),
    early_stopping_threshold: float = typer.Option(
        0.001, min=0, help="Minimum validation-loss improvement to reset patience.",
    ),
    limit: int | None = typer.Option(None, help="Cap evaluation sentences, for a smoke run."),
) -> None:
    """MT probe: fine-tune on MAFAND or local JSONL, then score FLORES+ and MAFAND test.

    Every system gets the same training cap, early-stopping policy and decoding
    budget through one shared ProbeConfig. Validation loss stops training;
    scoring reloads the model saved at that stopping step or the cap.
    Akuapem and Asante are reported side by side and never averaged.

    chrF++ is primary and sacreBLEU secondary. This is the historical
    stop-step protocol; the frozen best-dev protocol is ``eval mt-freeze``,
    ``eval mt-select`` and ``eval mt-test``.
    """
    from akanlm.audit.sources import load_sources
    from akanlm.corpora.loaders import load_pairs
    from akanlm.eval.baselines import comparison_checkpoints
    from akanlm.eval.mt_probe import (
        ProbeConfig, build_pairs, finetune, generate,
        load_flores_pairs, load_local_jsonl_pairs, orient_evaluation_texts,
        render_table, report_path, score, system_dir, write_report,
    )

    systems = comparison_checkpoints(checkpoints) if baselines else checkpoints
    typer.echo("MT systems: " + ", ".join(systems))
    sources = load_sources(corpora_dir)
    if flairs_six_epoch_setup:
        if (
            max_steps != 3000 or eval_steps != 100
            or early_stopping_patience != 5 or early_stopping_threshold != 0.001
            or max_length != 128 or num_beams != 4 or batch_size != 16
        ):
            raise typer.BadParameter(
                "The FLAIRS six-epoch setup fixes the training and decoding "
                "settings; remove conflicting budget and decoding flags."
            )
        config = ProbeConfig.flairs_six_epoch_setup(direction=direction)
    else:
        config = ProbeConfig(
            direction=direction, max_steps=max_steps, eval_steps=eval_steps,
            generation_max_length=max_length, num_beams=num_beams,
            generation_batch_size=batch_size,
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
        )
    ensure_dir(output_dir)

    mafand = sources["mafand_en_twi"]
    training_data = None
    if train_data_dir is None:
        train_rows = build_pairs(load_pairs(mafand.model_copy(update={"hf_split": "train"})), config)
        eval_rows = build_pairs(
            load_pairs(mafand.model_copy(update={"hf_split": "validation"})), config
        )
    else:
        import hashlib

        train_path = train_data_dir / "train.json"
        dev_path = train_data_dir / "dev.json"
        train_rows = load_local_jsonl_pairs(train_path, config)
        eval_rows = load_local_jsonl_pairs(dev_path, config)
        # The provided split repeats some identical pairs across train and dev.
        # Keep dev unseen during weight updates so it can govern early stopping.
        dev_pairs = {(row["source"], row["target"]) for row in eval_rows}
        before_exclusion = len(train_rows)
        train_rows = [
            row for row in train_rows
            if (row["source"], row["target"]) not in dev_pairs
        ]
        if not train_rows:
            raise typer.BadParameter("all local training pairs overlap development")
        training_data = {
            "source": str(train_data_dir.resolve()),
            "translation_columns": {"en": "English", "sw": "Twi in source files"},
            "train": {
                "path": str(train_path.resolve()),
                "sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
                "raw_rows": before_exclusion,
                "used_rows": len(train_rows),
                "excluded_dev_duplicates": before_exclusion - len(train_rows),
            },
            "dev": {
                "path": str(dev_path.resolve()),
                "sha256": hashlib.sha256(dev_path.read_bytes()).hexdigest(),
                "rows": len(eval_rows),
            },
        }
        typer.echo(
            f"MT training: {len(train_rows):,} local pairs; "
            f"development: {len(eval_rows):,}; "
            f"removed {before_exclusion - len(train_rows)} train/dev duplicates"
        )
    test_pairs = list(load_pairs(mafand, limit=limit))

    # (label, variety, english, twi)
    eval_sets = [
        ("flores_akuapem", "akuapem", *load_flores_pairs(sources, "akuapem", limit)),
        ("flores_asante", "asante", *load_flores_pairs(sources, "asante", limit)),
        (
            "mafand_test", None,
            [p.src_text for p in test_pairs], [p.tgt_text for p in test_pairs],
        ),
    ]

    results = []
    training_runs = []
    for checkpoint in systems:
        # Not Path(checkpoint).name: every run's leaf is "final", so the
        # probe's per-system directories and report rows would collide
        # across mixtures.
        system = run_label([checkpoint])
        fine_tune_dir = system_dir(output_dir, system, direction)
        model, tokenizer = finetune(
            checkpoint, train_rows, eval_rows, config, fine_tune_dir, resume=resume,
        )
        training_runs.append(json.loads(
            (fine_tune_dir / "training_summary.json").read_text(encoding="utf-8")
        ))
        for label, variety, english, twi in eval_sets:
            sources, references = orient_evaluation_texts(english, twi, direction)
            # score() canonicalizes target text and retains raw predictions.
            hypotheses = generate(model, tokenizer, sources, config, normalize_outputs=False)
            results.append(score(hypotheses, references, system, label, direction, variety))

    typer.echo("\n" + render_table(results))
    report = write_report(
        results, config, report_path(output_dir, direction),
        training_data=training_data, training_runs=training_runs,
    )
    typer.echo(f"\nwrote {report}")


@eval_app.command("mt-freeze")
def eval_mt_freeze(
    checkpoints: list[str] = typer.Argument(..., help="Probed model paths or hub ids."),
    study: Path = typer.Option(..., help="Study directory; one per language and protocol."),
    task: str = typer.Option("twi", help="'twi' (MAFAND train/validation, FLORES+ and MAFAND test) or 'local'."),
    language: str | None = typer.Option(None, help="Language code for --task local (kpe, vai, twi)."),
    data_dir: Path | None = typer.Option(
        None, help="--task local: train.jsonl, dev.jsonl and test_<name>.jsonl."
    ),
    corpora_dir: Path = typer.Option(CORPORA_DIR),
    baselines: bool = typer.Option(True, help="Add the recorded CPT init, matching mT5 and Afri-mT5."),
    limit: int | None = typer.Option(None, help="Cap every split; marks the study as a smoke study."),
    dropout_rate: float | None = typer.Option(
        None, help="Fine-tuning dropout for every system (0 <= rate < 1); frozen in the study."
    ),
) -> None:
    """Freeze the best-dev MT protocol for a study and register its systems.

    Runs the preflight on training and development text only: deduplication
    against development and test text, the 256/512 length rule, tokenizer
    coverage and the normalization guard. Rerunning with more checkpoints
    registers them under the existing protocol.
    """
    from akanlm.eval.baselines import comparison_checkpoints
    from akanlm.eval.mt_protocol import freeze, local_task_spec, twi_task_spec

    if task == "twi":
        spec = twi_task_spec(corpora_dir, limit)
    elif task == "local":
        if language is None or data_dir is None:
            raise typer.BadParameter("--task local requires --language and --data-dir")
        spec = local_task_spec(language, data_dir, limit)
    else:
        raise typer.BadParameter(f"unknown task {task}")
    systems = comparison_checkpoints(checkpoints) if baselines else checkpoints
    protocol = freeze(study, spec, systems, dropout_rate=dropout_rate)
    typer.echo(
        f"{study}: {protocol['procedure']['version']}, max length "
        f"{protocol['lengths']['resolved']}, {len(systems)} systems registered"
    )
    for system in systems:
        typer.echo(f"  {system}")


@eval_app.command("mt-select")
def eval_mt_select(
    system: str = typer.Argument(..., help="A registered model path or hub id."),
    study: Path = typer.Option(...),
    direction: str = typer.Option(..., help="en2twi, twi2en, en2kpe, ..."),
    keep_weights: bool = typer.Option(False, help="Keep weights of rates not selected."),
) -> None:
    """Choose the learning rate on development chrF++; never reads test data."""
    from akanlm.eval.mt_protocol import select

    document = select(study, system, direction, keep_weights=keep_weights)
    for candidate in document["candidates"]:
        typer.echo(
            f"lr {candidate['learning_rate']:.0e}: dev chrF++ {candidate['dev_chrf2pp']:.2f}, "
            f"restored step {candidate['restored_step']} of {candidate['completed_steps']}"
        )
    typer.echo(f"selected {document['selected']['learning_rate']:.0e}; "
               f"flags: {', '.join(document['flags']) or 'none'}")


@eval_app.command("mt-test")
def eval_mt_test(
    system: str = typer.Argument(..., help="A registered model path or hub id."),
    study: Path = typer.Option(...),
    direction: str = typer.Option(...),
    keep_weights: bool = typer.Option(False, help="Keep fine-tuned weights after decoding."),
) -> None:
    """Seeded test runs at the selected rate; writes the receipt first."""
    from akanlm.eval.mt_protocol import test

    document = test(study, system, direction, keep_weights=keep_weights)
    for name, values in document["scores"].items():
        typer.echo(f"{name}: chrF++ {values['chrf2pp_mean']:.2f} ± {values['chrf2pp_sd']:.2f}, "
                   f"BLEU {values['bleu_mean']:.2f} ± {values['bleu_sd']:.2f}")


@eval_app.command("mt-report")
def eval_mt_report(study: Path = typer.Option(...)) -> None:
    """Seed means and paired bootstrap comparisons from saved predictions."""
    from akanlm.eval.mt_protocol import render_report, report

    typer.echo(render_report(report(study)))
    typer.echo(f"\nwrote {study / 'report.json'}")


if __name__ == "__main__":  # pragma: no cover
    app()
