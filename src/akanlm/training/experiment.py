"""Standard CPT experiment: document splits, two vocabularies, reference eval.

Every system is scored on the same held-out mixture documents under two
metrics. Google's token-span perplexity ranks systems only within one
tokenizer. Character-span conditional BPC ranks systems across tokenizers,
which is what AkanT5 against Google mT5 requires. The starting systems are
scored before either arm trains (evaluation/pre_cpt.json); the final report
reuses those rows after re-verifying weights, split and implementation.
"""

import gc
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path

from akanlm.data.cpt_splits import DEFAULT_SPLIT_CONFIG, prepare, read_manifest
from akanlm.eval import corpus_bpc
from akanlm.eval.baselines import AFRICAN_REFERENCE, MT5_BACKBONES
from akanlm.eval.denoising import DenoisingConfig, compare
from akanlm.training.cpt import TrainingConfig, train
from akanlm.utils.paths import digest, file_sha256, write_json

PRE_CPT_REPORT = Path("evaluation/pre_cpt.json")
# Afri-mT5 exists only at base; its tokenizer is Google mT5-base's.
AFRICAN_BACKBONE = "google/mt5-base"
# Adjacent roles share a tokenizer, so frozen token-span tasks are reused.
SYSTEM_ORDER = ("akan_initialization", "akan_cpt", "google_mt5", "original_tokenizer_cpt", "afri_mt5")
PRE_CPT_ROLES = ("akan_initialization", "google_mt5", "afri_mt5")
# Same tokenizer: native perplexity is valid, and compare() refuses otherwise.
NATIVE_COMPARISONS = (
    ("akan_cpt", "akan_initialization"),
    ("original_tokenizer_cpt", "google_mt5"),
    ("afri_mt5", "google_mt5"),
    ("original_tokenizer_cpt", "afri_mt5"),
)
# Identical masked characters: valid across tokenizers.
CHARACTER_COMPARISONS = (
    ("akan_initialization", "google_mt5"),
    ("akan_initialization", "afri_mt5"),
    ("afri_mt5", "google_mt5"),
    ("akan_cpt", "google_mt5"),
    ("akan_cpt", "akan_initialization"),
    ("akan_cpt", "original_tokenizer_cpt"),
    ("original_tokenizer_cpt", "google_mt5"),
    ("akan_cpt", "afri_mt5"),
    ("original_tokenizer_cpt", "afri_mt5"),
)


def read_plan(root):
    plan = json.loads((Path(root) / "experiment_plan.json").read_text())
    expected = plan.pop("sha256")
    if digest(plan) != expected:
        raise ValueError("Experiment plan hash mismatch")
    plan["sha256"] = expected
    if "pre_cpt_systems" not in plan or "char_span_evaluation" not in plan:
        raise ValueError("Experiment plan predates pre-CPT evaluation; start a fresh experiment")
    allowed = {"name", "base_checkpoint", "expected_tokenizer"}
    if any(value != plan["control_config"][key] for key, value in plan["akan_config"].items()
           if key not in allowed):
        raise ValueError("CPT arms differ in training settings")
    split = read_manifest(plan["split_manifest"])
    if split["sha256"] != plan["split_sha256"]:
        raise ValueError("Experiment corpus split changed")
    return plan


def validate_completed_arm(destination, spec, split_sha256):
    card = json.loads((destination / "run_card.json").read_text())
    if (card["config"] != asdict(spec) or card["validation"].get("split_manifest_sha256") != split_sha256
            or (spec.early_stopping_patience == 0 and card["metrics"]["completed_steps"] != spec.max_steps)
            or not (destination / "final/model.safetensors").exists()):
        raise ValueError(f"Completed CPT arm does not match the experiment: {destination}")
    training_plan = json.loads((destination / "training_plan.json").read_text())
    weights = training_plan["initial_weights"]
    if not weights or any(file_sha256(Path(spec.base_checkpoint) / name) != checksum
                          for name, checksum in weights.items()):
        raise ValueError(f"CPT initialization changed since training: {spec.base_checkpoint}")


def backbone(checkpoint):
    config = json.loads((Path(checkpoint) / "config.json").read_text())
    shape = tuple(config.get(k) for k in ("d_model", "d_ff", "num_layers", "num_heads"))
    if config.get("model_type") != "mt5" or shape not in MT5_BACKBONES:
        raise ValueError("Cannot identify a supported mT5 backbone")
    return MT5_BACKBONES[shape]


def original_initialization(model_id, destination, *, sentencepiece_model=None):
    """Save Hub weights and their tokenizer, optionally restoring native byte fallback.

    Supply the pinned native SentencePiece model when a fast-tokenizer
    conversion dropped its byte-fallback behavior. Existing copies are never
    silently rewritten; use a new destination for a corrected initialization.
    """
    from transformers import AutoModelForSeq2SeqLM

    from akanlm.eval.denoising import tokenizer_fingerprint
    from akanlm.tokenizer.surgery import (
        assert_embeddings_shared,
        load_tokenizer,
        normalize_tying,
    )

    destination = Path(destination)
    compatibility = None
    if sentencepiece_model is not None:
        compatibility = {"kind": "native_sentencepiece_byte_fallback",
                         "sentencepiece_sha256": file_sha256(Path(sentencepiece_model))}
    receipt = destination / "original_initialization.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["model"] != model_id or any(
                file_sha256(destination / name) != checksum for name, checksum in saved["files"].items()):
            raise ValueError("Original-tokenizer initialization has changed")
        if compatibility is not None and saved.get("tokenizer_compatibility") != compatibility:
            raise ValueError("Original initialization lacks the requested tokenizer correction; use a new path")
        return str(destination)
    if destination.exists():
        raise FileExistsError(f"Incomplete or unverified original initialization: {destination}")
    tokenizer = load_tokenizer(model_id)
    if sentencepiece_model is not None:
        from transformers import PreTrainedTokenizerFast

        from akanlm.tokenizer.compat import restore_sentencepiece_byte_fallback
        restore_sentencepiece_byte_fallback(tokenizer, sentencepiece_model)
        # T5Tokenizer in Transformers 5.16.1 rebuilds the backend with byte
        # fallback disabled even when tokenizer.json enables it. The generic
        # fast class preserves this verified backend when the copy is reloaded.
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer.backend_tokenizer,
            **tokenizer.special_tokens_map,
            model_max_length=tokenizer.model_max_length,
            model_input_names=tokenizer.model_input_names,
            padding_side=tokenizer.padding_side,
            truncation_side=tokenizer.truncation_side,
            clean_up_tokenization_spaces=tokenizer.clean_up_tokenization_spaces,
        )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    if normalize_tying(model):
        raise ValueError("Original mT5 output head must remain untied")
    assert_embeddings_shared(model)
    destination.mkdir(parents=True)
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    saved = {"model": model_id, "hub_revision": getattr(model.config, "_commit_hash", None),
             "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
             "files": {p.name: file_sha256(p) for p in destination.iterdir()
                       if p.is_file()}}
    if compatibility is not None:
        saved["tokenizer_compatibility"] = compatibility
    receipt.write_text(json.dumps(saved, indent=2))
    return str(destination)


def local_copy(initializer_root, model_id):
    return Path(initializer_root) / (model_id.replace("/", "-") + "-original-v1")


def initial_systems(akan_checkpoint, reference, initializer_root):
    """Starting points scored before CPT: AkanT5 init, Google mT5, and Afri-mT5 at base only."""
    systems = {"akan_initialization": str(akan_checkpoint),
               "google_mt5": original_initialization(reference, local_copy(initializer_root, reference))}
    if reference == AFRICAN_BACKBONE:
        systems["afri_mt5"] = original_initialization(
            AFRICAN_REFERENCE, local_copy(initializer_root, AFRICAN_REFERENCE))
    return systems


def verify_reference_copies(systems, reference):
    original_initialization(reference, systems["google_mt5"])
    if reference == AFRICAN_BACKBONE:
        original_initialization(AFRICAN_REFERENCE, systems["afri_mt5"])
    elif "afri_mt5" in systems:
        raise ValueError("Afri-mT5 is a base-size reference only")


def score_systems(systems, split_manifest, *, token_config, char_config, batch_size=4, char_batch_size=16):
    """Load each system once and score both metrics on validation and test."""
    from akanlm.data.cpt_splits import evaluation_corpora
    from akanlm.eval.denoising import (
        load_checkpoint,
        release_memory,
        score_checkpoint,
        tokenizer_fingerprint,
    )

    unknown = set(systems) - set(SYSTEM_ORDER)
    if unknown:
        raise ValueError(f"Unknown evaluation roles: {sorted(unknown)}")
    manifest = read_manifest(split_manifest)
    corpora = {**evaluation_corpora(manifest, "validation"), **evaluation_corpora(manifest, "test")}
    token_rows, char_rows, tasks = [], [], {}
    for role in (r for r in SYSTEM_ORDER if r in systems):
        checkpoint = systems[role]
        tokenizer, model = load_checkpoint(checkpoint)
        fingerprint = tokenizer_fingerprint(tokenizer)
        # Cached token-span tasks hold ~10K padded rows per corpus; keep only one tokenizer's.
        for key in [k for k in tasks if k[0] != fingerprint]:
            del tasks[key]
        for row in score_checkpoint(model, tokenizer, checkpoint, corpora, manifest["sha256"],
                                    config=token_config, batch_size=batch_size, tasks=tasks):
            token_rows.append({"role": role, **row})
        identity = token_rows[-1]["weights"]
        for row in corpus_bpc.score_corpora(model, tokenizer, corpora, checkpoint,
                                            config=char_config, batch_size=char_batch_size):
            char_rows.append({"role": role, "model": checkpoint, "weights": identity,
                              "tokenizer_sha256": fingerprint, "split_manifest_sha256": manifest["sha256"],
                              "implementation_sha256": corpus_bpc.implementation_sha256(), **row})
        del model
        release_memory()
    # Stored form, so fresh rows compare equal to rows reloaded from pre_cpt.json.
    return as_json(token_rows), as_json(char_rows)


def build_comparisons(token_rows, char_rows, char_config):
    """Declared pairs only; rows absent at this stage are skipped, mismatches raise."""
    native = {(r["role"], r["dataset"], r["split"]): r for r in token_rows}
    character = {(r["role"], r["dataset"], r["split"]): r for r in char_rows}
    comparisons = []
    for metric, rows, pairs in (("native_perplexity", native, NATIVE_COMPARISONS),
                                ("character_bpc", character, CHARACTER_COMPARISONS)):
        for candidate, baseline in pairs:
            for (role, dataset, split), row in rows.items():
                other = rows.get((baseline, dataset, split))
                if role != candidate or other is None:
                    continue
                result = (compare(row, other) if metric == "native_perplexity" else corpus_bpc.compare(
                    row, other, samples=char_config.bootstrap_samples, seed=char_config.bootstrap_seed))
                comparisons.append({"metric": metric, "role": f"{candidate}_vs_{baseline}",
                                    "dataset": dataset, **result})
    return comparisons


def as_json(value):
    """Compare frozen settings as stored: JSON turns tuples into lists."""
    return json.loads(json.dumps(value))


def pre_cpt_identity(systems, split_manifest, token_config, char_config):
    from akanlm.eval.denoising import PROTOCOL
    return as_json({"systems": dict(systems), "split_sha256": read_manifest(split_manifest)["sha256"],
                    "token_span": {"protocol": PROTOCOL, "config": asdict(token_config)},
                    "char_span": {"protocol": corpus_bpc.PROTOCOL, "config": asdict(char_config)}})


def evaluate_pre_cpt(systems, split_manifest, destination, *, token_config, char_config, batch_size=4):
    """Score the starting systems before any CPT update touches them."""
    if set(systems) - set(PRE_CPT_ROLES):
        raise ValueError("Pre-CPT evaluation scores only starting checkpoints")
    token_rows, char_rows = score_systems(systems, split_manifest, token_config=token_config,
                                          char_config=char_config, batch_size=batch_size)
    report = {"stage": "pre_cpt", **pre_cpt_identity(systems, split_manifest, token_config, char_config),
              "split_manifest": str(split_manifest), "primary_split": "test",
              "comparison_rule": ("native_perplexity requires one tokenizer; character_bpc requires "
                                  "identical masked text and is the only cross-tokenizer ranking."),
              "scope": "Held-out documents from the declared mixture sources, not FLORES+ or mC4.",
              "results": token_rows, "character_results": char_rows,
              "comparisons": build_comparisons(token_rows, char_rows, char_config)}
    return write_json(destination, report)


def reusable_pre_cpt(path, systems, split_manifest, *, token_config, char_config):
    """Reuse pre-CPT rows only if systems, split, settings, weights and code still match."""
    from akanlm.eval.denoising import implementation_sha256, weights_identity

    report = json.loads(Path(path).read_text())
    expected = pre_cpt_identity(systems, split_manifest, token_config, char_config)
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"Pre-CPT report {path} does not match this experiment ({key}); "
                             "move it aside to re-evaluate")
    identities = {}
    for row in report["results"] + report["character_results"]:
        if row["model"] not in identities:
            identities[row["model"]] = weights_identity(row["model"])
        if row["weights"] != identities[row["model"]]:
            raise ValueError(f"Weights changed since pre-CPT evaluation: {row['model']}")
    code = implementation_sha256()
    if any(row["scoring"]["implementation_sha256"] != code for row in report["results"]):
        raise ValueError("Token-span evaluation code changed since pre-CPT evaluation; move the report aside")
    code = corpus_bpc.implementation_sha256()
    if any(row["implementation_sha256"] != code for row in report["character_results"]):
        raise ValueError("Character-span evaluation code changed since pre-CPT evaluation; move the report aside")
    return report


def pre_cpt(output, systems, split_manifest, *, token_config, char_config, batch_size=4):
    destination = Path(output) / PRE_CPT_REPORT
    kwargs = {"token_config": token_config, "char_config": char_config}
    if destination.exists():
        return reusable_pre_cpt(destination, systems, split_manifest, **kwargs)
    return evaluate_pre_cpt(systems, split_manifest, destination, batch_size=batch_size, **kwargs)


def evaluate_experiment(root, batch_size=4):
    root = Path(root)
    plan = read_plan(root)
    for path, spec in ((root, plan["akan_config"]), (root / "original_tokenizer_control", plan["control_config"])):
        if not (path / "run_card.json").exists():
            raise ValueError(f"Finish both CPT arms before primary evaluation: {path}")
        validate_completed_arm(path, TrainingConfig(**spec), plan["split_sha256"])
    initial = plan["pre_cpt_systems"]
    verify_reference_copies(initial, plan["reference_model"])
    token_config = DenoisingConfig(**plan["evaluation"])
    char_config = corpus_bpc.CharSpanConfig(**plan["char_span_evaluation"])
    kwargs = {"token_config": token_config, "char_config": char_config}
    before = pre_cpt(root, initial, plan["split_manifest"], batch_size=batch_size, **kwargs)
    trained = {"akan_cpt": str(root / "final"),
               "original_tokenizer_cpt": str(root / "original_tokenizer_control/final")}
    token_rows, char_rows = score_systems(trained, plan["split_manifest"], batch_size=batch_size, **kwargs)
    token_rows = before["results"] + token_rows
    char_rows = before["character_results"] + char_rows
    report = {"protocol": "mt5_token_span_cpt_v1", "character_protocol": corpus_bpc.PROTOCOL,
              "primary_split": "test", "reference_model": plan["reference_model"],
              "systems": {**initial, **trained}, "pre_cpt_report": str(root / PRE_CPT_REPORT),
              "scope": "Held-out documents from the declared mixture sources, not original mC4 or FLORES+.",
              "implementation": "NumPy corruption sampling; Mesh metric definitions; PyTorch scoring.",
              "tokenizer_exposure": "Existing Akan tokenizer predates the split; holdout applies to CPT weight updates.",
              "experiment_plan_sha256": plan["sha256"],
              "comparison_rule": ("native_perplexity requires identical tokenizer, corpus, masks and settings; "
                                  "character_bpc requires identical masked text and ranks across tokenizers."),
              "results": token_rows, "character_results": char_rows,
              "comparisons": build_comparisons(token_rows, char_rows, char_config)}
    return write_json(root / "evaluation/cpt_reference.json", report)


def run(config, output, *, corpora_dir=Path("configs/corpora"),
        mixtures_dir=Path("configs/mixtures"), manifest_sha256=None, resume=False,
        split_config=DEFAULT_SPLIT_CONFIG, initializer_root=Path("runs/init"), split_manifest=None,
        char_span_config=corpus_bpc.DEFAULT_CHAR_SPAN_CONFIG, pre_cpt_only=False):
    """Score starting systems, then train both arms at equal update and encoder-token budgets.

    Configs and split identity are frozen before either arm starts. Completed
    arms are reused only when their config and split match; an interrupted arm
    resumes solely on explicit --resume. Old single-arm runs are not upgraded.
    ``pre_cpt_only`` scores the starting systems without freezing a training plan,
    so the step budget can still change before training.
    """
    from akanlm.data.mixture import MixtureConfig

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "experiment_plan.json"
    reference = backbone(config.base_checkpoint)
    mixture = MixtureConfig.from_yaml(mixtures_dir / f"{config.mixture}.yaml")
    evaluation = DenoisingConfig.from_training(config)
    if plan_path.exists():
        plan = read_plan(output)
        if (plan["akan_config"] != asdict(config) or plan["manifest_sha256"] != manifest_sha256
                or plan["char_span_evaluation"] != as_json(asdict(char_span_config))):
            raise ValueError("Experiment settings changed; use a fresh output")
        split_path = Path(plan["split_manifest"])
        if split_manifest is not None and Path(split_manifest).resolve() != split_path.resolve():
            raise ValueError("Experiment split manifest path changed")
        split = read_manifest(split_path)
        frozen_mixture = {"replay": None, **split["mixture"]}
        if split["sha256"] != plan["split_sha256"] or frozen_mixture != asdict(mixture):
            raise ValueError("Experiment corpus split changed")
        if split["config"] != asdict(split_config):
            raise ValueError("Experiment split settings changed")
        control = TrainingConfig(**plan["control_config"])
        systems = plan["pre_cpt_systems"]
        verify_reference_copies(systems, reference)
        evaluation = DenoisingConfig(**plan["evaluation"])
    else:
        if (output / "final").exists() or list(output.glob("checkpoint-*")):
            raise ValueError("Output already holds checkpoints without an experiment plan; use a fresh output")
        split_path = Path(split_manifest) if split_manifest else output / "cpt_split.json"
        split = read_manifest(split_path) if split_path.exists() else prepare(
            mixture, corpora_dir, split_path, split_config)
        frozen_mixture = {"replay": None, **split["mixture"]}
        if frozen_mixture != asdict(mixture) or split["config"] != asdict(split_config):
            raise ValueError("Prepared split does not match the requested mixture and split settings")
        systems = initial_systems(config.base_checkpoint, reference, initializer_root)
        control = replace(config, name=reference.replace("/", "-") + "-cpt-control",
                          base_checkpoint=systems["google_mt5"], expected_tokenizer=systems["google_mt5"])
        if not pre_cpt_only:
            plan = {"akan_config": asdict(config), "control_config": asdict(control),
                    "split_manifest": str(split_path), "split_sha256": split["sha256"],
                    "reference_model": reference, "manifest_sha256": manifest_sha256,
                    "evaluation": asdict(evaluation), "char_span_evaluation": as_json(asdict(char_span_config)),
                    "pre_cpt_systems": systems,
                    "encoder_tokens_per_arm": config.max_steps * config.effective_batch_size * config.input_length,
                    "note": "Vocabulary sizes differ; token budgets match, text exposure and compute need not."}
            plan["sha256"] = digest(plan)
            plan_path.write_text(json.dumps(plan, indent=2))
    # Before any update: this is the reference point for every CPT gain.
    before = pre_cpt(output, systems, split_path, token_config=evaluation, char_config=char_span_config)
    if pre_cpt_only:
        return before
    for spec, destination in ((config, output), (control, output / "original_tokenizer_control")):
        card_path = destination / "run_card.json"
        if card_path.exists():
            validate_completed_arm(destination, spec, split["sha256"])
            continue
        checkpoints = list(destination.glob("checkpoint-*"))
        if checkpoints and not resume:
            raise ValueError(f"Interrupted CPT arm requires --resume: {destination}")
        train(spec, destination, corpora_dir=corpora_dir, mixtures_dir=mixtures_dir,
              manifest_sha256=manifest_sha256, split_manifest=split_path,
              resume_from_checkpoint=True if checkpoints else None)
    return evaluate_experiment(output)


# Single-vocabulary full runs share the same training and evaluation primitives.

def score_validation_checkpoint(checkpoint: Path, destination: Path, manifest, char_config, batch_size: int) -> list[dict]:
    """Character BPC on the validation split, cached; a stale cache raises."""
    from akanlm.data.cpt_splits import evaluation_corpora
    from akanlm.eval import corpus_bpc
    from akanlm.eval.denoising import (
        load_checkpoint,
        release_memory,
        tokenizer_fingerprint,
        weights_identity,
    )

    identity = {"model": str(checkpoint), "weights": weights_identity(checkpoint),
                "split_manifest_sha256": manifest["sha256"], "config": as_json(asdict(char_config)),
                "implementation_sha256": corpus_bpc.implementation_sha256()}
    if destination.exists():
        cached = json.loads(destination.read_text())
        if cached["identity"] != identity:
            raise ValueError(f"{destination} was scored with different weights, split or settings")
        return cached["rows"]

    tokenizer, model = load_checkpoint(checkpoint)
    rows = corpus_bpc.score_corpora(model, tokenizer, evaluation_corpora(manifest, "validation"),
                                    str(checkpoint), config=char_config, batch_size=batch_size)
    rows = as_json([{"model": str(checkpoint), "weights": identity["weights"],
                     "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
                     "split_manifest_sha256": manifest["sha256"], **row} for row in rows])
    del model
    release_memory()
    write_json(destination, {"identity": identity, "rows": rows})
    return rows


def single_implementation_identity():
    root = Path(__file__).parents[1]
    names = ("training/cpt.py", "training/validation.py", "training/optimization.py",
             "training/experiment.py", "training/checkpoints.py", "training/coverage.py", "training/distributed.py",
             "data/cpt_splits.py", "data/packing.py", "data/mixture.py", "data/collator.py",
             "eval/denoising.py", "eval/corpus_bpc.py", "eval/lm.py", "eval/uncertainty.py",
             "corpora/loaders.py", "corpora/extract.py", "corpora/normalize.py",
             "utils/parallel.py", "tokenizer/surgery.py", "tokenizer/compat.py", "audit/sources.py")
    return {name: file_sha256(root / name) for name in names}


def check_single_plan(output):
    from akanlm.training.checkpoints import runtime_framework

    plan = json.loads((output / "full_run_plan.json").read_text())
    if plan.pop("sha256") != digest(plan):
        raise ValueError("Full-run plan content hash mismatch")
    if plan["framework"] != runtime_framework():
        raise ValueError("Full-run framework changed")
    if plan["implementation_sha256"] != single_implementation_identity():
        raise ValueError("Full-run implementation changed; review before continuing")
    for name, checksum in plan["frozen_files"].items():
        if file_sha256(output / name) != checksum:
            raise ValueError(f"Frozen input changed: {name}")
    config = TrainingConfig(**plan["config"])
    if asdict(config) != json.loads((output / "inputs/config.json").read_text()):
        raise ValueError("Full-run config differs from frozen model and accelerator settings")
    for name, checksum in plan["initialization_files"].items():
        if file_sha256(Path(config.base_checkpoint) / name) != checksum:
            raise ValueError(f"Fresh initialization changed: {name}")
    return plan, config


def setup_single(config, output, *, corpora_dir=Path("configs/corpora"),
                 mixtures_dir=Path("configs/mixtures"), manifest_sha256=None,
                 split_manifest=None, split_config=DEFAULT_SPLIT_CONFIG,
                 char_span_config=corpus_bpc.DEFAULT_CHAR_SPAN_CONFIG):
    """Freeze reusable single-arm CPT inputs before CPU preparation or training."""
    from akanlm.data.mixture import MixtureConfig, validate
    from akanlm.training.checkpoints import runtime_framework
    from akanlm.training.cpt import check_expected_tokenizer

    output = Path(output)
    if (output / "full_run_plan.json").exists():
        plan, frozen = check_single_plan(output)
        if (asdict(config) != asdict(frozen) or plan["manifest_sha256"] != manifest_sha256
                or plan["split_config"] != asdict(split_config)
                or plan["character_evaluation"] != as_json(asdict(char_span_config))):
            raise ValueError("Full-run settings changed; use a fresh output")
        if split_manifest is not None and str(Path(split_manifest).resolve()) != plan.get("requested_split_manifest"):
            raise ValueError("Full-run split path changed")
        return plan
    # The launcher may have written its receipt before the CPU allocation starts.
    if output.exists() and any(p.name != "slurm_jobs.json" for p in output.iterdir()):
        raise FileExistsError(f"Unverified full-run output already exists: {output}")
    mixture = MixtureConfig.from_yaml(mixtures_dir / f"{config.mixture}.yaml")
    validate(mixture, corpora_dir)
    check_expected_tokenizer(config)
    initial = Path(config.base_checkpoint)
    if not list(initial.glob("*.safetensors")):
        raise ValueError("Single-arm CPT requires a local verified initialization")
    output.mkdir(parents=True, exist_ok=True)
    inputs = output / "inputs"
    shutil.copytree(corpora_dir, inputs / "corpora")
    (inputs / "mixtures").mkdir()
    shutil.copy2(mixtures_dir / f"{config.mixture}.yaml", inputs / "mixtures" / f"{config.mixture}.yaml")
    write_json(inputs / "config.json", asdict(config))
    audit = Path("runs/pilot/early-validation-small-m1-m1x/prerequisites/m1x_audit.json")
    if config.mixture == "m1x" and audit.exists():
        shutil.copy2(audit, inputs / "m1x_audit.json")
    split_path = output / "cpt_split.json"
    if split_manifest is not None:
        # Copy the complete caller-selected split rather than binding a mutable external file.
        candidate = read_manifest(split_manifest)
        if candidate["config"] != asdict(split_config):
            raise ValueError("Requested split fractions or seed differ from the supplied manifest")
        from akanlm.training.coverage import validate_full_source_coverage
        validate_full_source_coverage(mixture, candidate)
        del candidate
        shutil.copy2(split_manifest, split_path)
    plan = {"config": asdict(config), "framework": runtime_framework(),
            "manifest_sha256": manifest_sha256,
            "initialization_files": {p.name: file_sha256(p) for p in initial.iterdir() if p.is_file()},
            "frozen_files": {str(p.relative_to(output)): file_sha256(p) for p in inputs.rglob("*") if p.is_file()},
            "implementation_sha256": single_implementation_identity(),
            "character_evaluation": as_json(asdict(char_span_config)),
            "split_config": asdict(split_config),
            "requested_split_manifest": str(Path(split_manifest).resolve()) if split_manifest else None,
            "split_manifest": str(split_path), "sources": [c.source for c in mixture.components],
            "selection": "Per-source native loss and target-token-weighted eval_cpt_loss on validation. "
                         "Early stopping restores the best checkpoint and its patience across resumes.",
            "scope": "One selected vocabulary arm, held-out validation from every mixture source. "
                     "Existing tokenizer exposure is not a weight-training holdout. Test split is not used for selection."}
    plan["sha256"] = digest(plan)
    write_json(output / "full_run_plan.json", plan)
    return plan


def prepare_single(output, workers=4):
    """Create the full-mixture split and packed cache on CPU; fail on missing coverage."""
    from transformers import AutoTokenizer

    from akanlm.data.cpt_splits import SplitConfig, build_validation, read_manifest
    from akanlm.data.cpt_splits import prepare as prepare_split
    from akanlm.data.mixture import MixtureConfig
    from akanlm.training.coverage import (
        validate_character_tokenization,
        validate_full_source_coverage,
    )
    from akanlm.training.cpt import (
        PACKING_CACHE_ROOT,
        build_dataset,
        training_dataset_key,
    )

    plan, config = check_single_plan(output)
    corpora, mixtures = output / "inputs/corpora", output / "inputs/mixtures"
    mixture = MixtureConfig.from_yaml(mixtures / f"{config.mixture}.yaml")
    split_path = Path(plan["split_manifest"])
    manifest = read_manifest(split_path) if split_path.exists() else prepare_split(
        mixture, corpora, split_path, SplitConfig(**plan["split_config"]))
    coverage = validate_full_source_coverage(mixture, manifest)
    write_json(output / "preparation/split_coverage.json", coverage)
    del manifest
    gc.collect()
    tokenizer = AutoTokenizer.from_pretrained(config.base_checkpoint, local_files_only=True)
    datasets, protected, validation = build_validation(config, tokenizer, corpora, split_path)
    manifest = read_manifest(split_path)
    coverage = validate_full_source_coverage(mixture, manifest, validation)
    encoding = validate_character_tokenization(manifest, tokenizer, corpus_bpc.CharSpanConfig(**plan["character_evaluation"]))
    write_json(output / "preparation/character_tokenization.json", encoding)
    del manifest, datasets
    gc.collect()
    write_json(output / "preparation/validation_coverage.json", coverage)
    for label, source in validation["sources"].items():
        print(f"[VALIDATION] {label}: {source['n_documents']} documents, {source['n_examples']} fixed tasks")
    key = training_dataset_key(config, tokenizer, validation, plan["manifest_sha256"], corpora, mixtures)
    dataset, summary, _ = build_dataset(config, tokenizer, corpora, mixtures, protected=protected,
                                      cache_dir=PACKING_CACHE_ROOT / key, fingerprint=key, workers=workers)
    packed = summary["preparation"]
    seen = packed["exclusion"]["records_seen"]
    removed = packed["exclusion"]["records_removed"]
    duplicates = packed["exact_duplicates"]["removed_by_source_and_owner"]
    retained = {name: seen.get(name, 0) - removed.get(name, 0) - sum(duplicates.get(name, {}).values())
                for name in plan["sources"]}
    if any(count <= 0 for count in retained.values()):
        raise ValueError(f"A mixture source has no retained training documents: {retained}")
    coverage["retained_training_documents"] = retained
    write_json(output / "preparation/ready.json", {
        "dataset_key": key, "chunks": len(dataset), "split_sha256": validation["split_manifest_sha256"],
        "validation": validation, "coverage": coverage, "preparation": summary["preparation"],
        "full_run_plan_sha256": file_sha256(output / "full_run_plan.json"),
    })
    print(f"[READY] All mixture sources validated; {len(dataset):,} packed training chunks")


def resumable_checkpoint(output, world_size=1):
    """Quarantine interrupted saves and select the newest complete trainer checkpoint."""
    rng_files = (["rng_state.pth"] if world_size == 1 else
                 [f"rng_state_{rank}.pth" for rank in range(world_size)])
    required = ("model.safetensors", "optimizer.pt", "scheduler.pt", *rng_files,
                "trainer_state.json", "tokenizer.json", "tokenizer_config.json", "config.json")
    complete = []
    for checkpoint in sorted(output.glob("checkpoint-*")):
        if not checkpoint.is_dir() or not checkpoint.name.removeprefix("checkpoint-").isdigit():
            continue
        try:
            if any(not (checkpoint / name).is_file() or not (checkpoint / name).stat().st_size
                   for name in required):
                raise ValueError("Missing saved state")
            state = json.loads((checkpoint / "trainer_state.json").read_text())
            step = int(checkpoint.name.removeprefix("checkpoint-"))
            if state["global_step"] != step:
                raise ValueError("Checkpoint step differs from trainer state")
            if world_size > 1:
                marker = json.loads((checkpoint / "distributed_complete.json").read_text())
                if marker != {"world_size": world_size, "global_step": step}:
                    raise ValueError("Distributed checkpoint has no matching completion marker")
        except (ValueError, KeyError, FileNotFoundError):
            destination = output / "interrupted-checkpoints" / checkpoint.name
            destination.parent.mkdir(exist_ok=True)
            suffix = 1
            while destination.exists():
                destination = destination.with_name(f"{checkpoint.name}-{suffix}")
                suffix += 1
            checkpoint.rename(destination)
            print(f"[RESUME] Preserved interrupted save at {destination}")
            continue
        complete.append((step, checkpoint))
    return str(max(complete)[1]) if complete else None


def run_single(output, *, resume=False):
    """Train the selected vocabulary, restoring early-stop state after requeues."""
    from akanlm.data.cpt_splits import read_manifest
    from akanlm.data.mixture import MixtureConfig
    from akanlm.training.checkpoints import restore_milestones
    from akanlm.training.coverage import validate_full_source_coverage
    from akanlm.training.cpt import resolve_milestone_steps, train
    from akanlm.training.distributed import on_main_process, training_state

    output = Path(output)
    # Read only the small config to initialize the process group. Rank zero
    # verifies all hashes, owns report writes, and selects the resume point.
    config = TrainingConfig(**json.loads((output / "inputs/config.json").read_text()))
    training_state(config)

    def initialize():
        plan, config = check_single_plan(output)
        ready = json.loads((output / "preparation/ready.json").read_text())
        if ready["full_run_plan_sha256"] != file_sha256(output / "full_run_plan.json"):
            raise ValueError("Prepared cache belongs to another full-run plan")
        mixtures = output / "inputs/mixtures"
        split_path = Path(plan["split_manifest"])
        manifest = read_manifest(split_path)
        validate_full_source_coverage(MixtureConfig.from_yaml(mixtures / f"{config.mixture}.yaml"), manifest, ready["validation"])
        char_config = corpus_bpc.CharSpanConfig(**plan["character_evaluation"])
        (output / "evaluation").mkdir(exist_ok=True)
        score_validation_checkpoint(Path(config.base_checkpoint), output / "evaluation/initialization_char_bpc.json",
              manifest, char_config, 4)
        split_sha = manifest["sha256"]
        del manifest
        gc.collect()
        checkpoint = None
        completed = (output / "run_card.json").exists()
        if completed:
            validate_completed_arm(output, config, split_sha)
            print("[RESUME] Training already finished; completing final evaluation only")
        else:
            if list(output.glob("checkpoint-*")) and not resume:
                raise ValueError("Interrupted CPT run requires --resume")
            checkpoint = resumable_checkpoint(output, config.world_size)
            for step in restore_milestones(output, resolve_milestone_steps(config)):
                print(f"[RESUME] Restored interrupted milestone copy of checkpoint-{step}")
            # A preparation/startup interruption can leave a plan but no updates.
            if (output / "training_plan.json").exists() and not checkpoint and (output / "final").exists():
                raise ValueError("Final model exists without a completed run card or resumable checkpoint")
            print(f"[TRAIN] {'Resume ' + checkpoint if checkpoint else 'Fresh configured initialization'}")
        return plan, config, checkpoint, completed

    plan, config, checkpoint, completed = on_main_process(initialize)
    split_path = Path(plan["split_manifest"])
    char_config = corpus_bpc.CharSpanConfig(**plan["character_evaluation"])
    if not completed:
        train(config, output, corpora_dir=output / "inputs/corpora",
              mixtures_dir=output / "inputs/mixtures", split_manifest=split_path,
              manifest_sha256=plan["manifest_sha256"],
              resume_from_checkpoint=checkpoint, require_cache=True)

    def finish():
        manifest = read_manifest(split_path)
        score_validation_checkpoint(output / "final", output / "evaluation/final_char_bpc.json", manifest, char_config, 4)
        card = json.loads((output / "run_card.json").read_text())
        write_json(output / "completed.json", {
            "completed_steps": card["metrics"]["completed_steps"],
            "best_checkpoint": card["metrics"]["best_loss_checkpoint"],
            "early_stopped": card["metrics"]["completed_steps"] < config.max_steps,
            "sources": plan["sources"], "final_weights_sha256": file_sha256(output / "final/model.safetensors"),
        })
        print(f"[DONE] {output}; exported selected final model")

    on_main_process(finish)
