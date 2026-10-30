"""Two-process CPU training worker for the distributed CPT workflow check."""

import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer, TrainerCallback

from akanlm.data import cpt_splits
from akanlm.data.collator import DataCollatorForT5MLM
from akanlm.training import cpt, validation
from akanlm.training.checkpoints import restore_milestones
from akanlm.training.distributed import (
    on_main_process,
    require_identical_across_ranks,
    training_state,
)
from akanlm.training.experiment import resumable_checkpoint

# Every token of packed row i is FIRST_ROW_TOKEN + i, so the row stays
# identifiable from any token that survives corruption in inputs or targets.
FIRST_ROW_TOKEN = 5


def row_ids(inputs, rows):
    """Packed-row identities present in each example of a collated batch."""
    found = []
    for source, target in zip(inputs["input_ids"].tolist(), inputs["labels"].tolist()):
        found.append(sorted({token - FIRST_ROW_TOKEN for token in source + target
                             if FIRST_ROW_TOKEN <= token < FIRST_ROW_TOKEN + rows}))
    return found


def interrupted_copy(complete, step, marker):
    """Copy a complete checkpoint to ``step`` with a consistent trainer state."""
    copy = complete.with_name(f"checkpoint-{step}")
    shutil.copytree(complete, copy)
    state_path = copy / "trainer_state.json"
    trainer_state = json.loads(state_path.read_text())
    trainer_state["global_step"] = step
    state_path.write_text(json.dumps(trainer_state))
    (copy / "distributed_complete.json").unlink()
    if marker is not None:
        (copy / "distributed_complete.json").write_text(json.dumps(marker))
    return copy


def main():
    # Exercise two workers sharing one CPU core per training rank.
    cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpus[int(os.environ["LOCAL_RANK"]) % len(cpus)]})
    initial, output = map(Path, sys.argv[1:])
    spec = cpt.TrainingConfig(
        "tiny-ddp", str(initial), world_size=2, input_length=32, target_length=16,
        max_steps=2, warmup_steps=1, per_device_train_batch_size=2,
        gradient_accumulation_steps=2, gradient_checkpointing=True, bf16=False,
        dataloader_num_workers=2, eval_steps=1, save_steps=1, logging_steps=1,
        ignore_data_skip=False, milestone_steps="doubling", save_total_limit=2,
    )
    state = training_state(spec)
    tokenizer = AutoTokenizer.from_pretrained(initial)
    raw, target = spec.resolved_lengths()
    rows = [{"input_ids": [FIRST_ROW_TOKEN + i] * raw} for i in range(18)]
    collator = DataCollatorForT5MLM(tokenizer, input_length=32, target_length=target)
    fixed = [{key: value[0].tolist() for key, value in collator([row]).items()}
             for row in rows[:5]]
    # Unequal target counts check token weighting; one source leaves rank 1
    # empty, and five examples cannot be split evenly across two ranks.
    fixed[0]["labels"][2:] = [-100] * (target - 2)
    datasets = {"uneven": Dataset.from_list(fixed), "singleton": Dataset.from_list(fixed[:1])}
    cpt_splits.build_validation = lambda *a: (datasets, None, {"sha256": "tiny-ddp-fixed"})
    cpt.training_dataset_key = lambda *a: "tiny-ddp-packed"
    cpt.build_dataset = lambda *a, **k: (Dataset.from_list(rows), {"components": [], "preparation": {}}, None)
    trainer_class, early_stop_class = validation.trainer_classes()
    records, examples, saves = [], [], []

    class ObserveUpdates(TrainerCallback):
        def on_pre_optimizer_step(self, args, step_state, control, model=None, **kwargs):
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            assert any(torch.count_nonzero(g) for g in grads)

        def on_step_end(self, args, step_state, control, model=None, **kwargs):
            digest = hashlib.sha256()
            for parameter in model.parameters():
                digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
            records.append({"step": step_state.global_step, "weights_sha256": digest.hexdigest()})

        def on_save(self, args, step_state, control, **kwargs):
            # Runs after PreserveMilestones on rank zero: a marker must never
            # exist before every rank's RNG file.
            if step_state.is_world_process_zero:
                checkpoint = Path(args.output_dir) / f"checkpoint-{step_state.global_step}"
                saves.append({"step": step_state.global_step,
                              "marker": (checkpoint / "distributed_complete.json").is_file(),
                              "rng_files": [(checkpoint / f"rng_state_{rank}.pth").is_file()
                                            for rank in range(args.world_size)]})

    class ObservedTrainer(trainer_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.add_callback(ObserveUpdates())

        def training_step(self, model, inputs, num_items_in_batch=None):
            examples.append({"update": self.state.global_step + 1, "rows": row_ids(inputs, len(rows))})
            return super().training_step(model, inputs, num_items_in_batch)

        def _save_rng_state(self, output_dir):
            # Rank 1 finishes its RNG file last; without the barrier in
            # PreserveMilestones, rank zero would mark the save complete first.
            if self.args.process_index == 1:
                time.sleep(1)
            super()._save_rng_state(output_dir)

    validation.trainer_classes = lambda: (ObservedTrainer, early_stop_class)
    card = cpt.train(spec, output, split_manifest=output / "split.json")
    assert card.metrics["completed_steps"] == 2
    checkpoint = on_main_process(resumable_checkpoint, output, 2)
    assert checkpoint.endswith("checkpoint-2")
    card = cpt.train(replace(spec, max_steps=3), output,
                     split_manifest=output / "split.json", resume_from_checkpoint=checkpoint)
    assert card.metrics["completed_steps"] == 3
    (output / f"rank-{state.process_index}-updates.json").write_text(json.dumps(records, indent=2))
    aligned, _ = cpt.align_dataset_to_full_updates(Dataset.from_list(rows), spec.effective_batch_size, spec.seed)
    (output / f"rank-{state.process_index}-examples.json").write_text(json.dumps({
        "aligned_rows": [row["input_ids"][0] - FIRST_ROW_TOKEN for row in aligned],
        "microbatches": examples,
    }, indent=2))
    if state.is_main_process:
        (output / "rank-0-saves.json").write_text(json.dumps(saves, indent=2))

    def verify_recovery():
        complete = Path(resumable_checkpoint(output, 2))
        assert complete.name == "checkpoint-3"
        # Each case has every other property of a complete save, so only the
        # named defect can cause quarantine. Cases are checked one at a time.
        def missing_rng():
            copy = interrupted_copy(complete, 4, {"world_size": 2, "global_step": 4})
            (copy / "rng_state_1.pth").unlink()
            return copy

        cases = {"missing_rank_1_rng": missing_rng,
                 "missing_marker": lambda: interrupted_copy(complete, 5, None),
                 "wrong_marker": lambda: interrupted_copy(complete, 6, {"world_size": 1, "global_step": 6})}
        recovery = {}
        for name, make in cases.items():
            case = make()
            selected = resumable_checkpoint(output, 2)
            recovery[name] = {"selected": Path(selected).name, "checkpoint_left": case.exists(),
                              "quarantined": (output / "interrupted-checkpoints" / case.name).is_dir()}
            if case.exists():
                shutil.rmtree(case)
        # A copy interrupted after the marker: no milestone, a partial copy.
        milestone = output / "milestones" / complete.name
        shutil.rmtree(milestone)
        partial = milestone.with_name(milestone.name + ".tmp")
        partial.mkdir()
        os.link(complete / "config.json", partial / "config.json")
        restored = restore_milestones(output, cpt.resolve_milestone_steps(replace(spec, max_steps=3)))
        recovery["milestone_restore"] = {
            "restored_steps": restored, "partial_copy_left": partial.exists(),
            "files_match": sorted(p.name for p in milestone.iterdir()) == sorted(p.name for p in complete.iterdir()),
            "hard_linked": os.path.samefile(milestone / "model.safetensors", complete / "model.safetensors"),
        }
        (output / "recovery.json").write_text(json.dumps(recovery, indent=2))
        (output / "fixed_validation.json").write_text(json.dumps({"uneven": fixed, "singleton": fixed[:1]}))

    on_main_process(verify_recovery)
    # A rank-zero error must reach the peer rather than strand a collective.
    def fail():
        raise ValueError("fixture shared-file failure")

    try:
        on_main_process(fail)
    except (ValueError, RuntimeError) as error:
        assert "fixture shared-file failure" in str(error)
    else:
        raise AssertionError("Rank-zero failure was not propagated")
    # Ranks holding different copies of a shared input must all stop.
    try:
        require_identical_across_ranks("fixture input", {"rank": state.process_index})
    except ValueError as error:
        assert "fixture input" in str(error)
        (output / f"rank-{state.process_index}-mismatch.txt").write_text(str(error))
    else:
        raise AssertionError("A cross-rank input mismatch was not detected on this rank")
    state.wait_for_everyone()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
