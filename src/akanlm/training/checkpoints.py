"""Training-plan provenance, resume compatibility, and milestone checkpoints.

A run records its configuration, data identity, initial weights and runtime
versions; a resume must match them. Milestone checkpoints are hard-linked out
of Trainer rotation for later evaluation.
"""

import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

# Evaluation modules hash their own source into stored scores, so they keep
# importing file_sha256 from this module rather than from utils.paths.
from akanlm.utils.paths import file_sha256, write_json


def runtime_framework():
    """Library versions and reference commits that a resume must match."""
    import accelerate
    import datasets
    import torch
    import transformers

    return {
        "implementation": "pytorch_huggingface_mesh_semantic_port",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "datasets": datasets.__version__,
        "google_mt5_commit": "9dcd60fc43c31a8651461f9a21894a134ba22166",
        "mesh_tensorflow_reference_commit": "e6798a2610a2c2f4c4cd236d8214422cb1ecc00a",
        "numerical_parity_claimed": False,
    }


def bind_training_plan(output_dir, config, dataset_key, validation, resume, framework=None):
    """Write ``training_plan.json``, or check that a resume matches it.

    A resume must keep the configuration (except a larger ``max_steps``),
    training data, initial weights and runtime framework. Without ``resume``
    the output must hold no trained checkpoints.

    Raises:
        ValueError: The resume changes the run, or a fresh run would
            overwrite trained checkpoints.
    """
    output_dir = Path(output_dir)
    path = output_dir / "training_plan.json"
    initial = Path(config.base_checkpoint)
    weights = sorted(initial.glob("*.safetensors"))
    if not weights:
        raise ValueError("CPT requires a local initialization with safetensors weights")
    plan = {
        "config": asdict(config), "dataset_key": dataset_key,
        "validation": validation,
        "initial_weights": {p.name: file_sha256(p) for p in weights},
        "framework": framework or runtime_framework(),
    }
    previous = json.loads(path.read_text()) if path.exists() else None
    if resume:
        if previous is None:
            raise ValueError("Checkpoint output has no training plan; start a new output")
        old_config = dict(previous["config"])
        new_config = dict(plan["config"])
        old_steps, new_steps = old_config.pop("max_steps"), new_config.pop("max_steps")
        if (old_config != new_config or previous["dataset_key"] != dataset_key
                or previous["initial_weights"] != plan["initial_weights"]
                or previous.get("framework") != plan["framework"]):
            raise ValueError(
                "Resume would change training data, initialization, optimization, "
                "or PyTorch framework settings"
            )
        if new_steps < old_steps:
            raise ValueError("Resume cannot shrink the recorded training budget")
    elif list(output_dir.glob("checkpoint-*")) or (output_dir / "final").exists():
        raise ValueError("Output already contains trained checkpoints; use --resume or a fresh output")
    write_json(path, plan)


def preserve_milestone(root, step):
    """Hard-link ``checkpoint-<step>`` into ``milestones/`` unless already preserved.

    A partial copy left by an interrupted earlier attempt is replaced.

    Returns:
        bool: Whether a new milestone copy was made.
    """
    root = Path(root)
    source = root / f"checkpoint-{step}"
    destination = root / "milestones" / source.name
    if destination.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    # Hard links survive Trainer checkpoint rotation without duplicating
    # large weights and optimizer files on this shared filesystem.
    shutil.copytree(source, temporary, copy_function=os.link)
    temporary.rename(destination)
    return True


def restore_milestones(root, steps):
    """Finish milestone copies interrupted after their checkpoint was saved.

    Call only after incomplete checkpoints have been quarantined: every
    remaining ``checkpoint-<step>`` is then complete.

    Returns:
        list[int]: Steps whose milestone copy was made now.
    """
    root = Path(root)
    return [step for step in steps
            if (root / f"checkpoint-{step}").is_dir() and preserve_milestone(root, step)]


def preserve_milestones_callback(steps):
    from transformers import TrainerCallback

    class PreserveMilestones(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            # Trainer saves one RNG file per rank. Wait for all of them before
            # rank zero preserves the checkpoint with hard links.
            if args.world_size > 1:
                import torch.distributed as dist

                dist.barrier()
            if not state.is_world_process_zero:
                return
            root = Path(args.output_dir)
            source = root / f"checkpoint-{state.global_step}"
            if args.world_size > 1:
                # A positive file size alone cannot distinguish a complete
                # torch.save from one interrupted partway through a write.
                write_json(source / "distributed_complete.json",
                           {"world_size": args.world_size, "global_step": state.global_step})
            # If the job stops during this copy, the checkpoint is already
            # complete; run_single restores the milestone before resuming.
            if state.global_step in steps:
                preserve_milestone(root, state.global_step)

    return PreserveMilestones()
