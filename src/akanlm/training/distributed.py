"""Distributed CPT initialization and shared-artifact ownership."""

import os
from datetime import timedelta


def training_state(config):
    """Require the declared process count before loading any model weights."""
    from accelerate import PartialState
    from accelerate.utils import parse_flag_from_env

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if config.world_size < 1 or world_size != config.world_size:
        raise ValueError(
            f"CPT config requires WORLD_SIZE={config.world_size}; launched {world_size}. "
            "Use the matching accelerator profile and distributed launcher."
        )
    # Initial/final character evaluation may run on rank zero for longer than
    # NCCL's default timeout while the other rank waits for the shared report.
    state = PartialState(cpu=parse_flag_from_env("ACCELERATE_USE_CPU"), timeout=timedelta(hours=2))
    if state.num_processes != config.world_size:
        raise ValueError("Accelerate process count differs from the frozen CPT config")
    if state.num_processes == 1 and state.device.type == "cuda":
        import torch

        if torch.cuda.device_count() > 1:
            raise ValueError("CPT requires one process per visible GPU; use distributed launch")
    return state


def describe_rank(state):
    """Record the devices and CPU cores this process received, in JSON-safe types."""
    import torch

    record = {
        "rank": state.process_index, "local_rank": state.local_process_index, "device": str(state.device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_step_gpus": os.environ.get("SLURM_STEP_GPUS"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "visible_gpus": 0, "gpu_name": None, "gpu_uuid": None, "gpu_pci_bus_id": None,
    }
    if state.device.type == "cuda":
        # The device name is identical on both B200s; the UUID and PCI bus
        # show whether the ranks hold distinct physical GPUs.
        properties = torch.cuda.get_device_properties(state.device)
        record.update(visible_gpus=torch.cuda.device_count(), gpu_name=str(properties.name),
                      gpu_uuid=str(properties.uuid), gpu_pci_bus_id=int(properties.pci_bus_id))
    return record


def require_identical_across_ranks(label, value):
    """Fail on every rank when the ranks hold different copies of a shared input."""
    from accelerate import PartialState
    from accelerate.utils import gather_object

    if PartialState().num_processes == 1:
        return
    values = gather_object([value])
    if any(other != values[0] for other in values[1:]):
        raise ValueError(f"Ranks disagree about {label}: {values}")


def on_main_process(function, *args, **kwargs):
    """Run a shared-file operation once; broadcast its result or failure."""
    from accelerate import PartialState
    from accelerate.utils import broadcast_object_list

    state = PartialState()
    payload = [None, None]
    error = None
    if state.is_main_process:
        try:
            payload[0] = function(*args, **kwargs)
        except Exception as caught:  # noqa: BLE001 - any failure must reach the waiting ranks
            error = caught
            payload[1] = f"{type(caught).__name__}: {caught}"
    broadcast_object_list(payload)
    if error is not None:
        raise error
    if payload[1] is not None:
        raise RuntimeError(f"CPT rank zero failed: {payload[1]}")
    return payload[0]
