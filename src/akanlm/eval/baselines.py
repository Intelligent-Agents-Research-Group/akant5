"""Build comparisons against the unadapted mT5 backbone and CPT initialization."""

import json
from collections.abc import Sequence
from pathlib import Path

AFRICAN_REFERENCE = "masakhane/afri-mt5-base"
MT5_BACKBONES = {
    (512, 1024, 8, 6): "google/mt5-small",
    (768, 2048, 12, 12): "google/mt5-base",
    (1024, 2816, 24, 16): "google/mt5-large",
}


def comparison_checkpoints(checkpoints: Sequence[str]) -> list[str]:
    """Use recorded initialization and matching mT5 backbone as controls.
    Vocabulary surgery changes total parameter count. Afri-mT5 is an external
    reference; it does not determine the mT5 control size.
    """
    systems = []
    seen = set()

    def add(checkpoint: str) -> None:
        path = Path(checkpoint)
        key = str(path.resolve()) if path.exists() else checkpoint.rstrip("/")
        if key not in seen:
            systems.append(checkpoint)
            seen.add(key)

    for checkpoint in checkpoints:
        add(checkpoint)
    for checkpoint in checkpoints:
        if checkpoint.rstrip("/") == AFRICAN_REFERENCE:
            continue
        if checkpoint.rstrip("/") in MT5_BACKBONES.values():
            continue
        path = Path(checkpoint)
        if path.is_dir():
            config = json.loads((path / "config.json").read_text())
            for parent in (path, *path.parents):
                card = parent / "run_card.json"
                if card.is_file():
                    initial = json.loads(card.read_text())["config"]["base_checkpoint"]
                    if not Path(initial).is_dir():
                        raise ValueError(
                            f"Recorded CPT initialization is missing: {initial}. "
                            "Restore it or use --no-baselines with explicit controls."
                        )
                    add(initial)
                    break
        else:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(checkpoint).to_dict()
        shape = tuple(config.get(k) for k in ("d_model", "d_ff", "num_layers", "num_heads"))
        if config.get("model_type") != "mt5" or shape not in MT5_BACKBONES:
            raise ValueError(
                f"Cannot identify the mT5 backbone for {checkpoint}. "
                "Use --no-baselines and supply explicit controls."
            )
        add(MT5_BACKBONES[shape])
    add(AFRICAN_REFERENCE)
    return systems
