"""Offline two-process CPT: updates, sharding, global metrics, checkpoints and resume."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers import MT5Config, MT5ForConditionalGeneration

from akanlm.eval.denoising import score


def test_distributed_cpt_trains_evaluates_and_resumes(tmp_path, tiny_tokenizer):
    initial, output = tmp_path / "initial", tmp_path / "distributed"
    config = MT5Config(vocab_size=len(tiny_tokenizer), d_model=16, d_ff=32, d_kv=4,
                       num_heads=2, num_layers=1, num_decoder_layers=1,
                       dropout_rate=0., decoder_start_token_id=0)
    config.tie_word_embeddings = False
    model = MT5ForConditionalGeneration(config)
    model.encoder.embed_tokens = model.shared
    model.decoder.embed_tokens = model.shared
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    model.save_pretrained(initial)
    tiny_tokenizer.save_pretrained(initial)
    environment = {**os.environ, "ACCELERATE_USE_CPU": "true", "ACCELERATE_TORCH_DEVICE": "cpu",
                   "CUDA_VISIBLE_DEVICES": "",
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}
    worker = Path(__file__).with_name("cpt_distributed_worker.py")
    with (tmp_path / "distributed.log").open("w") as log:
        result = subprocess.run(
            [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
             "--nproc_per_node=2", str(worker), str(initial), str(output)],
            env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=240, check=False,
        )
    assert result.returncode == 0, (tmp_path / "distributed.log").read_text()[-16000:]
    card = json.loads((output / "run_card.json").read_text())
    assert card["framework"]["world_size"] == 2
    assert card["framework"]["distributed_backend"] == "gloo"
    assert card["framework"]["dataloader_start_method"] == "spawn"
    resources = card["framework"]["rank_resources"]
    assert [row["rank"] for row in resources] == [0, 1]
    assert all(row["dataloader_workers"] == 2 and len(row["cpu_affinity"]) == 1 for row in resources)
    assert all({"cuda_visible_devices", "gpu_uuid", "gpu_pci_bus_id"} <= set(row) for row in resources)
    assert card["effective_batch_size"] == 8
    assert card["tokens_per_step"] == 8 * 32
    batching = card["mixture_summary"]["preparation"]["optimizer_batching"]
    assert batching["usable_chunks"] == 16 and batching["complete_updates_per_epoch"] == 2
    rank_updates = [json.loads((output / f"rank-{rank}-updates.json").read_text()) for rank in range(2)]
    assert rank_updates[0] == rank_updates[1]
    assert [row["step"] for row in rank_updates[0]] == [1, 2, 3]
    assert len({row["weights_sha256"] for row in rank_updates[0]}) == 3

    # Sharding: within each update the ranks train on disjoint examples, and
    # the first epoch uses every aligned packed row exactly once.
    sharding = [json.loads((output / f"rank-{rank}-examples.json").read_text()) for rank in range(2)]
    aligned = sharding[0]["aligned_rows"]
    assert aligned == sharding[1]["aligned_rows"] and len(set(aligned)) == 16
    by_update = {}
    for rank, record in enumerate(sharding):
        for microbatch in record["microbatches"]:
            assert all(len(found) == 1 for found in microbatch["rows"]), microbatch
            by_update.setdefault(microbatch["update"], {}).setdefault(rank, []).extend(
                found[0] for found in microbatch["rows"])
    assert sorted(by_update) == [1, 2, 3]
    for update, ranks in by_update.items():
        assert sorted(ranks) == [0, 1] and all(len(rows) == 4 for rows in ranks.values()), (update, ranks)
        assert not set(ranks[0]) & set(ranks[1]), (update, ranks)
        assert len(set(ranks[0]) | set(ranks[1])) == 8, (update, ranks)
    first_epoch = [row for update in (1, 2) for rows in by_update[update].values() for row in rows]
    assert sorted(first_epoch) == sorted(aligned)
    assert set(by_update[3][0] + by_update[3][1]) <= set(aligned)

    # Completion protocol: the marker appears only after every rank's RNG
    # file, and a save missing either is never selected for resume.
    saves = json.loads((output / "rank-0-saves.json").read_text())
    assert [save["step"] for save in saves] == [1, 2, 3]
    assert all(save["marker"] and all(save["rng_files"]) for save in saves), saves
    for rank in range(2):
        assert (output / f"checkpoint-3/rng_state_{rank}.pth").is_file()
        assert (output / f"milestones/checkpoint-2/rng_state_{rank}.pth").is_file()
    assert json.loads((output / "checkpoint-3/distributed_complete.json").read_text()) == {
        "world_size": 2, "global_step": 3,
    }
    recovery = json.loads((output / "recovery.json").read_text())
    for case in ("missing_rank_1_rng", "missing_marker", "wrong_marker"):
        assert recovery[case] == {"selected": "checkpoint-3", "checkpoint_left": False,
                                  "quarantined": True}, (case, recovery[case])
    assert recovery["milestone_restore"] == {"restored_steps": [3], "partial_copy_left": False,
                                             "files_match": True, "hard_linked": True}

    # A cross-rank input mismatch stops both ranks.
    for rank in range(2):
        assert "fixture input" in (output / f"rank-{rank}-mismatch.txt").read_text()

    restored = MT5ForConditionalGeneration.from_pretrained(output / "final")
    assert any(not torch.equal(before[name], p) for name, p in restored.named_parameters())
    values = [score(restored, rows, device="cpu") for rows in
              json.loads((output / "fixed_validation.json").read_text()).values()]
    expected = sum(v["total_nats"] for v in values) / sum(v["num_eval_tokens"] for v in values)
    assert card["metrics"]["eval_cpt_loss"] == pytest.approx(expected, abs=1e-6)
    (tmp_path / "verification.json").write_text(json.dumps({
        "world_size": 2, "backend": "gloo", "completed_steps": 3,
        "rank_weights_identical": True, "global_eval_loss": expected,
        "examples_per_update": {update: {rank: sorted(rows) for rank, rows in ranks.items()}
                                for update, ranks in sorted(by_update.items())},
        "first_epoch_rows": sorted(first_epoch),
        "saves_observed_by_rank_zero": saves,
        "recovery": recovery,
        "cross_rank_mismatch_raised_on_ranks": [0, 1],
    }, indent=2))
