"""Held-out document protection and the CPT validation trainer.

Text from ``Role.EVAL_ONLY`` and ``Role.HELDOUT`` sources is excluded from
training. The trainer scores the corpus-split validation sets with the
token-span scorer in ``akanlm.eval.denoising``. Transformers is imported only
inside ``trainer_classes()`` so configuration loading stays lightweight.
"""

from __future__ import annotations

import re
from collections import Counter

from akanlm.audit.decontaminate import EvalItem, NgramIndex
from akanlm.utils.paths import digest

WORDS = re.compile(r"\w+", re.UNICODE)


def heldout_documents(sources):
    """Load declared held-out and eval-only text for exclusion only.

    The audit ledger records corpus counts rather than training-record
    removal IDs. This function rebuilds the exclusion guard directly from
    the pinned source declarations and protects both language sides of any
    parallel corpus entries.

    Args:
        sources: Mapping of source names to declared modality/role metadata.

    Returns:
        Tuple of ``(items, provenance)`` where items are evaluation-only
        protected text records and provenance stores document-level
        provenance metadata.

    Raises:
        ValueError: If a declared held-out source does not yield records.
    """
    from akanlm.audit.sources import Modality, Role
    from akanlm.corpora.loaders import load_pairs, load_texts

    items, provenance = [], {}
    for name, source in sorted(sources.items()):
        if source.role not in (Role.EVAL_ONLY, Role.HELDOUT):
            continue
        prefix = f"{name}/{source.hf_split}"
        if source.modality is Modality.PARALLEL:
            records = list(load_pairs(source))
            text_rows = [(r.index, r.src_text, r.tgt_text) for r in records]
            for record in records:
                items.append(EvalItem(prefix + "/source", record.index, record.src_text))
                items.append(EvalItem(prefix + "/target", record.index, record.tgt_text))
        else:
            records = list(load_texts(source))
            text_rows = [(r.index, r.text) for r in records]
            items.extend(EvalItem(prefix, r.index, r.text) for r in records)
        if not records:
            raise ValueError(f"No held-out documents loaded for exclusion: {prefix}")
        provenance[name] = {
            "split": source.hf_split, "documents": len(records),
            "declaration": source.model_dump(mode="json"),
            "documents_sha256": digest(text_rows),
        }
    return items, provenance


class ProtectedDocuments:
    """Reject records that overlap any protected evaluation or held-out text.

    The guard verifies both language sides before allowing a parallel record
    to be rendered. It also indexes short protected sentences as embedded
    word paths so that a sentence-level match inside a longer document is
    detected even when the 13-gram n-gram finder cannot see it.
    """

    def __init__(self, items):
        self.index = NgramIndex(13)
        self.short = {}
        self.seen = Counter()
        self.removed = Counter()
        for item in items:
            self.index.add(item)
            words = WORDS.findall(item.text.casefold())
            if 0 < len(words) < 13:
                node = self.short
                for word in words:
                    node = node.setdefault(word, {})
                node[None] = True

    def matches(self, text: str) -> bool:
        if self.index.find(text) is not None:
            return True
        words = WORDS.findall(text.casefold())
        for start in range(len(words)):
            node = self.short
            for word in words[start:start + 12]:
                node = node.get(word)
                if node is None:
                    break
                if None in node:
                    return True
        return False

    def accept(self, source: str, *texts: str) -> bool:
        return self.record(source, self.check(*texts))

    def check(self, *texts: str) -> bool:
        """Whether no text matches, without counting; safe to call in worker processes."""
        return not any(self.matches(text) for text in texts)

    def record(self, source: str, outcome: bool) -> bool:
        """Count one :meth:`check` outcome and return it."""
        self.seen[source] += 1
        if not outcome:
            self.removed[source] += 1
        return outcome

    def report(self):
        return {"records_seen": dict(self.seen), "records_removed": dict(self.removed),
                "ngram_size": 13, "embedded_short_sentences_protected": True}


class TrainingAndFixedEvalCollator:
    """Collate fixed validation rows (which carry labels) as-is; corrupt training rows."""

    def __init__(self, training_collator):
        self.training_collator = training_collator

    def __call__(self, examples):
        from transformers import default_data_collator
        if "labels" in examples[0]:
            return default_data_collator(examples)
        return self.training_collator(examples)


def trainer_classes():
    """Return ``(ValidationTrainer, CPTEarlyStopping)``."""
    from transformers import EarlyStoppingCallback, Seq2SeqTrainer

    class CPTEarlyStopping(EarlyStoppingCallback):
        """Early stopping that acts only on evaluations reporting ``metric_for_best_model``."""

        def on_evaluate(self, args, state, control, metrics, **kwargs):
            if args.metric_for_best_model in metrics:
                return super().on_evaluate(args, state, control, metrics, **kwargs)

    class ValidationTrainer(Seq2SeqTrainer):
        """Seq2SeqTrainer with the CPT optimizer, schedule, training loss and validation scorer."""

        def __init__(self, *args, cpt_config=None, **kwargs):
            self.cpt_config = cpt_config
            super().__init__(*args, **kwargs)
            if cpt_config:
                from akanlm.training.optimization import adafactor_class
                # Seq2SeqTrainer does not expose Trainer's optimizer-class
                # constructor argument in the installed release.
                self.optimizer_cls_and_kwargs = (adafactor_class(
                    cpt_config.adafactor_min_dim_size_to_factor), {
                    "lr": cpt_config.learning_rate,
                    "scale_parameter": cpt_config.adafactor_scale_parameter,
                    "relative_step": False,
                    "warmup_init": False,
                    # These currently equal transformers' defaults, but are
                    # part of Google's resolved Mesh Adafactor recipe. Pin
                    # them so a dependency upgrade cannot alter optimization.
                    "decay_rate": cpt_config.adafactor_decay_rate,
                    "beta1": cpt_config.adafactor_beta1,
                    "clip_threshold": cpt_config.adafactor_clip_threshold,
                    "eps": (cpt_config.adafactor_epsilon1,
                            cpt_config.adafactor_epsilon2),
                    "weight_decay": cpt_config.weight_decay,
                })

        def create_scheduler(self, num_training_steps, optimizer=None):
            """Use the rsqrt plateau schedule when ``rsqrt_no_ramp_down`` is configured."""
            if self.cpt_config and self.cpt_config.lr_scheduler_type == "rsqrt_no_ramp_down":
                from functools import partial

                from torch.optim.lr_scheduler import LambdaLR

                from akanlm.training.optimization import rsqrt_multiplier

                if self.lr_scheduler is None:
                    self.lr_scheduler = LambdaLR(
                        optimizer if optimizer is not None else self.optimizer,
                        partial(rsqrt_multiplier, plateau_steps=self.cpt_config.warmup_steps),
                    )
                    self._created_lr_scheduler = True
                return self.lr_scheduler
            return super().create_scheduler(num_training_steps, optimizer)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            """Add z-loss and target-capacity normalization, during training only."""
            labels = inputs.get("labels")
            loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
            if model.training and self.cpt_config and self.cpt_config.z_loss:
                from akanlm.training.optimization import training_z_loss
                loss = loss + training_z_loss(outputs.logits, labels, self.cpt_config.z_loss)
            if (model.training and self.cpt_config
                    and self.cpt_config.loss_normalization == "target_capacity"):
                loss = loss * (labels.ne(-100).sum() / labels.numel())
            return (loss, outputs) if return_outputs else loss

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            """Score each validation source and report token-weighted NLL per source and overall.

            Each rank scores every ``world_size``-th example; nats and target
            tokens are summed across ranks before dividing.
            """
            import torch

            from akanlm.eval.denoising import score

            datasets = self.eval_dataset if eval_dataset is None else eval_dataset
            # Score the local replica directly: unequal validation shard sizes
            # must not invoke DDP forward collectives or pad/repeat examples.
            model = self.accelerator.unwrap_model(self.model)
            rank, world_size = self.args.process_index, self.args.world_size
            was_training = self.model.training
            metrics, nats, tokens = {}, 0., 0
            try:
                for name, dataset in datasets.items():
                    rows = [dataset[i] for i in range(rank, len(dataset), world_size)]
                    values = (score(model, rows, batch_size=min(4, self.args.per_device_eval_batch_size),
                                    device=self.args.device) if rows else
                              {"total_nats": 0., "num_eval_tokens": 0})
                    totals = torch.tensor([values["total_nats"], values["num_eval_tokens"]],
                                          dtype=torch.float64, device=self.args.device)
                    if world_size > 1:
                        torch.distributed.all_reduce(totals)
                    source_nats, source_tokens = totals.tolist()
                    if source_tokens <= 0:
                        raise ValueError(f"No validation target tokens: {name}")
                    loss = source_nats / source_tokens
                    metrics[f"{metric_key_prefix}_{name}_loss"] = loss
                    metrics[f"{metric_key_prefix}_{name}_neg_log_perplexity"] = -loss
                    nats += source_nats
                    tokens += int(source_tokens)
            finally:
                self.model.train(was_training)
            metrics[f"{metric_key_prefix}_cpt_loss"] = nats / tokens
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics

    return ValidationTrainer, CPTEarlyStopping
