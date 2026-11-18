import torch
import numpy as np

def unpack_batch(batch):
    """Return features, class indices, and sample indices from a dataset batch.

    Returns a 3-tuple ``(features, class_idx, sample_path)`` for the common cases.
    When the batch is a 4-tuple (the bias-mitigation pipeline's
    ``(x, label, path, center)`` record), the hospital/center id is returned as a
    4th element. Existing 2-/3-tuple callers are unaffected -- they keep receiving
    exactly 3 return values.
    """
    if isinstance(batch, torch.Tensor):
        return batch, None, None
    if isinstance(batch, np.ndarray):
        return torch.from_numpy(batch), None, None
    if isinstance(batch, (tuple, list)):
        if len(batch) == 2:
            features, class_idx = batch
            return features, class_idx, None
        if len(batch) == 3:
            features, class_idx, sample_path = batch
            return features, class_idx, sample_path
        if len(batch) == 4:
            features, class_idx, sample_path, center = batch
            return features, class_idx, sample_path, center
    raise TypeError(f"Unsupported batch type: {type(batch)}")

class PersistentBatcher:
    """Yield a FIXED number of batches per epoch from one long-lived loader iterator.

    WHY this exists.  The paper fixes "50 epochs", but its two-stage DualSAE warm-up is fixed in
    STEPS (2,500).  The two are only mutually coherent at the data scale the paper used: 1.28M
    ImageNet embeddings at batch 2048 is 625 steps per epoch, 31,250 steps total, and 2,500 is
    8% of that.  Text corpora differ 50x in size, so a literal 50 epochs would give the smallest
    corpus ~200 optimizer steps -- FEWER than the warm-up.  The poly branch would then never
    unfreeze and DualSAE would silently degenerate into a MonoSAE while still reporting a full
    set of metrics.  Fixing the step budget instead makes every cell comparable and keeps the
    warm-up meaningful.

    WHY the iterator is PERSISTENT.  The naive implementation re-creates ``iter(loader)`` at the
    top of every epoch and takes the first N batches.  On a corpus with fewer than N batches per
    pass that replays the SAME prefix every epoch, so the model would only ever see a fraction
    of the data while the logs claim 50 epochs.  Holding one iterator across epochs and
    re-creating it only on StopIteration means consecutive epochs continue through the dataset,
    and each new pass is reshuffled by the DataLoader as usual.

    Edge cases and limitations:
      * ``steps_per_epoch=None`` reproduces today's behaviour exactly -- one full pass per
        epoch -- so every existing vision run is unaffected.
      * An empty loader raises rather than spinning forever.
      * ``passes_completed`` counts how many times the underlying loader was exhausted, which is
        the honest "true epochs" number for a caption.
    """

    def __init__(self, loader, steps_per_epoch=None):
        self.loader = loader
        self.steps_per_epoch = steps_per_epoch
        self._it = None
        self.passes_completed = 0

    def __len__(self) -> int:
        """Batches this batcher yields per epoch (what tqdm's progress bar needs)."""
        return int(self.steps_per_epoch) if self.steps_per_epoch else len(self.loader)

    def epoch(self):
        """Generator over exactly ``len(self)`` batches."""
        if self.steps_per_epoch is None:
            # Unchanged path: a plain pass over the loader.
            self.passes_completed += 1
            for batch in self.loader:
                yield batch
            return

        remaining = int(self.steps_per_epoch)
        if remaining <= 0:
            raise ValueError(f"steps_per_epoch must be positive, got {self.steps_per_epoch}")
        while remaining > 0:
            if self._it is None:
                self._it = iter(self.loader)
            try:
                batch = next(self._it)
            except StopIteration:
                # One full pass finished.  Re-create the iterator so the next batch continues
                # into a freshly shuffled pass rather than restarting this epoch's prefix.
                self._it = None
                self.passes_completed += 1
                # Guard against an empty loader, which would otherwise loop forever.
                if len(self.loader) == 0:
                    raise RuntimeError("train loader yielded no batches; cannot fill an epoch")
                continue
            remaining -= 1
            yield batch


def resolve_step_budget(num_epochs: int, steps_per_epoch, warmup_steps: int, label: str = "run"):
    """Validate the step budget against the warm-up and return the total step count.

    Raises rather than warns: a warm-up longer than training is not a degraded run, it is a
    DIFFERENT MODEL (a MonoSAE) reported under the DualSAE name, and it fails silently.
    """
    if steps_per_epoch is None:
        return None
    total = int(num_epochs) * int(steps_per_epoch)
    if warmup_steps and int(warmup_steps) >= total:
        raise ValueError(
            f"{label}: warmup_steps={warmup_steps} >= total steps={total} "
            f"({num_epochs} epochs x {steps_per_epoch} steps). The poly branch would never "
            f"unfreeze and the run would be a plain SAE, not a DualSAE. Lower the "
            f"warm-up or raise --steps_per_epoch."
        )
    return total


# Opt-in TrainingConfig fields, mapped to the default that means "the flag was not passed". A
# field holding EXACTLY that default is dropped from the serialized payload (config_payload
# below), so runs that do not use an opt-in flag carry no key for it. Listed by NAME (not "every
# None/False-valued field") on purpose: many other fields legitimately default to None or False
# (wandb_project, tb_dir, skip_extra_eval, ...) and are always serialized.
#   jump_threshold_init                --jump_threshold_init; train_other + train_dual;
#                                      None = model default (theta = 1.0).
#   dual_train_threshold_post_warmup   --dual_train_threshold_post_warmup; train_dual only;
#                                      False = default recipe (threshold frozen after warm-up).
#   poly_arch, poly_final_act,         residual-branch ablation options; train_dual only; the
#   poly_warmup_zero,                  defaults are the standard recipe (see TrainingConfig).
#   freeze_sparse_from, snapshot_steps
# Add a field here only if it is opt-in and its default leaves every other run unchanged.
OPT_IN_DEFAULT_CONFIG_FIELDS = {
    "jump_threshold_init": None,
    "dual_train_threshold_post_warmup": False,
    "poly_arch": "mlp",
    "poly_final_act": "auto",
    "poly_warmup_zero": False,
    "freeze_sparse_from": None,
    "snapshot_steps": None,
}
# Kept for readers/comments written against the step-0 name (the None-default subset).
OPT_IN_NONE_CONFIG_FIELDS = tuple(k for k, v in OPT_IN_DEFAULT_CONFIG_FIELDS.items() if v is None)


def _is_unset_default(value, default) -> bool:
    """True iff ``value`` is the opt-in field's "flag not passed" default, compared TYPE-EXACTLY.

    Why type-exact: in Python ``0 == False`` and ``0.0 == False``, so a plain ``==`` would also
    drop a (hypothetical) numeric 0 written into a bool field, and ``None == None`` is the only
    case where identity and equality agree anyway.  A value of another type is never "unset":
    it is kept, so a misuse stays VISIBLE in training_config.json instead of vanishing.
    """
    return type(value) is type(default) and value == default


def config_payload(config) -> dict:
    """The dict every trainer serializes for its config: vars(config) minus UNSET opt-in fields.

    Used for training_config.json (main.py), the checkpoint 'config' entry (autoencoder_final.pt,
    per-epoch checkpoints) and the W&B config.  When an opt-in flag IS passed the key is kept,
    so the setting travels with the run (training_config.json is the provenance record the
    loaders and collectors read).

    Returns a NEW dict: vars() is the instance's live __dict__, and deleting from it would delete
    the attribute from the config object the trainer is still using.  Readers that rebuild a
    config from the payload are unaffected either way: an absent key falls back to the
    dataclass default (None / False), i.e. to exactly what the run used.
    """
    payload = dict(vars(config))
    for name, default in OPT_IN_DEFAULT_CONFIG_FIELDS.items():
        if name in payload and _is_unset_default(payload[name], default):
            del payload[name]
    return payload
