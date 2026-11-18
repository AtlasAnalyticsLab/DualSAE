"""Lightweight TensorBoard logger for the SAE trainers.

WHY this exists: on compute nodes without internet access W&B can only run in offline mode and
is viewable only after `wandb sync`. TensorBoard instead writes event files to disk, which
`tensorboard --logdir …` reads live. This wraps a SummaryWriter so the trainers can log the SAME
per-epoch / per-step metric dicts they send to W&B (when W&B is enabled).

DESIGN / edge cases:
- No-op when `tb_dir` is falsy → call sites need no `if` guards (`RunLogger(None)` is a safe sink),
  so a run can be TB-only, W&B-only, or both with identical call sites.
- Import fallback: `torch.utils.tensorboard.SummaryWriter` hard-requires the `tensorboard` pip
  package; `tensorboardX` is the write-side fallback (same add_scalar API, no protobuf<6 pin —
  useful with protobuf 6.x).
- We `flush()` after each log() so a live viewer sees new points within seconds (the "real-time" part).
- Non-numeric values and the `'epoch'` key (used as the step axis) are skipped; NaN passes through
  (TensorBoard renders a gap), which is what we want for skip_extra_eval runs that leave some
  train-set metrics as NaN.
"""

from typing import Optional
import numbers

# torch's writer needs the `tensorboard` package installed; tensorboardX is the protobuf-6-safe
# write-side fallback with the same SummaryWriter API. SummaryWriter stays None if neither is present.
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # ImportError, or a protobuf/runtime issue inside tensorboard
    try:
        from tensorboardX import SummaryWriter  # type: ignore
    except Exception:
        SummaryWriter = None  # type: ignore


class RunLogger:
    """Mirror of the W&B logging surface, backed by a TensorBoard SummaryWriter.

    No-op when `tb_dir` is None/empty, so trainers may call .log()/.summary()/.close()
    unconditionally regardless of whether TensorBoard logging was requested.
    """

    def __init__(self, tb_dir: Optional[str], flush_secs: int = 10):
        self.writer = None
        if tb_dir:
            if SummaryWriter is None:
                # tb_dir was requested but no backend is importable — fail loud rather than
                # silently dropping every metric.
                raise ImportError(
                    "tb_dir was set but neither torch.utils.tensorboard nor tensorboardX is "
                    "importable. Install tensorboardX (write side) into the venv."
                )
            # flush_secs is a backstop; we also flush() after each log() for near-real-time updates.
            self.writer = SummaryWriter(log_dir=tb_dir, flush_secs=flush_secs)

    @property
    def enabled(self) -> bool:
        return self.writer is not None

    def _add(self, data: dict, step: int, prefix: str = "") -> None:
        for k, v in data.items():
            if k == "epoch":  # this is the step axis, not a scalar to plot
                continue
            # bool is a subclass of int — exclude it; keep ints/floats (NaN is a float → kept).
            if isinstance(v, bool) or not isinstance(v, numbers.Real):
                continue
            self.writer.add_scalar(f"{prefix}{k}", float(v), global_step=int(step))

    def log(self, data: dict, step: int) -> None:
        """Log scalar metrics at `step` (per-epoch: step=epoch; per-step: step=global_step)."""
        if self.writer is None:
            return
        self._add(data, step)
        self.writer.flush()  # make just-written points visible to a live login-node viewer

    def summary(self, data: dict, step: int = 0) -> None:
        """Log best/final metrics under a `best/` tag prefix (e.g. best/MONOSCORE)."""
        if self.writer is None:
            return
        self._add(data, step, prefix="best/")
        self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
