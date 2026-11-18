#!/usr/bin/env python
"""Per-epoch L0 / MonoScore trajectories for the runs of the Matryoshka/Gated grid.

WHY THIS EXISTS. `branch_metrics` scores `best_val_monoscore_model.pt` (the
evaluation protocol). For a run whose sparsity DRIFTS during training, that
checkpoint's L0 is not the L0 of the trained model -- and MonoScore frequently
peaks at epoch 1, so the selected checkpoint can predate the swept knob acting
at all (e.g. a collapsing branch can select epoch 1, the least dead point of a
monotone decline).

So the reported L0 alone cannot distinguish "sparse" from "dying". This emits
the trajectory endpoints beside it, which can:

    l0_first   L0 at the first validated epoch
    l0_best    L0 at the epoch the protocol SELECTS (what the table reports)
    l0_max     the largest L0 ever reached
    l0_final   L0 at the last validated epoch  <- the trained model
    drift      l0_final / l0_best, the tell

A healthy run has l0_best ~= l0_final (drift ~= 1). A dying branch has
drift ~= 0 however good its reported number looks.

Needs the `tensorboard` package (tensorboardX is write-side only). Output is
plain JSON so the pickers can read it without that dependency. CPU only -- it
reads only event files written with --tb_dir.
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
from pathlib import Path

TB_DIR = os.path.abspath(os.environ.get("MAIN_TB_DIR", "./outputs/main/tensorboard"))
OUT = os.path.join(os.path.abspath(os.environ.get("MAIN_STATE_DIR", "./outputs/main/selection")),
                   "l0_trajectory.json")

# L0 and MonoScore as the dual/other trainers log them.
TAG_L0, TAG_MS, TAG_R2, TAG_R2M = "val_active", "val_mono_mean", "val_r2", "val_r2_mono"


WANTED = (TAG_L0, TAG_MS, TAG_R2, TAG_R2M, "best/EPOCH")


def _read_scalars(d):
    """{tag: [values in step order]} for WANTED tags only.

    Uses the raw record iterator rather than EventAccumulator. EventAccumulator
    indexes EVERY scalar in the run (this project logs ~35 of them), which made
    a full pass over hundreds of directories slow and CPU-limit prone.
    Filtering to five tags at read time is roughly an order of magnitude less
    work.
    """
    # tensorboard's bundled proto (no tensorflow needed).
    from tensorboard.compat.proto import event_pb2
    out = {t: [] for t in WANTED}
    for fn in sorted(glob.glob(os.path.join(d, "*tfevents*"))):
        for rec in _tfrecords(fn):
            try:
                ev = event_pb2.Event.FromString(rec)
            except Exception:
                continue
            for v in getattr(ev, "summary", None).value if ev.HasField("summary") else []:
                if v.tag in out and v.HasField("simple_value"):
                    out[v.tag].append((ev.step, v.simple_value))
    return {t: [val for _, val in sorted(vs, key=lambda kv: kv[0])]
            for t, vs in out.items()}


def _tfrecords(path):
    """Yield raw record payloads from a TFRecord file (length, crc, data, crc)."""
    import struct
    with open(path, "rb") as fh:
        while True:
            head = fh.read(8)
            if len(head) < 8:
                return
            (n,) = struct.unpack("<Q", head)
            if fh.read(4) is None:
                return
            data = fh.read(n)
            if len(data) < n:
                return
            fh.read(4)
            yield data


def _flush(out_path, rows):
    """Atomic partial write, so a killed pass still advances the cache."""
    tmp = Path(str(out_path) + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(rows, indent=1, sort_keys=True))
    tmp.replace(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tb-dir", default=TB_DIR)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--glob", default="t3x*")
    ap.add_argument("--full", action="store_true",
                    help="ignore the cached entries and reparse everything")
    args = ap.parse_args()
    try:
        from tensorboard.compat.proto import event_pb2  # noqa: F401
    except ImportError:
        sys.exit("needs the `tensorboard` package (pip install tensorboard)")

    # INCREMENTAL. Parsing hundreds of event directories takes minutes, so a
    # full rescan on every call is wasteful. Each entry records the newest event-file mtime it was
    # built from; a directory whose files have not changed since is reused.
    # Deleted runs drop out because the output is rebuilt from the
    # directories that exist now, not merged into the old dict blindly.
    prev = {}
    if not args.full:
        try:
            prev = json.loads(Path(args.out).read_text())
        except Exception:
            prev = {}

    def _src_mtime(d):
        ts = [os.path.getmtime(f) for f in glob.glob(os.path.join(d, "*"))]
        return max(ts) if ts else 0.0

    rows, reused = {}, 0
    for d in sorted(glob.glob(os.path.join(args.tb_dir, args.glob))):
        name = os.path.basename(d)
        mt = _src_mtime(d)
        old_row = prev.get(name)
        if (old_row and "error" not in old_row
                and old_row.get("src_mtime") is not None
                and mt <= old_row["src_mtime"]):
            rows[name] = old_row
            reused += 1
            continue
        try:
            sc = _read_scalars(d)
        except Exception as exc:                 # a run still mid-write
            rows[name] = {"error": f"{type(exc).__name__}: {exc}",
                          "src_mtime": None}
            continue
        l0 = sc.get(TAG_L0) or []
        if not l0:
            continue                             # nothing logged yet
        ms, r2, r2m = (sc.get(TAG_MS) or [], sc.get(TAG_R2) or [],
                       sc.get(TAG_R2M) or [])
        be = sc.get("best/EPOCH") or []
        best_ep = int(be[-1]) if be else None
        # FLUSH AS WE GO: a full pass that writes only at the end loses
        # everything when it is killed; with periodic flushes plus the mtime
        # cache above, every invocation makes progress.
        if len(rows) % 40 == 0:
            _flush(args.out, rows)
        # best/EPOCH is 1-indexed; the series is 0-indexed.
        l0_best = (l0[best_ep - 1] if best_ep and 0 < best_ep <= len(l0) else None)
        rows[name] = {
            "src_mtime": mt,
            "n_epochs_validated": len(l0),
            "best_epoch": best_ep,
            "l0_first": round(l0[0], 3),
            "l0_best": (None if l0_best is None else round(l0_best, 3)),
            "l0_max": round(max(l0), 3),
            "l0_final": round(l0[-1], 3),
            # The tell. Guard the zero denominator: a run whose SELECTED
            # checkpoint already has L0 0 is degenerate either way, so report
            # drift 0.0 rather than dividing by zero or emitting null.
            "drift_final_over_best": (
                None if l0_best is None else
                (0.0 if l0_best <= 1e-9 else round(l0[-1] / l0_best, 4))),
            "ms_first": (round(ms[0], 4) if ms else None),
            "ms_max": (round(max(ms), 4) if ms else None),
            "ms_final": (round(ms[-1], 4) if ms else None),
            "r2_final": (round(r2[-1], 4) if r2 else None),
            "r2_mono_final": (round(r2m[-1], 4) if r2m else None),
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=1, sort_keys=True))
    ok = [r for r in rows.values() if "error" not in r]
    print(f"wrote {args.out}: {len(rows)} runs ({len(ok)} with L0 series, "
          f"{reused} reused, {len(rows) - reused} reparsed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
