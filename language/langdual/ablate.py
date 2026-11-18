"""Registry of the language-model ablation cells (one knob moved from the main-grid setting).

Three axes are shipped:

  warmupB    DualSAE warm-up length (the number of steps the Residual Branch stays frozen),
             swept over {0, 1250, 5000, 10000} for all three activations at blocks 6/12/18 on
             OpenWebText.  The main grid holds the 2,500-step default, which is therefore NOT
             retrained here.
  tkuntied   TopK baseline with an UNTIED decoder (--no_tied_weights), blocks 6/12/18.
  dg0B1250   DualSAE with the decorrelation loss switched off (gamma = span_coef = 0) at
             warm-up 1250, TopK/BatchTopK, blocks 6/12/18.

The last two are not part of any language table; they are the standard-SAE / gamma=0 arms
that the LLM-steering component (<repo>/llm_steering) loads, together with the main-grid
BatchTopK baselines and the warmupB=1250 Duals.

Every ablation cell inherits EVERYTHING it does not name from the main-grid cell at the same
coordinates (langdual.config) and writes into its own root, so an ablation cell can never
overwrite a main-grid result.  Tags: ``abl_<axis><value>_B_L<n>_<corpus>_<act>_<variant>_seed<S>``
(e.g. ``abl_warmupB1250_B_L12_owt_topk_dual_seed42``) or, for single-value axes declared with
``tag_value=False``, ``abl_<axis>_B_...`` (e.g. ``abl_tkuntied_B_L18_owt_topk_baseline_seed42``).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

from langdual import config

# Everything the ablation writes lives under its own root.
ROOT = os.path.abspath(os.environ.get("LANGDUAL_ABL_ROOT", os.path.join(config.ROOT, "ablation")))
RUNS_DIR = os.path.join(ROOT, "runs")
RESULT_DIR = os.path.join(ROOT, "results")
REPORT_DIR = os.path.join(ROOT, "reports")
LOG_DIR = os.path.join(ROOT, "logs")
TB_ROOT = os.path.join(config.TB_ROOT, "ablation")

TOPK_ACTS = ("topk", "batch_topk")
ALL_ACTS = ("topk", "batch_topk", "jump_relu")
DOWNSTREAM_LAYERS: Tuple[int, ...] = (6, 12, 18)


@dataclass(frozen=True)
class Axis:
    """One ablation axis: which knob moves, over what values, and for which arms.

    `env` is the LD_* variable train_cell.sh reads for that knob, so an axis needs no change to
    the training path -- the worker is driven entirely by the environment.
    """
    name: str
    env: str                       # the LD_* override the worker consumes
    values: Tuple[str, ...]        # excludes the default, which the main grid already has
    default: str                   # the main-grid value (for provenance)
    activations: Tuple[str, ...]
    variants: Tuple[str, ...]      # ('dual',) for DualSAE-only knobs
    layers: Tuple[int, ...] = DOWNSTREAM_LAYERS
    lm_corpus: str = "owt"
    extra_env: Dict[str, str] = field(default_factory=dict)
    # True = abl_<name><value>_...; False = abl_<name>_..., allowed only for a single-value axis.
    tag_value: bool = True
    # Whether the axis's cells are evaluated with lmsae/eval_trackb by the ablation driver.
    # The steering arms are evaluated by the steering component instead (at their final
    # checkpoint), so their cells are trained here but not scored.
    score: bool = True
    note: str = ""

    def __post_init__(self) -> None:
        if not self.tag_value and len(self.values) != 1:
            raise ValueError(f"axis {self.name!r}: tag_value=False needs exactly one value")
        if self.lm_corpus not in config.LM_CORPORA:
            raise ValueError(f"axis {self.name!r}: unknown corpus {self.lm_corpus!r}")


AXES: Tuple[Axis, ...] = (
    Axis("warmupB", "LD_WARMUP_STEPS", ("0", "1250", "5000", "10000"), "2500",
         ALL_ACTS, ("dual",),
         note="two-stage schedule: steps during which the Residual Branch is frozen. "
              "warmup=0 trains both branches jointly from step 0."),
    Axis("tkuntied", "LD_TIED_WEIGHTS", ("untied",), "tied", ("topk",), ("baseline",),
         tag_value=False, score=False,
         note="TopK baseline with an untied decoder (--no_tied_weights; the encoder is set to "
              "the decoder transpose at init and for every latent revived by the dead-latent "
              "reinit)."),
    Axis("dg0B1250", "LD_SPAN_COEF", ("0",), "0.01", TOPK_ACTS, ("dual",),
         extra_env={"LD_WARMUP_STEPS": "1250"}, tag_value=False, score=False,
         note="DualSAE with gamma=0 (decorrelation loss off) at warm-up 1250."),
)

AXES_BY_NAME: Dict[str, Axis] = {a.name: a for a in AXES}


@dataclass(frozen=True)
class AblCell:
    axis: str
    value: str
    track: str
    key: str            # layer tag, e.g. "L12"
    corpus: str
    activation: str
    variant: str
    seed: int

    @property
    def safe_value(self) -> str:
        """Filesystem-safe rendering of the value."""
        return self.value.replace(",", "-").replace(".", "p")

    @property
    def tag(self) -> str:
        """Globally unique, flat leaf; distinct from every main-grid tag (prefix ``abl_``)."""
        ax = AXES_BY_NAME.get(self.axis)
        middle = self.axis if (ax is not None and not ax.tag_value) else f"{self.axis}{self.safe_value}"
        return (f"abl_{middle}_{self.track}_{self.key}_{self.corpus}"
                f"_{self.activation}_{self.variant}_seed{self.seed}")

    @property
    def layer(self) -> int:
        return int(self.key[1:])

    @property
    def scored(self) -> bool:
        return AXES_BY_NAME[self.axis].score

    @property
    def run_dir(self) -> str:
        return os.path.join(RUNS_DIR, self.tag)

    @property
    def done_sentinel(self) -> str:
        return os.path.join(self.run_dir, "CELL_DONE.json")

    @property
    def tb_dir(self) -> str:
        return os.path.join(TB_ROOT, self.axis, self.tag)

    @property
    def overrides(self) -> Dict[str, str]:
        """LD_* environment overrides that turn the main-grid cell into this ablation cell."""
        ax = AXES_BY_NAME[self.axis]
        env = {ax.env: self.value}
        env.update(ax.extra_env)
        return env


def iter_axis_cells(axis: Axis) -> Iterator[AblCell]:
    # Depth is the OUTERMOST loop (then value, activation, variant, seed).
    for layer in axis.layers:
        for value in axis.values:
            for act in axis.activations:
                for var in axis.variants:
                    for seed in config.SEEDS:
                        yield AblCell(axis.name, value, "B", f"L{layer}", axis.lm_corpus,
                                      act, var, seed)


def iter_cells(axes: Optional[Tuple[str, ...]] = None) -> Iterator[AblCell]:
    for ax in AXES:
        if axes and ax.name not in axes:
            continue
        yield from iter_axis_cells(ax)


def trackb_result(cell: AblCell) -> str:
    return os.path.join(RESULT_DIR, "trackb", f"{cell.tag}.json")


def _audit() -> int:
    cells = list(iter_cells())
    problems: List[str] = []
    tags = [c.tag for c in cells]
    if len(set(tags)) != len(tags):
        problems.append("tag is not injective")
    clash = set(tags) & {c.tag for c in config.iter_cells()}
    if clash:
        problems.append(f"tags collide with the main grid: {sorted(clash)[:3]}")
    for c in cells:
        tw = c.overrides.get("LD_TIED_WEIGHTS")
        if tw is not None and (tw != "untied" or c.activation != "topk" or c.variant != "baseline"):
            problems.append(f"{c.tag}: LD_TIED_WEIGHTS={tw!r} (only 'untied', TopK baseline)")
    print(f"{'axis':10s} {'env knob':18s} {'values':22s} {'cells':>6s}")
    for ax in AXES:
        n = len(list(iter_axis_cells(ax)))
        print(f"{ax.name:10s} {ax.env:18s} {','.join(ax.values):22s} {n:>6d}"
              + ("" if ax.score else "   (train only)"))
    print(f"root: {ROOT}")
    if problems:
        print("\nFAILED:")
        for p in problems:
            print("  -", p)
        return 1
    print("\nAUDIT OK")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", action="store_true")
    ap.parse_args()
    raise SystemExit(_audit())
