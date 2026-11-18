"""Stable seed derivation and RNG hygiene (D-11, D-22).

D-22, verbatim: every derived seed is

    int.from_bytes(hashlib.sha256(f'{base}|{key}'.encode()).digest()[:4], 'little')

Why sha256 and never Python ``hash()``: ``hash(str)`` is salted per process (unless
PYTHONHASHSEED is set), so a seed built from it changes on every run and a resumed task would
not reproduce an uninterrupted one.  The first 4 bytes
give an unsigned 32-bit value, which every consumer accepts: ``random.seed``, ``np.random.seed``
(max 2**32-1), ``torch.manual_seed`` and vLLM ``SamplingParams(seed=...)``.

D-11: ``BaseAutoencoder.__init__`` calls ``torch.manual_seed(cfg['seed'])``, so loading a model
silently re-seeds torch's global RNG.  Every stage re-seeds explicitly AFTER all model loads with
``seed_everything``; ``sae_io.load_arm`` additionally makes the load itself RNG-neutral.

Self-test:  python -m common.seeds --selftest
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import numbers
import random
import sys
from typing import Iterator, Optional, Union

import numpy as np

SeedPart = Union[str, int, float]

# Upper bound (exclusive) of a derived seed: 4 bytes.  Exported so callers can assert ranges.
SEED_MODULUS = 2 ** 32

# The ONE rendering of a float inside a seed key: 6 significant digits, format spec 'g'.
# Pinned by the selftest; changing it changes every float-keyed seed (e.g. every D-49
# generation seed, whose key contains the KL budget).
FLOAT_KEY_FORMAT = ".6g"


def canon_float(x: float) -> str:
    """Canonical text of a float seed-key component: ``format(float(x), '.6g')``.

    Why a fixed format and not ``repr``: D-49 keys contain KL budgets (0.05 ... 1.6 nats), and the
    same budget reaches the key along different paths -- a literal in a grid, a value read back
    from a calibration JSON, a numpy float32 column.  ``repr`` renders those differently
    (``repr(float(np.float32(0.1)))`` is '0.10000000149011612'), which would silently change
    every generation seed between calibration, generation and a resumed job.  '.6g' maps all of
    0.1, 0.10, 1e-01 and float32(0.1) to '0.1', and the text is independent of platform and
    numpy version (it is Python's own correctly-rounded float formatting).
    Edge cases: -0.0 renders as '0' (same budget); NaN and +-inf are rejected (a missing value
    must never become a seed).  Known limitation (by design): two values that agree to 6
    significant digits share a key -- irrelevant for pre-registered grids; a key that genuinely
    needs more precision must be passed as an explicit string.
    A float with an integral value renders like the int ('1.0' -> '1'), so a budget of 1 nat
    gives the same seed whether it was typed 1 or 1.0.
    """
    v = float(x)
    if v != v or v in (float("inf"), float("-inf")):
        raise ValueError(f"non-finite float {x!r} cannot be part of a seed key")
    if v == 0.0:
        v = 0.0            # fold -0.0 into 0.0 so both render '0'
    return format(v, FLOAT_KEY_FORMAT)


def _canon(part: object, what: str) -> str:
    """Render one seed component as the exact text D-22 hashes.

    Accepted: ``str`` (verbatim), integers (``str(int(x))``) and floats (``canon_float``).
    Why so strict: the D-22 formula hashes the f-string rendering of its inputs, and many
    "obvious" key types render unstably --
      * a tuple renders its elements with ``repr``, and ``repr(np.int64(3))`` is ``'3'`` under
        numpy 1.x but ``'np.int64(3)'`` under numpy 2.x, so a tuple key would silently change
        every seed on a numpy upgrade;
      * a float's ``repr`` depends on its dtype history (float32 vs float64), so floats go
        through the ONE fixed rendering ``canon_float`` (D-49 keys contain KL budgets);
      * ``bool`` is an int subclass but renders as 'True', which is almost always a bug here.
    Integers (including numpy integers) are converted with ``int()`` first, so ``np.int64(3)``
    and ``3`` give the same seed; numpy floats likewise go through ``float()``.  Multi-part keys
    go through ``derive_seed_parts``.
    """
    if isinstance(part, (bool, np.bool_)):
        raise TypeError(f"{what} must be str, int or float, got bool {part!r} (almost certainly a "
                        f"bug)")
    if isinstance(part, str):
        return part
    if isinstance(part, numbers.Integral):
        return str(int(part))
    if isinstance(part, (float, np.floating)):
        return canon_float(part)
    raise TypeError(f"{what} must be str, int or float, got {type(part).__name__} {part!r}; "
                    f"use derive_seed_parts(base, *parts) for multi-part keys")


def derive_seed(base: SeedPart, key: SeedPart) -> int:
    """D-22 exactly: sha256(f'{base}|{key}'), first 4 bytes, little-endian, as an int in
    [0, 2**32).

    ``base`` names the purpose (e.g. ``'a2.interpret'``) or is an integer run seed; ``key``
    identifies the item (e.g. ``'news20|test'``).  Both must be str, int or float (see
    ``_canon``; floats are rendered by ``canon_float``).
    """
    text = f"{_canon(base, 'base')}|{_canon(key, 'key')}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def derive_seed_parts(base: SeedPart, *parts: SeedPart) -> int:
    """Multi-part key: the parts are canonicalised individually and joined with '|', then fed
    to ``derive_seed`` as ONE key.  So ``derive_seed_parts(b, 'x', 3) == derive_seed(b, 'x|3')``.

    '|' is also D-22's base/key separator, so a part containing '|' could collide with a
    different split of the same text; parts containing '|' are therefore rejected.
    """
    if not parts:
        raise ValueError("derive_seed_parts needs at least one key part")
    canon = [_canon(p, f"part[{i}]") for i, p in enumerate(parts)]
    for i, c in enumerate(canon):
        if "|" in c:
            raise ValueError(f"part[{i}]={c!r} contains '|', which would make keys ambiguous")
    return derive_seed(base, "|".join(canon))


def seed_everything(seed: int, *, cuda: bool = True) -> None:
    """Seed Python ``random``, numpy's GLOBAL RNG and torch (CPU and, if available, all CUDA
    devices) with one integer.  Call it after all model loads (D-11).

    torch is imported lazily so that CPU-only consumers (stats, firth) never pay for it.
    """
    seed = int(seed)
    if not 0 <= seed < SEED_MODULUS:
        raise ValueError(f"seed {seed} outside [0, 2**32) (numpy's global seed range)")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is always present in the topic_model venv
        return
    torch.manual_seed(seed)  # also seeds every CUDA device's default generator
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng(seed: int) -> np.random.Generator:
    """A private numpy Generator (PCG64).  Preferred over the global RNG everywhere in new code:
    it cannot be disturbed by, and cannot disturb, any library that seeds ``np.random``."""
    return np.random.Generator(np.random.PCG64(int(seed)))


@contextlib.contextmanager
def isolated_numpy_rng(seed: Optional[int] = None) -> Iterator[None]:
    """Run a block with numpy's GLOBAL RNG isolated from the caller.

    Why: HypotheSAEs' samplers (``sample_percentile_bins``, ``sample_top_zero``;
    interpret_neurons.py:28-29, :76-77) call ``np.random.seed(...)`` on the global RNG.  Without
    isolation, every later ``np.random.*`` call in the app would silently follow the sampler's
    seed, so the app's own randomness would depend on how many samplers ran before it (and a
    resumed job, which skips finished samplers, would diverge from an uninterrupted one).

    Behaviour: the global state is saved on entry and restored on exit, including when the block
    raises.  If ``seed`` is given the global RNG is seeded with it inside the block, so the block
    itself is reproducible regardless of what ran before.  Python's ``random`` module is NOT
    touched (HypotheSAEs does not seed it); use ``isolated_python_rng`` for that.
    """
    state = np.random.get_state()
    try:
        if seed is not None:
            np.random.seed(int(seed))
        yield
    finally:
        np.random.set_state(state)


@contextlib.contextmanager
def isolated_python_rng(seed: Optional[int] = None) -> Iterator[None]:
    """Same contract as ``isolated_numpy_rng`` for Python's ``random`` module (SAE-TM's
    evaluation.py draws intruder words with ``random``)."""
    state = random.getstate()
    try:
        if seed is not None:
            random.seed(int(seed))
        yield
    finally:
        random.setstate(state)


# --------------------------------------------------------------------------- selftest
def _selftest() -> int:
    failures = []

    def check(cond: bool, msg: str) -> None:
        print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond:
            failures.append(msg)

    # 1. The formula is D-22 literally (recomputed here independently of derive_seed).
    ref = int.from_bytes(hashlib.sha256("a2.interpret|news20|test".encode()).digest()[:4], "little")
    check(derive_seed("a2.interpret", "news20|test") == ref, "derive_seed == literal D-22 formula")
    check(derive_seed_parts("a2.interpret", "news20", "test") == ref,
          "derive_seed_parts joins parts with '|' into one key")
    # A pinned value: if this ever changes, every derived seed in every app changed with it.
    check(derive_seed(42, "x") == 902298685, f"pinned value derive_seed(42,'x') = {derive_seed(42, 'x')}")
    check(derive_seed(np.int64(42), np.int32(7)) == derive_seed(42, 7), "numpy ints == python ints")
    check(0 <= derive_seed("b", "k") < SEED_MODULUS, "range [0, 2**32)")
    for bad in ((1, 2), True, np.bool_(True), None):
        try:
            derive_seed("b", bad)  # type: ignore[arg-type]
            check(False, f"rejects key {bad!r}")
        except TypeError:
            check(True, f"rejects key {bad!r}")
    # Floats (D-49 KL budgets): one canonical rendering, whatever path the value took.
    kl = [0.1, 0.10, 1e-01, np.float32(0.1), np.float64(0.1), "0.1"]
    check(len({derive_seed_parts("a16.gen", "kl", v) for v in kl}) == 1,
          "float keys: 0.1 == 0.10 == 1e-01 == float32(0.1) == '0.1' (canon_float '.6g')")
    check([canon_float(v) for v in (0.05, 0.2, 1.6, 1.0, -0.0, 1e-5, 0.0001)] ==
          ["0.05", "0.2", "1.6", "1", "0", "1e-05", "0.0001"],
          "canon_float renderings pinned")
    # A pinned float-keyed value: if this changes, every D-49 generation seed changed.
    check(derive_seed_parts("a16.gen", "kl", 0.4) == derive_seed("a16.gen", "kl|0.4") == 367852051,
          f"pinned derive_seed_parts('a16.gen','kl',0.4) = {derive_seed_parts('a16.gen', 'kl', 0.4)}")
    for bad in (float("nan"), float("inf")):
        try:
            derive_seed("b", bad)
            check(False, f"rejects non-finite float {bad!r}")
        except ValueError:
            check(True, f"rejects non-finite float {bad!r}")
    try:
        derive_seed_parts("b", "a|b")
        check(False, "rejects '|' inside a part")
    except ValueError:
        check(True, "rejects '|' inside a part")

    # 2. Isolation: the block's np.random.seed does not leak; the outer stream continues as if
    # the block never ran.
    np.random.seed(123)
    a1 = np.random.rand(3)
    np.random.seed(123)
    _ = np.random.rand(1)  # consume one draw, then run an intrusive block
    with isolated_numpy_rng(seed=999):
        np.random.seed(5)  # what HypotheSAEs does
        inner = np.random.rand(2)
    a2 = np.random.rand(2)
    check(np.array_equal(a1[1:], a2), "outer numpy stream unaffected by an inner np.random.seed")
    with isolated_numpy_rng(seed=999):
        np.random.seed(5)
        inner2 = np.random.rand(2)
    check(np.array_equal(inner, inner2), "inner block reproducible")
    np.random.seed(7)
    before = np.random.get_state()[1].copy()
    try:
        with isolated_numpy_rng():
            np.random.seed(1)
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    check(np.array_equal(np.random.get_state()[1], before), "state restored after an exception")
    random.seed(1)
    r1 = [random.random() for _ in range(2)]
    random.seed(1)
    _ = random.random()
    with isolated_python_rng(seed=3):
        random.random()
    r2 = random.random()
    check(r1[1] == r2, "isolated_python_rng restores python random")

    # 3. seed_everything: two calls give the same torch/numpy/random draws.
    seed_everything(derive_seed("selftest", "everything"))
    import torch
    t1, n1, p1 = torch.rand(2), np.random.rand(2), random.random()
    seed_everything(derive_seed("selftest", "everything"))
    t2, n2, p2 = torch.rand(2), np.random.rand(2), random.random()
    check(torch.equal(t1, t2) and np.array_equal(n1, n2) and p1 == p2,
          "seed_everything reproduces torch/numpy/random")
    print("SEEDS SELFTEST", "OK" if not failures else f"FAILED ({len(failures)})")
    return 0 if not failures else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(_selftest())
    ap.print_help()
