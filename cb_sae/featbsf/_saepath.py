"""sys.path shim for importing the vision/ core modules from featbsf.

The vision/ directory is NOT an installed/qualified package: its trainers run
with vision/ as the working directory and import bare top-level names
(`from dataset import LMDBFeatureDataset`, `from utils import unpack_batch`,
`from models.matryoshka_sae import MatryoshkaSAE`). To reuse those modules from
featbsf we put the vision/ directory itself on sys.path so the same bare
import names resolve.

Kept in ONE place (this module) so featbsf.data / adapters / cb_data share
identical resolution and the shim can be audited/changed in a single spot.

Edge cases / known limitations:
- vision/ defines generic top-level names ('dataset', 'utils', 'models',
  'loss'). We insert at the FRONT of sys.path so vision's modules win over any
  same-named package elsewhere; consequently featbsf code must never rely on a
  DIFFERENT top-level module called e.g. 'utils'. featbsf itself is always
  imported as the qualified 'featbsf.*' package, so it cannot be shadowed.
- Importing 'dataset' executes vision/dataset/__init__.py, which also imports
  open_images.py (pandas/PIL/tqdm at module level -- plain imports, no network).
"""
import sys

from featbsf.config import REPO_ROOT

SAE_DIR = REPO_ROOT / "vision"


def ensure_sae_on_path() -> None:
    """Idempotently put vision/ at the front of sys.path.

    Idempotence matters because every featbsf module that touches vision/
    calls this at import time; repeated calls must not grow sys.path.
    """
    p = str(SAE_DIR)
    if p not in sys.path:
        # Front insertion: vision's 'dataset'/'models' must beat any
        # same-named third-party package (see module docstring).
        sys.path.insert(0, p)
