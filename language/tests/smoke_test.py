#!/usr/bin/env python3
"""End-to-end smoke test of the language pipeline on SYNTHETIC data (no downloads).

What it builds, all under --work (default ./outputs/language_smoke):
  * a tiny randomly initialised GPT-NeoX causal LM (hidden size 1024 = Pythia-410m's, 13
    blocks so block 12 exists) and a word-level tokenizer, saved where prepare_assets.sh would
    put the Pythia snapshot;
  * a synthetic two-shard parquet corpus where prepare_assets.sh would put OpenWebText
    (skipped, with the token streams written directly, if pyarrow is not installed).

Then it runs the real code paths with tiny budgets:
  registry self-checks -> activation-shard reader self-test -> pack -> extract (block 12) ->
  train (TopK baseline and Dual-TopK through vision/main.py, argv built by train_cell.sh) ->
  evaluate (dCE, KL, R^2, MonoScore, L0) -> table builder -> one top-activating-contexts page.

    python language/tests/smoke_test.py [--work DIR] [--device auto|cpu|cuda]

The numbers produced are meaningless (random model, random text); only the plumbing is tested.
Exit status 0 iff every step succeeded.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LANG_DIR = HERE.parent
VISION_DIR = LANG_DIR.parent / "vision"

# Tiny budgets.  The encode / training batch must divide the scoring set (BatchTopK and the
# evaluator's batch trimming depend on it).
TRAIN_TOKENS = 32_768
EVAL_TOKENS = 8_192
BATCH = 256
TINY = {"--n_latents": "8192", "--topk_k": "8", "--batch_size": str(BATCH),
        "--num_epochs": "2", "--steps_per_epoch": "4", "--warmup_steps": "3",
        "--num_workers": "0"}


def step(msg: str) -> None:
    print(f"\n=== {msg}", flush=True)


def build_fake_assets(rng_seed: int = 0) -> None:
    """Tiny LM + tokenizer at the Pythia snapshot path; synthetic corpus at the OWT path."""
    import numpy as np
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, PreTrainedTokenizerFast
    from langdual import assets, config

    words = [f"w{i}" for i in range(500)]
    vocab = {"<|endoftext|>": 0, "<unk>": 1, **{w: i + 2 for i, w in enumerate(words)}}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>",
                                  unk_token="<unk>")
    torch.manual_seed(rng_seed)
    cfg = GPTNeoXConfig(vocab_size=512, hidden_size=config.LM_D_MODEL, num_hidden_layers=13,
                        num_attention_heads=8, intermediate_size=256,
                        max_position_embeddings=config.LM_CTX_LEN, eos_token_id=0,
                        bos_token_id=0)
    model = GPTNeoXForCausalLM(cfg).eval()
    lm_dir = assets.local_snapshot(config.LM_REPO)
    os.makedirs(lm_dir, exist_ok=True)
    model.save_pretrained(lm_dir, safe_serialization=True)
    tok.save_pretrained(lm_dir)

    rng = np.random.default_rng(rng_seed)
    docs = [" ".join(rng.choice(words, size=int(rng.integers(20, 200))))
            for _ in range(800)]
    docs[5] = ""                           # an empty document must be skipped by the packer
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return None
    files = assets.lm_corpus_files("owt")
    half = len(docs) // 2
    for f, chunk in zip(files, (docs[:half], docs[half:])):
        os.makedirs(os.path.dirname(f), exist_ok=True)
        pq.write_table(pa.table({"text": chunk}), f)
    return docs


def write_streams_without_pyarrow(docs) -> None:
    """Fallback when pyarrow is missing: write the packed streams directly (pack is skipped)."""
    import numpy as np
    from transformers import AutoTokenizer
    from langdual import assets, config
    tok = AutoTokenizer.from_pretrained(assets.local_snapshot(config.LM_REPO), local_files_only=True)
    ids = []
    for d in docs:
        if d:
            ids += tok(d, add_special_tokens=False)["input_ids"] + [0]
    ids = np.asarray(ids, dtype=np.uint16)
    os.makedirs(config.TOKENS_DIR, exist_ok=True)
    for split, a, b in (("train", 0, TRAIN_TOKENS), ("val", TRAIN_TOKENS, TRAIN_TOKENS + EVAL_TOKENS),
                        ("test", TRAIN_TOKENS + EVAL_TOKENS, TRAIN_TOKENS + 2 * EVAL_TOKENS)):
        np.save(config.token_stream(split), ids[a:b])


def train_tiny(tag: str, device: str, env: dict) -> None:
    """Train one cell with tiny overrides; argv otherwise exactly what train_cell.sh builds."""
    from langdual import config
    out = subprocess.run(["bash", str(LANG_DIR / "train_cell.sh"), tag],
                         env=dict(env, LD_DRY_RUN="1"), capture_output=True, text=True, check=True)
    lines = out.stdout.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("--- LD_DRY_RUN"))
    words = [shlex.split(l)[0] if l != "''" else "" for l in lines[i + 1:]]
    argv, j = [], 0
    while j < len(words):
        if words[j] in TINY:
            argv += [words[j], TINY[words[j]]]
            j += 2
        else:
            argv.append(words[j])
            j += 1
    argv += ["--device", device]
    res = subprocess.run([sys.executable] + argv, cwd=str(VISION_DIR), env=env,
                         capture_output=True, text=True)
    log = Path(config.RUNS_DIR) / f"{tag}.smoke.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(res.stdout + res.stderr)
    if res.returncode != 0:
        raise SystemExit(f"training {tag} failed; see {log}")
    run = Path(config.RUNS_DIR) / tag
    for f in ("best_val_monoscore_model.pt", "training_config.json"):
        assert (run / f).exists(), f"{tag}: {f} missing"
    if "dual" in tag:
        assert "WARM START COMPLETE" in res.stdout, f"{tag}: Residual Branch never unfroze"
    print(f"  trained {tag}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=os.path.join("outputs", "language_smoke"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--keep", action="store_true", help="do not wipe --work first")
    a = ap.parse_args()

    work = Path(a.work).resolve()
    if work.exists() and not a.keep:
        shutil.rmtree(work)
    root = work / "root"
    os.environ.update(LANGDUAL_ROOT=str(root), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      WANDB_MODE="offline", TOKENIZERS_PARALLELISM="false")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    sys.path.insert(0, str(LANG_DIR))
    sys.path.insert(1, str(VISION_DIR))
    env = dict(os.environ, PYTHON=sys.executable,
               PYTHONPATH=f"{LANG_DIR}:{VISION_DIR}:" + os.environ.get("PYTHONPATH", ""))
    env.pop("LD_ENV_READY", None)

    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device

    from langdual import ablate, config, jobs
    from langdual.lmsae import resid_shards

    step("registry self-checks")
    assert config._audit() == 0 and ablate._audit() == 0 and jobs.selftest() == 0

    step("activation-shard reader self-test")
    assert resid_shards._self_test() == 0

    step("synthetic assets (tiny random GPT-NeoX, word-level tokenizer, parquet corpus)")
    docs = build_fake_assets()

    if docs is None:
        step("pack: SKIPPED (pyarrow not installed); writing token streams directly")
        import numpy as np
        rng = np.random.default_rng(0)
        words = [f"w{i}" for i in range(500)]
        write_streams_without_pyarrow([" ".join(rng.choice(words, size=100)) for _ in range(800)])
    else:
        step("pack")
        from langdual import materialize_corpora as MC
        MC.main(["pack", "--lm-corpus", "owt", "--train-tokens", str(TRAIN_TOKENS),
                 "--eval-tokens", str(EVAL_TOKENS)])
        meta = json.load(open(MC.token_meta_path("train")))
        assert meta["n_tokens"] == TRAIN_TOKENS
        val = json.load(open(MC.token_meta_path("val")))
        assert val["held_out_after_doc"] >= meta["next_doc_index"], "held-out overlaps train"

    step("extract block 12 (train, val, test)")
    from langdual.lmsae import extract_resid as ER
    for split in ("train", "val", "test"):
        ER.main(["--corpus", "owt", "--layers", "12", "--split", split, "--device", device,
                 "--batch-seqs", "8", "--norm-tokens", "4096"])
        man = json.load(open(os.path.join(config.resid_dir(12, split), "manifest.json")))
        assert man["norm_factor"] > 0 and man["layer"] == 12
    assert ER.main(["--corpus", "owt", "--layers", "12", "--split", "train", "--verify",
                    "--device", device]) == 0

    step("train (TopK baseline + Dual-TopK, tiny budgets)")
    for tag in ("B_L12_owt_topk_baseline_seed42", "B_L12_owt_topk_dual_seed42"):
        train_tiny(tag, device, env)

    step("evaluate")
    from langdual.lmsae import eval_trackb as EV
    for tag in ("B_L12_owt_topk_baseline_seed42", "B_L12_owt_topk_dual_seed42"):
        EV.main(["--tag", tag, "--score-tokens", str(EVAL_TOKENS // 2), "--encode-batch-size",
                 str(BATCH), "--ce-tokens", "2048", "--device", device])
        r = json.load(open(config.trackb_result(jobs.cell_from_tag(tag))))
        for path in (("ce", "kl_clean_vs_reconstructed"), ("recon", "r2_global_a4"),
                     ("monoscore", "minmax", "mean_active_paper_a4"), ("sparsity", "l0_mean")):
            v = r
            for k in path:
                v = v[k]
            assert isinstance(v, float) and math.isfinite(v), (tag, path, v)

    step("table builder")
    rc = subprocess.run([sys.executable, str(LANG_DIR / "build_language_table.py"),
                         "--layers", "12", "--corpora", "owt"], env=env).returncode
    assert rc == 0 and (Path(config.REPORT_DIR) / "language_table.tex").exists()

    step("top-activating contexts page")
    from langdual.qualitative import top_contexts as TC
    TC.main(["--layer", "12", "--corpus", "owt", "--activation", "topk", "--seed", "42",
             "--rows", str(EVAL_TOKENS // 2), "--concepts", "2"])
    page = Path(config.REPORT_DIR) / "qualitative" / "contexts_L12_owt_topk_seed42.html"
    assert page.exists() and page.stat().st_size > 0

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
