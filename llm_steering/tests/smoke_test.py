#!/usr/bin/env python3
"""End-to-end smoke test of the steering pipeline on SYNTHETIC data (no downloads, CPU, minutes).

What is synthetic / stubbed (everything else is the real pipeline code):
  * the language model: a tiny randomly initialised GPT-NeoX (14 blocks, width 16) with a byte-
    level BPE tokenizer trained here, saved where the real pythia snapshot would be;
  * the language-pipeline inputs: packed token streams, residual shards (computed with the tiny
    model), AG News / GoEmotions corpora and their document-embedding LMDBs, and randomly
    initialised SAE / DualSAE checkpoints (real model classes from <repo>/vision) for block 12,
    TopK and BatchTopK, all three arms, seeds 42-44;
  * the Amazon parquet reader, the bge encoder and the phi-4 judge (a deterministic fake that
    rates DiffMean pilot generations 2 and random-direction ones 0, so the P1 rule decides a
    budget; every other rating is a hash of the prompt).  The AxBench parser/harmonic mean are
    replaced by simple equivalents because the official files are a download.
  * the health-gate thresholds (random SAEs cannot pass them) and the calibration sizes are
    relaxed for speed.  Grid: block 12 only (A16_LAYERS), TopK + BatchTopK (A16_ACTIVATIONS).

The stages run in-process, in waves chosen by the real planner (steering/plan.py), with the smoke
size options of each stage, then the report (s9), the table builder and the example selection.
The test asserts that every stage produced its outputs.

    python tests/smoke_test.py [--workdir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]           # <repo>/llm_steering

D_MODEL = 16
N_LAYERS = 14            # block 12 and the block after it must exist
VOCAB = 600
DICT = 512
K = 64
SEEDS = (42, 43, 44)
N_TEST_WINDOWS = 1040    # >= CALIB_START_WINDOW (1000) + calibration windows (8)
N_TRAIN_WINDOWS = 360    # D-44 threshold batches come from the train shards
WORDS = ("house road meal letter window garden table river morning street door book pen "
         "village train lunch recipe floor stairs bakery station photograph corner drawer "
         "kitchen map evening family story school bus market bridge field").split()
CLASS_WORDS = {"World": "government election minister", "Sports": "team match goal",
               "Business": "market stocks earnings", "Sci/Tech": "software internet research",
               "joy": "happy glad delighted", "anger": "angry furious annoyed",
               "sadness": "sad grief tears", "fear": "afraid scared worried",
               "positive": "great excellent love", "negative": "terrible broken refund"}


# ============================================================================ fixtures
def _rng(seed):
    import numpy as np
    return np.random.default_rng(seed)


def _sentence(rng, cls_word=None, n=12):
    w = [WORDS[i] for i in rng.integers(0, len(WORDS), n)]
    if cls_word:
        for t in cls_word.split():
            w.insert(int(rng.integers(0, len(w))), t)
    return " ".join(w).capitalize() + "."


def build_lm(lm_dir: Path, prompts):
    """Tiny GPT-NeoX + byte-level BPE tokenizer (eos = id 0, as pythia)."""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, PreTrainedTokenizerFast
    import torch
    rng = _rng(0)
    corpus = list(prompts) + [_sentence(rng, w) for w in CLASS_WORDS.values() for _ in range(20)]
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=["<|endoftext|>"],
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(corpus, tr)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|endoftext|>",
                                   bos_token="<|endoftext|>", unk_token="<|endoftext|>")
    assert fast.eos_token_id == 0
    lm_dir.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(lm_dir)
    torch.manual_seed(0)
    cfg = GPTNeoXConfig(vocab_size=len(fast), hidden_size=D_MODEL, num_hidden_layers=N_LAYERS,
                        num_attention_heads=2, intermediate_size=32,
                        max_position_embeddings=256, rotary_pct=0.25, eos_token_id=0,
                        bos_token_id=0, tie_word_embeddings=False)
    lm = GPTNeoXForCausalLM(cfg)
    # A freshly initialised unembedding (std 0.02) gives near-uniform logits that no residual
    # edit can move by 0.05 nats; a unit-scale unembedding makes the KL calibration reachable.
    with torch.no_grad():
        lm.embed_out.weight.normal_(0.0, 1.0)
    lm.save_pretrained(lm_dir, safe_serialization=True)
    return len(fast)


def build_tokens_and_shards(root: Path, lm_dir: Path, vocab: int, layer: int = 12):
    """Packed token streams + the tiny model's block-``layer`` residuals as fp16 shards."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    lm = AutoModelForCausalLM.from_pretrained(lm_dir, local_files_only=True).eval()
    (root / "tokens").mkdir(parents=True, exist_ok=True)
    rng = _rng(1)
    nf = None
    for split, n_win in (("train", N_TRAIN_WINDOWS), ("test", N_TEST_WINDOWS)):
        toks = rng.integers(1, vocab, n_win * 128).astype(np.uint16)
        np.save(root / "tokens" / f"{split}.u16.npy", toks)
        out = []
        cap = []
        h = lm.gpt_neox.layers[layer].register_forward_hook(
            lambda m, a, o: cap.append((o[0] if isinstance(o, tuple) else o).detach()))
        with torch.no_grad():
            for s in range(0, n_win, 64):
                ids = torch.from_numpy(toks[s * 128:min(n_win, s + 64) * 128].astype(np.int64)).view(-1, 128)
                cap.clear()
                lm(input_ids=ids)
                out.append(cap[0].reshape(-1, D_MODEL).float().numpy())
        h.remove()
        X = np.concatenate(out)
        if nf is None:                       # the norm factor comes from the TRAIN split
            nf = float(np.sqrt((X ** 2).sum(1)).mean() / np.sqrt(D_MODEL))
        d = root / "resid" / f"L{layer}" / split
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "shard_00000.npy", (X / nf).astype(np.float16))
        man = {"ctx_len": 128, "d_model": D_MODEL, "dtype": "float16", "layer": layer,
               "split": split, "norm_factor": nf, "norm_convention": "divide",
               "shards": [{"file": "shard_00000.npy", "n_rows": int(X.shape[0])}]}
        (d / "manifest.json").write_text(json.dumps(man))


def build_corpora(root: Path, bge_dim: int = 1024):
    """AG News / GoEmotions jsonl + labels, and their document-embedding LMDBs."""
    import lmdb
    import numpy as np
    from steering import concepts
    rng = _rng(2)
    for corpus, n_per in (("agnews", {"train": 30, "test": 12}), ("goemotions", {"train": 30, "test": 12})):
        for split, n in n_per.items():
            texts, labels = [], []
            if corpus == "agnews":
                for ci, name in enumerate(("World", "Sports", "Business", "Sci/Tech")):
                    for _ in range(n):
                        texts.append(_sentence(rng, CLASS_WORDS[name]))
                        labels.append(ci)
                y = np.asarray(labels, np.int64)
            else:
                names = concepts.GOEMO_NAMES
                for ci, name in enumerate(names):
                    for _ in range(n if name in CLASS_WORDS else 2):
                        texts.append(_sentence(rng, CLASS_WORDS.get(name)))
                        labels.append([ci])
                texts.append(_sentence(rng))                     # one multi-label row (dropped)
                labels.append([names.index("joy"), names.index("sadness")])
                y = np.zeros((len(labels), len(names)), np.int64)
                for r, ls in enumerate(labels):
                    y[r, ls] = 1
            d = root / "corpora" / corpus
            d.mkdir(parents=True, exist_ok=True)
            with open(d / f"{split}.jsonl", "w") as fh:
                for t in texts:
                    fh.write(json.dumps({"text": t}) + "\n")
            np.save(d / f"{split}_labels.npy", y)
            X = rng.normal(size=(len(texts), bge_dim)).astype(np.float32)
            p = root / "lmdb" / "bge" / corpus / f"{split}.lmdb"
            p.parent.mkdir(parents=True, exist_ok=True)
            env = lmdb.open(str(p), map_size=1 << 30)
            with env.begin(write=True) as txn:
                txn.put(b"__len__", str(len(texts)).encode())
                txn.put(b"__shape__", str(X.shape).encode())
                for i, row in enumerate(X):
                    txn.put(str(i).encode(), row.tobytes())
                    txn.put(f"__path__{i}".encode(), f"{corpus}/{split}/{i}".encode())
            env.close()


def build_saes(layer: int = 12):
    """Randomly initialised checkpoints in the trainer's run-directory layout."""
    import torch
    from common import arms, sae_load  # noqa: F401  (sae_load puts <repo>/vision on sys.path)
    from models import BatchTopKSAE, DualSAE, TopK, TopKSAE
    log = ("Epoch {e}/50 - Val Loss: 1.0e-03, Val {key}: 1.0e-03, Val Active: 64.0, "
           "Val Binary Purity: 1.0000\n")
    for act in ("topk", "batch_topk"):
        for arm in arms.ARMS:
            for s in SEEDS:
                ref = arms.resolve_arm("B", layer, None, act, arm, s)
                rd = ref.run_dir
                rd.mkdir(parents=True, exist_ok=True)
                torch.manual_seed(s)
                base_cfg = {"act_size": D_MODEL, "dict_size": DICT, "top_k": K, "seed": s,
                            "device": "cpu", "dtype": torch.float32, "input_unit_norm": False,
                            "n_batches_to_dead": 4882}
                if arm == "baseline" and act == "topk":
                    m = TopKSAE(n_latents=DICT, n_inputs=D_MODEL, activation=TopK(k=K),
                                tied=False, normalize=False)
                    tc = {"model_type": "topk", "topk_k": K, "tied_weights": False,
                          "normalize": False, "seed": s}
                    cfg, key = None, "NMSE"
                elif arm == "baseline":
                    cfg = dict(base_cfg, l1_coeff=0.0, top_k_aux=256, aux_penalty=0.03125)
                    m = BatchTopKSAE(cfg)
                    tc = {"model_type": "batch_topk", "topk_k": K, "normalize": False, "seed": s}
                    key = "L2"
                else:
                    cfg = dict(base_cfg, l1_coeff=1e-4, bandwidth=1e-3, poly_hidden=512,
                               poly_layers=1, poly_hidden_dims=None, sparsity_mode=act)
                    m = DualSAE(cfg)
                    tc = {"model_type": "dual", "sparsity_mode": act, "top_k": K,
                          "normalize": False, "span_coef": 0.0 if arm == "dual_g0" else 0.01,
                          "warmup_steps": 1250, "seed": s}
                    key = "L2"
                sd = m.state_dict()
                torch.save({"epoch": 50, "step": 31250, "model_state_dict": sd,
                            "optimizer_state_dict": {}, "config": tc}, rd / "autoencoder_final.pt")
                torch.save({"epoch": 50, "model_cfg": cfg, "model_state_dict": sd,
                            "val_monoscore": 0.5}, rd / "best_val_monoscore_model.pt")
                (rd / "training_config.json").write_text(json.dumps(tc))
                (rd / "train_stdout.log").write_text("".join(log.format(e=e, key=key)
                                                             for e in range(1, 51)))
                (rd / "CELL_DONE.json").write_text(json.dumps({"tag": ref.tag}))


# ============================================================================ stubs
class FakeLLM:
    """Stands in for common.llm_backend.TextLLM (no vLLM, no phi-4)."""

    def __init__(self, **kw):
        self.cache = None
        self.last_info = []

    @staticmethod
    def _rating(prompt: str) -> int:
        if "[smoke-dirset=diffmean]" in prompt:
            return 2
        if "[smoke-dirset=random]" in prompt:
            return 0
        return int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % 3

    def generate_retry_length(self, prompts, max_tokens, retry_max_tokens, temperature=0.0):
        self.last_info = [{} for _ in prompts]
        return [(f"Explanation.\nRating: {self._rating(p)}", "stop", False) for p in prompts]

    def describe(self):
        return {"model_path": "fake-judge"}

    def stats(self):
        return {}

    def close(self):
        pass


class FakeEncoder:
    """Stands in for the bge SentenceTransformer: a fixed random projection of hashed words."""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def encode(self, texts, **kw):
        import numpy as np
        out = np.zeros((len(texts), self.dim), np.float32)
        for i, t in enumerate(texts):
            for w in (t or "").lower().split():
                h = int(hashlib.sha256(w.encode()).hexdigest()[:8], 16)
                out[i, h % self.dim] += 1.0
        return out


def install_stubs():
    import numpy as np
    from common import llm_backend as LB, text_encoder
    from steering import base, concepts, s7_judge
    # Health gates: random SAEs cannot pass them; relax every removal rule for the smoke.
    base.GATE_ACTIVE_MIN = -1.0
    base.GATE_CE_MIN = -1e9
    base.GATE_CE_DUAL_MARGIN = 1e9
    base.GATE_LOSS_RATIO = 1e9
    # Two of the six KL budgets, and a smaller calibration set (the grid search is unchanged).
    base.BUDGETS = (0.05, 0.2)
    # (WindowKL scores in chunks of 8 windows, so the window count stays a multiple of 8.)
    base.CALIB_WINDOWS = 8
    base.N_BISECT = 2
    rng = _rng(3)

    def amazon_split(split):
        n = 40 if split == "train" else 16
        texts = [_sentence(rng, CLASS_WORDS["positive" if i % 2 == 0 else "negative"]) for i in range(2 * n)]
        return texts, np.asarray([5.0 if i % 2 == 0 else 1.0 for i in range(2 * n)])
    concepts._amazon_split = amazon_split
    text_encoder.load_encoder = lambda enc, device: FakeEncoder(enc.dim)
    LB.TextLLM = FakeLLM
    LB.load_axbench_templates = lambda: {
        "concept": "Concept: {concept}\nSentence: {sentence}\nRate.",
        "instruction": "Instruction: {instruction}\nSentence: {sentence}\nRate.",
        "fluency": "Sentence: {sentence}\nRate fluency."}

    def parse(text, min_rating=0.0, max_rating=2.0):
        try:
            v = float((text or "").rsplit("Rating:", 1)[1].strip().split()[0])
        except (IndexError, ValueError):
            return 0.0
        return v if min_rating <= v <= max_rating else 0.0
    LB.parse_axbench_rating = parse
    LB.axbench_rating_status = lambda text, *a, **k: "ok"
    LB.axbench_harmonic_mean = lambda s: 0.0 if any(x == 0 for x in s) else len(s) / sum(1.0 / x for x in s)
    # Tag each judge prompt with its direction family, so the fake judge can tell DiffMean
    # pilot generations from random-direction ones.
    orig = s7_judge._prompts_for_row

    def tagged(row, tpl, concept_only):
        return [(k, c, p + f"\n[smoke-dirset={row.get('dirset')}]")
                for k, c, p in orig(row, tpl, concept_only)]
    s7_judge._prompts_for_row = tagged


# ============================================================================ driver
SMOKE_ARGS = {"s0": ["--d44-batches", "10", "--allow-short", "--ce-batches", "2"],
              "s1": ["--n-train", "12", "--n-test", "6"],
              "s2": ["--max-latents", "8"],
              "s4a": ["--n-prompts-cap", "2"],
              "s6": ["--n-prompts-cap", "2"],
              "s7c": ["--cap-per-class", "30", "--n-amazon", "40"]}


def run_module(mod: str, args):
    m = importlib.import_module(mod)
    old = sys.argv
    sys.argv = [mod] + list(args)
    try:
        rc = m.main()
    finally:
        sys.argv = old
    if rc not in (0, None):
        raise RuntimeError(f"{mod} {args} returned {rc}")


def cascade():
    from steering import base, jobs, plan, s7_judge
    order = ("s0", "s1", "s2", "s4a", "s4j", "s5", "s6", "s7c", "s7j", "s9")
    for wave in range(1, 20):
        for f in (plan._refs, plan._pair, plan._prereg):
            f.cache_clear()
        ran = 0
        for kind in order:
            for i in plan.classify(kind)["todo"]:
                env = dict(jobs.shell_env(kind, i))
                args = env["A16_ARGS"].split() + SMOKE_ARGS.get(kind, [])
                print(f"[smoke] wave {wave} {kind}[{i}] {env['A16_UNIT']}", flush=True)
                run_module(env["A16_MODULE"], args)
                ran += 1
        if not os.path.isfile(base.p1_json()) and s7_judge.unit_done("pilot"):
            run_module("steering.s4_pilot", ["--rule"])
            ran += 1
        print(f"[smoke] wave {wave}: {ran} unit(s)", flush=True)
        if ran == 0:
            return
    raise RuntimeError("cascade did not converge")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default=None, help="default: a fresh temporary directory")
    ap.add_argument("--keep", action="store_true", help="keep the work directory")
    a = ap.parse_args()
    work = Path(a.workdir or tempfile.mkdtemp(prefix="steering_smoke_"))
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    lang, out = work / "lang", work / "out"
    os.environ.update({"LANGDUAL_ROOT": str(lang), "A16_ROOT": str(out), "A16_SMOKE": "1",
                       "A16_LAYERS": "12", "A16_ACTIVATIONS": "topk,batch_topk",
                       "LANG_LM_D_MODEL": str(D_MODEL), "HF_HUB_OFFLINE": "1",
                       "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                       "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "CUDA_VISIBLE_DEVICES": ""})
    sys.path.insert(0, str(HERE))
    import numpy as np
    from steering import base, s3_prompts
    from common import lang_config
    print(f"[smoke] work dir {work}", flush=True)
    vocab = build_lm(Path(lang_config.local_snapshot(lang_config.LM_REPO)), s3_prompts.PROMPTS)
    build_tokens_and_shards(lang, Path(lang_config.local_snapshot(lang_config.LM_REPO)), vocab)
    build_corpora(lang)
    build_saes()
    install_stubs()
    cascade()
    run_module("steering.make_table", [])
    run_module("steering.select_examples", ["--direction", "dual"])
    run_module("steering.select_examples", ["--direction", "baseline"])

    # ------------------------------------------------------------------ checks
    import csv
    bad = []
    p1 = base.read_json(base.p1_json())
    if not p1 or p1.get("primary_budget") in (None, "none"):
        bad.append(f"P1 not decided: {p1 and p1.get('primary_budget')}")
    rows = list(csv.DictReader(open(base.p("s9", "per_seed.csv"))))
    tests = list(csv.DictReader(open(base.p("s9", "tests.csv"))))
    fam = [t for t in tests if t["family"] == "True" and t["q"] not in ("", "nan")]
    arms_seen = {r["arm"] for r in rows}
    if not {"baseline", "dual", "dual_g0"} <= arms_seen:
        bad.append(f"per_seed.csv arms {arms_seen}")
    if not fam:
        bad.append("no confirmatory test reached BH")
    for f in ("steering_table.tex", "steering_table.md", "steering_table.json",
              "examples_dual_wins.json", "examples_baseline_wins.json", "report.md"):
        if not os.path.isfile(base.p("s9", f)):
            bad.append(f"missing s9/{f}")
    tex = open(base.p("s9", "steering_table.tex")).read()
    if "Standard" not in tex or "DualSAE" not in tex:
        bad.append("table columns not labelled from the arm field")
    print(f"[smoke] per_seed rows {len(rows)}, family tests in BH {len(fam)}, P1 "
          f"{p1 and p1.get('primary_budget')}")
    print("SMOKE TEST", "OK" if not bad else f"FAILED: {bad}")
    if not a.keep and not bad and not a.workdir:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
