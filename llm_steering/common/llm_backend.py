"""The in-process phi-4 judge backend (D-20, D-22, D-50).

Engine
  * Local snapshot of microsoft/phi-4 at the pinned revision (PHI4_REVISION; prepare_assets.sh
    downloads exactly that revision).  Never the hub id: runtime is offline.
  * In-process vLLM engine: VLLM_ENABLE_V1_MULTIPROCESSING=0 is set (setdefault) at IMPORT of
    this module, i.e. before the lazy ``import vllm`` -- the V1 multiprocessing EngineCore can
    deadlock at bootstrap (the front-end then waits forever for its READY handshake), while the
    in-process engine is correct for synchronous batched ``LLM.chat``.  Any other value is
    refused.
  * One engine per process, ever (a second vLLM init in one process deadlocks / double-
    allocates): a module-level guard raises on a second construction.  The engine is built
    LAZILY, on the first request that is not served from the cache, so a fully cached resume
    never loads it.
  * At engine build a one-token probe checks that vLLM's prompt-token count equals this
    module's count for the same chat; otherwise the length guard would be invalid and the run
    stops before sending anything.

generate(prompts, max_tokens, temperature, seeds=None, top_p=1.0) -> [(text, finish_reason)]
  * Batched ``llm.chat`` with ONE SamplingParams per prompt (per-request seeds, D-22).
  * T > 0 requires a seed for every item: an unseeded stochastic answer cannot be cached (two
    samples of one prompt would share a key) and would not reproduce on resume.
  * LENGTH GUARD: every rendered chat prompt is tokenized with the pinned tokenizer's chat
    template; if n_prompt + max_tokens > max_model_len the caller's ``retruncate(i, attempt)``
    callback is asked for a shorter prompt (attempt 1..5; callers halve max_chars each time).
    Still too long, or no callback: the item is SKIPPED -- returned as (None,
    'skipped_overlength') and recorded -- and never sent, because one over-length prompt makes
    vLLM raise for the WHOLE batch, so a resumed job would die on the same batch forever.
  * CACHE: JSONL, one file per process in ``cache_dir`` (this process writes only its own file
    ``cache.<task>.jsonl`` and reads every ``cache.*.jsonl`` in the directory).  Key = sha256 of
    (model path, dtype, engine_kwargs, rendered prompt, sampling params, seed).  FIRST RECORD
    WINS, in memory exactly as on reload: a key that already has a record is never overwritten
    or re-appended, so a resumed process serves exactly what the uninterrupted one served (vLLM
    at T=0 is not batch-invariant, so a second generation of the same key can differ).  Each
    record is ONE os.write() of a complete line on an O_APPEND descriptor; fsync every 1000
    records and on close.  Readers skip malformed lines (a torn final line from a killed job); on
    reopening its own file a writer first terminates a torn tail with '\n'.

AxBench judge (module functions, no engine needed)
  parse_axbench_rating(text)      the OFFICIAL AxBench lm_judge parser, executed from the pinned
                                  asset file (not re-typed): DEFAULT_RATING 0.0 on failure or
                                  out-of-range, no clamping (D-50).
  axbench_harmonic_mean(scores)   the official harmonic mean (0 if any score is 0).
  load_axbench_templates()        runpy.run_path of the pinned prompt_templates.py (never
                                  ``import axbench``: its __init__ imports plotting packages).
"""

from __future__ import annotations

import os

# MUST precede any vllm import anywhere in the process (see the module docstring).
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


# MUST precede any vllm import anywhere in the process (see the module docstring).
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import ast  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import runpy  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union  # noqa: E402

from common import lang_config as _lc  # noqa: E402

# The judge: microsoft/phi-4 at a pinned revision, as a local snapshot (prepare_assets.sh puts
# it at <LANGDUAL_ROOT>/raw/models/microsoft__phi-4; ``A16_JUDGE_PATH`` overrides).
PHI4_REPO = "microsoft/phi-4"
PHI4_REVISION = "932b33c0ec9ca189badeb22480721a8de9d0e006"
PHI4_SNAPSHOT = os.environ.get("A16_JUDGE_PATH", _lc.local_snapshot(PHI4_REPO, "model"))

# Official AxBench judge assets (stanfordnlp/axbench @ AXBENCH_COMMIT; prepare_assets.sh fetches
# the two files).  The sha256 pins make a silently changed asset fail loudly instead of changing
# the judge.
AXBENCH_DIR = os.environ.get("A16_AXBENCH_DIR", os.path.join(_lc.RAW_DIR, "axbench"))
AXBENCH_COMMIT = "41c8332543e5a631f9a8c0a9df38799893ace758"
AXBENCH_LM_JUDGE = os.path.join(AXBENCH_DIR, "axbench/evaluators/lm_judge.py")
AXBENCH_TEMPLATES = os.path.join(AXBENCH_DIR, "axbench/evaluators/prompt_templates.py")
AXBENCH_LM_JUDGE_SHA256 = "dd496c1cf755373fb57fc99b1f4555ee2ee52d0f1986b1a2312d065c999094d7"
AXBENCH_TEMPLATES_SHA256 = "9d4eb22324b0e98bd867170073814998794efe9dfdf3cef795ec2000dda46a52"

MAX_RETRUNCATIONS = 5          # re-truncate an over-length prompt (max_chars halved) at most 5x
FSYNC_EVERY = 1000             # fsync every 1,000 cache records
DEFAULT_BATCH = 4096           # prompts per llm.chat call (cache is written after each call)

Prompt = Union[str, List[Dict[str, str]]]
Retruncate = Callable[[int, int], Optional[Prompt]]

_log = logging.getLogger("llm_steering.llm_backend")
_ENGINE_BUILT = False          # one vLLM engine per process (module-level guard)


# =========================================================================== AxBench judge
def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_pin(path: str, expected: str) -> None:
    got = _sha256_file(path)
    if got != expected:
        raise RuntimeError(f"{path}: sha256 {got} != pinned {expected}; the official AxBench "
                           f"asset changed (expected commit {AXBENCH_COMMIT})")


_AXB_CACHE: Dict[str, Any] = {}


def _official_axbench() -> Dict[str, Any]:
    """Execute the OFFICIAL rating logic from lm_judge.py without importing the axbench package.

    The file's AST is parsed and exactly these nodes are compiled into an isolated namespace:
      LMJudgeEvaluator.DEFAULT_RATING, ._get_rating_from_completion,
      ._get_ratings_from_completions, and the harmonic_mean closure of .compute_metrics.
    So the code that runs IS the official code at the pinned commit (sha256-checked), not a
    re-typed copy.  Its ``logger.warning`` calls go to a quiet logger (ERROR level) so millions
    of judge completions do not spam the job log; failure RATES are reported by the app via
    axbench_rating_status().
    """
    if _AXB_CACHE:
        return _AXB_CACHE
    _check_pin(AXBENCH_LM_JUDGE, AXBENCH_LM_JUDGE_SHA256)
    src = open(AXBENCH_LM_JUDGE).read()
    tree = ast.parse(src, filename=AXBENCH_LM_JUDGE)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LMJudgeEvaluator")
    wanted = {"_get_rating_from_completion", "_get_ratings_from_completions"}
    body: List[ast.stmt] = []
    for n in cls.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "DEFAULT_RATING"
                                             for t in n.targets):
            body.append(n)
        elif isinstance(n, ast.FunctionDef) and n.name in wanted:
            body.append(n)
    got = {n.name for n in body if isinstance(n, ast.FunctionDef)}
    if got != wanted or not any(isinstance(n, ast.Assign) for n in body):
        raise RuntimeError(f"lm_judge.py no longer defines {wanted} + DEFAULT_RATING")
    cm = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "compute_metrics")
    hm = [n for n in ast.walk(cm) if isinstance(n, ast.FunctionDef) and n.name == "harmonic_mean"]
    if len(hm) != 1:
        raise RuntimeError("lm_judge.py compute_metrics no longer defines harmonic_mean")
    new_cls = ast.ClassDef(name="_OfficialLMJudgeRating", bases=[], keywords=[], body=body,
                           decorator_list=[])
    mod = ast.Module(body=[new_cls, hm[0]], type_ignores=[])
    ast.fix_missing_locations(mod)
    quiet = logging.getLogger("llm_steering.axbench_official")
    quiet.setLevel(logging.ERROR)
    ns: Dict[str, Any] = {"logger": quiet}
    exec(compile(mod, AXBENCH_LM_JUDGE, "exec"), ns)
    _AXB_CACHE.update(parser=ns["_OfficialLMJudgeRating"](), harmonic_mean=ns["harmonic_mean"],
                      default=ns["_OfficialLMJudgeRating"].DEFAULT_RATING)
    return _AXB_CACHE


def parse_axbench_rating(text: Optional[str], min_rating: float = 0.0,
                         max_rating: float = 2.0) -> float:
    """Official AxBench rating of one completion: ``_get_ratings_from_completions([text])[0]``
    (last 'Rating:' split, first line, brackets/quotes/asterisks stripped, float; unparsable or
    outside [min_rating, max_rating] -> DEFAULT_RATING 0.0; no clamping).  A skipped item
    (text None) gets the same DEFAULT_RATING, and should be counted as a failure by the app."""
    off = _official_axbench()
    return float(off["parser"]._get_ratings_from_completions([text if text is not None else ""],
                                                              min_rating, max_rating)[0])


def axbench_rating_status(text: Optional[str], min_rating: float = 0.0,
                          max_rating: float = 2.0) -> str:
    """Why a rating is what it is, for the D-50 parse-failure rates: 'ok', 'no_marker' (no
    'Rating:'), 'unparsable' (float() failed) or 'out_of_range'.  Uses the official single-
    completion function, so 'ok' <=> the official rating is a parsed value, not the default."""
    if text is None or "Rating:" not in text:
        return "no_marker"
    try:
        r = _official_axbench()["parser"]._get_rating_from_completion(text)
    except Exception:
        return "unparsable"
    return "ok" if (r is not None and min_rating <= r <= max_rating) else "out_of_range"


def axbench_harmonic_mean(scores: Sequence[float]) -> float:
    """Official AxBench aggregate (lm_judge.py compute_metrics.harmonic_mean): 0 if any score is
    0, else len/sum(1/s).  Scores = (concept, instruction, fluency) ratings, each 0-2."""
    return float(_official_axbench()["harmonic_mean"](list(scores)))


def load_axbench_templates() -> Dict[str, str]:
    """The three official judge templates, used verbatim (D-50).  Keys 'concept', 'instruction',
    'fluency' plus the official constant names."""
    _check_pin(AXBENCH_TEMPLATES, AXBENCH_TEMPLATES_SHA256)
    ns = runpy.run_path(AXBENCH_TEMPLATES)
    out = {k: v for k, v in ns.items() if k.startswith("UNIDIRECTIONAL_PAIRWISE_EVALUATION_")}
    out["concept"] = ns["UNIDIRECTIONAL_PAIRWISE_EVALUATION_CONCEPT_RELEVANCE_TEMPLATE"]
    out["instruction"] = ns["UNIDIRECTIONAL_PAIRWISE_EVALUATION_INSTRUCTION_RELEVANCE_TEMPLATE"]
    out["fluency"] = ns["UNIDIRECTIONAL_PAIRWISE_EVALUATION_FLUENCY_TEMPLATE"]
    return out


# =========================================================================== GPU helper
def suggest_gpu_memory_utilization(frac_of_free: float = 0.9, cap: Optional[float] = None,
                                   physical_index: Optional[int] = None) -> float:
    """gpu_memory_utilization = 0.9 * free / total on the chosen GPU, for a shared GPU
    (vLLM refuses to start unless free >= util * total).  Queried with nvidia-smi so that no CUDA
    context is created in this process before vLLM's.  ``physical_index`` defaults to the single
    GPU named in CUDA_VISIBLE_DEVICES.  ``cap`` bounds it."""
    if physical_index is None:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
        physical_index = int(cvd) if cvd.isdigit() else 0
    out = subprocess.run(["nvidia-smi", f"--id={physical_index}",
                          "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout.strip()
    free, total = (float(v) for v in out.split(","))
    util = frac_of_free * free / total
    return round(min(util, cap) if cap is not None else util, 3)


# =========================================================================== cache
class _JsonlCache:
    """Per-task JSONL cache.  In-memory dict key -> record, filled from every
    ``cache.*.jsonl`` in the directory at construction; appends go ONLY to this task's own file.

    Only ``cache.*.jsonl`` is read: an app may keep its own JSONL outputs next to the cache, and
    any such file with a 'k' field would otherwise be ingested as cache records."""

    def __init__(self, cache_dir: str, task: str):
        cache_dir = os.path.abspath(cache_dir)
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+", task):
            raise ValueError(f"cache task name {task!r} must be a plain file-name token")
        os.makedirs(cache_dir, exist_ok=True)
        self.dir = cache_dir
        self.path = os.path.join(cache_dir, f"cache.{task}.jsonl")
        self.mem: Dict[str, Dict[str, Any]] = {}
        self.n_loaded = 0
        self.n_malformed = 0
        self.n_files = 0
        for p in sorted(glob.glob(os.path.join(cache_dir, "cache.*.jsonl"))):
            self._load(p)
        self._fd: Optional[int] = None
        self._since_fsync = 0

    def _load(self, path: str) -> None:
        self.n_files += 1
        with open(path, "rb") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = rec["k"]
                except (ValueError, KeyError, TypeError):
                    # A torn record (killed job) or a fragment terminated by the repair below.
                    self.n_malformed += 1
                    continue
                if key not in self.mem:          # first record wins (identical by construction)
                    self.mem[key] = rec
                    self.n_loaded += 1

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self.mem.get(key)

    def _open(self) -> None:
        if self._fd is not None:
            return
        self._fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o640)
        # Torn-tail repair: if our own file ends mid-record, terminate that fragment so the next
        # record starts on a fresh line (the fragment is then skipped by every reader).
        size = os.fstat(self._fd).st_size
        if size:
            with open(self.path, "rb") as fh:
                fh.seek(size - 1)
                if fh.read(1) != b"\n":
                    os.write(self._fd, b"\n")

    def put(self, key: str, rec: Dict[str, Any]) -> bool:
        """Append a record unless the key already has one (first record wins, in memory AND on
        disk, so this process and any later reload agree).  Returns True iff it was written."""
        if key in self.mem:
            return False
        self._open()
        rec = dict(rec, k=key)
        data = (json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        # ONE write() per record.  A short write (disk full / signal) is completed in a loop; if
        # the process dies in between, the reader skips the torn line.
        view = memoryview(data)
        while view:
            n = os.write(self._fd, view)
            view = view[n:]
        self.mem[key] = rec
        self._since_fsync += 1
        if self._since_fsync >= FSYNC_EVERY:
            os.fsync(self._fd)
            self._since_fsync = 0
        return True

    def flush(self) -> None:
        """Per-batch flush.  Records are written with os.write (no user-space buffer), so they are
        already in the kernel; nothing to do beyond the periodic fsync.  Kept so the per-batch
        contract is explicit at the call site."""

    def close(self) -> None:
        if self._fd is not None:
            os.fsync(self._fd)
            os.close(self._fd)
            self._fd = None


def _default_task_name() -> str:
    """Unique per process: array task + job + host + pid, so no two live processes can ever
    append to the same cache file (the design's 'one cache file per array task')."""
    return (f"t{os.environ.get('SLURM_ARRAY_TASK_ID', 'x')}"
            f".j{os.environ.get('SLURM_ARRAY_JOB_ID') or os.environ.get('SLURM_JOB_ID') or 'local'}"
            f".{socket.gethostname().split('.')[0]}.p{os.getpid()}")


# =========================================================================== the backend
class TextLLM:
    """One phi-4 engine per process with the length guard and the per-task cache.

    Parameters
      model_path              PHI4_SNAPSHOT (D-20); any other path is allowed but recorded.
      seed                    engine seed (LLM(seed=...)); per-request seeds come via generate.
      max_model_len           context length (the steering judge uses 4096).
      gpu_memory_utilization  vLLM fraction; see suggest_gpu_memory_utilization.
      enable_prefix_caching   default on (many judge prompts share the template preamble).
      cache_dir / cache_task  JSONL cache (None disables caching); task defaults to a unique name.
      batch_size              prompts per llm.chat call; the cache is persisted after each call.
      engine_kwargs           extra LLM(...) keyword arguments (recorded in describe()).
    """

    def __init__(self, *, model_path: str = PHI4_SNAPSHOT, seed: int = 0, max_model_len: int = 4096,
                 gpu_memory_utilization: float = 0.9, enable_prefix_caching: bool = True,
                 dtype: str = "auto", cache_dir: Optional[str] = None,
                 cache_task: Optional[str] = None, batch_size: int = DEFAULT_BATCH,
                 engine_kwargs: Optional[Dict[str, Any]] = None):
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"model path {model_path} not found (offline: pass a local "
                                    f"snapshot path, never a hub id)")
        self.model_path = model_path
        self.seed = int(seed)
        self.max_model_len = int(max_model_len)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.enable_prefix_caching = bool(enable_prefix_caching)
        self.dtype = dtype
        self.batch_size = int(batch_size)
        self.engine_kwargs = dict(engine_kwargs or {})
        self._llm = None
        self._tok = None
        self._stats: Dict[str, Any] = dict(
            n_requested=0, n_cached=0, n_sent=0, n_dedup=0, n_skipped_overlength=0,
            n_retruncated=0, n_retruncate_calls=0, n_len_mismatch=0, n_finish_length=0,
            n_retry=0, n_retry_replaced=0, n_retry_skipped=0, n_cache_kept_first=0,
            engine_load_s=None, skipped=[])
        self.cache = _JsonlCache(cache_dir, cache_task or _default_task_name()) if cache_dir else None
        self.last_info: List[Dict[str, Any]] = []
        # The exact messages each item of the last generate() call was answered on (after any
        # re-truncation), aligned with its prompts; generate_retry_length re-asks THESE.
        self._last_messages: List[List[Dict[str, str]]] = []
        self._call = 0

    # ------------------------------------------------------------------ tokenizer / engine
    @property
    def tokenizer(self):
        """The pinned snapshot's tokenizer (local files only).  Same files vLLM loads."""
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        return self._tok

    def _engine(self):
        global _ENGINE_BUILT
        if self._llm is not None:
            return self._llm
        if _ENGINE_BUILT:
            raise RuntimeError("a vLLM engine was already built in this process; one engine per "
                               "process")
        if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
            raise RuntimeError("VLLM_ENABLE_V1_MULTIPROCESSING must be '0' (in-process engine; the "
                               "multiprocessing EngineCore can deadlock at bootstrap)")
        t0 = time.time()
        from vllm import LLM
        _ENGINE_BUILT = True
        self._llm = LLM(model=self.model_path, tokenizer=self.model_path, dtype=self.dtype,
                        seed=self.seed, max_model_len=self.max_model_len,
                        gpu_memory_utilization=self.gpu_memory_utilization,
                        enable_prefix_caching=self.enable_prefix_caching, **self.engine_kwargs)
        self._stats["engine_load_s"] = round(time.time() - t0, 1)
        self._probe_token_count()
        return self._llm

    def _probe_token_count(self) -> None:
        """The length guard counts tokens with the tokenizer; vLLM counts its own.  Verify once,
        on a real chat, that the two agree (else the guard could pass a prompt vLLM rejects)."""
        from vllm import SamplingParams
        msgs = [{"role": "user", "content": "Length-guard probe: reply with OK."}]
        mine = self._count([self._render(msgs)])[0]
        out = self._llm.chat([msgs], [SamplingParams(max_tokens=1, temperature=0.0)],
                             use_tqdm=False)
        theirs = len(out[0].prompt_token_ids)
        if mine != theirs:
            raise RuntimeError(f"token-count mismatch between the length guard ({mine}) and vLLM "
                               f"({theirs}); refusing to run with an invalid guard")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _to_messages(p: Prompt) -> List[Dict[str, str]]:
        if isinstance(p, str):
            return [{"role": "user", "content": p}]
        if isinstance(p, list) and p and all(isinstance(m, dict) and "role" in m and "content" in m
                                             for m in p):
            return p
        raise TypeError("a prompt is a str (one user turn) or a list of {'role','content'} dicts")

    def _render(self, msgs: List[Dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _count(self, rendered: List[str]) -> List[int]:
        if not rendered:
            return []
        enc = self.tokenizer(rendered, add_special_tokens=False)["input_ids"]
        return [len(e) for e in enc]

    def _key(self, rendered: str, sp: Dict[str, Any]) -> str:
        """Cache key.  Everything that can change the ANSWER is in it: model path, dtype and the
        extra engine kwargs (e.g. quantization; json with sorted keys, non-JSON values via
        str) besides the prompt and sampling parameters.  Deliberately NOT in it: max_model_len,
        gpu_memory_utilization and prefix caching (capacity knobs; they do not change a
        completion that fits) and the engine seed (every T>0 request carries its own seed)."""
        blob = json.dumps({"model": self.model_path, "dtype": str(self.dtype),
                           "engine_kwargs": self.engine_kwargs, "prompt": rendered,
                           "sampling": sp, "seed": sp.get("seed")},
                          sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------ main entry
    def generate(self, prompts: Sequence[Prompt], max_tokens: int, temperature: float = 0.0,
                 seeds: Optional[Sequence[Optional[int]]] = None, top_p: float = 1.0, *,
                 retruncate: Optional[Retruncate] = None, stop: Optional[Sequence[str]] = None,
                 use_cache: bool = True, use_tqdm: bool = False
                 ) -> List[Tuple[Optional[str], str]]:
        """Generate one completion per prompt; returns [(text, finish_reason)] aligned with
        ``prompts``.  Skipped (over-length) items are (None, 'skipped_overlength').
        Per-item details of this call are left in ``self.last_info`` (status 'cached' | 'sent' |
        'dup' | 'skipped', n_prompt, n_retruncations, n_prompt_vllm).

        ``use_cache=False`` bypasses the cache LOOKUP only (e.g. to re-sample on purpose): the
        fresh answers are returned, and stored only for keys that have no record yet.  A key's
        FIRST record always wins, in this process and after any reload, so a later cached call
        (and a resumed job) serves the same answer as an uninterrupted run would."""
        n = len(prompts)
        self._call += 1
        if seeds is not None and len(seeds) != n:
            raise ValueError(f"{len(seeds)} seeds for {n} prompts")
        seeds_l: List[Optional[int]] = list(seeds) if seeds is not None else [None] * n
        if temperature > 0 and any(s is None for s in seeds_l):
            raise ValueError("temperature > 0 needs a per-request seed for EVERY item (D-22): "
                             "an unseeded sample neither reproduces nor caches")
        if int(max_tokens) >= self.max_model_len:
            raise ValueError(f"max_tokens {max_tokens} >= max_model_len {self.max_model_len}")
        stop_l = list(stop) if stop else None
        self._stats["n_requested"] += n

        def sp_of(i: int) -> Dict[str, Any]:
            return {"max_tokens": int(max_tokens), "temperature": float(temperature),
                    "top_p": float(top_p), "seed": None if seeds_l[i] is None else int(seeds_l[i]),
                    "stop": stop_l}

        results: List[Tuple[Optional[str], str]] = [(None, "")] * n
        info: List[Dict[str, Any]] = [{} for _ in range(n)]
        msgs = [self._to_messages(p) for p in prompts]
        rendered = [self._render(m) for m in msgs]
        keys = [self._key(r, sp_of(i)) for i, r in enumerate(rendered)]

        # 1. Cache lookup on the ORIGINAL prompt (a hit means it fit when it was sent).
        todo: List[int] = []
        for i in range(n):
            rec = self.cache.get(keys[i]) if (self.cache and use_cache) else None
            if rec is not None:
                results[i] = (rec["text"], rec["finish_reason"])
                info[i] = {"status": "cached", "n_prompt": rec.get("n_prompt"), "n_retruncations": 0}
                self._stats["n_cached"] += 1
            else:
                todo.append(i)

        # 2. Length guard (tokenize only the misses), with the caller's re-truncation callback.
        counts = dict(zip(todo, self._count([rendered[i] for i in todo])))
        send: List[int] = []
        for i in todo:
            npr, attempts = counts[i], 0
            while npr + int(max_tokens) > self.max_model_len and attempts < MAX_RETRUNCATIONS \
                    and retruncate is not None:
                attempts += 1
                self._stats["n_retruncate_calls"] += 1
                newp = retruncate(i, attempts)
                if newp is None:
                    break
                msgs[i] = self._to_messages(newp)
                rendered[i] = self._render(msgs[i])
                npr = self._count([rendered[i]])[0]
            if npr + int(max_tokens) > self.max_model_len:
                results[i] = (None, "skipped_overlength")
                info[i] = {"status": "skipped", "n_prompt": npr, "n_retruncations": attempts}
                self._stats["n_skipped_overlength"] += 1
                self._stats["skipped"].append({"call": self._call, "index": i, "n_prompt": npr,
                                               "max_tokens": int(max_tokens), "attempts": attempts})
                continue
            if attempts:
                self._stats["n_retruncated"] += 1
                keys[i] = self._key(rendered[i], sp_of(i))
                rec = self.cache.get(keys[i]) if (self.cache and use_cache) else None
                if rec is not None:     # the truncated version was answered in an earlier run
                    results[i] = (rec["text"], rec["finish_reason"])
                    info[i] = {"status": "cached", "n_prompt": npr, "n_retruncations": attempts}
                    self._stats["n_cached"] += 1
                    continue
            counts[i] = npr
            info[i] = {"status": "sent", "n_prompt": npr, "n_retruncations": attempts}
            send.append(i)

        # 3. De-duplicate identical requests inside the call (same key -> one generation).
        first_of: Dict[str, int] = {}
        uniq: List[int] = []
        for i in send:
            if keys[i] in first_of:
                info[i]["status"] = "dup"
                self._stats["n_dedup"] += 1
            else:
                first_of[keys[i]] = i
                uniq.append(i)

        # 4. Generate in batches; persist each batch before starting the next.
        if uniq:
            from vllm import SamplingParams
            llm = self._engine()
            for s in range(0, len(uniq), self.batch_size):
                chunk = uniq[s:s + self.batch_size]
                sps = []
                for i in chunk:
                    d = sp_of(i)
                    sps.append(SamplingParams(max_tokens=d["max_tokens"], temperature=d["temperature"],
                                              top_p=d["top_p"], seed=d["seed"], stop=d["stop"]))
                outs = llm.chat([msgs[i] for i in chunk], sps, use_tqdm=use_tqdm)
                self._stats["n_sent"] += len(chunk)
                for i, o in zip(chunk, outs):
                    c = o.outputs[0]
                    text, fr = c.text, str(c.finish_reason)
                    results[i] = (text, fr)
                    if len(o.prompt_token_ids) != counts[i]:
                        self._stats["n_len_mismatch"] += 1
                    if fr == "length":
                        self._stats["n_finish_length"] += 1
                    info[i]["n_prompt_vllm"] = len(o.prompt_token_ids)
                    if self.cache is not None:
                        wrote = self.cache.put(keys[i], {
                            "text": text, "finish_reason": fr, "n_prompt": counts[i],
                            "n_out": len(c.token_ids), "sampling": sp_of(i),
                            "model": os.path.basename(self.model_path.rstrip("/")),
                            "t": round(time.time(), 1)})
                        if not wrote:
                            # Only reachable with use_cache=False: the fresh answer is returned
                            # to the caller, the stored first record stays authoritative.
                            self._stats["n_cache_kept_first"] += 1
                if self.cache is not None:
                    self.cache.flush()
        for i in send:
            if info[i]["status"] == "dup":
                results[i] = results[first_of[keys[i]]]
        self.last_info = info
        self._last_messages = msgs
        return results

    def generate_retry_length(self, prompts: Sequence[Prompt], max_tokens: int,
                              retry_max_tokens: int, temperature: float = 0.0,
                              seeds: Optional[Sequence[Optional[int]]] = None, top_p: float = 1.0,
                              **kw) -> List[Tuple[Optional[str], str, bool]]:
        """D-50 rule: a completion with finish_reason == 'length' is re-asked ONCE at
        ``retry_max_tokens``, with the same seed and EXACTLY the prompt the first ask was
        answered on (after any re-truncation the first ask needed).

        Returns [(text, finish_reason, retried)], retried=True iff the retry's answer replaced
        the first one.  Per item, ``self.last_info[i]`` carries ``retry`` ('none' | 'replaced' |
        'skipped') and ``retry_prompt_changed`` (always False: the retry never re-truncates).

        Why no re-truncation on the retry: the rule re-asks the SAME question with more room.
        Re-truncating would judge a shorter document than the first ask (a different question),
        and would do so only for long answers -- an arm-dependent change of the input.
        So a first prompt that fits at ``max_tokens`` but not at ``retry_max_tokens`` (e.g.
        2560-3328 prompt tokens at max_model_len 4096 with 768 -> 1536) is NOT re-asked: its
        first, length-truncated completion is KEPT (finish_reason 'length', retried False,
        retry 'skipped') instead of being thrown away for a skipped item.  Apps count these per
        arm from ``stats()['n_retry_skipped']`` or last_info; the official parsers then see the
        truncated text exactly as they would without a retry rule."""
        first = self.generate(prompts, max_tokens, temperature, seeds, top_p, **kw)
        info1 = [dict(d, retry="none", retry_prompt_changed=False) for d in self.last_info]
        sent_msgs = list(self._last_messages)
        redo = [i for i, (_, fr) in enumerate(first) if fr == "length"]
        out = [(t, fr, False) for t, fr in first]
        if redo:
            self._stats["n_retry"] += len(redo)
            sub_seeds = None if seeds is None else [seeds[i] for i in redo]
            kw2 = {k: v for k, v in kw.items() if k != "retruncate"}      # never re-truncate
            second = self.generate([sent_msgs[i] for i in redo], retry_max_tokens, temperature,
                                   sub_seeds, top_p, retruncate=None, **kw2)
            info2 = self.last_info
            for j, i in enumerate(redo):
                if second[j][0] is None and second[j][1] == "skipped_overlength":
                    # The retry does not fit: keep the first completion (never lose an answer).
                    self._stats["n_retry_skipped"] += 1
                    info1[i] = dict(info1[i], retry="skipped", retry_n_prompt=info2[j].get("n_prompt"))
                    continue
                self._stats["n_retry_replaced"] += 1
                out[i] = (second[j][0], second[j][1], True)
                info1[i] = dict(info2[j], retry="replaced", retry_prompt_changed=False,
                                first_n_prompt=info1[i].get("n_prompt"))
            # The skipped-overlength entries the retry recorded are NOT skipped items of this
            # call (the first completion was kept): take them back out of the counters so
            # n_skipped_overlength keeps meaning "items that got no answer".
            n_back = sum(1 for j in range(len(redo)) if second[j][1] == "skipped_overlength")
            if n_back:
                self._stats["n_skipped_overlength"] -= n_back
                self._stats["skipped"] = [s for s in self._stats["skipped"]
                                          if s["call"] != self._call]
        self.last_info = info1
        self._last_messages = sent_msgs
        return out

    # ------------------------------------------------------------------ reporting
    def stats(self) -> Dict[str, Any]:
        """Cumulative counters of this backend (the skipped list is capped at 1000 entries)."""
        s = dict(self._stats)
        s["skipped"] = list(self._stats["skipped"][:1000])
        if self.cache is not None:
            s.update(cache_path=self.cache.path, cache_records_loaded=self.cache.n_loaded,
                     cache_files_read=self.cache.n_files, cache_malformed_lines=self.cache.n_malformed)
        return s

    def describe(self) -> Dict[str, Any]:
        """Provenance record for every output file (the judge path is recorded in every output)."""
        return {"model_path": self.model_path, "seed": self.seed, "max_model_len": self.max_model_len,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "enable_prefix_caching": self.enable_prefix_caching, "dtype": self.dtype,
                "engine_kwargs": self.engine_kwargs,
                "VLLM_ENABLE_V1_MULTIPROCESSING": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")}

    def close(self) -> None:
        if self.cache is not None:
            self.cache.close()

