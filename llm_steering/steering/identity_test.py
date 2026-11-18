"""Identity-hook precondition (D-49): run before trusting ANY steered number.

Checks, per layer (hard failures unless stated):
  1. greedy decoding of the neutral prompts gives IDENTICAL tokens with and without the identity
     hook (the full EditHook masking path with a zero edit), with the KV cache ...
  2. ... and again without the KV cache (full recompute each step);
  3. the raw -> shard -> raw round trip spliced at positions >= 1 on owt test windows stays below
     next-token KL 1e-6 (measured 8.9e-8 on CPU);
  4. the edit mask is exact: an 'add' hook changes the layer output at every real position >= 1
     and at NO pad or position-0 slot, in prefill and in every KV-cached decode step (hidden
     states captured from the layer output);
  informational: cached vs uncached greedy agreement (different GEMM shapes may round
  differently under TF32; not part of the precondition).

CLI:  python -m steering.identity_test --layers 12 [--n-prompts 4] [--device cuda:0]
Writes base.identity_record() (JSON) and exits non-zero on any hard failure.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List

import torch

from steering import base, steer
from common import loss_recovered_hooks as H
from common import lang_config as config


def run(layers: List[int], n_prompts: int, device: str, max_new: int = base.MAX_NEW_TOKENS
        ) -> Dict:
    t0 = time.time()
    lm, tok = steer.load_lm(device)
    prompts_txt = base.check_prompts(require_pin=False)[:n_prompts]
    prompts = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts_txt]
    res: Dict = {"layers": {}, "n_prompts": len(prompts), "max_new": max_new, "passed": True,
                 "torch": torch.__version__}
    for layer in layers:
        nf, _ = steer.norm_factor(layer)
        hook = steer.EditHook(lm, layer, nf)
        r: Dict = {}
        g_plain_kv = steer.generate(lm, prompts, None, None, max_new, use_cache=True, greedy=True)
        with hook.installed():
            hook.configure("identity")
            g_id_kv = steer.generate(lm, prompts, hook, None, max_new, use_cache=True, greedy=True)
            g_id_nc = steer.generate(lm, prompts, hook, None, max_new, use_cache=False, greedy=True)
        g_plain_nc = steer.generate(lm, prompts, None, None, max_new, use_cache=False, greedy=True)
        r["identity_kv_equal"] = bool(torch.equal(g_plain_kv, g_id_kv))
        r["identity_nocache_equal"] = bool(torch.equal(g_plain_nc, g_id_nc))
        r["cache_vs_nocache_token_agreement"] = float((g_plain_kv == g_plain_nc).float().mean())

        # 3. round trip at positions >= 1 on 8 owt test windows.
        win = steer.owt_windows(0, 8).to(device)
        pos = torch.arange(win.shape[1], device=device)[None].expand(win.shape[0], -1)
        with torch.no_grad():
            clean = lm(input_ids=win, position_ids=pos, use_cache=False).logits

            def rt(hidden):
                x = H.from_shard_space(H.to_shard_space(hidden.float(), nf, "divide"), nf, "divide")
                keep = hidden.clone()
                keep[:, 1:] = x[:, 1:].to(hidden.dtype)
                return keep
            with H.splice_activation(hook.module, rt):
                spl = lm(input_ids=win, position_ids=pos, use_cache=False).logits
        r["roundtrip_kl"] = H.next_token_kl(clean, spl)
        r["roundtrip_ok"] = bool(r["roundtrip_kl"] < 1e-6)

        # 4. mask exactness, within ONE run: forward hooks fire in registration order, so a
        #    capture hook registered on layer L BEFORE the EditHook sees the UNEDITED output, and a
        #    forward pre-hook on layer L+1 sees the EDITED one.  Their difference must be non-zero
        #    exactly on real positions >= 1 (prefill) and on every new token (decode steps).
        # residual width of the LM (1024 for pythia-410m; read from the config so the synthetic
        # smoke model can be narrower)
        d = torch.randn(len(prompts), config.LM_D_MODEL, device=device) * 10.0
        before: List[torch.Tensor] = []
        after: List[torch.Tensor] = []
        h_cap = hook.module.register_forward_hook(
            lambda m, a, o: before.append(H._split_output(o)[0].detach()))
        nxt = lm.gpt_neox.layers[layer + 1]
        h_nxt = nxt.register_forward_pre_hook(
            lambda m, a, kw: after.append((a[0] if a else kw["hidden_states"]).detach()),
            with_kwargs=True)
        try:
            with hook.installed():
                hook.configure("add", delta_raw=d)
                steer.generate(lm, prompts, hook, None, 4, use_cache=True, greedy=True)
        finally:
            h_cap.remove()
            h_nxt.remove()
        ids, att = steer.left_pad(prompts, device)
        want = (steer.positions(att) >= 1) & att.bool()          # prefill edit set
        diff0 = (after[0] - before[0]).abs().amax(-1) > 0
        ok_prefill = bool(torch.equal(diff0, want))
        ok_decode = all(bool(((a_ - b_).abs().amax(-1) > 0).all()) and a_.shape[1] == 1
                        for a_, b_ in zip(after[1:], before[1:]))
        ok_mask = ok_prefill and ok_decode and len(after) == len(before) == 4
        r["mask_prefill_exact"], r["mask_decode_all_edited"] = ok_prefill, ok_decode
        r["n_decode_steps_checked"] = len(after) - 1
        r["n_pads_and_sinks_untouched"] = int((~want).sum())
        r["passed"] = bool(r["identity_kv_equal"] and r["identity_nocache_equal"]
                           and r["roundtrip_ok"] and ok_mask)
        res["layers"][str(layer)] = r
        res["passed"] = res["passed"] and r["passed"]
        base.log(f"identity L{layer}: {r}")
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", type=int, nargs="+", default=list(base.LAYERS))
    ap.add_argument("--n-prompts", type=int, default=base.N_PROMPTS_OTHER)
    ap.add_argument("--max-new", type=int, default=base.MAX_NEW_TOKENS)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    res = run(a.layers, a.n_prompts, a.device or base.device_default(), a.max_new)
    base.atomic_write_json(base.identity_record(), res)
    print(f"IDENTITY TEST {'PASSED' if res['passed'] else 'FAILED'} -> {base.identity_record()}")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
