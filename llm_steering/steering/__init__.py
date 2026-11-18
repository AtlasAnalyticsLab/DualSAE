"""LLM steering of pythia-410m-deduped with residual-stream SAEs and DualSAEs.

Protocol: PROTOCOL.md (the D-xx / Cxx / Rx / Px identifiers cited in the code comments are
defined there).  Stages (each a resumable unit with a skip-guard; see jobs.py for the unit-index
contract and plan.py for the completion-aware planner):

  s0   rescore + D-43 health gate per SAE at the final checkpoint (D-44 threshold for BatchTopK);
       gates43.py: primary gate = (a), (c), D-44 (rule (b) a reported diagnostic); the strict
       gate incl. (b) is reported by s9 as a sensitivity analysis
  s1   concept data (D-45) + per-latent AUROC selection (D-46) + DiffMean/random directions
  s2   analytic poly cancellation for every live Dual latent (D-47)
  s3   the neutral prompt file (P2; shipped at data/neutral_prompts.txt, --check validates it)
  s4   arm-free L12 pilot (D-48) -> P1 rule (s4a generate, s4j judge, s4p rule)
  s5   KL calibration per SAE x direction x mode x budget (C29; armfree_L12 owned by s4a)
  s6   generation (D-49): modes (i), (ii'), (iii); DiffMean/random; unsteered null row -- plus
       the judge-free metrics (s7m merged into the same task, D-51)
  s7   s7c bge concept classifier; s7j judging (phi-4, official AxBench templates; D-50; the
       live-matched shards are never judged, D-46)
  s8   200-item hand-label sheet (+ optional InternVL3.5 second judge)
  s9   report (Welch per contrast, BH within the confirmatory family, verdict words, kappa);
       every confirmatory table twice: primary gate (headline) and the strict-gate sensitivity,
       BH separately per version

Post-processing (no GPU): make_table.py (the quantitative steering table from s9/per_seed.csv and
s9/tests.csv) and select_examples.py (the qualitative example pool).

Every output goes under ``$A16_ROOT`` (base.py).
"""
