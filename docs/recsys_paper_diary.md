# Recsys paper (WWW) — working diary

Full plan: `C:\Users\Julia\.claude\plans\wild-munching-stonebraker.md` (approved 2026-08-22).
Read that first for the why; this file tracks the where-are-we-now.

---

## CURRENT STATUS / NEXT STEP
*(this block is overwritten each session — always current, read this first)*

**2026-09-17: caught and fixed a real bug - a self-sustaining fork-and-
crash loop that burned ~20h of GPU time on 3 of the 4 parallel seed runs
after they'd already finished legitimately. No training data was lost.**

**Root cause:** none of the `run_*.py` driver scripts (the small
per-experiment launchers like `run_amazon_beauty_frozenproj_scaletest_
3000x2500_seed1.py`) guarded their top-level `sweep.main(...)` call with
`if __name__ == "__main__":`. This had been harmless for years because
nothing in the pipeline used real multiprocessing - until this session's
`diagnostics_workers=4` (added 2026-09-15 for the incremental+parallel
`analyze_run` fix) got threaded into these scripts for the first time.
Windows' multiprocessing `spawn` start method re-imports the top-level
script inside every worker process it creates; with no `__main__` guard,
that re-import re-executes `sweep.main(...)` too - so each "diagnostics
worker" was actually re-running the ENTIRE ~10-12h bilevel training
pipeline from scratch instead of the lightweight per-snapshot diagnostics
task it was supposed to do. Most such workers died quickly (GPU/CPU
contention from 4+ processes fighting over one GPU), and
`multiprocessing.Pool` silently replaces dead workers by default - so the
whole thing repeated automatically, forever, needing no external trigger.

**How it was caught:** a routine "как дела?" progress check showed the
outer-step counter jumping backward and forward across log lines
(`[Outer 42/100]` then `[Outer 6/100]`) - the tell that more than one
process was writing to the same log file. `Get-CimInstance Win32_Process`
confirmed real `--multiprocessing-fork` children under each of the 3
training parents' PIDs, and the parents' CPU time had gone completely flat
(one sample-pair showed 48723.1875 -> 48723.65625 across several hours -
essentially zero new work), meaning each parent was permanently blocked
in `Pool.imap_unordered()` waiting for results that would never arrive.
Counting `"[Save] Начальная D_input сохранена"` occurrences (printed once
per fresh `train()` invocation) in each polluted log gave 45-74 - i.e.
45-74 separate full pipeline re-executions per job, not just 4.

**Recovery:** killed the 3 hung parent process trees (`taskkill /T /F`) -
GPU dropped from ~15GB/93% util back to ~1GB/1% immediately, confirming
the runaway work stopped. Checked disk state before assuming anything was
lost: all 3 affected runs (1000u/1200i seed2, 3000u/2500i seed1, 3000u/2500i
seed2) had already written a complete `history.npz`, `ncf_model_final.pt`,
and - critically - `results_*.json` BEFORE getting stuck (the hang is in
`run_one()`'s *post*-training `analyze_run()` call, strictly after
`run_experiment.main()` already returned and saved results) - so real,
valid results were recovered with zero retraining:

| scale | seed | test HR@10 | test NDCG@10 |
|---|---|---|---|
| 1000u/1200i | 0 | 0.2221 | 0.1106 |
| 1000u/1200i | 2 | 0.2272 | 0.1151 |
| 3000u/2500i | 0 | 0.3119 | 0.1612 |
| 3000u/2500i | 1 | 0.3072 | 0.1566 |
| 3000u/2500i | 2 | 0.3052 | 0.1561 |

Seed-to-seed spread is small at both scales (reassuring for the "quality
rises with scale" finding - not an artifact of one lucky seed). The 4th
run (1000u/1200i seed1) crashed separately, mid-training, on an unrelated
`MemoryError` during a `np.savez_compressed` checkpoint write (likely
system memory pressure from all 4 parallel jobs at once, not the fork
bug) - lost 299/300 epochs' worth of progress since no `history.npz`
had been written yet; relaunched clean from scratch (solo, no contention
this time) after the fix.

**Fix:** added `if __name__ == "__main__":` around the top-level call in
all 12 `run_*.py` driver scripts in `my_verification/` (commit `bff1207`),
not just the 4 that triggered this - the same landmine was live in every
one of them, just unfired because they'd never been run with
`diagnostics_workers>1` before.

**Rebuilt `hyperbolicity_scaling_comparison.png` (2026-09-17) with GINCF,
Poincare, and the fair Euclidean baseline all computed at all 3 scales
under equal conditions** (`full_hyperbolicity_table_scaling.py`, now
extended to take multiple seeds per scale plus per-scale Poincare/
Euclidean rows via the same `diagnostics_for_D` pass). New finding, clearly
visible on the right panel (HR@10 vs scale, one line per geometry):
**GINCF's margin over the Euclidean baseline narrows as scale grows** -
300u/800i: ~0.19 vs ~0.12 (GINCF well ahead); 1000u/1200i: ~0.22-0.23 vs
~0.20 (narrower); 3000u/2500i: ~0.31 vs ~0.29 (narrowest yet). GINCF still
wins at every scale tested so far, but the gap shrinking is worth watching
as a possible ceiling effect - not yet enough scale points to know if it
keeps shrinking, flattens, or reverses. Poincare stays worst at every
scale (consistent with earlier findings) but also improves with scale
like the other two. Left panel confirms hyperbolicity (delta_rel) still
shows no clean relationship to quality at any scale, for any geometry.

**1000u/1200i seed1 needed 3 attempts before landing clean** - first
`MemoryError` mid-checkpoint (RAM pressure from 4 parallel jobs), second
`torch.AcceleratorError: CUDA out of memory` ~1 minute after an unrelated
process on this shared machine (`ArcticCompendium/.venv - analysis.
pixel_maps_daily --workers 12`) started - almost certainly external
contention, not a pipeline bug. Third attempt (solo, no contention) ran
clean end to end in 15.96h, INCLUDING the automatic post-training
`analyze_run` diagnostics sweep (n_workers=4) completing correctly this
time - confirms the `__main__`-guard fix actually resolved the fork-bomb,
not just papered over one instance of it. Result: HR@10=0.2140,
NDCG@10=0.1089. Worth remembering: this machine is shared, and other
users' jobs can kill a training run with no code-level cause - if this
keeps recurring, basic OOM-retry/checkpoint-resume in
`GradientIsomapCF.train()` would be worth the engineering lift, but two
isolated incidents so far isn't yet a clear pattern.

**Final 3-seed picture at every scale (`full_hyperbolicity_table_scaling.py`,
`hyperbolicity_scaling_comparison.png`, rebuilt 2026-09-18):**

| scale | GINCF seeds (HR@10) | GINCF mean | Poincare | Euclidean (fair) |
|---|---|---|---|---|
| 300u/800i | 0.187, 0.190, 0.207 | 0.195 | 0.100 | 0.123 |
| 1000u/1200i | 0.214, 0.222, 0.227 | 0.221 | 0.116 | 0.205 |
| 3000u/2500i | 0.305, 0.307, 0.312 | 0.308 | 0.172 | 0.292 |

The narrowing-gap finding holds up with full seed coverage, not just the
earlier partial picture: GINCF's margin over Euclidean is ~0.07 at
300u/800i, narrows to ~0.02 at both larger scales and stays there (not a
monotonically-shrinking-to-zero trend past 1000u/1200i, at least not yet -
3000u/2500i's gap is about the same size as 1000u/1200i's, not smaller).
GINCF's own seed-to-seed spread stays tight at every scale (a good sign -
the "equal conditions" comparison isn't being won/lost on noise). Poincare
stays worst at every scale, improving in absolute terms with scale like
everything else but never catching up. Hyperbolicity (delta_rel) still
shows no clean relationship to quality anywhere on this plot.

**Open follow-ups, not yet done:**
- The narrowing-then-plateauing gap pattern is only 3 scale points - not
  enough to know if it keeps flat, or would keep narrowing/reverse with a
  4th, larger scale point. Not yet statistically tested either (means
  from 2-3 seeds each, no formal comparison).
- Poincare/Euclidean baselines at the 2 larger scales are still n=1 each
  (only GINCF has seed repeats) - same standing caution as everywhere
  else in this investigation.
- The narrowing-gap pattern above is worth a dedicated look once more
  scale points or seeds exist - not yet statistically tested, just visible
  on the plot.

Running (all with `OMP_NUM_THREADS=3`/`MKL_NUM_THREADS=3` to avoid CPU
thread oversubscription across 8 concurrent processes; GPU has headroom -
CUDA does the heavy compute, confirmed via `device=cuda` in every script):
- `run_amazon_beauty_frozenproj_scaletest_{1000x1200,3000x2500}_seed{1,2}.py`
  (new driver scripts, seed=1/seed=2, otherwise identical config to the
  existing seed=0 runs) - the 4 expensive ones, ~10-12h each serially, so
  parallelized specifically to compress this from ~45h sequential.
- `poincare_baseline.py --tag amazon_beauty_{1000x1200,3000x2500}` and
  `save_euclidean_baseline_geometry.py --dropout 0.2 --tag
  amazon_beauty_{1000x1200,3000x2500}_dropout02fair` - much cheaper
  (single NCF training pass, no bilevel loop), should land within 1-2h.

**Hit and fixed a launch bug**: all 4 GINCF jobs crashed immediately on
`UnicodeEncodeError` from a Cyrillic print in `GradientIsomapCF_log.py`'s
`train()` (`[Save] Начальная D_input сохранена → ...`) - stdout redirected
to a file without a console attached defaults to the system codepage
(cp1251), not UTF-8. This is the same class of issue as the known
"PYTHONIOENCODING=utf-8 needed for Cyrillic prints" gotcha (see Windows
shell gotchas memory) but had not actually bitten a background run before
now. Fixed by adding `PYTHONIOENCODING=utf-8` to the launch command;
relaunched cleanly, confirmed all 4 passed the crash point into real
training. GPU at ~91% VRAM (14.9/16.3GB) and 93% util, CPU ~51% - healthy,
no more headroom for additional concurrent jobs though.

**Once these land:** extend `full_hyperbolicity_table_scaling.py` and
`plot_hyperbolicity_scaling.py` to include the new seeds/baselines - 3
seeds per scale (mean+std, matching 300u/800i's presentation) instead of
n=1, and Poincare/Euclidean reference points recomputed per-scale instead
of reusing the 300u/800i-only ones.

---

**2026-09-15: 3000u/2500i result landed (best quality across all scales
tested), machine rebooted mid-diagnostics but training results survived,
diagnostics recomputation made crash-resilient and parallelized.**

**3000u/2500i result:** HR@10=0.3119, NDCG@10=0.1612, 100 outer epochs,
best absolute quality seen at any scale tested so far (beats both
300u/800i and 1000u/1200i). train_loss dropped from 0.366 (first-10 mean)
to 0.311 (last-10 mean), val_hr from 0.162 to 0.205 (first-10 vs last-10
mean), best val_hr=0.314 at epoch 87/100 (near the end, not mid-run like
1000u/1200i). 4/100 outer steps triggered the bad-gradient guard (handled,
not fatal). **Still n=1 at this scale** - same standing caution as
1000u/1200i.

A machine reboot hit partway through the post-hoc geometry-diagnostics
pass for this run (`geometry_diagnostics.csv`, the ORC/persistent-homology/
spectral table - separate from and irrelevant to the HR/NDCG numbers
above, which are saved live during training and were untouched by the
reboot). ~8h of diagnostics computation was lost because `analyze_run`
only wrote the CSV once at the very end. Fixed: `analyze_run` now writes
`out_csv` incrementally after every processed snapshot, and optionally
parallelizes across snapshots via `n_workers` (multiprocessing.Pool with
a top-level picklable worker, kept spawn-safe for Windows - see the
documented past multiprocessing-freeze incident in
`geometry_diagnostics.py`'s `ollivier_ricci_curvature` docstring, which
this deliberately avoids repeating). Verified byte-identical output
against the existing serial 300-row baseline before trusting it on the
real target (commit `efab0dd`).

**Diagnostics recomputation completed** (100/100 snapshots, `n_workers=8`,
well under the ~8h serial estimate). H1 (persistent homology) had to be
skipped at BOTH 1000u/1200i and 3000u/2500i - the 1-skeleton's estimated
triangle count (2.3M at 3000u/2500i) exceeds `geometry_diagnostics.py`'s
300k safety cap by far, so H0-only at these scales. This was already true
at 800i scale for some configs (see the 2026-09-12 hyperbolicity entry
below) but is now the norm, not the exception, above ~1000 items - H1
comparisons across scale are not available with the current cap.

**Built the scale-extended hyperbolicity/quality comparison
(`full_hyperbolicity_table_scaling.py`, `plot_hyperbolicity_scaling.py`,
per the user's earlier-deferred idea), reusing `diagnostics_for_D` on each
scale's restored-best-by-val_hr epoch for an apples-to-apples comparison
with the existing 300u/800i 3-seed table:**

| scale | delta_rel | test HR@10 |
|---|---|---|
| 300u/800i (3 seeds) | 0.237-0.247 | 0.187-0.207 |
| 1000u/1200i (n=1) | 0.311 | 0.222 |
| 3000u/2500i (n=1) | 0.338 | 0.312 |

Quality rises monotonically with scale. Hyperbolicity does NOT explain
this - if anything delta_rel also rises with scale (geometry becomes
LESS hyperbolic as scale grows), the opposite of what a "more hyperbolic
= better" story would predict, while quality improves anyway. This is
the same "hyperbolicity doesn't track quality" finding as the 2026-09-12
entry below, now confirmed to hold (not reverse or resolve) across scale.
Plot saved to `process_docs/hyperbolicity_scaling_comparison.png`
(2026-09-15) - two panels: delta_rel-vs-HR@10 scatter with scale points
as diamonds, and a direct HR@10-vs-scale panel (replacing the original
H1 bar chart, since H1 isn't available above 300u/800i - see above).

**Next steps:**
- Seed repeats at 1000u/1200i and 3000u/2500i, per the standing n=1
  caution - not yet done at either larger scale.
- Decide whether to raise `geometry_diagnostics.py`'s H1 safety cap (or
  subsample nodes before persistent homology) to get H1 comparable across
  scales - currently only available at 300u/800i.
- Everything else carried over below is still open.

---

**2026-09-14, later: measured training-time vs post-hoc-diagnostics-time
scaling separately - they scale completely differently.** User asked to
track both components with scale to estimate growth rate, after noticing
the post-hoc geometry diagnostics phase (per-snapshot ORC/persistent-
homology/spectral analysis, run once via `analyze_run` after training -
NOT part of the method itself, only feeds the paper's diagnostic tables)
was taking unexpectedly long at 3000u/2500i.

| items | train s/outer-step | diagnostics s/snapshot |
|---|---|---|
| 800 | 114.8 | 3.05 |
| 1200 (1.5x) | 140.7 (1.2x) | 3.75 (1.2x) |
| 2500 (2.1-3.1x) | 361.7 (2.6-3.2x) | **~296.6 (79-97x!)** |

Training time scales roughly proportionally with data size, as expected.
Post-hoc diagnostics time is dramatically superlinear - a ~2-3x increase
in item count produced a ~80-100x increase in per-snapshot diagnostics
cost, suggesting some sub-computation (likely all-pairs shortest paths
for Ollivier-Ricci curvature, and/or persistent homology's simplicial
complex construction - the log already shows the triangle-count safety
cap being hit at this scale) crosses into a much more expensive
complexity regime around n=2500. This diagnostics cost is separable from
and irrelevant to the method's actual downstream quality (HR@10/NDCG@10)
- it only matters for building the geometry/hyperbolicity comparison
tables. At larger scales, this phase could be skipped or subsampled
(e.g. only analyze every Nth snapshot, or just epoch0+restored-best)
without affecting any quality conclusion - not yet decided/implemented.

**2026-09-14: 1000u/1200i scale result in (single run, mixed picture),
3000u/2500i retry launched with the flush fix.**

**1000u/1200i result:** HR@10=0.2221, NDCG@10=0.1106, 12.0h, zero guard
triggers - the best absolute quality seen anywhere in this investigation.
train_loss trend still robustly significant and decreasing (r²=0.134,
p=6.2e-11) - reproduces the core finding from 300u/800i. But val_hr trend
is significant in the OPPOSITE direction this time (r²=0.022, p=0.0097,
declining: first100 mean=0.200 -> last100 mean=0.172), unlike all 3 seeds
at 300u/800i (which rose). Best checkpoint at step 137/300 (mid-run, not
early or late). Restore-best-not-last still clearly earned its keep
(last-step val_hr=0.189 vs restored 0.251). Plot:
`scaletest_1000x1200_convergence.png` in process_docs. **Caveat
explicitly flagged by the user: this is n=1 at this scale - no seed
repeats yet, same standing caution as everywhere else in this
investigation.**

**User's idea for later (not yet built):** extend the hyperbolicity/
quality scatter plot (`hyperbolicity_frozenproj_comparison.png`) to
include runs from every scale tested, once there's enough data - to show
whether quality trends up with scale and whether same-scale seeds
cluster together on the plot. Revisit once more scale points exist.

**Now running:** 3000u/2500i retry (`run_amazon_beauty_frozenproj_scaletest.py`,
outer_epochs=100, seed=0) - same config as the original recon that
appeared to hang (see the correction below: it hadn't actually hung, just
the log wasn't flushing). Relaunched fresh with the flush fix now in
place, so this time progress should be visible in real time and the
run can be trusted to complete or fail honestly.

**Still open / not yet decided (carried over):**
- 3000u/2500i result - pending.
- Whether to get seed repeats at 1000u/1200i and/or 3000u/2500i once both
  land, given the user's standing caution about n=1 results.
- A fair (dropout=0.2) Euclidean baseline at these larger scales, for a
  true apples-to-apples comparison (currently only exists at 300u/800i).
- Spectral/connectivity-vs-quality correlation check across other
  datasets/scales - deferred, revisit later per the user.
- ML-1M/ML-10M still sit at the pre-dropout, pre-frozen-projection
  generation - resweep decision deferred until scale-robustness settles.
- Item-representation/parameter-count table still not added to main.tex.
- Gromov delta section (4.1) and empty Conclusion - still deferred.

---

**2026-09-13: CORRECTION - the "instability at scale" finding below was
WRONG. Root cause was a logging bug (missing `flush=True`), not a real
hang. Fixed, and both scale tests relaunched.**

Both the 3000u/2500i recon and the 1000u/1200i full test appeared to
"hang" (log stopped advancing for 3+ hours / 50+ minutes respectively).
Both were killed based on that stale-log signal - the second one (per
the user's explicit "перезапустить если что" instruction) via an
automated health-check cron job using the same log-staleness heuristic.

**Both kills were mistakes.** Checked the actual `matrices_epochN.npz`
files (real, unbuffered disk writes) after the fact: the 3000u/2500i run
had really reached epoch 56 (vs ~26 visible in the log) and the
1000u/1200i run had reached epoch 285/300 - 95% done - at the moment each
was killed. Neither was stuck; both were healthy and actively
progressing. Root cause: none of the hot-path print statements
(`[Inner NCF] ep N/200`, `[Outer N/M]`, `[Final NCF]`, `[Guard]`) had
`flush=True` - when stdout is redirected to a file (not a terminal),
Python fully block-buffers instead of line-buffering, so long silences in
the log are normal, not evidence of a hang. At larger data scale, each
buffered chunk covers more wall-clock time (slower per-epoch), making a
perfectly healthy run look stuck to anyone watching the log file. The
earlier "post-guard subnormal-float slowdown" hypothesis was an
overcomplicated wrong explanation for a much more mundane cause.

**Fixed (commit `08e1b6b`):** added `flush=True` to all the hot-path
prints in `GradientIsomapCF_log.py`. Cancelled the flawed health-check
cron job (`ccea4135`) and replaced it (`af7848c8`, every 30 min) with a
corrected one that NEVER kills on log-staleness alone - it now requires
independent confirmation from BOTH the actual `matrices_epochN.npz` file
progress (numeric-sorted, not lexicographic) AND a PowerShell
`Get-Process` CPU-time-growth check before concluding a run is truly
stuck.

**Relaunched `run_amazon_beauty_frozenproj_scaletest_1000x1200.py` from
scratch** (no mid-run resumability, so the 95%-complete progress from the
killed run is lost) with the flush fix in place. 3000u/2500i retry still
pending behind this one, per the user's stated order (safer scale first).

**Still open / not yet decided (carried over):**
- Scale-robustness of the adopted config (freeze_item_projection=True,
  dropout=0.2, eta=0.001) at 1000u/1200i - result pending (relaunched).
- 3000u/2500i retry - queued behind 1000u/1200i.
- Spectral/connectivity-vs-quality correlation check across other
  datasets/scales - deferred, revisit later per the user.
- ML-1M/ML-10M still sit at the pre-dropout, pre-frozen-projection
  generation - resweep decision deferred until scale-robustness settles.
- Item-representation/parameter-count table still not added to main.tex.
- Gromov delta section (4.1) and empty Conclusion - still deferred.

---

**2026-09-12/13, SUPERSEDED BY THE CORRECTION ABOVE - kept for the
record, do not trust this diagnosis:** scale-robustness recon test found
a new, real instability risk - a single gradient-guard trigger can be
followed by a severe, silent slowdown, not just a one-off self-corrected
blip.

Launched `run_amazon_beauty_frozenproj_scaletest.py` (3000u/2500i, eta=0.001,
freeze_item_projection=True, dropout=0.2, outer_epochs=100, single seed) as
a cheap recon before committing to the full 200-300 step horizon at this
scale (would cost ~27-28h). Ran cleanly for the first 26 outer steps
(~350-450s/step, one gradient-guard trigger at outer epoch 20 - 22036
non-finite entries zeroed, seemingly self-corrected since outer steps
21-26 completed normally). Then the log stopped advancing entirely.

**Diagnosed via PowerShell `Get-Process`, not just log-watching:** the
actual Windows process (distinct from the git-bash wrapper PID) had
accumulated ~45,000 CPU-seconds over ~6 hours wall-clock (~2 cores'
worth) - i.e. NOT hung/deadlocked, but genuinely still computing,
just catastrophically slower than before (no progress visible in the log
for 3+ hours on a single inner-loop epoch that should take seconds).
Leading hypothesis: the guard-triggered near-degenerate weights pushed
some downstream computation into the subnormal/denormalized float range,
which is a well-known cause of 10-100x silent slowdowns on most hardware
with no error or crash. Stopped the task (`TaskStop`), verified via
`Get-Process` that the real PID was gone (no orphan) and GPU memory
returned to baseline.

---

**2026-09-12: METHOD FIXED - `freeze_item_projection=True` + `dropout=0.2` +
`eta_outer=0.001` is the adopted configuration going forward. Hyperbolicity
diagnostics rerun on it (same "geometry doesn't track quality" finding as
always); next direction is spectral/connectivity properties; scale-
robustness of this exact config is an explicitly open gap.**

**3-seed replication of eta=0.001 (300 steps, extended from 200 to check
a plateau) landed:** test HR@10 = 0.194 ± 0.009, beats the properly fair
(dropout=0.2) Euclidean baseline (0.123) with t=11.5, p=0.0075. train_loss
trend significantly decreasing in ALL 3 seeds (p from 1e-7 to 1e-19) -
robust, reproducible. val_hr trend weaker on replication (sig in 1/3 of
seeds) than the flagship single run suggested, but consistently positive
direction. Plateau confirmed real (last-100 mean not above mid-100 in any
seed) - extending past ~300 steps likely has diminishing returns. Full
writeup in memory (`project_gincf_outer_loop_noise_warmstart`).

**Hyperbolicity/geometry diagnostics table rebuilt for this final config**
(`full_hyperbolicity_table_frozenproj.py`/`.csv`), using the RESTORED-BEST
epoch's geometry per seed (not last, since that distinction now matters -
best epochs were 187/216/89 out of 300, never the last). Same core
finding as throughout the whole project: hyperbolicity doesn't track
quality. Poincare-pretrained: most hyperbolic (delta_rel=0.169), worst
quality (HR@10=0.10). Winning GINCF: mid-range hyperbolicity (0.240-0.247,
barely moved from its own pre-optimization ~0.230-0.237), best quality
(0.19-0.21). Euclidean (dropout=0.2 fair baseline): least hyperbolic
(0.383), by far the most topological complexity (H1=968, 3-13x every
other arm), middling quality (0.123). Plot:
`hyperbolicity_frozenproj_comparison.png` in process_docs.

**New direction, requested by the user:** since hyperbolicity doesn't
explain the winner, check whether connectivity/spectral properties do
instead - GINCF's lambda2 roughly doubles during optimization (0.20-0.25
-> 0.43-0.58) and H1 count roughly quadruples (70-80 -> 270-360), both
far more dramatic shifts than delta_rel's near-flat movement. Next: check
whether spectral_gap/lambda2/H1 correlate with downstream quality (a
cheap re-analysis of ALREADY-collected geometry_diagnostics.csv data, no
new training needed) and whether this generalizes across other datasets/
scales - not started yet.

**Explicitly flagged gap:** the adopted config's behavior at larger scale
is UNVERIFIED. The one large-scale run we have (6000u/2500i, HR@10=0.2827,
2026-09-09) predates `freeze_item_projection` entirely (dropout=0.2 only)
and was a single run - it does not test whether the CURRENT method scales.
Needs a dedicated test before any scale-robustness claim.

**Still open / not yet decided:**
- Spectral/connectivity-vs-quality correlation check (cheap, existing
  data) - not yet run.
- Scale-robustness test of the final adopted config - not yet run,
  probably the more important of the two before writing anything into
  main.tex.
- ML-1M/ML-10M still sit at the pre-dropout, pre-frozen-projection
  generation - resweep decision deferred until scale-robustness settles.
- Item-representation/parameter-count table still not added to main.tex.
- Gromov delta section (4.1) and empty Conclusion - still deferred.

---

**2026-09-11, even later: eta=0.001 frozen-projection result IS the
strongest convergence signal in the whole investigation - now verifying
with 3 more seeds (extended to 300 steps) and a fair (dropout=0.2)
baseline recompute, all running in parallel.**

**Single-run headline result:** eta=0.001, freeze_item_projection=True,
dropout=0.2, 200 outer steps: val_hr trend r²=0.088, **p=2e-5**;
train_loss trend r²=0.203, **p=2e-11** - both highly significant AND
consistent direction (loss down, HR@10 up). Best checkpoint at step
187/200 (late, not an early fluke - first time this has happened in the
whole investigation, every other test's best step landed in the first
15-30% of the run). Plot: `frozenproj_eta001_outer200_convergence.png`.
Checked for a plateau (rolling-12 mean of the last ~30 steps sits flat
around 0.18-0.19, even ticking down slightly at the very end) - looks
close to plateaued, not clearly still climbing, so extending much further
may have diminishing returns, but not conclusive from one run.

**Final test metrics: HR@10=0.2067, NDCG@10=0.0896** - beats the existing
Euclidean baseline (HR@10=0.1767, NDCG@10=0.0778) by ~17% relative. BUT
that baseline number predates `dropout` support in
`train_and_eval_euclidean_baseline` (never had the parameter at all until
now, commit `693f5d0`) - not an apples-to-apples comparison yet.

**Now running in parallel (4 processes, deliberately fewer than the usual
~5-8 since the GPU currently has ~72-80% utilization from unrelated
external jobs - `examples.gradient_isomap.synthetic.e8_euclidean_init_drift`,
not ours, confirmed via /proc cmdline inspection - keeping to 4 to avoid
badly contending with that):**
1-3. `run_amazon_beauty_frozenproj_eta001_seed.py --seed {0,1,2}` - same
   config, outer_epochs raised 200→300 to test the plateau question with
   a longer horizon, for a real multi-seed comparison (the whole
   investigation's standing lesson: single runs aren't reliable evidence).
4. `save_euclidean_baseline_geometry.py --dropout 0.2` (tag
   `amazon_beauty_dropout02fair`) - a genuinely fair baseline number to
   compare the frozen-projection result against.

**Still open / not yet decided:**
- Does the strong eta=0.001 trend replicate across seeds, and does 300
  steps confirm or break the apparent plateau? Results pending.
- Does the frozen-projection method still beat the Euclidean baseline
  once the baseline also gets dropout=0.2? Result pending.
- If both hold up: this eta=0.001/frozen_item_projection configuration is
  a strong candidate for the paper's main reported result, a real
  departure from every earlier generation (original/hrfix/realfix/p30,
  all unfrozen, all without a real outer-loop trend).
- ML-1M/ML-10M still sit at the pre-dropout, pre-frozen-projection
  generation - resweep decision deferred until this settles.
- Item-representation/parameter-count table still not added to main.tex.
- Gromov delta section (4.1) and empty Conclusion - still deferred.

---

**2026-09-11, later: frozen-projection outer-loop test (eta=0.03/60 steps)
did NOT show a clearer convergence trend than unfrozen - user asked to
re-verify this wasn't a bug, then to retry with a smaller outer lr.**

**Result:** val_hr trend r²=0.034, p=0.158 (not significant), std=0.041 -
comparable to or weaker than the unfrozen dropout=0.2 control at the same
horizon (r²=0.114, p=0.008, significant). Plot:
`frozenproj_vs_unfrozen_outer_trend.png`. So `freeze_item_projection`
fixed the FIXED-Z diagnostic (geometry causally matters when Z is held
still) but did not, on this one run, make the OUTER loop's own trajectory
more directional - the outer loop still recreates a fully fresh NCF (user
embeddings + MLP tower still free and reinitialized) every step, so the
basin-hopping noise source identified earlier isn't touched by this fix.

**User's sharp catch:** the control run's p=0.008 "significant downward
trend" looked suspiciously clean given nothing like it had shown up
before - asked to re-check the implementation for a bug rather than
assume it was real. Compiled every outer-loop trend measured across this
whole investigation (8 runs) side by side:

| config | r² | p | direction | had_bad_grad |
|---|---|---|---|---|
| eta=0.00001, 500 steps | 0.001 | 0.48 | down | False |
| eta=0.03, 200 steps | 0.039 | **0.005** | down | False |
| eta=0.1, 200 steps | 0.002 | 0.51 | down | False |
| eta=0.0001, 200 steps | 0.026 | **0.024** | **up** | False |
| 6000u/2500i, 60 steps | 0.006 | 0.54 | down | False |
| unfrozen dropout=0.2, 60 steps (control) | 0.114 | **0.008** | down | False |
| warm_start, 60 steps | 0.009 | 0.46 | up | False |
| frozen+dropout, 60 steps | 0.034 | 0.16 | up | False |

3/8 "significant" but with INCONSISTENT sign (2 down, 1 up) - exactly the
pattern expected from noise under repeated significance testing across
~8 runs, not a real systematic effect (a real bug or genuine trend would
push the sign the same way most of the time). Verified epoch numbering is
sequential (no off-by-one), zero gradient-guard triggers anywhere (no
numerical corruption). Conclusion: not a bug, just one noisy draw that
happened to land at p<0.05 - same lesson as the whole warm-start A/B
saga, single runs aren't reliable evidence here.

**In progress:** user's next hypothesis - smaller outer lr under
`freeze_item_projection=True` should perturb the manifold less per step,
so the now-geometry-sensitive critic's gradient direction should show up
more clearly (even if slower), unlike the earlier exhaustive small-eta
sweep under the UNFROZEN architecture (down to eta=0.00001/500 steps,
never significant, r²=0.001). Launched
`run_amazon_beauty_frozenproj_smalleta_test.py`: eta=0.001 (10x smaller
than the eta=0.03 test above), outer_epochs=200 (matches the horizon used
for the eta=0.1/eta=0.0001 unfrozen probes in the table, for direct
comparability), same dropout=0.2/select_by=hr/patience=30/cap=200. First
step confirmed healthy (82.7s). Estimated ~5.5-6h. Not yet analyzed.

**Still open / not yet decided:**
- Does eta=0.001 reveal a trend under frozen_item_projection where
  eta=0.03 didn't? Result pending.
- If still no trend at any eta: the basin-hopping noise (free user-side
  embeddings + MLP tower reinitializing every step) is confirmed as the
  dominant, still-unaddressed noise source - would need an even more
  invasive fix (e.g. constrain/warm-start those too) to test further, or
  accept this as a documented limitation of the bilevel search itself.
- ML-1M/ML-10M still sit at the pre-dropout, pre-frozen-projection
  generation - resweep decision deferred.
- Item-representation/parameter-count table still not added to main.tex.
- Gromov delta section (4.1) and empty Conclusion - still deferred.

---

**2026-09-11: frozen-projection hypothesis CONFIRMED on the fixed-Z
diagnostic (real Z significantly beats shuffled Z), now wired into the
production pipeline and testing whether the outer loop finally converges.**

**Decisive result (reran with per-epoch history capture for convergence
plots, commit `6c50d82`, identical numbers to the first run - fully
reproducible):**

| condition | test HR@10 | val_loss | inner-loop epochs used |
|---|---|---|---|
| real Z | 0.159 ± 0.012 | 0.367 ± 0.013 | 42-67 (mean 50.8) |
| shuffled Z | 0.097 ± 0.019 | 0.453 ± 0.075 | 31-33 (mean 32.4) |
| random Z | 0.107 ± 0.026 | 0.517 ± 0.058 | 31-97 (mean 60.0) |

real vs shuffled: t=5.583, **p=0.0005**. Convergence plots
(`frozen_projection_convergence_all_conditions.png`) show all runs
early-stop cleanly (no cap hits, monotonic loss curves) - real Z takes
longer to converge (more to learn from real structure) and reaches a
clearly higher val_hr plateau; shuffled Z converges fast to a lower
ceiling. This is the first time in the whole investigation that swapping
in a wrong-but-structurally-identical Z has actually hurt - confirming
`freeze_item_projection` closes the "escape hatch" (see below).

**Code changes this investigation, for the record:**
- `recsys/GradIsomapCF_movielens/NeuMFOnManifold.py`: new
  `freeze_item_projection`/`item_projection_init_data` params. When
  enabled: MLP branch uses `z_i` unchanged (identity - `latent_dim` is
  deliberately set equal to `mlp_user_dim` for exactly this reason, a
  fair "our embedding vs. baseline's" comparison, confirmed intentional
  by the user, not an oversight); GMF branch uses a FIXED PCA projection
  (SVD of the actual `item_Z`, not random - user flagged random-frozen as
  risky/poorly-conditioned) down to `factor_num` dims. Only user-side
  embeddings + the downstream MLP tower/predict layer stay trainable.
- `recsys/my_verification/ablation_geometry_vs_optimization.py`:
  `train_and_eval_ncf_on_fixed_Z` gained `dropout`, `weight_decay`,
  `freeze_item_projection` (previously `dropout` hardcoded 0.0,
  `weight_decay` left at AdamW's unexamined 0.01 default).
- `recsys/GradIsomapCF_movielens/GradientIsomapCF_log.py` (production):
  `freeze_item_projection` threaded into BOTH `NeuMFOnManifold`
  construction points (inner loop + final-NCF stage), passing
  `item_projection_init_data=item_Z_epoch`/`item_Z_final` so the frozen
  GMF projection is recomputed fresh from whatever the manifold's CURRENT
  state is at each point (not stale). Optimizers now filter to
  `requires_grad` params only. (Also carries `warm_start_inner`, tested
  and closed out earlier - no significant effect - and the DataLoader/
  cudnn determinism fixes from the outer-loop noise investigation.)
- `recsys/my_verification/run_experiment.py` /
  `recsys/my_verification/run_eta_outer_sweep.py`: `dropout` and
  `freeze_item_projection` exposed as pass-through params (previously
  `dropout` hardcoded 0.0 at both call sites; `outer_epochs`/`seed` were
  hardcoded too, fixed earlier in this investigation).

**In progress:** smoke-testing `freeze_item_projection=True` in the real
outer loop (2 outer steps) before committing to the full 60-step run
(`run_amazon_beauty_frozen_proj_outer_test.py`, eta=0.03, dropout=0.2,
same protocol as every other A/B test here). This is THE question the
whole investigation has been building toward: does the outer loop's
val_hr trajectory finally show a real trend once the inner critic
actually depends on Z's geometry (unlike every prior test, all of which
used the unfrozen architecture and topped out at r²<=0.04, never
significant)? Result pending.

**Still open / not yet decided:**
- Does the outer loop converge under freeze_item_projection=True? The
  central open question right now.
- If yes: main.tex's whole framing shifts - frozen projection may need to
  become the primary reported architecture, not a side diagnostic.
- If no: the noise is confirmed to come from elsewhere (fresh-reinit
  multimodality itself, already documented) rather than "critic blind to
  geometry" - still a valuable, publishable negative result either way.
- ML-1M/ML-10M still sit at the pre-dropout, pre-frozen-projection
  generation - far behind the current investigation; resweep decision
  deferred until the frozen-projection outer-loop result is in.
- Item-representation/parameter-count table still not added to main.tex.
- Gromov delta section (4.1) and empty Conclusion - still deferred.

---

**2026-09-09, later: root-caused WHY "any topology works" - no channel for
pairwise-distance information exists anywhere in NCF's training objective
- and built a fix (frozen item projection) to test tomorrow.**

After the scaled (6000u/2500i) run also showed no outer-loop convergence
trend, user reframed the question: "NCF converges to a decent optimum on
literally any topology - that's a strange conclusion, there shouldn't be
this many degrees of freedom." Investigated via code inspection (not
speculation):

- Exhaustively grepped every `loss_fn(`/`bce_loss`/`loss_cf` call across
  `GradientIsomapCF_log.py` and `ablation_geometry_vs_optimization.py`:
  every one is a plain BCE loss on individual (user, item) interaction
  labels. No regularization term, no distance-preserving penalty, no
  graph-Laplacian smoothness anywhere - confirmed exhaustively, not by
  sampling.
- The ONLY channel for Z's geometric information to reach NCF is the raw
  per-item lookup `z_i = item_Z[item]` in `NeuMFOnManifold.forward()`,
  which immediately passes through freely-retrained linear layers
  (`item_GMF_linear`, `item_MLP_linear` - xavier-init, retrained from
  scratch every time a new NeuMFOnManifold is created) before reaching
  user-side free embeddings and a fully-trainable MLP tower. Nothing in
  the objective requires the network to preserve or exploit Z's actual
  metric structure - it's free to fit any per-item assignment that helps
  the BCE loss, using Z's raw values only as an arbitrary (if
  information-carrying) per-item code.
- Traced why `latent_dim=64`: confirmed by the user it was DELIBERATELY
  set equal to `mlp_user_dim = factor_num*(2**(num_layers-1)) = 16*4 = 64`
  for a fair "our embedding vs. baseline's embedding" comparison - not an
  oversight, so the earlier "wasted dimensions" theory doesn't apply.
  Checked whether the existing intrinsic-dimension-estimation tooling
  (`utils/intrinsic_dim_estimators.py`, Levina-Bickel MLE) was ever
  applied to the recsys data - it wasn't (only synthetic geometries/MNIST),
  but this is now a moot point given the dimension choice was intentional.

**User ruled out random-frozen projection** ("может плохо сжимать") -
implemented instead: `freeze_item_projection=True` on `NeuMFOnManifold`
(commit `5b6346f`) - MLP branch uses `z_i` unchanged (identity map, exact
since latent_dim==mlp_user_dim by design, no information lost), GMF branch
uses a FIXED PCA projection of the actual item_Z data (not random) down to
factor_num dims via SVD. Only user-side embeddings and the downstream MLP
tower/predict layer stay trainable. Sanity-checked on CPU (no GPU load,
resources busy with other work) - identity/frozen weights correct,
gradients flow only to trainable params.

**Test prepared (`recsys/my_verification/test_frozen_projection_geometry.py`,
commit `3eaa649`):** real Z vs shuffled-rows Z (same vectors, wrong item
assignment) vs fresh-random Z, 5/5/3 seeds, freeze_item_projection=True +
dropout=0.2. If real clearly beats shuffled/random now, that confirms
freezing the projection forces genuine dependence on geometry (the escape
hatch is closed). If real still doesn't win, the escape hatch is
downstream (MLP tower / GMF's free user-side multiplication) and would
need a further architectural fix there too.

**Scheduled, not yet run:** rescheduled from tomorrow noon to TODAY
~16:03 (cron job id `b3bcd5ff`, replaces the cancelled `6be345d5`) - user
said the machine reboots tonight, so waiting until tomorrow wouldn't
survive that anyway. Will check `nvidia-smi` is actually free, then launch
the test and report back. Session-only cron (not persisted to disk) - if
the session ends before it fires, needs manual re-launch.

**Still open / not yet decided:**
- The decisive result: does freezing the item projection make the model
  actually sensitive to whether Z's geometric assignment is correct?
  Result pending tomorrow.
- If frozen-projection doesn't fix it either, next suspect is the
  downstream MLP tower / GMF free user-multiplication - would need a
  further, more invasive architectural constraint.
- Step 2 (more data, 6000u/2500i single run) result still stands
  (HR@10=0.2827) but multi-seed variance at that scale is still unmeasured.
- ML-1M/ML-10M still sit at the pre-dropout, pre-frozen-projection
  generation - far behind the current investigation.
- Item-representation/parameter-count table still not added to main.tex.
- Gromov delta section (4.1) and empty Conclusion - still deferred.

---

**2026-09-09: dropout=0.2 validated in the real outer loop (step 1
complete), now running step 2 (more data) at an empirically-probed
6000u/2500i scale, ~7.7h estimated.**

**Step 1 validation result (real outer loop, not just fixed-Z):**
eta=0.03/outer_epochs=60, 5 seeds each. dropout=0.0 (control, from the
warm-start A/B test): test_hr=0.170±0.018. dropout=0.2: test_hr=0.181±0.009
- cross-seed std more than halved (matches/exceeds the fixed-Z diagnostic's
prediction), mean also higher. Within-run outer-trajectory noise (val_hr
std across the 60 steps of a single run) was NOT reduced by dropout
(0.035 vs 0.038, ns) - dropout fixes cross-seed reproducibility of the
final result, not the outer loop's own step-to-step noise. Plot:
`dropout_outerloop_validation.png` in process_docs. dropout=0.2 adopted as
the new default going forward (`dropout` param, previously hardcoded to
0.0 in `run_experiment.py`, now threaded through both call sites and
`run_eta_outer_sweep.py`, commit `20ae812`).

**Step 2: does relaxing data sparsity further reduce the residual
variance (std=0.009, still not zero)?** User's framing: ensembling/
checkpoint-averaging were explicitly ruled out as fixes (masks the noise
being investigated rather than fixing the method) - "estimate what fits
in memory and computes in 6-8h, launch on that" for a bigger-data test.

**Empirically probed scale/timing/stability (not guessed) before
committing the full budget:**

| users | items | s/step | had_bad_grad |
|---|---|---|---|
| 300 | 800 | ~120 (baseline, under GPU contention) | - |
| 1000 | 1200 | ~137 | - |
| 3000 | 2500 | ~330 | False (clean) |
| 5000 | 4000 | ~459 | **True - EVERY step**, 35843 non-finite grad entries zeroed each time |
| 6000 | 2500 | ~455 | False (clean) |

5000u/4000i's gradient-guard triggering on both probed steps (identical
count both times) indicates a structural eigenvalue-degeneracy issue at
that item count (the known `eigh` backward instability, see
`docs/gradient_isomap_mnist_regression_report.md`/earlier diary entries),
not a rare blip - decided NOT to build the 6-8h budget on an unstable
config, since a noisy/degraded result there would be uninterpretable
(genuine multimodality vs. gradient corruption). 6000u/2500i gives
comparable computational cost (~455s/step) WITHOUT the instability, since
items (not users) drive the eigenvalue-degeneracy risk - and users are
the more directly relevant axis for testing NCF's own underdetermination
anyway.

**Launched:** `run_amazon_beauty_scaled_dropout_test.py` - 6000u/2500i
(20x/3.1x the original 300u/800i), outer_epochs=60 (same as all prior A/B
tests here, for direct comparability), dropout=0.2, eta=0.03, select_by=hr,
patience=30, cap=200. Estimated ~7.7h (60×455s + ~450s overhead). First
step confirmed healthy. Not yet analyzed.

**Still open / not yet decided:**
- Whether more data (step 2) further tightens the residual cross-seed
  variance, or whether dropout alone already captured most of the
  achievable improvement - result pending.
- Whether to also multi-seed-repeat the scaled config once this single run
  finishes (expensive at this scale - would need to weigh against just
  accepting single-run evidence here given the cost).
- ML-1M/ML-10M still sit at the hrfix (2026-08-27) generation, and now
  also the dropout=0.0 pre-fix generation - decision on resweeping with
  dropout=0.2 deferred.
- Item-representation/parameter-count table still not added to main.tex.
- Gromov delta section (4.1) and empty Conclusion - still deferred.

---

**2026-08-30, git history cleanup + new direction: too many degrees of
freedom in the NCF critic, regularization (dropout) looks like a real
fix.** Side track: user pushed the branch, asked to strip
`Co-Authored-By: Claude` trailers from 47 unpushed commits (done via
`git filter-branch --msg-filter`, verified tree/diff identical, author
already correctly `chrislisbon` throughout - 9 separate commits by a real
collaborator "Diana" left untouched) and then to also strip 124.3MB of
large old-run artifacts (`run48/images/D_epochs/*.npy`, `ratings.dat`,
heatmap/collage PNGs) from the same 68→69 unpushed commits via
`git filter-branch --index-filter` + `git rm --cached`. **Caught and fixed
my own mistake here:** filter-branch's final sync actually deletes matched
files from the working directory too (not just untracks them, contrary to
what I'd told her) - `ratings.dat` (the real, needed MovieLens-1M dataset)
was gone from disk; restored it immediately from the filter-branch backup
ref (`refs/original/...`, still present). The other 63 old-run artifacts
were left un-restored (not needed, recoverable from the same backup ref if
ever wanted). Added `.gitignore` entries for both so they don't get
re-committed. Push is now ~10.5MB (was ~135MB).

**Back to the actual research question:** user's framing - "NCF converges
to a local optimum on ANY topology, meaning the algorithm has too many
degrees of freedom relative to the data - how do we treat that?" Checked
the already-collected fixed-Z 5-seed data more closely instead of
guessing: `best_val_loss` (continuous, not HR@10's coarse top-10 binning)
varies 0.331-0.438 across seeds on an IDENTICAL manifold - ~29% relative
spread, confirming genuinely different-quality local optima, not just
measurement noise on equivalent solutions. Separately noticed
`val_hr_training` (best HR@10 seen during training) only varies 13% while
final `test_hr` varies 68% - early stopping is itself overfitting to the
300-user val set's own sampling noise, a second, distinct noise source.

**User explicitly ruled out ensembling/checkpoint-averaging as fixes** -
her reasoning: the goal here is verifying the METHOD works, not producing
clean-looking metrics by construction-averaging away the very noise being
investigated. Agreed plan: (1) cheap - try regularization (dropout was
hardcoded to 0.0 everywhere, weight_decay left at AdamW's unexamined
default 0.01) on the fixed-Z diagnostic first; (2) expensive - more data
(larger user/item subsample) if (1) isn't sufficient.

**Step (1) result - clean, strong win.** Same fixed-Z 5-seed test,
`dropout=0.2` (weight_decay left at 0.01): val_loss relative spread
0.288 -> **0.037** (8x tighter), test_hr std 0.038 -> **0.020** (near
half), and mean test_hr even rose slightly (0.150 -> 0.172). Also tested
`weight_decay=0.10` alone (spread 0.114, worse than dropout) and
`dropout=0.2 + weight_decay=0.10` together (spread 0.091, worse than
dropout alone) - dropout is doing essentially all the work, more
regularization is not simply better. `dropout` was hardcoded to `0.0` at
both `run_experiment.py` call sites (never exposed) - now a proper
parameter, threaded through `run_eta_outer_sweep.py` too (commit
`20ae812`).

**In progress:** validating this in the REAL outer bilevel loop, not just
the isolated fixed-Z diagnostic - `run_amazon_beauty_dropout_test_seed.py`,
eta=0.03/outer_epochs=60/dropout=0.2, 5 seeds run in parallel. Directly
comparable to the warm-start A/B test's control arm (identical config,
dropout=0.0): baseline test_hr=0.170±0.018, within-run val_hr
std=0.0352±0.0025. Not yet analyzed.

**Still open / not yet decided:**
- Whether dropout=0.2's fixed-Z benefit holds up once the manifold is also
  being optimized (outer loop result pending).
- Step (2), more data, still queued behind step (1)'s validation.
- ML-1M/ML-10M still sit at the hrfix (2026-08-27) generation.
- Item-representation/parameter-count table still not added to main.tex.
- Gromov delta section (4.1) and empty Conclusion - still deferred.

---

**2026-08-30, final for the day: warm-start investigation CLOSED OUT -
properly powered 5-seed-per-arm test found no statistically significant
effect.** Ran control (fresh reinit) vs `warm_start_inner=True` at 5 seeds
each (0-4; seed=0 reused from the earlier single-run test, seeds 1-4 run as
8 parallel OS processes - GPU had ample headroom for this model scale,
~3.2GB/16.3GB VRAM, 58% util, completed in well under the ~9-11h a
sequential run would have taken):

| metric | control (n=5) | warm_start (n=5) | t-test p |
|---|---|---|---|
| test HR@10 | 0.170 ± 0.018 | 0.150 ± 0.021 | 0.19 (ns) |
| within-run val_hr std | 0.0352 ± 0.0025 | 0.0333 ± 0.0032 | 0.37 (ns) |
| mean\|Δstep\| train_loss | 0.0921 | 0.0689 | not tested, but consistent across all 5 seeds |

Neither the downstream noise metric nor the final result showed a
significant difference - if anything control looked slightly (non-
significantly) better on HR@10. Only the outer TRAIN loss's own step-to-
step volatility was consistently lower under warm-start (~25%), but this
never propagated to val_hr or the final metric. **Decision (agreed with
the user, recorded to memory
`project_gincf_outer_loop_noise_warmstart`):** close out warm-starting as
"tested properly, no measurable benefit" - do not adopt it. The underlying
diagnosis (seed variance/basin-hopping explains most of the outer-loop
noise, confirmed via the fixed-Z 5-seed experiment) remains a real and
well-evidenced finding for the paper; warm-starting the inner NCF just
isn't an effective practical fix for it. Plot:
`warmstart_5seed_final_comparison.png` in process_docs.

**This closes a ~2-day investigative arc** (outer-lr sweep -> DataLoader/
cudnn determinism fix -> geometry-vs-downstream correlation analysis ->
fixed-Z seed-variance experiment -> warm-start A/B, single-run then
5-seed) into why the outer bilevel loop's trajectory looks noisy. Net
takeaways to carry into the paper: (1) the manifold's own geometric
diagnostics DO move with a strong, real, statistically significant trend
under gradient descent (r² up to 0.94) - the optimizer is not aimless;
(2) downstream ranking quality (val_hr) is statistically decoupled from
that geometric trend at this dataset scale, most likely because NCF
training is highly multimodal here (seed-only std=0.038 in test HR@10 on
a completely fixed manifold) and the inner critic is freshly resampled
every outer step; (3) warm-starting does not fix this in practice; (4) any
single-seed result at this dataset scale (300u/800i) should be treated
with real skepticism - multi-seed repetition is now demonstrated necessary,
not just theoretically prudent.

**Still open / not yet decided:**
- Whether/how to write this whole investigation up for the paper (as a
  methodology/limitations point - "downstream ranking quality is decoupled
  from the geometric optimization objective at small dataset scale" is a
  substantive, honest finding, not just a null result to hide).
- Whether to formalize multi-seed repetition (e.g. 5 seeds, report mean±std)
  as standard practice for ALL reported numbers on this pipeline going
  forward - raised by the user, not yet decided/implemented broadly.
- ML-1M/ML-10M still sit at the hrfix (2026-08-27) generation - decision on
  resweeping deferred, now also entangled with the multi-seed-repetition
  question above.
- Item-representation/parameter-count table still not added to main.tex.
- Gromov delta section (4.1) and empty Conclusion - still deferred.

---

**2026-08-30, even later: the single-run warm-start A/B test was
inconclusive (didn't reduce val_hr noise), correctly flagged by the user
as itself vulnerable to the same seed-variance confound - now running a
5-seeds-per-arm repeat, in parallel.**

Single-run result (seed=0 only): control std(val_hr)=0.0328, warm_start
std(val_hr)=0.0339 - essentially identical, no reduction; control's own
val_hr trend (r²=0.114) was even stronger than warm-start's (r²=0.009).
Only a partial signal: mean|step-to-step train_loss change| dropped ~22%
(0.093→0.072) under warm-start, but that didn't show up in val_hr at all.
Final test HR@10: control 0.1700, warm-start 0.1367 (warm-start worse).

User's catch: a single run per arm can't be trusted given the just-proven
seed variance (std=0.038 from seed alone on a fixed Z) - this A/B
comparison needed its own repeats to mean anything, exactly the same
methodological point she raised generally after the fixed-Z experiment
(see [[project_gincf_outer_loop_noise_warmstart]] in the memory system).

**Now running 5 repeats per arm** (seed=0 already have from the single-run
test; launched seeds 1-4 for both control and warm_start_inner=True as 8
separate OS processes in parallel - commit `bcd3601` added `seed` as a
pass-through param to `run_eta_outer_sweep.py`, was hardcoded to 0 before).
GPU has ample headroom for this model scale (300u/800i): 8 concurrent
processes use ~3.2GB/16.3GB VRAM, 58% utilization, comfortable margin.
Estimated ~1.5-2h wall clock for all 8 (vs ~9-11h sequential). Not yet
analyzed - once done, compare mean/std of val_hr-noise and final HR@10
across the 5 seeds per arm (proper statistical comparison, not point
estimates).

**Still open / not yet decided:**
- Whether warm-starting reduces noise once judged over 5 real repeats per
  arm, not 1 - result pending.
- Whether/how to formalize multi-seed repetition as standard practice more
  broadly on this pipeline (the user's point applies beyond this one A/B
  test).
- ML-1M/ML-10M still sit at the hrfix (2026-08-27) generation - decision on
  resweeping deferred until the outer-loop investigation concludes.
- Item-representation/parameter-count table still not added to main.tex.
- Gromov delta section (4.1) and empty Conclusion - still deferred.

---

**2026-08-30, latest: root-caused the outer-loop noise (not lr, not step
count - NCF training seed variance / basin-hopping), implemented a
togglable `warm_start_inner` fix, A/B test in progress.** User pushed on
"the optimizer must be following SOME loss - go look at the actual code
again." Traced the real mechanics in `GradientIsomapCF_log.py`: the outer
gradient step backprops `bce_loss` (full train set, frozen best-inner-NCF)
through `isomap_model` - this `bce_loss` IS what's logged as the outer
loop's "train" value (`history['train_loss']`), which I had never actually
examined on its own (only val_hr/val_loss, downstream evaluation metrics
computed after the step). Plotted it directly
(`outer_actual_train_loss_convergence.png`): it does NOT show clean
convergence either, and for eta=0.03/0.1 it actually INCREASES over the
horizon. Decisive tell: mean|step-to-step change| in this train_loss is
~0.09-0.11 regardless of eta (0.00001 to 0.1, a 10,000x range) - if the
noise came from Z's movement size, tiny eta should show far smaller
swings. It doesn't, ruling out step size as the cause.

**Root cause, confirmed by a clean controlled experiment:** `ncf_model =
NeuMFOnManifold(...)` is created fresh, randomly initialized, at EVERY
outer step (`GradientIsomapCF_log.py` outer loop) - not continued from the
previous step. Held Z completely fixed (one real snapshot) and trained NCF
from scratch with 5 different seeds only: test HR@10 ranged 0.097-0.197
(std=0.038) - a 2x spread from seed alone, on an IDENTICAL manifold. This
matches the ENTIRE outer-loop step-to-step noise magnitude observed at
every eta/horizon tested (std 0.019-0.042). At this dataset scale (300
users/800 items), NCF training is highly multimodal; fresh reinit
resamples a random local optimum every single outer step, which very
likely dominates over any real signal from Z's actual movement. Also
checked step-by-step Spearman correlation between all 11 geometry
diagnostics and val_hr/val_loss across 3 runs (66 tests): nothing, all
|r|<0.20 - geometry itself moves with a strong, real trend (r² up to 0.94
for eta=0.1's spectral gap) that's statistically decoupled from downstream
ranking quality. Plots: `geometry_trend_vs_hr_noise.png`,
`geometry_drift_eta01_dramatic.png`, `fixed_Z_seed_variance_vs_outer_noise.png`.

**User's verdict and decision, recorded to memory
(`project_gincf_outer_loop_noise_warmstart` in the auto-memory system,
not just this diary):** warm-starting the inner NCF across outer steps is
now well-motivated - NOT for inner-loop convergence speed (already fine),
but to stop resampling a random basin every step, restoring the continuity
a valid outer/hypergradient signal needs. Known tradeoff stated explicitly:
this may sacrifice some of the "lucky best step" upside the current
repeated-random-restart-like behavior provides (e.g. the best result found
so far, val_hr=0.2367 at step 120/500 for eta=0.00001, may be exactly this
kind of luck). Separately, the user drew a broader methodological
conclusion: given seed variance alone is comparable to any effect size
under investigation, single-seed results at this dataset scale are not
reliable - future reported numbers on this pipeline should average over
multiple seeds, not rely on one run.

**Implemented (commit `4292c69`):** `warm_start_inner: bool = False` on
`GradientIsomapCF` (default preserves existing behavior exactly - only
the NCF *weights* carry over between outer steps when enabled, the AdamW
optimizer state is still reset each step, to isolate "same basin" from
"same optimizer trajectory"). Threaded through `run_experiment.py` and
`run_eta_outer_sweep.py`.

**In progress:** `run_amazon_beauty_warmstart_ab_test.py` - control (fresh
reinit, current default) vs `warm_start_inner=True`, identical otherwise
(eta=0.03, outer_epochs=60, select_by=hr, patience=30, cap=200), run
back-to-back for a fair comparison. ~2h total estimated. Not yet analyzed
- compare std(val_hr) and mean step-to-step train_loss swing between arms
once both finish.

**Still open / not yet decided:**
- Whether warm-starting actually reduces the noise as hypothesized, and by
  how much - first direct empirical test, result pending.
- Whether/how to add multi-seed repetition as standard practice for
  reported numbers on this pipeline (raised by the user, not yet
  implemented anywhere).
- ML-1M/ML-10M still sit at the hrfix (2026-08-27) generation - decision on
  resweeping deferred until the outer-loop investigation concludes.
- Item-representation/parameter-count table still not added to main.tex.
- Gromov delta section (4.1) and empty Conclusion - still deferred.

---

**2026-08-30, later: corrected a misread instruction - "больший шаг" meant a
LARGER outer lr (eta=0.1, to shake the manifold harder), not more outer
steps.** Had already launched `run_amazon_beauty_outer1000_eta1e5_test.py`
(eta=0.00001, 1000 steps) on the wrong reading; stopped it immediately once
corrected (task had not logged its first outer step yet, so ~0 wasted
compute). Relaunched as `run_amazon_beauty_outer200_eta01_test.py`:
eta_outer=0.1 (the largest tried in any of the longer-horizon confirmation
tests so far - 0.1 was in the original 4-point sweep but only ever at 30
outer steps), outer_epochs=200 for direct comparability with the eta=0.03
and eta=0.0001 confirmation tests. Rationale: even eta=0.03's fairly large
geometry swings (ORC −0.31→−0.99) showed zero step-by-step correlation with
val_hr; testing whether an even stronger perturbation changes that picture,
either by surfacing a real correlation or more decisively confirming the
geometry/downstream decoupling found so far. In progress, not yet analyzed.

---

**2026-08-30: the outer=500/eta=1e-5 confirmation test finished (best
result so far, HR@10=0.2233), which reframed the whole outer-loop-noise
investigation - the manifold itself DOES move with a strong, real trend;
val_hr just doesn't track it.** User's sharp question: "an optimizer can't
be indifferent to direction - it's optimizing something." Checked the
per-outer-epoch `geometry_diagnostics.csv` (already logged, no new
training needed) rather than only the downstream val_hr/val_loss:

| metric | eta=0.00001, 500 steps | eta=0.03, 200 steps |
|---|---|---|
| Kruskal stress | r²=0.78, falling (0.307→0.304) | r²=0.83, **rising** (0.286→0.478) |
| Spearman ρ (D_latent vs D_geodesic) | r²=0.65, rising | r²=0.50, **falling** (0.80→0.77) |
| ORC mean | r²=0.89 | r²=0.02, big raw swing (−0.31→−0.99) |
| H1 count (latent) | r²=0.48, rising | r²=0.48, rising sharply |

Geometry moves with r² up to 0.89 (vs val_hr's r²≤0.04 at every eta tested
so far) - the outer optimizer is genuinely, statistically strongly directed
on its own loss surface. Direction/magnitude differs by eta: tiny eta
slowly IMPROVES manifold self-consistency (lower stress, higher rho), large
eta pushes it away from that fast. Plot: `geometry_trend_vs_hr_noise.png`.

**Checked whether ANY geometry metric correlates step-by-step with
val_hr/val_loss (Spearman, all 11 diagnostics x 3 runs x 2 targets = 66
tests):** essentially nothing - every |r|<0.20, only 2/66 hit p<0.01 (both
r≈0.20), consistent with pure multiple-testing noise (~0.66 false positives
expected at that threshold). So the geometric drift and the downstream
ranking-quality noise are, in this data, statistically independent - the
disconnect isn't "the optimizer wanders aimlessly," it's "the geometric
loss surface and downstream HR@10 don't track each other moment-to-moment,"
most likely because HR@10 is measured through a freshly-reinitialized inner
NCF proxy every single outer step.

**In progress:** `run_amazon_beauty_outer1000_eta1e5_test.py` - same
eta=0.00001 config, doubled to 1000 outer steps, testing whether val_hr's
trend is real-but-weak (needs a longer horizon to resolve from noise, same
as the geometry trend needed ~500 steps to become clearly significant) or
genuinely flat. Very expensive: ~18h estimated (32800s/500-step-run rate).
Not yet analyzed.

**All new plots** (`geometry_trend_vs_hr_noise.png`,
`inner_loop_stopping_histogram_amazon_eta1e5_outer500.png`,
`outer_loop_trajectory_500steps_amazon_eta1e5.png`,
`noise_vs_eta_outer_summary.png`) delivered to process_docs.

**Still open / not yet decided (carried over):**
- Whether outer_epochs=1000 resolves a real (if weak) val_hr trend, or
  confirms it's fundamentally decoupled from the geometric optimization.
- Whether to add a correlation/geometry-vs-downstream analysis section to
  the paper - this reframing (geometry converges, downstream doesn't track
  it) may be a more defensible and interesting narrative than "outer loop
  doesn't converge."
- ML-1M/ML-10M still sit at the hrfix (2026-08-27) generation - decision on
  resweeping deferred until the outer-loop investigation concludes.
- Item-representation/parameter-count table still not added to main.tex.
- Gromov delta section (4.1) and empty Conclusion - still deferred.

---

**2026-08-29, even later: found and fixed real cross-run non-determinism,
launched outer_epochs=500/eta=0.00001 as the next confirmation test.**
Investigating the eta=0.0001 outer=200 run's result (0.1767, LOWER than the
same eta's outer=30 result of 0.2100), compared step-by-step val_hr/val_loss
between the two independent runs of that config: identical for outer steps
1-2 (same val_loss to 8 decimal places), then diverge completely from step 3
onward. Root cause: `GradientIsomapCF_log.py`'s `inter_loader` used
`DataLoader(..., shuffle=True)` with no explicit `generator`, so its shuffle
order came from the ambient global RNG stream rather than a value fixed to
the run's seed - combined with GPU floating-point non-determinism (cuDNN
kernels are not bit-exact by default), this let two `seed=0` runs diverge
after enough training steps that a tiny numeric difference changes which
epoch the inner loop's HR-based early stopping fires on, which then shifts
everything downstream.

**Fixed (commit `0964024`):** `inter_loader` now uses an explicit
`torch.Generator` seeded from `self.ng_seed` (was implicitly global-RNG-
dependent); `run_experiment.main()` now sets `cudnn.deterministic=True`,
`cudnn.benchmark=False`, `use_deterministic_algorithms(True,
warn_only=True)`. Honest caveat, not oversold: this is best-effort GPU
determinism, not a bit-exact guarantee - some CUDA kernels still lack a
deterministic implementation and `warn_only=True` lets those through with
just a warning rather than crashing the run.

**User's read on the eta=0.0001/outer=200 result:** rejected warm-starting
the inner NCF proxy between outer steps as a fix (correctly - the manifold
changes every outer step, so carried-over weights would fight the new
input distribution rather than help). Instead asked for: even smaller
outer lr, outer_epochs above 200, and the determinism fix above (already
applied). Launched `run_amazon_beauty_outer500_eta1e5_test.py`:
eta_outer=0.00001 (10x below 0.0001), outer_epochs=500, same inner settings
(select_by=hr, patience=30, cap=200). In progress - first outer step
confirmed healthy (train=0.3244, differs from the pre-fix runs' train=0.4694
at step 1, confirming the shuffle-order fix actually changed execution).
Estimated ~8.5h at 61.2s/step. Not yet analyzed.

**Still open / not yet decided (carried over):**
- Whether outer_epochs=500/eta=1e-5 finally reveals a real trend, or the
  outer search is fundamentally noise-dominated regardless of step size
  (current best guess after the eta=0.0001 test: mostly noise, since std
  dropped ~2x from eta=0.01→0.0001 but r^2 of any linear trend stayed
  ~0.03 at both outer=200 tests so far).
- ML-1M/ML-10M still sit at the hrfix (2026-08-27) generation - decision on
  resweeping deferred until the outer-loop investigation concludes.
- Item-representation/parameter-count table still not added to main.tex.
- Gromov delta section (4.1) and empty Conclusion - still deferred.

---

**2026-08-29, later: investigating whether the outer loop's lr (`lr_isomap`
= eta_outer, AdamW on IsomapNN's weights) is simply too large to preserve
directionality - user's hypothesis after seeing the outer_epochs=200 test
below stay noisy.** Reasoning: each outer AdamW step perturbs `D_input`'s
underlying weights; if the step is large/undirected relative to the
manifold's structure, consecutive outer steps produce near-unrelated
manifolds, which would look exactly like the noise we've been seeing
regardless of how long the loop runs.

**Quick test, 30 outer steps each, same inner settings (select_by=hr,
patience=30, cap=200), two etas two orders of magnitude below the
smallest tried before (0.01):**

| eta_outer | std(val_hr) over 30 steps | mean \|step-to-step Δ\| | final test HR@10 |
|---|---|---|---|
| 0.01 (existing p30) | 0.0415 | 0.0441 | 0.167 |
| 0.001 (new) | 0.0238 | 0.0289 | 0.197 |
| 0.0001 (new) | 0.0225 | 0.0326 | **0.210** |

**Partial confirmation.** Trajectory std roughly halves going from 0.01 to
0.001 (supports the hypothesis - smaller step, smaller perturbation,
smaller noise amplitude) but does NOT keep monotonically dropping from
0.001 to 0.0001, and neither smaller eta shows a genuinely smooth/monotonic
trajectory - the "restored best" step still lands unpredictably (28/30 for
eta=0.001, 3/30 for eta=0.0001). So: lower lr reduces noise MAGNITUDE, but
doesn't turn the search into a directed one over just 30 steps. Useful side
effect: both smaller etas beat eta=0.01 outright, and eta=0.0001
(HR@10=0.210) already beats the freshly-corrected Euclidean baseline
(0.1767) - the best GINCF result on this dataset so far, any generation.
Plot: `outer_loop_smaller_lr_comparison.png` in process_docs.

**In progress:** eta_outer=0.0001 at outer_epochs=200 (same setup as the
eta=0.03/outer_epochs=200 test above), to see whether the lower noise floor
reveals a real smooth trend over a much longer horizon, consistent with the
user's "this will slow convergence but may preserve directionality"
prediction. `run_amazon_beauty_outer200_test_eta0001.py`
(n_run=`amazon_beauty_outer200_test_eta0001_0.0001`). ETA ~3-3.5h based on
the eta=0.03 run's 11757s. Not yet analyzed.

**Still open / not yet decided (carried over):**
- Whether/how to isolate inner-loop-patience effect from final-NCF-patience
  effect.
- ML-1M/ML-10M still sit at the hrfix (2026-08-27) generation - decision on
  resweeping deferred until the outer-lr investigation settles.
- Item-representation/parameter-count table still not added to main.tex.
- Gromov delta section (4.1) and empty Conclusion - still deferred.

---

**2026-08-29: two more generations of the convergence fix on Amazon
Beauty (realfix, then patience=30/"p30"), Euclidean baseline finally
recomputed under a matching criterion, and an outer_epochs=200 confirmation
test in progress.** Context: after the 2026-08-27 hrfix numbers below, the
user pushed further - found two more real bugs (inner loop's `should_stop`
was still loss-based even under `select_by="hr"`; outer loop never
restored the best-scoring `isomap_model`, always used the last of 30 outer
steps). Both fixed (commit `391c525`). Then tested raising inner+final
patience from 8 to 30 (cap 60→200) - single-eta test showed a genuine
rise-plateau-overfit cycle for the first time. Full 4-eta "p30" sweep
completed cleanly (zero guard triggers).

**Amazon Beauty, final test HR@10 across all four generations:**

| eta | original | hrfix (2026-08-27) | realfix | p30 |
|---|---|---|---|---|
| 0.01 | 0.087 | 0.123 | 0.157 | 0.167 |
| 0.03 | 0.117 | 0.150 | 0.177 | 0.147 |
| 0.05 | 0.080 | 0.180 | 0.140 | **0.193** |
| 0.10 | 0.117 | 0.157 | 0.180 | 0.170 |

Not uniform: eta=0.01/0.05 keep improving generation over generation,
eta=0.03/0.10 regress slightly from realfix to p30 - most likely outer-loop
noise (fresh random NCF-proxy init every outer step), not a real patience
regression, but not isolated/confirmed.

**Euclidean baseline was stale (0.06) for the wrong reason.** The
`save_euclidean_baseline_geometry.py` CLI already had `--select_by
--patience --epochs` (fixed earlier, commit `3a0d03b`, contrary to a
mid-session note that it was still missing them - false alarm, verified by
reading the file). Rerun explicitly under patience=30/epochs=200/select_by=hr
(`euclidean_baseline_geometry_amazon_beauty_p30fix.npz`) gives **HR@10=0.1767,
NDCG@10=0.0778** - much higher than the 2026-08-27 table's 0.0600 (which
used patience=8/epochs=60, matching that generation's GINCF settings, so it
was internally consistent for hrfix, just not for later generations). The
jump makes sense: Euclidean NeuMF has ~12x more free parameters per item
(80 vs ~6.5 shared-projection, see the item-representation table) and
plausibly needs the longer budget more than the manifold arms do.

**p30-generation GINCF vs the freshly-corrected Euclidean baseline
(0.1767):** eta=0.05 (0.193) still wins; eta=0.01/0.10 (0.167/0.170) now
lose narrowly; eta=0.03 (0.147) loses more clearly. So on 3 of 4 eta the
corrected comparison says Euclidean wins or ties - this is the fair,
apples-to-apples version of what the user suspected all along ("оптимизированная
геометрия проигрывает евклидовой").

**Outer-loop restored-best-checkpoint fix is doing real work, confirmed
again on p30:** last-outer-step val_hr vs restored-best val_hr per eta:
0.01: 0.080 vs 0.217; 0.03: 0.153 vs 0.200; 0.05: 0.097 vs 0.173; 0.10:
0.083 vs 0.157 - using the last step instead of the best would have roughly
halved every result.

**In progress:** `run_amazon_beauty_outer200_test.py` (new `outer_epochs`
param added to `run_eta_outer_sweep.run_one`/`main`) - single eta (0.03),
outer loop run for 200 steps instead of 30 (inner loop unchanged:
select_by=hr, patience=30, cap=200), to check whether the outer trajectory
shows a real trend over a much longer horizon or stays noisy, and whether
the inner NCF proxy converges well at every one of those 200 steps
regardless of which manifold it's fitting. Estimated ~3-4h wall clock.
First launch attempt crashed instantly on a `UnicodeEncodeError` (a `→`
character in a print statement, Windows console cp1251 codepage) -
relaunched with `PYTHONIOENCODING=utf-8`, confirmed past the crash point.
Not yet analyzed - full inner+outer convergence graph summary still to be
built once it finishes, plots to go in `process_docs` as usual.

**All plots for this update** (`inner_loop_stopping_histogram_amazon_p30.png`,
`outer_loop_trajectories_all4_amazon_p30.png`, `four_generations_comparison_amazon.png`)
already delivered to `process_docs`.

**Still open / not yet decided:**
- Whether/how to isolate inner-loop-patience effect from final-NCF-patience
  effect (both were raised from 8→30 together).
- Whether the p30 generation's non-uniform eta=0.03/0.10 regression is real
  outer-loop noise or something to dig into further.
- ML-1M/ML-10M still sit at the hrfix (2026-08-27) generation - not yet
  resweeped with realfix or p30. Decision deferred until the outer_epochs=200
  confirmation test settles what "correct" methodology to standardize on.
- Item-representation/parameter-count table still not added to main.tex's
  experimental-setup section (explicitly requested, still outstanding).
- Gromov delta section (4.1) and empty Conclusion - still deferred.

---

**2026-08-27: HR-fix propagated to all three datasets - controlled re-eval
done, main.tex fully updated with final numbers.** Both long sweeps
(ML-1M ~7h, ML-10M ~15h) finished cleanly (zero guard triggers
throughout). Ran the same controlled seed=0 post-hoc re-eval used for
Amazon Beauty against both: `ablation_geometry_vs_optimization.py
--select_by hr --patience 8 --epochs 60` per eta config, plus fresh
Poincare/Euclidean baselines under identical settings
(`save_euclidean_baseline_geometry.py` needed the same CLI params added,
commit `3a0d03b`).

**Final corrected numbers, all three datasets, all under the same
HR@10-based selection criterion:**

| Dataset | Best GINCF | Pure init | Poincare | Euclidean (winner) |
|---|---|---|---|---|
| ML-1M | eta=0.03: 0.3933/0.2583 | 0.3400/0.2218 | 0.2767/0.1691 | **0.4867/0.3469** |
| ML-10M | eta=0.01: 0.3344/0.2058 | 0.3144/0.2032 | 0.1906/0.1186 | **0.4950/0.3586** |
| Amazon Beauty | eta=0.05: **0.1800/0.0836** | 0.1233/0.0551 | 0.1000/0.0367 | 0.0600/0.0288 |

**Important correction to my in-process report from earlier today:** I
initially got excited that ML-1M's in-process eta=0.01 number (0.3700)
nearly matched the OLD Euclidean baseline number (0.3633), suggesting
GINCF might be closing the gap. This was premature - the Euclidean
baseline ALSO jumped dramatically once recomputed under the same fixed
criterion (0.3633 -> 0.4867 for ML-1M, 0.4415 -> 0.4950 for ML-10M). The
old criterion was shortchanging every arm, not just GINCF, so the
*relative* gap between GINCF and Euclidean is essentially unchanged on
both MovieLens datasets - Euclidean still wins decisively. Determinism
re-confirmed throughout: pure_init identical across all 4 eta reruns per
dataset, Poincare/Euclidean reproduce their earlier-session numbers
exactly when rerun with matching settings.

**What changed vs. what didn't, per dataset:**
- **ML-1M:** the eta-ablation is no longer monotonic - eta=0.03 (not
  eta=0.01) is now the best GINCF config, and it's the only one that
  beats pure_init (previously none did). Direction of "which eta is
  worst" also isn't simply eta=0.10 anymore.
- **ML-10M:** eta=0.01 still wins among GINCF configs (unchanged), but
  the ordering of the others changed - eta=0.10 now beats eta=0.03 and
  eta=0.05, breaking the old monotonic "higher eta worse" pattern.
- **Amazon Beauty:** already reported previous update - eta=0.05 (not
  eta=0.01) wins, monotonicity fully gone.
- **Universal across all three:** Euclidean vs GINCF's *qualitative*
  outcome is unchanged (Euclidean wins on both MovieLens scales, GINCF
  wins on Amazon Beauty) - only the exact eta rankings and absolute
  numbers shifted. The core paper narrative survives fully intact and
  the evidence for it is now more rigorous.

**main.tex fully updated** (`tab:corrected_results`, `tab:full_diagnostics`,
`tab:ml10m_results`, `tab:ml10m_diagnostics` all replaced with final
numbers; interpretive paragraphs rewritten to match - e.g. the old "eta=0.01
is the exception, hyperbolicity doesn't increase for it" claim no longer
holds with the new data, replaced with an accurate description of the new
delta_rel/ORC pattern). Moved the HR@10-selection methodology explanation
to one central place (Section 3.2.5, "Evaluation Protocol and Metrics")
instead of repeating it per-dataset; `subsec:amazon`'s local copy trimmed
to a one-line back-reference. Verified brace balance (0 issues) and
grepped for stale old numbers elsewhere in the paper (none found) after
every edit.

**Still open:**
- Gromov delta section (4.1, "Experiment 49-52" numbering) and empty
  Conclusion - still deferred, unchanged from before, now probably the
  main remaining task before the paper is feature-complete.
- The outer-loop trajectory noise question (flagged a few updates ago,
  Amazon Beauty specific) - never revisited under the new criterion.
- No LaTeX compiler on this machine - user should compile main.tex to
  confirm it renders correctly before treating any of this as final.

---

**2026-08-27: ML-1M + ML-10M HR-fix resweeps running (long background
job); added a Limitations subsection to main.tex while waiting.**

User confirmed (2026-08-26 night) she wants the HR-based model-selection
fix propagated to ML-1M and ML-10M too, not just Amazon Beauty. Launched
`run_ml1m_sweep_hrfix.py` then `run_ml10m_sweep_hrfix.py` sequentially in
one background Monitor (commit `557a5a1`) - same settings as the Amazon
Beauty fix (`select_by="hr"`, `inner_patience=8`, `final_patience=8`,
`cf_epochs=60`, `final_cf_epochs=60`), distinct n_run_prefixes
(`eta_sweep_hrfix_*`, `ml10m_eta_sweep_hrfix_*`) so the originals stay
available for comparison.

**ML-1M HR-fix sweep: COMPLETE.** All 4 configs finished cleanly (zero
guard triggers), ~1h40m-2h25m per config (slower than the original
~50min-1h given the wider budget). In-process summary (NOT yet the
controlled seed=0 re-eval used for every other "corrected" table in this
paper - see caveat below):

| eta | HR@10 | NDCG@10 |
|---|---|---|
| 0.01 | 0.3700 | 0.2349 |
| 0.03 | 0.3433 | 0.2288 |
| 0.05 | 0.3167 | 0.1708 |
| 0.10 | 0.2933 | 0.1851 |

Direction (higher eta worse) is UNCHANGED from the original ML-1M
finding - unlike Amazon Beauty, this pattern survived the fix. But the
magnitude jumped enormously: old corrected eta=0.01 was 0.2567/0.1519,
clearly behind the Euclidean baseline's 0.3633/0.2354; new in-process
eta=0.01 is 0.3700/0.2349 - now essentially tied with the old Euclidean
number. This is a big, promising signal but NOT yet confirmed - the
final-NCF stage inside the live sweep does not reset its RNG seed the
way `ablation_geometry_vs_optimization.py`'s post-hoc re-eval does, so
these in-process numbers are not directly comparable/reproducible the
way every other "corrected" result in this paper is. **Next step once
ML-10M finishes: run the same post-hoc seed=0 re-eval (`ablation_geometry_vs_optimization.py --select_by hr --patience 8 --epochs 60`)
against the saved `eta_sweep_hrfix_*` Z snapshots, plus fresh
Poincare/Euclidean baselines under matching settings, exactly as already
done for Amazon Beauty - only then update `tab:corrected_results`/
`tab:full_diagnostics` in main.tex.**

**ML-10M HR-fix sweep: in progress.** eta=0.01/0.03/0.05 done (~3h35m-3h44m
each, zero guard triggers), eta=0.10 running as of this update. Given the
per-config pace, expect total sweep time roughly 14-16h - deliberately
NOT running the ML-1M post-hoc re-eval concurrently with this (would
contend for the same GPU); doing both re-evals together once ML-10M's
sweep itself finishes.

**Added to main.tex while waiting** (new `subsec` "Limitations: Memory,
Scalability, and Cold Start", before Conclusion, in blue): three points,
prompted by the user's own questions about production applicability -
(1) the shared-projection architecture reduces the CF head's own
trainable-parameter/optimiser-state footprint vs.\ a free embedding
table, but does NOT reduce the stored per-item representation size ($Z$
is still $O(n \times d)$); (2) the real scalability bottleneck is the
geometry-\emph{search} stage itself ($D_{\text{input}} \in
\mathbb{R}^{n\times n}$, Floyd--Warshall) - quadratic memory, cubic
compute, infeasible at production catalogue sizes without algorithmic
changes (landmark/sparse-graph Isomap) not implemented here; (3)
cold-start is asymmetric - item representations come from rating-profile
features (a genuine, untested potential advantage for new items), but
user representations are ordinary randomly-initialised learnable
embeddings in every tested configuration, so new-user cold start is
exactly as unsolved as in vanilla NeuMF. Verified brace balance (0
issues), no bug-narrative language.

**Still open:** the post-hoc re-eval for both ML-1M and ML-10M (see
above), then update `tab:corrected_results`/`tab:ml10m_results` and
their diagnostics tables with final numbers. Gromov delta section (4.1)
and empty Conclusion still deferred.

---

**2026-08-26, night: Amazon Beauty fully resweept with the HR-fix - the
winning eta CHANGED, confirming the user's concern that fixing only
evaluation (not the actual bilevel search) wasn't enough.** User asked to
propagate the select_by="hr" fix to all 4 eta configs and literally
rerun the geometry-search sweep (not just re-evaluate saved Z snapshots),
specifically to see what happens to hyperbolicity once the search itself
uses the corrected inner-loop criterion throughout.

Ran `run_amazon_beauty_sweep_hrfix.py` (new driver, `select_by="hr"`,
`inner_patience=8`, `final_patience=8`, `cf_epochs=60`,
`final_cf_epochs=60` - up from patience=3-5/cap=30), n_run_prefix
`amazon_beauty_hrfix_eta_sweep_*` (kept separate from the original
`amazon_beauty_eta_sweep_*` results). All 4 configs finished cleanly,
zero guard triggers, ~45-55min total (slightly slower than the original
~48min sweep due to the larger epoch/patience budget, but not
dramatically - the inner loop mostly still exits well before the new
60-epoch cap).

**Corrected results (300u/800i, HR@10/NDCG@10/val_loss):**

| Model | Old (patience=3) | New (patience=8, full resweep) |
|---|---|---|
| pure_init | 0.0400/0.0162/0.3617 | 0.1233/0.0551/0.3679 |
| GINCF eta=0.01, converged | 0.1133/0.0500 | 0.1233/0.0535/0.3603 |
| GINCF eta=0.03, converged | 0.0500/0.0199 | 0.1500/0.0775/0.3833 |
| **GINCF eta=0.05, converged** | 0.0667/0.0264 | **0.1800/0.0836/0.3440** |
| GINCF eta=0.10, converged | 0.0467/0.0198 | 0.1567/0.0782/0.3931 |
| Poincare-Pretrained | 0.0600/0.0270 | 0.1000/0.0367/0.5637 |
| Euclidean NeuMF (baseline) | 0.0567/0.0253 | 0.0600/0.0288/0.6077 |

**The winning eta changed from 0.01 to 0.05.** This is the key finding
that validates the user's instinct: my earlier "quick pilot" (re-evaluating
the OLD sweep's already-saved eta=0.01 Z snapshot with the new criterion)
found eta=0.01 still winning (0.1767) - but that Z was found by a bilevel
search that used the OLD, flawed inner-loop criterion throughout its 30
outer steps. Once the ENTIRE search is redone with the fixed criterion,
the actual optimum shifts to eta=0.05 (0.1800), and the "higher eta is
worse" monotonic pattern (robust across every other config/dataset in
this paper) disappears entirely for Amazon Beauty. Fixing only the final
evaluation stage was NOT equivalent to fixing the actual optimization -
exactly what the user flagged before I ran this.

**The core finding survives, and strengthens further:** GradientIsomapNCF
(now eta=0.05) still beats the Euclidean baseline outright (0.1800 vs
0.0600 - a 3x margin, up from ~2x). Poincare still doesn't win (0.1000).
Validation loss still tracks ranking quality throughout (winning
eta=0.05 has the lowest val_loss of the whole table, 0.3440; Euclidean
has the highest, 0.6077). Hyperbolicity still doesn't explain the
winner - Euclidean's ORC (-0.1895, even more negative than before) and
H1 count (1552, up from 1318) are still the most extreme in the table,
Poincare is still the most tree-like by delta_rel (0.1694, unchanged
since its fit doesn't depend on the sweep), yet neither wins downstream.

**Also fixed along the way:** `train_and_eval_euclidean_baseline` could
previously return either item embeddings OR training history but not
both in the same call - needed both to persist a consistent geometry +
convergence record for this rerun (commit `bb04d3c`). Regenerated
`poincare_fitted_geometry_amazon_beauty.npz` and
`euclidean_baseline_geometry_amazon_beauty.npz` with the corrected
downstream metadata (D matrices unchanged - both fits are deterministic
and reproduced their prior numbers exactly, confirming no drift).

**main.tex's `subsec:amazon` fully rewritten** with the new numbers
(both `tab:amazon_results` and `tab:amazon_diagnostics`), plus one new
paragraph explaining the HR@10-based model-selection choice (methodology,
not a bug narrative - matches the user's "clean results only" standard).
Verified brace balance (0 issues), no bug/checkpoint narrative language.
Comparison plot (`old_vs_hrfix_sweep_amazon_beauty.png`) delivered to
`process_docs` alongside the rest of this investigation's plots.

**Still open / not yet decided:**
- ML-1M and ML-10M still use the OLD loss-based criterion throughout the
  paper - not yet decided whether/when to redo those (each is a much
  larger time commitment: ML-1M ~3.5h, ML-10M ~8h, vs. Amazon Beauty's
  ~50min). User has not yet asked for this explicitly.
- The outer-loop trajectory noise (flagged two updates ago) - not
  re-examined with the new criterion; could revisit if it becomes
  relevant.
- Gromov delta section (4.1, "Experiment 49-52" numbering) and empty
  Conclusion - still deferred, unchanged from before.

---

**2026-08-26, evening: model-selection criterion investigation - a real
methodological fix, and it turned out to STRENGTHEN the Amazon Beauty
finding rather than undermine it.** User noticed the "final NCF" loss
convergence plots looked wrong (val loss below train loss from epoch 1,
noisy early-stopping trigger) and pushed back hard on my first two
explanations before we found the real cause.

**Root cause, confirmed in code:** train batches use `num_ng=2` (1
positive : 2 negatives, ~33% positive rate); val/test batches use
`num_ng=99` (1:99, ~1% positive rate, the standard sampled-ranking
protocol). BCE loss averaged over these two batch compositions is not on
a comparable scale - a batch dominated by 99 easy negatives has
structurally lower average loss almost regardless of ranking quality.
Both the "final NCF" stage's hand-rolled early stopping (`patience_final
= 3`, loss-based) and the actual bilevel sweep's `EarlyStopping` class
used this scale-mismatched val loss for "best checkpoint" selection.
Literature check confirmed the standard fix: He et al.'s NCF and most
follow-ups select on val HR@10/NDCG@10 (computed on the same sampled
batches) instead of raw val loss.

**User's sharper catch (initially I got this wrong):** I first claimed
fixing this would be "cheap" (just re-evaluate already-saved Z snapshots)
- she correctly pointed out that if the *inner* per-outer-step NCF proxy
was also under-trained by the same flawed criterion, the gradient signal
used to update `D_input` at every one of the 30 outer steps would be
unreliable too, meaning the whole bilevel search trajectory - not just
the final evaluation - could be compromised. Checked this directly:
- The **final-NCF stage** (produces every reported test_hr/test_ndcg,
  all 3 datasets) uses the simple hand-rolled criterion - confirmed
  broken, this is the part that needed fixing.
- The **inner per-outer-step loop** uses the more sophisticated
  `EarlyStopping` class (window-based, restores best-of-last-6-epochs,
  not global best) - checked actual `cf_history.json` lengths from the
  real Amazon Beauty sweep: mostly ran 22-30 of the 30-epoch cap (not the
  2-9 epochs seen in the final-stage bug), so the inner proxy was
  reasonably well-trained, not severely undertrained. This means the
  outer bilevel search trajectory (the saved Z snapshots) is probably
  more trustworthy than initially feared - a full resweep of the
  expensive multi-hour outer loops is likely NOT required, though the
  inner loop's criterion was fixed too for consistency/future runs.

**Fix implemented** (commit `8eda4d6`): added `select_by="loss"/"hr"` to
`GradientIsomapCF` (new `final_patience`/`inner_patience` params, both
loops now also track val HR@10), threaded through `run_experiment.py` and
`poincare_baseline.py`. Relabeled "pure_init" to "pure_init (euclidean)"
in diagnostic table row labels (disambiguation, since a Poincare
convergence curve now exists alongside it).

**Quick verification on Amazon Beauty** (`select_by="hr"`, patience=8,
epoch cap=60, up from patience=3/cap=30) - all 4 final-NCF arms
recomputed with real per-epoch history saved:

| Model | HR@10 (old, patience=3) | HR@10 (new, patience=8, HR-select) | NDCG@10 (new) |
|---|---|---|---|
| Pure init (euclidean) | 0.0400 | 0.1233 | 0.0551 |
| **GINCF eta=0.01, converged** | 0.1133 | **0.1767** | **0.0810** |
| Poincare-Pretrained | 0.0600 | 0.1000 | 0.0367 |
| Euclidean NeuMF (baseline) | 0.0567 | 0.0600 | 0.0288 |

**The gap did not close - it widened.** GINCF was already the best arm
under the old (flawed) criterion; under fair, well-tuned selection it
pulls further ahead of every alternative, including the Euclidean
baseline (which barely moves: 0.0567->0.0600, converges/plateaus within
~9 epochs regardless of patience budget - genuinely fast-converging on
this sparse dataset, not cut off early). Convergence curves (saved to
`process_docs`, see below) show GINCF's val HR@10 still rising smoothly
out to epoch ~26-34, while Poincare and the Euclidean baseline plateau
by epoch ~9 - real evidence of headroom the old criterion was missing,
not noise.

**Plots delivered to** `C:\Users\Julia\Documents\NSS_lab\документы\2027
WWW Recsys\process_docs\` (SendUserFile doesn't open for the user - see
[[feedback_show_via_process_docs_folder]] memory):
- `convergence_final_ncf_amazon_beauty.png` - original 3-arm loss curves
  that triggered the investigation.
- `outer_loop_trajectory_amazon_beauty.png` - all 4 eta outer-loop
  trajectories (still noisy across all 30 outer steps - a separate,
  smaller open question, see below).
- `convergence_loss_vs_hr_selection_amazon_beauty.png` /
  `final_results_loss_vs_hr_selection_amazon_beauty.png` - first
  pure_init-only diagnostic (patience=3, before widening patience too).
- `convergence_all_4_arms_fixed_amazon_beauty.png` /
  `before_after_fix_amazon_beauty.png` - final 4-arm verification
  (patience=8, cap=60) referenced above.

**Added to main.tex** (Section 3.2, new "Model Comparison Summary"
subsubsection + `tab:model_comparison`, in blue): item-representation
source and parameter count per arm - Euclidean baseline has ~12x more
item-specific free parameters (64,000 vs 5,200 shared) at n=800, growing
linearly with item count while the geometry-based arms' capacity stays
fixed. Makes the capacity-mismatch hypothesis already in
`subsec:loss`/`subsec:amazon` quantitative rather than qualitative.
Verified brace balance (0 issues), fixed a hardcoded "Eq.~2" reference to
use a proper `\label`/`\ref` instead (no compiler available to verify
numbering by hand).

**Still open:**
- The outer-loop trajectory plot is still noisy across all 30 steps even
  after this investigation - not yet fully explained (could be genuine
  geometry-search noise, could be residual noise from re-initializing a
  fresh NCF proxy every outer step regardless of selection criterion).
  Not urgent per the finding above (inner loop trains close to budget),
  but flagged as a real open question if revisited.
- Amazon Beauty's `select_by="hr"` verification used only 1 of the 4 eta
  configs (0.01, the winner) plus pure_init/Poincare/Euclidean - the
  other 3 etas (0.03/0.05/0.10) and the ML-1M/ML-10M numbers throughout
  the paper still reflect the OLD loss-based criterion. Not yet decided
  whether/how to propagate this fix to those (diary + user should discuss
  scope before doing a full repaint of every table).
- `verify_hr_fix_amazon_beauty.json`, `poincare_convergence_history*.json`
  not yet committed as of this being written (code is committed, these
  result artifacts are not) - low priority, matches existing pattern of
  leaving raw run outputs untracked.

---

**2026-08-26, later: Amazon Beauty sweep complete + corrected + written up
- the paper's headline new finding this session.** The real sweep
(Monitor `bzjxeueqh`, filtered) finished all 4 configs in ~48 minutes
total (much faster than either MovieLens scale - Amazon interactions are
sparser per user, so the negative-sampling/training loop does less work
per batch). Zero guard triggers, zero crashes.

Recomputed with the checkpoint-bug fix (same playbook as ML-10M) - full
output captured to log files this time from the start, no truncated
`tail` mistake repeated. Also fit Poincare + Euclidean baselines
(`--tag amazon_beauty`) and built the unified table via the new
**generalized** `full_hyperbolicity_table_generic.py` (replaces the
per-dataset-copy-paste pattern flagged as tech debt in the previous
update - takes an eta-folder prefix + a downstream-metrics JSON side file
instead of being hardcoded per dataset). Also fixed a real gap found
along the way: `ablation_geometry_vs_optimization.py`'s **argparse CLI**
(`main()`) was missing the `--dataset_type`/`--amazon_category` flags -
only the underlying `build_data()` function had them from the earlier
generalization commit, so the first recompute attempt failed with
"unrecognized arguments" until this was caught and fixed (commit
`bea3d61`).

**Corrected results (HR@10/NDCG@10), 300 users/800 items:**

| Model | HR@10 | NDCG@10 | val_loss |
|---|---|---|---|
| Pure Euclidean-init Isomap $Z$ | 0.0400 | 0.0162 | 0.3617 |
| GINCF $\eta$=0.01, epoch0 | 0.0533 | 0.0232 | 0.3601 |
| **GINCF $\eta$=0.01, converged** | **0.1133** | **0.0500** | **0.3483** |
| GINCF $\eta$=0.03, converged | 0.0500 | 0.0199 | 0.3620 |
| GINCF $\eta$=0.05, converged | 0.0667 | 0.0264 | 0.3654 |
| GINCF $\eta$=0.10, converged | 0.0467 | 0.0198 | 0.3673 |
| Poincare-Pretrained | 0.0600 | 0.0270 | 0.3703 |
| Euclidean NeuMF (baseline) | 0.0567 | 0.0253 | 0.3748 |

**This is the first case across all three scale/domain combinations tested
in this paper where GradientIsomapNCF outright beats the Euclidean
baseline** - not just narrows the gap (ML-10M) or loses cleanly (ML-1M).
$\eta=0.01$ converged wins on both HR@10/NDCG@10 AND has the lowest
val_loss of the whole table - genuinely converged, not noise. The
Euclidean baseline, which won decisively at both MovieLens scales, has
the *highest* val_loss here.

**Crucially, this win is NOT explained by hyperbolicity** - pure_init
already has a more negative ORC mean than the winning eta=0.01 geometry,
and Poincare (still the most tree-like row by delta_rel=0.1694) doesn't
win either. The Euclidean baseline actually has the most negative ORC AND
the most H1 cycles (1318) of any row here, despite the worst loss -
hyperbolicity and downstream quality are essentially decoupled at this
scale/domain. Working hypothesis written into the paper: Amazon's sparser
interactions leave fewer effective training positives at this pool size
than MovieLens does, so a free embedding table's extra capacity becomes a
liability (overfits) rather than an asset, and the manifold constraint
acts as an implicit regulariser - same capacity-mismatch lens as
Section subsec:loss, just flipped by data density instead of by scale.

Added `subsec:amazon` to main.tex (after `subsec:ml10m`, before
Conclusion) - results table, full diagnostics table, two interpretive
paragraphs, all in blue. Added `hou2024bridging` (arXiv:2403.03952, the
Amazon Reviews'23 dataset paper) to main.bib. Verified brace balance in
both main.tex and main.bib (0 issues), no bug-narrative language.
Committed: `bea3d61` (code), diary not yet committed as of writing this -
do that next.

**This closes the "n=1 dataset" gap from the Abstract**: the paper now
has 3 scale/domain configurations (ML-1M, ML-10M-at-1800-items, Amazon
Beauty) across 2 distinct domains (movies, e-commerce reviews), with a
real, defensible cross-cutting finding (capacity vs. data density
determines whether geometry search or free embeddings win - not
hyperbolicity per se) rather than three disconnected results tables.

**Still open** (from before, largely unchanged): Gromov delta section
(4.1, "Experiment 49-52" numbering ambiguous, needs the user to clarify
before I touch original non-blue text), and the empty Conclusion (now has
even more material to synthesize - the capacity/density story above is
probably the single most citable insight to lead with).

---

**2026-08-26 update: second dataset chosen and integrated - Amazon Beauty.**
User confirmed Amazon over Yelp (structurally closer to the existing
rating+timestamp pipeline; Yelp's more interesting angle - the
user-friendship graph - would need pipeline changes to actually use, not
just a different data source), and picked the "Beauty" category. The
modern (2023) McAuley-Lab dataset renamed the old "Beauty" benchmark to
**Beauty_and_Personal_Care** (729.6K users / 207.6K items / 6.6M ratings
raw, before our subsampling) - "All_Beauty" is a different, much smaller
subcategory (253 users after 5-core filtering) and would not have worked.
Source: https://amazon-reviews-2023.github.io/ (5-core pure-ID CSV,
`user_id,parent_asin,rating,timestamp`), downloaded from
`mcauleylab.ucsd.edu` (112MB gzipped, well under the 20GB ceiling - first
download attempt was truncated by a dropped connection, caught via a
size check and retried with `-C -` resume until it matched the expected
Content-Length).

Pipeline changes (commit `fab9ffd`): added `load_amazon_ratings()` to
`run_experiment.py`, producing the same `userId/movieId/rating/timestamp`
column contract `load_movielens_ratings()` already produces (so
`prepare_sequences()`/`subsample_users_items()` work unmodified on either
source - string IDs sort/map fine, no int-ID assumption anywhere in that
path). Threaded a new `dataset_type` ("movielens"/"amazon") parameter
through `run_experiment.main()`, `run_eta_outer_sweep.py`, and
`ablation_geometry_vs_optimization.py`'s `build_data()` (needed later for
the corrected-checkpoint recompute + hyperbolicity table, same treatment
ML-10M got). Smoke-tested end-to-end (50 users/200 items, 1 outer epoch) -
clean run, no errors beyond the already-known Windows console
`UnicodeEncodeError` (fixed the same way as before, `PYTHONIOENCODING=utf-8`
- not a code issue).

**Real sweep launched** (`run_amazon_beauty_sweep.py`, Monitor task
`bzjxeueqh`, filtered this time to only surface config boundaries/warnings/
errors - the ML-10M sweep's unfiltered per-epoch stream was excessive):
300 users, 800 items - same scale as the paper's original ML-1M
configuration, chosen deliberately for the most direct "same pipeline,
different domain" comparison. 4 configs (eta in {0.01, 0.03, 0.05, 0.10}),
~3-4h estimated by analogy to ML-1M's own ~3.5h at this scale. Results land
in `logs_movielens_isomap_cf/amazon_beauty_eta_sweep_<eta>/` +
`amazon_beauty_eta_outer_sweep_summary.json`.

**Next steps once the sweep finishes** (same sequence already used twice
for ML-1M then ML-10M - this is now a repeatable playbook, not a new
design each time):
1. Recompute final HR@10/NDCG@10 via `ablation_geometry_vs_optimization.py
   --dataset_dir_name amazon_beauty --dataset_type amazon --max_movies 800`
   for the checkpoint-bug fix (the running sweep process has the old
   checkpoint code in memory, same as ML-10M's first pass did).
2. Fit a Poincare baseline and an Euclidean baseline at this scale/dataset
   (`poincare_baseline.py` / `save_euclidean_baseline_geometry.py`, both
   already `--tag`-parameterized - use `--tag amazon_beauty`).
3. Run `full_hyperbolicity_table_ml10m.py`'s pattern a third time
   (probably worth generalizing into one script with a `--tag`/config
   argument at this point, rather than a third near-duplicate file).
4. Add a `subsec:amazon` section to main.tex, in blue, same structure as
   `subsec:ml10m` - this closes the paper's "Датасетов: n=1" gap from
   n=1 (MovieLens only) to n=3 scales across 2 domains.

---

**2026-08-26 update: ML-10M full hyperbolicity+loss table done, closing the
"Still open" item from the previous block.** Parameterized
`poincare_baseline.py` and `save_euclidean_baseline_geometry.py` with a
`--tag` option (backward-compatible, empty tag preserves the original
ML-1M filenames) so a second scale's fitted geometries don't clobber the
first's. Ran both at ML-10M scale (300u/1800i) - reproduced the exact same
downstream numbers as the untagged earlier run (determinism confirmed
again). New `full_hyperbolicity_table_ml10m.py` (mirrors
`full_hyperbolicity_table.py`, reuses its `diagnostics_for_D()`) computed
all 11 rows. Also re-ran the 3 remaining eta configs' ablation with full
output captured to log files (the first pass only had `tail -6` output,
missing epoch0's exact val_loss for etas 0.03/0.05 - re-ran rather than
guess).

**Key finding, sharper than at ML-1M:** the Poincare-Pretrained geometry is
again the most hyperbolic by every measure ($\delta_{rel}=0.2631$, ORC
mean$=-0.2288$, f_neg$=0.985$) but now has both the highest validation loss
AND the worst HR@10/NDCG@10 of every row in the table - worse than every
GINCF configuration, not just the Euclidean baseline. At $n=1800$ persistent
homology's $H_1$ is skipped everywhere (safety threshold exceeded, all
rows show H1=-1) - noted as such in the paper table rather than silently
omitted or faked.

Added to main.tex's `subsec:ml10m`: the Poincare row folded into
`tab:ml10m_results`, plus a new `tab:ml10m_diagnostics` table (mirrors
`tab:full_diagnostics`) and one interpretive paragraph. Verified brace
balance (0 issues) and no bug-narrative language. Not yet committed as of
this being written - do that next, along with the parameterized-script
diffs (`poincare_baseline.py`, `save_euclidean_baseline_geometry.py`) and
the new `full_hyperbolicity_table_ml10m.py`.

---

**2026-08-25 evening update: ML-10M sweep complete + corrected + a genuinely
new finding.** The ML-10M eta_outer sweep (300 users, 1800 items - "a
different MovieLens scale" per user's earlier instruction) finished all 4
configs cleanly (zero guard triggers, zero crashes, ~8h total). Its
in-process HR@10/NDCG@10 numbers still had the checkpoint bug (the running
process had the old code loaded in memory even though the source was
already fixed) - recomputed all 4 configs plus a matching Euclidean
baseline via `ablation_geometry_vs_optimization.py --dataset_dir_name
ml-10m --max_movies 1800` (same corrected-checkpoint methodology already
used for ML-1M). Corrected results:

| Model | HR@10 | NDCG@10 |
|---|---|---|
| Euclidean NeuMF (baseline) | **0.4415** | **0.2985** |
| Pure Euclidean-init Isomap Z (0 outer steps) | 0.2542 | 0.1394 |
| GINCF eta=0.01, epoch0 | 0.2341 | 0.1296 |
| GINCF eta=0.01, converged | 0.2910 | 0.1658 |
| GINCF eta=0.03, epoch0 | 0.2676 | 0.1429 |
| GINCF eta=0.03, converged | 0.2475 | 0.1261 |
| GINCF eta=0.05, epoch0 | 0.2910 | 0.1771 |
| GINCF eta=0.05, converged | 0.2408 | 0.1265 |
| GINCF eta=0.10, epoch0 | 0.2375 | 0.1362 |
| GINCF eta=0.10, converged | 0.2107 | 0.1047 |

Two things carry over from ML-1M and one is new:
1. **Carries over:** Euclidean NeuMF still wins by a wide margin at this
   larger scale too - the "no geometry-aware model beats a plain learnable-
   table baseline" finding is not an ML-1M-specific artifact.
2. **Carries over:** the eta-ablation direction (higher eta = worse at
   convergence) still holds: 0.01 (0.2910) > 0.03 (0.2475) > 0.05 (0.2408)
   > 0.10 (0.2107) for HR@10.
3. **NEW, differs from ML-1M:** at ML-1M every eta's converged result was
   worse than pure_init. Here, eta=0.01's converged result (0.2910/0.1658)
   clearly BEATS pure_init (0.2542/0.1394) - the first case across both
   scales where the outer-loop optimisation genuinely helps over doing
   nothing. It still falls well short of the Euclidean baseline, but this
   is a real, scale-dependent signal, not noise (pure_init itself is
   identical across all 4 eta reruns, 0.2542/0.1394 - a good determinism
   sanity check, since it's recomputed independently each time from the
   same D_input_init.npy + seed).

Added this as a new subsection in main.tex (`subsec:ml10m`, in blue) -
second-scale validation, fulfilling "Используй другой масштаб MovieLens"
from the user's earlier instruction. Diary + code not yet committed as of
this block being written - do that next.

**Still open:**
- Second, distinct-domain dataset (Amazon/Yelp) - never confirmed with the
  user, still a real blocker per the earlier "Two real blockers" note below.
- The full hyperbolicity-diagnostics table (`full_hyperbolicity_table.py`
  pattern) has NOT yet been run for the ML-10M scale - only the ML-1M
  version is in the paper's `tab:full_diagnostics`. Worth doing if the
  ML-10M finding above becomes a permanent part of the paper's narrative,
  but not done yet.

---

**2026-08-25 update (responding to user's 4-part feedback on the previous
paper edit):** user asked for (a) bug-narrative language removed from
main.tex - only clean results+conclusions belong there, technical detail
stays in this diary; (b) investigate why hyperbolicity shows no downstream
benefit; (c) a full table with every hyperbolicity measure for both the
Poincare/geoopt manifold and our found manifolds, across sample sizes;
(d) loss values in that table too, since non-convergence (wrong inner/outer
lr) was a live worry.

Done, in order:
1. Renamed `subsec:reeval` from "A Critical Re-evaluation: Checkpoint Bug
   and Missing Baselines" to "A Refined Evaluation: Isolating the Outer
   Loop's Contribution", stripped the bug-narrative paragraph, replaced with
   clean framing. Verified (grep) no `bug|checkpoint|state_dict|deepcopy`
   language remains anywhere in the `\new{...}` blocks.
2. Convergence check (three representative points spanning the ranking -
   best case pure-init, worst case eta=0.10 converged, Euclidean baseline):
   all three converge cleanly, early-stopping well before the 30-epoch cap
   (pure_init: stop ep9, best val 0.2962 @ep6; eta=0.10 converged: stop ep6,
   best val 0.3356 @ep3; Euclidean: stop ep13, best val 0.2826 @ep10). No
   erratic/non-monotonic loss - rules out an lr/optimisation-failure
   explanation for the ranking. This was the final-CF-stage's own lr=1e-3;
   the bilevel outer loop's own lr *is* the swept eta_outer parameter
   (0.01-0.10) - not separately re-checked, but its geometric trajectories
   (Phase 6 log below) show smooth, gradual drift with no instability
   signature, which is itself evidence against an outer-loop lr problem.
3. Built `full_hyperbolicity_table.py` + persisted-geometry additions to
   `poincare_baseline.py` (new `poincare_fitted_geometry.npz` output) and a
   new `save_euclidean_baseline_geometry.py` (item-embedding geometry ->
   `euclidean_baseline_geometry.npz`) - now ALL 11 rows (pure_init, 4 etas x
   {epoch0, converged}, Poincare, Euclidean) run through the exact same
   `diagnostics_for_D()` pass: delta_rel, ORC mean/f_neg, lambda2/spectral
   gap, H1 count, val_loss, HR@10, NDCG@10. Reran Poincare/Euclidean fits
   fresh to persist them - both reproduced their previously-reported
   downstream numbers exactly (Poincare 0.2433/0.1322/0.3424,
   Euclidean 0.3633/0.2354/0.2826), confirming determinism.
   Committed: `21724a9`.
4. Wrote `\label{subsec:loss}` (previously just a forward-reference) in
   main.tex: convergence-check paragraph + the full 11-row table
   (`tab:full_diagnostics`, `table*`+`resizebox` for width) + two
   interpretive paragraphs, all in blue, no implementation narrative.
   Key finding surfaced there: validation loss tracks ranking quality
   throughout the table (Euclidean lowest loss+best ranking, Poincare
   highest loss despite being the single most hyperbolic geometry by every
   measure) - internally consistent, supports "genuine capacity-limited
   optimum" over "some arms didn't train properly". Also: the outer loop's
   effect on hyperbolicity is NOT uniform across eta - it increases
   hyperbolicity at eta=0.03/0.05/0.10 but *decreases* it at eta=0.01 (the
   single best-performing config) - suggests the small benefit from
   conservative outer steps isn't actually explained by increasing
   hyperbolicity. Verified brace balance (0 issues) after the edit.

**Still open from this round:**
- The bilevel outer loop's own inner-NCF lr (`lr_ncf` used during the
  actual 30-outer-epoch training, as opposed to this post-hoc
  fixed-Z ablation's lr=1e-3) was not directly re-verified against a
  convergence curve - worth doing if the capacity-mismatch explanation
  needs to be ruled out further.
- Two more datasets (per original instruction) still not started.
- ML-10M sweep (Monitor `b6fn4kmh8`) still running in background, unaffected
  by any of this - at outer epoch ~29/30 of its first (eta=0.01) config as
  of this update. Its final numbers will need the same corrected-checkpoint
  recompute once it finishes.
- `docs/recsys_paper_diary.md` itself was never actually committed to git
  despite being referenced by commit hash throughout this file - fixing
  that now alongside this update.

---

**Phase:** 3 done → in Phase 4 (build missing geometry diagnostics). Git
clean at `c6a0278` (nothing new committed yet this sub-session).

**Phase 4 is DONE and validated end-to-end on real n=800 data**
(`recsys/my_verification/geometry_diagnostics.py`, commits `40f711a`,
`0aa5551`, `a4915d6`). All 4 missing measures from the paper draft's
Section 3.3 now have real, working, tested code: Ollivier-Ricci curvature,
graph-Laplacian spectral gap, Kruskal stress/Spearman rho, persistent
homology H0/H1 - run against the real smoke-test matrices via `Monitor`
(live-streamed progress, no crash, ~5s/epoch) and producing sane numbers in
the same ballpark as the paper's own reported ranges (H1 count ~2000-2200 on
D_geodesic vs. the paper's 1291-2048).

**Phase 5 done** (commit `4a7dfab`): NaN/Inf-gradient guard ported into
`GradientIsomapCF`'s own outer loop (local code, `Adam/GradientIsomap.py`
untouched) - smoke-tested, `had_bad_grad` correctly logged per epoch and
inert (False) in the normal case.

**Phase 6 in progress:** `recsys/my_verification/run_eta_outer_sweep.py`
launched (Monitor task `brritxltc`, persistent, filtered) - sweeps
eta_outer in {0.01, 0.03, 0.05, 0.10}, 30 outer epochs each, full geometry
diagnostics per epoch. Full raw log:
`...scratchpad\eta_sweep_full_log.txt`. Per-config results land in
`recsys/my_verification/logs_movielens_isomap_cf/eta_sweep_<eta>/` +
`geometry_diagnostics.csv`; combined summary in
`recsys/my_verification/eta_outer_sweep_summary.json` (written
incrementally after each config).

**PHASE 6 COMPLETE (2026-08-24, ~18:05-21:38, ~3.5h, commit `e6a191d`).**
All 4 configs finished cleanly: zero guard triggers (`had_bad_grad=False`
throughout, all configs), zero crashes, zero hangs. Final held-out TEST-set
results:

| eta_outer | HR@10 (ours) | NDCG@10 (ours) | HR@10 (paper) | NDCG@10 (paper) |
|---|---|---|---|---|
| 0.01 | 0.3667 | 0.2286 | 0.2833 | 0.1621 |
| 0.03 | 0.2967 | 0.1675 | 0.2900 | 0.1515 |
| 0.05 | 0.2900 | 0.1578 | 0.2100 | 0.1093 |
| 0.10 | 0.2633 | 0.1502 | 0.2133 | 0.1138 |

**Key finding: our reproduction shows a clean MONOTONIC decrease in both
HR@10 and NDCG@10 as eta_outer increases - stronger/cleaner than the
paper's own numbers**, which are NOT monotonic (paper's eta=0.03 is
actually their single best config, slightly beating eta=0.01; eta=0.05 and
0.10 cluster much lower). Ours groups more gradually. Both agree on the
qualitative claim (low eta better than high eta) but disagree on the exact
shape - worth discussing honestly in the writeup as "directionally
consistent, quantitatively cleaner" rather than "identical reproduction".
Our absolute numbers also run systematically higher than the paper's across
every config (e.g. our eta=0.01 NDCG=0.2286 exceeds even the paper's best
config's 0.1515) - plausible causes, not yet disentangled: (a) documented
batch_size/cf_epochs deviation, (b) the Phase 1-5 stability/correctness
fixes changing the actual optimization trajectory, (c) different random
seed than whatever the paper used (paper doesn't state one). Needs a
dedicated ablation to attribute if this matters for the writeup.

**Geometric trajectories** (full detail in
`recsys/my_verification/logs_movielens_isomap_cf/eta_sweep_<eta>/geometry_diagnostics.csv`
per config): ORC mean drift accelerates sharply with eta - eta=0.01 reaches
only -0.16 by epoch 29; eta=0.03 reaches -0.25 to -0.32 range with one sharp
spike to -0.32 at epoch 22 (matches the paper's own description of "a
single large negative spike... episode of local geometric instability" for
their aggressive-eta experiment, though not guard-triggered here - a real
geometric fluctuation, not NaN corruption); eta=0.05 and 0.10 both undergo
dramatic H1-cycle-count collapse (from ~2000-2200 down to ~200-700) with
ORC mean diving to -1.0 to -1.6 range by mid-run. This is a much more
dramatic geometric transformation at high eta than the paper reports
(paper's ORC never goes below about -0.93 mean, ours goes to -1.58) -
consistent with the same qualitative "high eta destroys hyperbolic
structure faster" story, just quantitatively more extreme in our
reproduction.

Deviation from paper's exact hyperparameters (batch_size=2048 not 32,
cf_epochs=30 cap not 100) - documented in run_eta_outer_sweep.py's
docstring and this diary, not silently changed.

**CRITICAL FINDING (2026-08-25, commits `2c2563d`, `215baf2`):** the paper's
central empirical claim ("preserving initial geometry beats aggressive
optimization, and beats the NeuMF baseline") does not survive a careful,
bug-fixed, apples-to-apples re-evaluation.

1. Found and fixed a real bug: `GradientIsomapCF`'s "final NCF" retraining
   step (the one whose HR@10/NDCG@10 gets reported as the headline
   test_hr/test_ndcg) had the same bare-`state_dict()`-reference bug already
   fixed elsewhere - `best_state_final = final_ncf.state_dict()` stored a
   live reference, not a snapshot, so `load_state_dict()` afterward just
   reloaded whatever the LAST epoch happened to be, not the genuinely best
   one. This affected every previously reported number (the ml-1m sweep AND
   the in-progress ml-10m sweep, which was already running when this was
   found - its own final numbers will need the same recomputation once it
   finishes, geometry trajectories are unaffected).
2. Built `ablation_geometry_vs_optimization.py` to recompute corrected final
   metrics for all 4 ml-1m configs by reusing saved `matrices_epoch*.npz`
   (no need to redo the 30-outer-epoch bilevel loop). First attempt used the
   wrong negative-sampling scheme (`NCFTrainDatasetFutureBlind`, train-only
   exclusion) - `GradientIsomapCF` has its OWN internal negative-sampling
   method (full train+val+test history exclusion), never reuses
   new_datasets.py at all for training interactions. Fixed by constructing a
   real `GradientIsomapCF` instance just to reuse its `.inter_loader`.
3. Corrected results (HR@10/NDCG@10, pure_init / epoch0 / epoch29):
   eta=0.01: 0.2767/0.1565 -> 0.2867/0.1684 -> 0.2567/0.1519
   eta=0.03: 0.2767/0.1565 -> 0.2667/0.1665 -> 0.2400/0.1227
   eta=0.05: 0.2767/0.1565 -> 0.2533/0.1565 -> 0.2433/0.1325
   eta=0.10: 0.2767/0.1565 -> 0.2700/0.1652 -> 0.1833/0.0949
   Every config's FULLY CONVERGED (epoch29) result is worse than pure init,
   at every eta - not just the aggressive ones. The "higher eta = worse"
   direction still roughly holds (eta=0.10 is clearly worst), but "small eta
   gives real gains over doing nothing" does not.
4. Built `poincare_baseline.py`: a genuine a priori hyperbolic baseline
   (geoopt Poincare-ball embeddings fit via Riemannian Adam to the SAME
   D_input^(0) target GradientIsomapNCF starts from - "hyperbolic MDS" - no
   bilevel loop, matching the user's explicit design request), plus running
   our own geometry_diagnostics.py on the fitted manifold as a sanity check
   (confirmed: f_neg=0.9624, ORC mean=-0.165 - correctly and strongly more
   hyperbolic than anything GradientIsomapNCF's own trajectory reached,
   validating the diagnostic toolkit itself).
5. Also trained a genuinely fair Euclidean NeuMF baseline (learnable
   embedding tables) through the identical `inter_loader` harness (previous
   "NeuMF baseline" numbers, both the paper's and mine, used a DIFFERENT
   negative-sampling scheme via new_datasets.py's Pure-NCF branch - not
   apples-to-apples with GradientIsomapCF's own scheme).
6. **Full honest comparison, same harness, ml-1m 300u/800i, latent_dim=64:**
   - Euclidean NeuMF (learnable tables): HR@10=0.3633, NDCG@10=0.2354 - BEST
   - Pure Euclidean-init Isomap Z (0 outer steps): 0.2767/0.1565
   - GradientIsomapNCF, best case (eta=0.01, 1 outer step): 0.2867/0.1684
   - GradientIsomapNCF, converged, any eta: 0.1833-0.2567/0.0949-0.1519 (all worse than baseline)
   - Poincare-pretrained (a priori hyperbolic): 0.2433/0.1322 (also worse than baseline)

   **Neither geometry-aware approach (data-driven or a priori hyperbolic)
   beats a plain learnable-embedding-table NeuMF at this scale/config.**
   This is a real, honest result, not a bug - needs to reshape the paper's
   narrative, not be hidden. Possible explanations not yet disentangled:
   latent_dim=64 may over-provision NeuMF's free-table capacity relative to
   what a 300-user/800-item manifold-constrained representation can express;
   dataset may be too small for geometric structure to pay off over pure
   memorization; final_cf_epochs=30/patience=3 may be undertrained for the
   manifold-derived arms specifically. None confirmed.

**main.tex updated** (paper source found at `C:\Users\Julia\Documents\NSS_lab\
документы\2027 WWW Recsys\paper\main.tex` + `main.bib` - NOT a git repo, plain
files, edits are saved directly, nothing to commit there): added
`\usepackage{xcolor}` + `\new{...}` blue-highlight macro; new Section 3.2.4
describing Poincare-Pretrained NCF (with `nickel2017poincare`/`geoopt2020`
bib entries added); new Section 4.3 "A Critical Re-evaluation" with the full
corrected results table and honest discussion, all in blue. No LaTeX
compiler available on this machine to verify - checked brace balance
programmatically (0, balanced) and underscore escaping by hand instead.
Did NOT touch/delete any of the original (now partially superseded) text in
Section 4.2 - the correction is additive and clearly marked, not a silent
rewrite.

**Still open / not yet done:**
- Recompute ml-10m sweep's final metrics once it completes (same
  cheap-recompute approach, checkpoint bug now fixed for the code but not
  for the already-running process).
- Investigate WHY geometry-aware approaches underperform here (the
  capacity/undertraining hypotheses above) - would need deliberate
  additional ablations (vary latent_dim, vary final_cf_epochs) not yet run.
- Fill Gromov delta section (Section 4.1, still "[To be added]" in main.tex)
  and write Conclusion - deferred, lower priority than the correctness issue
  just found.
- No LaTeX compiler on this machine - user should compile/check main.tex
  before trusting it renders correctly.

**Phase 7 partial (commit `6f4301b`):** added MRR@K to evaluation.py
(genuinely distinct from NDCG - diverges for any hit below rank 1).
Deliberately did NOT add a separate "Recall@K": this dataset's leave-one-out
sampled-evaluation protocol (exactly one positive per user) makes Recall@K
mathematically identical to HR@K, so a second name for the same number
would be redundant, not new information.

**Two real blockers found, both need the user, not more autonomous work:**
1. **Second dataset for Phase 7** - plan suggested ML-10M/25M or a second
   domain (Amazon/Yelp), but never got a specific pick confirmed. This is a
   real decision (which dataset, which scale) plus a download commitment -
   not something to just guess and proceed on.
2. **Phase 8 (writing into the paper) has no target** - checked
   `C:\Users\Julia\Downloads\Telegram Desktop\` and the whole repo for a
   `.tex`/editable source of the draft; only the PDF exists
   (`WWW___Recommender_System_Manifold.pdf`). Can't "fill in the Gromov
   delta section" or "write the Conclusion" without knowing what format the
   user wants that in (LaTeX source if it exists somewhere else, a
   standalone markdown draft, Overleaf link, etc.).

Both logged here rather than guessed at. Sweep results (Phase 6) are the
real deliverable ready for the user regardless of how Phase 7/8 resolve.

**Incident notes (2026-08-24):** THREE incidents this session.
1. & 2. Two full-system hangs (`Kernel-Power` event 41, hard reset required)
   at ~10:58 and ~11:33 - investigated thoroughly, correlated with nothing
   of mine running or only a trivial op (tiny CUDA matmul, `nvidia-smi`
   query). Points to pre-existing driver/hardware instability on this
   machine (matches the `NVIDIA Overlay.exe`/`dwm.exe` crash cluster from
   2026-08-22 evening), not something my session triggered.
3. A third freeze (~12:36-12:44, no Kernel-Power 41 this time - system
   recovered on its own rather than hard-crashing, a different signature
   from 1&2) that WAS caused by my code: `ollivier_ricci_curvature()` called
   `GraphRicciCurvature.OllivierRicci` without overriding its default
   `proc=cpu_count()` - confirmed via `inspect.signature` that this
   evaluated to `proc=28` on this machine, and via reading the library
   source (`OllivierRicci.py:325`, `Pool(processes=_proc)`) that it
   unconditionally spawns a `multiprocessing.Pool` of that size - so even
   the tiny 12-node toy graph in check_geometry_diagnostics.py triggered a
   burst of 28 new Python interpreter processes on Windows (spawn, not
   fork - each fully re-imports numpy/scipy/gudhi/cvxpy's solver stack),
   which is a real, severe resource-contention spike, independent of the
   driver issue in 1&2. Fixed by forcing `proc=1` (confirmed via source
   that this still uses Pool, just with 1 worker - normal, safe
   multiprocessing, not a special/fragile code path - the danger was
   specifically the 28x burst, not multiprocessing per se).

   Lesson for future dependencies: **check for multiprocessing/parallelism
   defaults tied to cpu_count() in any new third-party library before first
   use**, especially on this machine (28 logical cores - defaults tuned for
   "use all cores" scale badly here for anything that spawns rather than
   just threads). `persistent_homology()`'s original unbounded Rips-complex
   construction (50th-percentile edge threshold, could blow up to millions
   of simplices at n=800) was also hardened earlier this session as a
   separate, unrelated precaution (staged construction + hard
   `max_simplices` cap) - not confirmed as a cause of any incident, but a
   real latent risk regardless.

User asked me to snapshot state before risky steps so any future crash can
be correlated. Practice now in effect: before any non-trivial run, capture
process list / GPU state / memory / git HEAD / event-log RecordId
high-water-mark to `...scratchpad\forensics\state_snapshot_latest.txt`
(PowerShell, overwritten each time - see that file for the exact commands),
then after the run check `Get-WinEvent` for anything past that RecordId.
Also now checking any new third-party library's parallelism defaults via
`inspect.signature`/source reading BEFORE first use, not just after an
incident.

**Open questions / blockers:** none blocking, but proceeding cautiously per
above - small steps, snapshot-check-run-check, not large unattended runs
until the driver/hardware stability question settles one way or the other.

**Do NOT:** modify `Adam/`, `regularizator/`, `structure_approximation/` beyond
what's already committed (see plan Context section) — these are finished
tooling per user instruction.

**Committed so far (chronological):** `6220397` `d41e8a9` `cfaa4d7`(reverted)
`6e25309` `8c9b13d` `7f8578c` `473edf5` `212f588` `7268342` `c1473a4`
`0e1b4b0` `e0d6804`(merge) `b4b6a2f` `c6a0278` `40f711a` `0aa5551` `a4915d6`.

**Progress-visibility practice (added 2026-08-24, user request):** any
run expected to take more than ~10-20s now goes through the `Monitor` tool
(tails the process's own stdout as live chat notifications) rather than a
plain blocking `Bash` call, specifically so a long run keeps producing
visible, timestamped evidence of activity instead of the terminal looking
frozen until it returns - understandable concern to have on a session with
genuine freeze incidents. `geometry_diagnostics.py` now has flushed,
timestamped `_progress()` calls at every stage/substep that could plausibly
take more than a couple seconds. Apply this same pattern (progress logging +
Monitor) to Phase 6's sweep runs.

**Left uncommitted on purpose (out of Phase-1 scope, not lost, revisit if asked):**
`examples/gradient_isomap/mnist/stages/{first,second}_stage.py`,
`examples/gradient_isomap/synthetic/curvature_diagnostics/*.py`,
`examples/gradient_isomap/mnist_regression/{diagnostic_convergence_check,run_full_pipeline_all_fixes}.py`,
`examples/sionna_isomap/mnist_law_samp_num_analys.py` (this last one's directory
name suggests an unrelated wireless-comms thread — didn't touch it, unclear
ownership/purpose). All still untracked on disk.

**Git hygiene note:** twice during Phase 1 committed extra files that were still
staged in the index from an earlier `git add`, under a commit message that didn't
mention them (caught both times via `git show --stat HEAD` right after
committing, fixed with `git reset --soft HEAD~1` + redo). Switched to
`git commit <explicit paths>` afterward, which only commits the named paths
regardless of index state — use that form going forward, and always sanity-check
`git show --stat HEAD` immediately after every commit.

---

## Log

### 2026-08-22 — Session 1: investigation + plan approval

- Read the WWW paper draft PDF, the repo's README, and three internal
  investigation docs (`docs/gradient_isomap_mnist_regression_report.md`,
  `docs/paper_experiment_noise_robustness.md`, `docs/regularization_investigation_journal.md`)
  — none of these mention recsys/MovieLens/NCF at all; they're a separate
  synthetic/MNIST-regression track.
- Could not fetch the OpenReview algorithm paper (id `GWOvA3wBuD`) — blocked by
  a bot-verification page even via the `/pdf?id=` endpoint. Relied on the repo's
  own README + investigation journals instead, which describe the same method
  (free pairwise-distance matrix, gradient-optimized under task loss, geodesics
  via differentiable Floyd-Warshall, kernel PCA embedding).
- Key finding via an Explore subagent + manual `git`/`grep` verification:
  - `recsys/GradIsomapCF_movielens/*.py` does not exist in the working tree of
    `mnist_reg_example` — only stale `__pycache__` + old logs. Real source is on
    branch `GradientIsomapNCF` (local + origin), forked at `1acbae7` (2026-01-19),
    last commit `bd1152c` (2026-06-18).
  - `GradientIsomapNCF` only touches `recsys/` + `requirements.txt` (encoding-only)
    outside itself — clean, low-conflict merge expected.
  - No code anywhere in the repo computes Ollivier-Ricci curvature, a real graph-
    Laplacian spectral gap, Kruskal stress, or persistent homology. Only Gromov
    δ-hyperbolicity exists (`recsys/my_verification/analyze_hyperbolicity.py`,
    offline script, not wired into training) — and the draft itself marks δ as
    "[To be added]". Yet the draft's Table 1-4/Figures 1-5 report numbers for
    ALL of these measures. This is the central gap the plan exists to close.
  - `GradientIsomapCF`'s bilevel loop reimplements everything from scratch and
    has none of the NaN/Inf-gradient guards hardened into `Adam/GradientIsomap.py`
    on 2026-08-01 — plus there are further uncommitted fixes sitting in the
    working tree right now (`near_best` kick-guard, kick disabled by default,
    Floyd-Warshall backward rewrite).
  - Working tree also had a critical, already-fixed-but-uncommitted bug in
    `regularizator/GraphRegTrainer.py`: `graph_loss` was computed via
    `sklearn`/numpy on detached tensors, so it carried zero gradient — graph
    regularization never affected any synthetic/MNIST training run regardless
    of lambda, until the (uncommitted) `torch.cdist` fix.
- Environment check: `_new_venv/Scripts/python.exe` (Python 3.14.7) has
  torch 2.11.0+cu128 (CUDA OK), sklearn 1.9.0, pandas 3.0.5, numpy 2.5.2 — this
  is the venv to use. Global `python` has no torch installed.
  GPU: RTX 5080, 16GB, ~2.9GB/46%util already used by other processes (browser)
  at check time — re-check `nvidia-smi` before each heavy run.
  Disk: 202GB free of 465GB — no concern for the <20GB download budget.
  ML-1M is already vendored in the `GradientIsomapNCF` branch (`data/ml-1m/`,
  small) — no download needed for the primary dataset.
- User directive: work in `mnist_reg_example`, merge `GradientIsomapNCF` in,
  commit what's needed, reproduce + fill in what's missing (don't cut any of
  the draft's ideas — implement the missing diagnostics for real), add
  datasets/metrics, don't touch `Adam/`/`regularizator/`/`structure_approximation/`
  beyond what's already staged, keep this diary, watch resources, ≤20GB downloads.
- Wrote and got approval for the phased plan (see path above). Starting execution.

### 2026-08-22 — Session 1 continued: Phase 1 (commit pending fixes)

Baseline: `pytest tests/` — 62 passed before touching anything. Committed in
logical groups (see `git log --oneline` from `6220397` to `0e1b4b0`):
1. `6220397` — GradientIsomap/KernelPCA/Isomap stability fixes + manifold_diagnostics.py
   (also swept up `higher_dim_geometries.py`, which was already staged - harmless,
   just landed in a different commit than planned).
2. `d41e8a9` — Floyd-Warshall backward regression tests (isolated after two
   index-mixup false starts, see hygiene note above).
3. `6e25309` — GraphRegTrainer autograd-detachment fix (the big one: graph_loss
   had zero gradient) + best-model deepcopy fix + adaptive_lambda crash fix,
   with all 3 corresponding test files.
4. `8c9b13d` — IntrinsicNN weight_decay fix.
5. `7f8578c` — MLE intrinsic dimension estimator + test.
6. `473edf5` — swiss_roll_tight/torus_fat synthetic geometries.
7. `212f588` — mnist_regression/synthetic orchestrator fixes (dead ReLU, LR,
   shuffle=False, deepcopy).
8. `7268342` — requirements.txt UTF-8 + trimmed to direct deps (verified against
   what `_new_venv` actually has installed - matches).
9. `c1473a4` — completed mnist_regression_cnn (had only committed 1/4 files in
   the group-5 commit; caught by re-checking untracked files afterward).
10. `0e1b4b0` — dims_estimation benchmark runner scripts.

Final check: `git status --porcelain | grep -v '^??'` empty (no more tracked-file
modifications), `pytest tests/` — still 62 passed, 15 subtests passed. Phase 1 done.

### 2026-08-22 — Session 1 continued: Phase 2 (merge GradientIsomapNCF)

`git merge GradientIsomapNCF --no-edit` — exactly one conflict as predicted,
`requirements.txt` (binary/encoding conflict, git left our-side UTF-8 content
with no markers). Checked recsys's actual imports (`grep` across
`recsys/**/*.py`): only stdlib + numpy/pandas/matplotlib/scipy/torch/PIL/sklearn
- scipy and Pillow aren't in requirements.txt as top-level pins but are already
satisfied transitively (scipy 1.18.0 via scikit-learn, Pillow 12.3.0 via
matplotlib/torchvision) - verified by importing both in `_new_venv`. Resolved
by keeping current branch's requirements.txt as-is (`git add`), no edits
needed. Merge commit `e0d6804`. `pytest tests/` still 62 passed after merge.

`recsys/GradIsomapCF_movielens/*.py` and `data/ml-1m/*` are real files again.
`recsys/my_verification/*.py` (already present, untracked before) unaffected
by the merge (no conflict there - they never existed on either branch's git
history, they were purely local untracked work sitting in the working tree).

### 2026-08-22 — Session 1 continued: Phase 3 (get recsys running on current stack)

Smoke-tested via `run_experiment.main(max_users=300, max_movies=800,
epochs_pure=2, gradisomap_epochs=2, cf_epochs=2, final_cf_epochs=2)`
(`_new_venv`, GPU had ~14.7GB free at the time). Two false starts:
1. First attempt used max_movies=60 for speed - hung forever, not just slow.
   Root cause: `NCFTestDatasetSampled` (new_datasets.py) rejection-samples
   num_ng=99 negatives per positive from items outside the user's full
   history; with only 60 items total this is often infeasible, so the
   `while negs < num_ng` loop never terminates. Not a real bug - the paper's
   actual config (800 items) always has enough headroom. Fixed by using the
   real item count with epoch counts cut down instead.
2. Second attempt crashed on `UnicodeEncodeError` (cp1251 can't encode a `→`
   character in a Russian print statement) - purely a Windows console-encoding
   artifact triggered by redirecting stdout to a file, not a real bug. Fixed
   by setting `PYTHONIOENCODING=utf-8` at invocation time rather than editing
   the print statement.
3. Third attempt: full success. Pure NCF baseline (HR@10=0.23, NDCG@10=0.13
   on test after 2 epochs) and GradientIsomapCF's bilevel loop (2 outer
   epochs, HR@10=0.26, NDCG@10=0.135) both trained and evaluated end-to-end.
   Results/matrices saved to `recsys/my_verification/logs_movielens_isomap_cf/
   smoketest_merge0822/` (local artifacts, not committed).

`check_floydwarshall_grad.py`: `torch.autograd.gradcheck` on a raw n x n
input tensor FAILED at first - looked alarming (did the Phase-1 backward
rewrite break something?). Investigated by hand: the failure is entirely on
the diagonal (d(dist[i,i])/d(graph[i,i])), which is never a real degree of
freedom in this pipeline (`IsomapNN.update_distance_matrix()` always
reconstructs the diagonal as exactly 0 from a zeros-init matrix, never from
the learnable `self.layer`). Verified off-diagonal gradient is correct to
~1e-9 (float64 noise) via manual finite-difference, and that the new fast
backward matches the OLD brute-force reference exactly (0.0 diff, including
on the diagonal) - so this is a pre-existing gradcheck-vs-real-usage mismatch,
not something the rewrite introduced. Fixed the test itself to gradcheck over
the actual free-parameter representation (symmetric, zero-diagonal,
reconstructed from upper-triangular params exactly like
`update_distance_matrix()`) instead of an unconstrained full matrix - now
passes cleanly, all 5 seeds, ~3e-9 max diff.

Committed: `b4b6a2f` (fixed gradcheck), `c6a0278` (analyze_hyperbolicity.py +
run_experiment.py - these had NEVER been committed anywhere before, confirmed
via `git log --all -- recsys/my_verification/` returning nothing). Verified
analyze_hyperbolicity.py against the smoke-test's real saved matrices - loads
correctly, reports sane delta_rel values (0.30-0.77 range across
D_input/D_geodesic/D_latent).

`pytest tests/` still 62 passed after all of Phase 3's commits.
