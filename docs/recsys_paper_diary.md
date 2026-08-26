# Recsys paper (WWW) — working diary

Full plan: `C:\Users\Julia\.claude\plans\wild-munching-stonebraker.md` (approved 2026-08-22).
Read that first for the why; this file tracks the where-are-we-now.

---

## CURRENT STATUS / NEXT STEP
*(this block is overwritten each session — always current, read this first)*

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
