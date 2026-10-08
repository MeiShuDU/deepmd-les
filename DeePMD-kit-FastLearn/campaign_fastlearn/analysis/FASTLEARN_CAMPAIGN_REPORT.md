# FastLearn campaign report: short-range vs LES long-range on water

## What this campaign is

Eight DeepMD-kit `hybrid_ener` arms trained on the same data, to ask one question: does adding the LES latent-charge Ewald long-range term to a short-range DeepMD model buy accuracy, and does it matter whether the charges are learned?

All eight arms were trained on the FastLearn 320-frame water set with the same schedule: 80,000 steps, `batch_size: 2` (so 160 steps per epoch, 500 epochs), the `exp` learning-rate staircase (1e-3 to 3.51e-8 in 16 levels of 5,000 steps), and the same fitting setup.

Four model families, two seeds each (`_sA`, `_sB`):

| family | long-range | charges |
|---|---|---|
| `deepmd` | none (short-range control) | - |
| `deepmd-les` | latent Ewald | learned per atom, initialised at SPC/E |
| `deepmd-les-claim-neutral` | latent Ewald | learned, projected to sum zero each frame |
| `deepmd-les-freeze-charge` | Ewald | fixed SPC/E table, no charge network |

Two further arms (`cace-sr`, `cace-lr`) come from the cace reference implementation and are reported separately, below.

Metric protocol, and why it is not the `lcurve.out` columns: `lcurve.out` reports validation error over `numb_btch: 3` frames per display step, which swings 10-13% over the last 20,000 steps and so cannot rank these arms.
Every number in this report instead scores a checkpoint over **all 80 validation frames** using DeepMD's own `test_ener` plus `weighted_average` (`deepmd/eval_campaign.py`), for each of the five kept late checkpoints (40k, 50k, 60k, 70k, 80k).

Reproduce with:

```bash
cd campaign_fastlearn
./run_extend.sh                                 # 80k -> 120k, all 8 arms (see "The 120k extension")

cd deepmd
python eval_campaign.py --steps auto --out ../analysis/data/valid_sweep.tsv --reset <arm> [...]
python eval_campaign.py --steps auto --out ../analysis/data/valid_sweep_120k.tsv --reset <arm> [...]
python decompose_energy.py --step 80000 --reset --out ../analysis/data/energy_decomp_valid.tsv <arm> [...]
python decompose_energy.py --split train --step 80000 --reset --out ../analysis/data/energy_decomp_train.tsv <arm> [...]
python lr_force_share.py                       # writes analysis/data/lr_mechanism_les.tsv

cd ../cace
PYTHON=<pod python> ./sweep_cace.sh             # all five checkpoints of both arms
python collect_sweep.py                        # writes both cace TSVs into analysis/data/

cd ../analysis
python extract_les_log.py
python -m nbconvert --to notebook --execute --inplace fastlearn_campaign.ipynb
```

`eval_campaign.py --steps auto` globs `model.ckpt-*.pt` numerically, so the same command that wrote the 40k-80k sweep picks up 90k-120k with no change; the 120k file is written separately only so the two run sets stay distinguishable.

`cace/sweep_cace.sh` is the full cace sweep (five checkpoints per arm), run on the pod with the pod's venv, which carries deepmd, cace and torch in one environment so both codes are scored against the same labels by the same interpreter.

---

## Summary

**Force error: adding the LES long-range term helps, and the ranking is converged.**

Every LES arm beats the short-range control, and the separation is clean: the worst LES run (5.434e-02 eV/A) is better than the best control run (5.609e-02), both on the same five-checkpoint mean.
Family means, on the 80-frame validation split averaged over the five late checkpoints:

| family | rmse_f [eV/A] | vs control |
|---|---|---|
| short-range only | 5.689e-02 | - |
| LES, learned charges | 5.352e-02 | -5.9% |
| LES, learned + neutral | 5.366e-02 | -5.7% |
| LES, fixed SPC/E charges | 5.394e-02 | -5.2% |

Replicate spread within a family is 0.1-2.8%, so a 5% effect is 2 to 50 times the noise floor.
The force result is also not sensitive to where training stopped: the 40k to 80k drift is under 0.7% for every arm.

**The notable negative result: learning the charges buys nothing here.**

Fixed SPC/E charges score 5.394e-02 against 5.352e-02 for learned charges - a 0.8% gap, well inside the 2.8% replicate spread of the control family.
Constraining the learned charges to be neutral per frame changes nothing either (5.366e-02).
On this data, at this configuration, the LES gain comes from having a long-range term at all, not from the network that predicts the charges.

**Energy error: the learned-charge arms are worse, and a 120k extension confirms it while narrowing it.**

At the 80,000 checkpoint the per-atom energy RMSE is 5.434e-04 eV for the control, 6.430e-04 for learned charges (+18%), 6.470e-04 for learned-neutral (+19%), and 5.449e-04 for fixed charges (+0.3%).
The catch, and the reason this is not reported as a mean over the late checkpoints: `rmse_e` is not converged.
The short-range control falls 28-30% from 40k to 80k, the learned arms are still dropping 19-21% over the last 10k alone (3 to 7 times the 3-6% spread between replicates), and the series is non-monotone - so no energy number here describes a settled checkpoint, and averaging the window would report a value describing no actual checkpoint.
Averaging the five late checkpoints would inflate the penalty to +23-26%.

The arms were therefore extended to 120,000 steps (see "The 120k extension" below).
Endpoint to endpoint the penalty narrows monotonically - **+24.1% at 40k, +18.3% at 80k, +15.5% at 120k** (6.099e-04 against 5.283e-04) - and the fixed-charge arm ends at **-2.4%**, slightly better than the control.
So the direction of the finding is confirmed at a second, longer endpoint, and the size shrinks rather than grows; the earlier guess that it "may shrink further" is what happened.
It is still not settled at 120k: over the last 20,000 steps the energy wanders 10-19% step to step on a net movement of 1-5%, which is what an LR at its floor looks like rather than a converged minimum.

**Level versus shape: the gap is a fit difference, not a shifted reference.**

Splitting the per-frame energy error into a level (bias) and a shape (spread) term at 80,000: `bias^2/rmse^2` is 3-20% for every arm, so a constant explains only a minority of the error anywhere.
The fitted slope is 1.0009-1.0179, so there is no scale error to absorb either, and `rmse_cal` lands within 0.5% of `spread` for every arm: once the level is removed, a linear recalibration buys essentially nothing.
The invariant worth checking is `rmse_cal <= rmse`: the fitted affine family contains `a=1, b=0`, so a correct least-squares calibration can never leave a larger residual than the raw error.
(An earlier draft cited `rmse_cal <= spread` instead. That is not an invariant - the `a=0` constant fit is not in the rescaled family - and the wrong wording is worth recording here because it concealed a real defect rather than exposing it.)

Comparing shapes, the learned-charge arms are 12.8% and 15.8% worse than the control while the fixed-charge arm is 1.1% better.
So the energy penalty, if it survives longer training, is a genuine fit-quality difference rather than a reference-offset artefact.

**Two defects found while validating this section, both now fixed.**

The first is a frame-count bug, and it was the visible one: `read_system` read only `set.000` of each system, so the train split was scored on 240 frames instead of 320 - `data_1` contributed 80 of its 160, because the FastLearn export split it across `set.000` and `set.001`.
The valid split is a single system with a single set, which is exactly why it was unaffected and why the error survived this long in plain sight.
Both files have since been regenerated with the fix; the train file is now 320 frames, and the valid columns moved only in their last digits, as predicted.

The second is the calibration fit, and its cause is now settled rather than merely defended.
The pre-fix train column reported `rmse_cal = 1.82e-03` against a raw `rmse = 5.14e-04`, which no least-squares affine fit can produce.
The cause is a dtype: `energy.npy` is written as **float32**, and `np.polyfit` takes its `rcond` from `finfo(x.dtype).eps`, which for float32 is `len(x) * 1.2e-7 ~ 3e-5`.
The uncentered Vandermonde `[E_true, 1]` on labels of magnitude `3e4` is far worse conditioned than that (`cond ~ 1e9`, so `1/cond ~ 8e-10`), so `lstsq` truncates the second singular value and returns the rank-1 minimum-norm solution instead of the least-squares optimum.
On this data that solution came out `a = 0.500001` exactly, with a residual above the raw `rmse`, which is precisely the pre-fix symptom; a float64 synthetic test with the same conditioning reproduces the reported `1.8243e-03` to 0.7%.
It also explains the second symptom: all eight arms reported nearly the same `rmse_cal` (1.824-1.909e-3) because they all received the same degenerate slope.

The fix is to cast the labels to float64 in `read_system`, which drops `rcond` to `~5e-14` and restores the true optimum; the centered fit is kept as well, conditioning on the label spread rather than the magnitude.
`decompose()` now asserts `rmse_cal <= rmse`, so this class of failure - or any other - fails loudly instead of writing a plausible-looking column.

Worth recording, because the first fix looked right for the wrong reason: centering the targets *without* the cast also works, since better conditioning is enough to clear the float32 `rcond`, which is why the two are documented together - the cast is the cure, the centering is belt-and-braces.
The comment originally written to justify the centering claimed ill-conditioning on the `-29945 eV` label magnitude, and a float64 test disproves that: `cond(V)` is `1.3e9` against a float64 `rcond` of `~5e-14`, so `lstsq` is far inside its tolerance and the old and new fits are numerically identical in float64.
The defect was never the label magnitude; it was the dtype driving `rcond`, and only the dtype explanation predicts the identical degenerate slope across all eight arms.

**A mechanism that fits the pattern.**

See the section on `remove_self_interaction` below: the kept self-interaction term is a function of the charges, so it is a harmless constant for fixed charges but a geometry-dependent spurious force for learned ones.
That is exactly the observed asymmetry - learned charges pay, fixed charges do not.

**Cross-code: the cace arms are more accurate, but not because of the long range.**

The two cace arms land about 2.5x better on force than the best LES arm (`cace-lr` 2.1711e-02 against 5.3047e-02), and the trap is to read that as a verdict on LES.
The labels are bit-identical between the codes, so the only remaining difference is the descriptor, and the control settles it: `cace-sr` has no long-range term at all and still beats every one of the eight deepmd arms.
The cross-code gap is a descriptor-and-recipe difference, not the physics under study.
What survives is the within-code ablation, and the two codes agree in sign: the long-range term helps in both, by 17.6% on force within cace (`cace-lr` 2.1711e-02 against `cace-sr` 2.6345e-02, and 25.9% over epochs 300-500 alone) and by 5.9% within deepmd.
Both deltas are positive; their size is architecture-dependent.

---

## The 120k extension: what a longer anneal changed, and what it did not

The eight DeepMD arms were extended from 80,000 to 120,000 steps (`run_extend.sh`, one `dp --pt train --restart` per arm from `model.ckpt-80000.pt`; all eight completed with no failures).
The extension is a *separate* run set whose restart reheats the learning rate, so its 40,000-step span is a re-anneal rather than a continuation of the 80k campaign's, and the extension's own late window is not directly comparable to the campaign's.

**The restart reheats the LR by 24x, so 80k-120k is a re-anneal, not a continuation.**

DeepMD's `exp` schedule derives its stop step from the config's `numb_steps`, so restarting from step 80,000 with `numb_steps: 120000` puts the LR back at `1.6e-06` - 23.9x above the `6.7e-08` the 80k run ended on.
Every arm's `lcurve.out` therefore contains **two rows at step 80000**: the 80k campaign's endpoint (`lr 6.7e-08`) and the extension's first row (`lr 1.6e-06`).
The same 23.9x jump appears in all eight arms, so it is the schedule, not a per-arm event.
The LR then steps back down over the next 40,000 steps (`1.6e-06 -> 1.1e-06 -> 7.0e-07 -> 3.0e-07 -> 5.4e-08`).

That reheat is visible in the full split, and it is why the window rule changed.
The short-range control's energy error jumps from 5.434e-04 at 80k to **8.771e-04 at 90k** (1.61x) while the LR is re-elevated at 7.0e-07, then recovers to 5.337e-04 by 100k as the LR re-anneals.
The force error does not move (5.68e-02 throughout, under 0.2%), so the 80k-120k window is contaminated for energy and clean for force.
**The only apples-to-apples energy comparison between the two run sets is endpoint to endpoint**, because both the 80k endpoint and the 120k endpoint sit at an LR floor.

**Force: converged, and it was already converged at 40k.**

Force moves by under 0.7% from 40k to 120k in every family, so the extension changed nothing about the campaign's force result.
At the 120k endpoint the replicate-averaged force is 5.6791e-02 for the control, 5.3404e-02 for learned charges (-6.0%), 5.3546e-02 for learned-neutral (-5.7%) and 5.3860e-02 for fixed charges (-5.2%) - the same ranking, within 0.1% of the 40k-80k five-checkpoint means in the Summary table.

**Energy: 120k is the best checkpoint for every family, but it is still not settled.**

Endpoint to endpoint (80k -> 120k) every family improves: control 5.4341e-04 -> 5.2827e-04 (-2.8%), learned 6.4295e-04 -> 6.0991e-04 (-5.1%), learned-neutral 6.4702e-04 -> 5.9859e-04 (-7.5%), fixed 5.4491e-04 -> 5.1543e-04 (-5.4%).
The series is still not converged, and what rules convergence out is the wander, not a trend: over the last 20,000 steps the energy moves 1.05-1.19x from 100k to 110k and then 0.82-0.94x from 110k to 120k, a 10-19% swing several times the 3-6% replicate spread, and it is non-monotone (both 100k and 110k are high outliers for every family).
The net movement over that window is only 1-5%, so the trend is exhausted while the series still wanders.

**The learned-charge energy penalty survives the extension, and it narrows.**

The penalty against the short-range control is +24.1% at 40k, +18.3% at 80k and **+15.5% at 120k** (6.0991e-04 against 5.2827e-04), with the neutral variant at +13.3% and the fixed-charge arm at -2.4%.
The penalty is also the same on train and validation (at 80k, +11.2% on train against +11.6% on valid), so it is a fit degradation rather than a generalisation gap.

**The cross-code force factor is unchanged by the extension.**

`cace-lr` still scores 2.78x better than the DeepMD LES family at its 120k endpoint (1.9217e-02 against 5.3404e-02), and `cace-sr` still beats every arm without a long-range term at all (2.5881e-02, 2.06x).
Since DeepMD's force moved 0.2% over the whole 40k-120k span, the extension cannot move this comparison, and the descriptor confound the cace section describes is untouched by it.

---

## LES diagnostics: is the charge channel learning?

From each LES arm's verbose log (every 100 steps).

The learned-charge families start exactly at SPC/E (`+0.4238` H / `-0.8476` O; charge standard deviation 0.600 e) and grow the magnitude of the charges as training proceeds.
The standard deviation reaches about 0.74-0.78 e by the final log.
The fixed-charge arm is flat at 0.600 e throughout, by construction.

So the charge network is doing something - it is not stuck at its initialisation.
What the accuracy numbers say is that the something it does is not worth anything on this data.

---

## The `remove_self_interaction` question

This is the most important finding for anyone reading the earlier comparison work, because it explains an anomalous energy scale.

The logged LES long-range energy `E_lr` is **positive and small**: +10.9 to +22.7 eV/frame at the first log, +11.6 to +18.0 at the end.
The same Ewald sum run with the physical convention is about **-385 eV** on a validation frame at SPC/E charges.
The two differ by a large positive self-interaction term that the kernel keeps.

The campaign inherited `remove_self_interaction: False` from the author's cace recipe, where it is both the constructor default and passed explicitly (`campaign_fastlearn/cace/fit_cace.py`).
With the flag off, the kernel retains

```
E_self = norm_factor / (sigma * (2*pi)^1.5) * sum_i q_i^2  =  +5.7446 eV * sum q^2
```

At SPC/E on a 192-atom frame, `sum q^2 = 68.97`, so that term is `+396.2 eV` against a physical `-384.8 eV`, leaving `+11.4 eV`.

Three independent checks agree:

1. The fixed-charge replicates log `+10.93` and `+11.26 eV` at their first checkpoint, which is the same chemistry and the same arithmetic, a few percent from the standalone `+11.43 eV` (the gap is a 2-frame batch mean and the net's rounding of SPC/E).
2. Flipping the flag on a trained LES model moves the same frame and charges from `-384.77 eV` (`True`) to `+11.43 eV` (`False`): a 396.2 eV shift, which is `5.7446 x 68.97` to five digits.
3. cace's own kernel in its own units gives `+0.1263` / `-4.2528`; times the 90.0474 conversion, that is `+11.37` / `-382.9 eV`.

**This is not a porting error.**

The flag mirrors the cace reference faithfully.
The consequence is that the term is not a constant for a learned-charge model: the logged charges grow (std 0.600 to ~0.78), so `sum q^2` grows from ~69 to ~106-117 and the self term from ~397 to ~660 eV.
Because it is quadratic in geometry-dependent charges, it contributes a spurious force through `dq/dr`.
Measured over all 80 validation frames by flipping the flag and differencing the total force, that force is 0.83-1.07 eV/A per component rms across the four learned-charge arms, which is **1.01-1.31x the total force rms** (`analysis/data/lr_mechanism_les.tsv`).
It is not a small correction: it is a force comparable to, and for three of the four arms larger than, the whole net force, which is possible only because the physical Ewald force nearly cancels it.
Both `deepmd-les-freeze-charge` arms measure exactly zero on the same flip (`~1e-15 eV/A`), which is the prediction for per-type constants - the term is a pure offset the fitting bias absorbs at no cost.
A fixed-charge model is therefore immune, and the two ends of the measurement, a real force against an exact zero, are what make this a mechanism rather than a correlation.

An earlier draft reported this force as `0.3510 eV/A rms (0.8426 max)`, `24%` of a `1.4519 eV/A` total.
That figure is not reproducible from the current data and its provenance is untraceable; `0.3510` is close to the *long-range* force rms, not the self force, so the earlier number appears to have differenced the wrong pair.
The numbers above come from the whole split and the checked-in TSV.

Recommendation: do not "fix" the flag to `True` on cace-fidelity grounds, and do not relaunch this campaign for it.
All six LES arms share the flag, so arm-versus-arm and les-versus-cace comparisons remain internally consistent.

---

## Why the long-range term is worth 3-5x more to cace than to DeepMD

The within-code ablations agree in sign but not in size: turning the long-range term on is worth 5.9% of force in DeepMD and 17.6% in cace.
The charge and `E_lr` diagnostics point at the structural difference behind the gap: the two codes end up asking the same term to do very different amounts of work on the same physical system.
Everything in the table below is a mechanism number, so it is checkpoint-robust - the DeepMD column is measured at 80,000, because `les_metrics.tsv` stops at step 80200, and the cace column spans all four phase checkpoints.
The extension does write charges, but only a partial trace whose step counter restarts with the run, and its last block still reads `q_O` and `q_H` within 0.3% of the 80k values - so the 80k column stands for the extension as well.
The accuracy contrast that goes with it is stated in prose, citing the 120k endpoint for DeepMD and the final phase checkpoint for cace.

**Both codes keep the self term; DeepMD's is 9x larger and its charge 3x stronger.**

| | `deepmd-les` (learned, 4 arms) | `deepmd-les-freeze-charge` | `cace-lr` (4 phase ckpts) |
|---|---|---|---|
| `f_lr/f_tot` | 0.370-0.405 | 0.263 | 0.0934-0.0938 |
| `f_self/f_tot` | 1.01-1.31 | exactly 0 | 0.550-0.594 |
| `sum q^2` native units | 105.5-116.4 | 68.97 | 1207.7-1228.1 |
| `sum q^2` physical e^2 | 105.5-116.4 | 68.97 | 13.4-13.6 |
| `abs(q)_rms` physical e | 0.741-0.779 | 0.5993 | 0.264-0.266 |
| `q_O` / `q_H` physical e | -1.05 to -1.08 / +0.52 to +0.57 | -0.8476 / +0.4238 | -0.365 to -0.371 / +0.191 to +0.193 |
| net `sum q` per frame, e | +4.2 / +4.0 (`les`), 0.000 (`claim-neutral`) | 0.000 | +0.73 to +1.28 |
| `E_self` eV/frame | 606-669 | 396.2 | 76.7-78.0 |
| `E_lr` eV/frame | 17.6-18.4 | 11.42 | 2.44-2.53 |
| `E_self / E_lr` | 34.5-37.2 | 34.7 | 30.8-31.4 |

Only the `sum q^2` row needs a conversion: cace and les use different Ewald prefactors (cace 1.0, les 90.4756), so the same physical charge reads 9.5x larger in cace's column.
Dividing cace's 1207.7-1228.1 by 90.4756 puts its physical charge at 13.4-13.6 e^2, i.e. `abs(q)_rms` 0.26 e against SPC/E's 0.60 - a charge **2.3x weaker** - while the DeepMD learned charge grows to 0.74-0.78 e, **1.3x stronger** than SPC/E.
Everything else in the table is already a dimensionless share or an eV scale, so it transfers between the codes directly.

Both codes produce a roughly 2:1 O:H split, cace -1.90 to -1.94 and DeepMD -1.88 to -1.89 in the plain `les` arms, so both learned a water-like charge geometry.
What differs is scale: `q_O` is 0.371 e in cace against 1.05 to 1.08 e in DeepMD, bracketing SPC/E's 0.8476 e from below and above, which is the same 2.3x gap the `abs(q)_rms` row shows.
Neither code enforces neutrality by default and both end non-neutral - cace at +0.73 to +1.28 e per frame, DeepMD's plain `les` arms at +4.0 to +4.2 e - while the `claim-neutral` arms sit at exactly 0.000.
In the `claim-neutral` arms the `q_O`/`q_H` ratio is exactly -2.00, but that is an arithmetic identity of the constraint over 64 O and 128 H rather than a learned result, so the plain `les` arms are the ones to read for the ratio.

**The outcome differs qualitatively, not just in size.**

In cace the long-range term is a *clean win on both metrics*: force 2.5881e-02 -> 1.9217e-02 (-25.8% at the final checkpoint) and energy 2.385e-04 -> 1.852e-04 (-22.4%).
In DeepMD it is a *trade*: force 5.6791e-02 -> 5.3404e-02 (-6.0%) while energy goes 5.2827e-04 -> 6.0991e-04 (+15.5%).
And the trade is not bought by the long-range *channel*, it is bought by the charge's *freedom*: pin the charges at SPC/E and the same channel delivers -5.2% force with -2.4% energy, a clean small win.
So the DeepMD charge network converts an energy-neutral force win into an energy loss, while cace's does the opposite.

**Three candidate factors, only partly separable from this data.**

1.  **How much of the force the long-range channel is asked to carry.**
    DeepMD's les Ewald carries 40% of its model's force (`f_lr/f_tot` 0.370-0.405); cace's carries 9.4%.
    A channel carrying 40% of the force has the leverage to reshape the energy fit; one carrying 9% is a correction to a model that is already good.
    This is the most robust difference in the table - it needs no unit conversion, and it separates the two codes cleanly.

2.  **Whether the charge is a free function or a pinned physical value.**
    In DeepMD the energy penalty exists *only* for the learned-charge arms; the pinned arm is neutral-to-helpful on energy while still winning on force.
    So the penalty is the price of the charge network's freedom - and that network is exactly the thing the campaign's force result says is worth nothing here (`deepmd-les-freeze-charge` 5.386e-02 against `deepmd-les` 5.340e-02, a 0.9% gap).
    Read together: in DeepMD the learned charge buys 0.9% of force and pays 15.5% of energy for it, which is a bad trade the model has no term to punish.

3.  **The kept self term's scale, which follows the charge strength.**
    `E_self = 5.7446 eV * sum q^2` is 606-669 eV/frame for the DeepMD learned arms against 77 eV for cace - an 8-9x difference that tracks the charge magnitude, since both codes run `remove_self_interaction: False`.
    Its force is exactly zero for pinned charges and 1.01-1.31x the *total* force for DeepMD's learned charges, against 0.55-0.59x for cace.
    This is the leading causal candidate for the energy penalty: it is a geometry-dependent spurious energy that the short-range net must cancel, and this campaign measures both ends of it (a real force against an exact zero).
    It is not proven, and the cace arms are the reason for care - cace keeps the term too and pays no energy penalty at all, so if the self term is the cause what matters would have to be its magnitude rather than its presence.

Factors 1 and 3 are not independent: a stronger charge raises both the long-range force share and the self term, so this campaign cannot say which of the two is doing the work.
The charge networks also differ by two orders of magnitude (DeepMD's adds 53,378 parameters, 1.08M total, against cace's `[24,12]` head and 2,610 total), a third way the implementations differ and a further reason not to attribute the gap to any single knob.

**One arm would separate them.**

Retrain a `deepmd-les` arm with `remove_self_interaction: True` - reachable from `les_params`, unlike the Ewald prefactor - and score it against `deepmd-les_sA`.
If the energy penalty disappears, factor 3 is the cause; if it survives, the penalty lives in the SR/LR split itself and factor 1 is.
About 3.3 hours of pod time; not run.

---

## Limits of this campaign

These bound every number above and should travel with any of them.

- **No `remove_self_interaction: True` control arm exists.**
  The cost of the flag is therefore not priced by this campaign, and it is the one experiment that would separate the candidate mechanism from the SR/LR split (see "Why the long-range term is worth 3-5x more to cace than to DeepMD").
  Measuring it needs one more arm (about 3.3 hours of pod time); it has not been run.
- **The energy result is not converged** for the DeepMD arms or for `cace-lr`, as described above.
  Only the force result is, and only for `cace-sr` is the energy converged too.
  The 120k extension does not settle it either: every DeepMD family ends at its best energy value yet (3-8% under its own 80k figure), but it is still wandering 10-19% step to step, so 120k is a better endpoint than 80k, not a converged one.
- **The 120k run set is a re-anneal, not a continuation.**
  Its restart reheats the LR 23.9x, so any statistic that spans step 80000 mixes two schedules; only endpoint-to-endpoint comparisons between the two run sets are clean.
- **Two seeds per DeepMD family, one seed per cace arm.**
  Differences below roughly 3% are not resolvable on the DeepMD side, and the cace arms have no replicate spread at all, so the cross-code factor of 2.4 should be read as an order of magnitude rather than a measurement.
- **The cross-code comparison is confounded by the descriptor.**
  The labels are bit-identical, but deepmd uses `se_a` (`sel` [39, 73], `rcut` 5.5) while cace uses `n_atom_basis` 3, `n_radial_basis` 12, `max_l` 3, `max_nu` 3 at the same cutoff.
  No arm separates the two, and `cace-sr` beating every DeepMD LES arm without a long-range term shows the confound is larger than the effect being studied.
  Only the within-code ablations are interpretable.
- **The data is a narrow, fixed-cell set**, with a total energy span of 3.96 eV over the 320 training frames (standard deviation 0.65 eV).
  That is why the energy question is fragile and the force question is not, and it is also why no linear energy calibration is meaningful here.
- **The relative-path trap.**
  DeepMD resolves a relative `systems` entry against the process working directory, not the config file.
  `eval_campaign.py` resolves them against the run directory explicitly; any new tool must do the same.

Outstanding data-quality items found while building the analysis tooling, none of which change the conclusions but all of which are worth fixing before the next campaign: the author's own loader has an ase-3.x energy-target bug, reusing the author's `atomic_energies` introduces a +72.9 eV/frame offset, and the written `sel` ([39, 73]) does not match what the descriptor measured ([31, 59]).

---

## cace arms

The two cace arms were trained on the same 320 frames and the same 500-epoch budget as the eight DeepMD arms, and are scored on the same 80 validation frames by the same harness (`cace/score_cace.py`, `cace/collect_sweep.py`).
One seed each, so unlike the DeepMD families they have no replicate spread of their own.

**On force, cace is much more accurate - and that is a statement about the descriptor, not about LES.**

At the same checkpoint rule (the mean of the four phase checkpoints, epochs 200-500) the cace arms score 2.1711e-02 (`cace-lr`) and 2.6345e-02 (`cace-sr`), against the best DeepMD arm's 5.3047e-02 - a factor of 2.4 and 2.0.
The label files are bit-identical between the codes (energy `max|dE| = 0.000e+00`; force to `5.0e-09 eV/A`, which is xyz decimal truncation), so the difference is not the data.
The control decides the interpretation: `cace-sr` has no long-range term at all, yet it beats every one of the eight DeepMD arms, including all six LES arms.
Whatever cace's descriptor and recipe contribute is worth more than the entire LES long-range channel on this system, so the cross-code factor of 2.4 must not be read as "LES is worse".
It is a confounded comparison, and this report records it as one.

**The unconfounded statement is the within-code ablation, and both codes agree in sign.**

Within cace, turning the long-range term on (`cace-lr` against `cace-sr`) improves force by 17.6% on the four phase checkpoints, and by 25.9% over epochs 300-500 alone.
That spread is informative rather than a choice of window: `cace-lr`'s epoch-200 checkpoint is its worst (`2.930e-02`, against `1.92e-02` at every later checkpoint), from the phase whose energy weight is still 0.1 while the charge head is barely trained, so the long-range head is a liability early and a large win once its charges train.
The DeepMD analogue, over its own late window (epochs 250-500), is a 5.9% gain (`5.352e-02` against `5.689e-02`).
Both codes improve when the long-range term is added; the size of the gain is architecture-dependent, and neither number transfers to the other.

**Energy is not compared across codes.**

`cace-sr`'s energy RMSE is converged by epoch 300 (`2.54e-04`, then `2.38e-04` and `2.39e-04`), but `cace-lr`'s is still falling at epoch 500 (`1.85e-04`, against a four-checkpoint mean of `3.387e-04`) - the same non-convergence the DeepMD learned-charge arms show, and for the same reason.
A cace-versus-DeepMD energy ratio would set a converged number against an unconverged one, which is why the cross-code panels are force shares and dimensionless ratios only.

**cace keeps the self term too, and it dominates cace's long-range force even more than les's.**

cace's kernel keeps the term (`remove_self_interaction: False`, the author's own default), and the same flag-flip measurement on `cace-lr` gives `f_lr_over_f_tot = 0.094` and `f_self_over_f_tot = 0.55-0.59`.
So in cace the kept self term is about six times the long-range force it sits inside, and the long-range channel carries under 10% of the total force - against about 40% for the DeepMD LES arms.
That is a real mechanism difference between the implementations, not a unit artefact: cace's Ewald term is a small correction to an already good short-range model, while the les Ewald term dominates its model's force budget.
The self term's energy scale is the same artefact in both codes: cace's `77.0 eV` self term against a `+2.53 eV` logged `E_lr` is a ratio of 31, against 37 for `deepmd-les_sA` (`669 eV` against `+18.0 eV`).
cace's charges are also not constrained to be neutral - the scored frames carry a net `6.9-12.2` in cace units per frame - so its self term is not even a fixed offset in species, unlike the DeepMD `freeze-charge` arms.

**The checkpoint rules differ between the codes, and the comparison has to say so.**

DeepMD keeps its last five checkpoints (40k-80k steps, epochs 250-500 at 160 steps/epoch); cace writes one at each phase end (epochs 200, 300, 400, 500) plus a `best_model.pth` selected by validation loss *within the final phase* on these same 80 frames.
A mean including `best_model` would be selected on the set it is scored against, so it is reported and flagged but excluded, and the notebook draws it as an open diamond.
The headline cace number is therefore the mean of four evenly spaced phase checkpoints against the DeepMD arms' mean of five late ones - comparable, but not the same rule, and one seed each against two.

**The harness needs its own long-range force pass, and that is where the first bug was.**

cace's modules cache their outputs in the batch dict, so a second forward on the same dict returns the first pass's tensor, and differentiating it hits a graph the model's own `Forces` module already consumed.
The long-range force therefore needs a fresh pass, obtained by swapping the `Forces` module's `energy_key` to `ewald_potential`, with an assertion that both passes visited the frames in the same order.
The DeepMD side needs no such care: `lr_weight` scales `E_lr` above the coordinate gradient, so `lr_weight = 0` removes the long-range force exactly and `F_LR = F(w=1) - F(w=0)` is exact rather than reconstructed.

**The split check is built so a mismatch cannot silently produce a number.**

The cace side scores against the campaign's own `xyz/valid.xyz`, and on every run the script re-derives that file's frames and compares them against the DeepMD arms' `data/data_3` on geometry.
The two agree to `5.0e-09 A` in positions and exactly in the cell, with the residual being the xyz text carrying fewer decimals than float64; a genuinely different frame set would show an order-1 A gap.
`train.xyz` is likewise exactly `data_0` + `data_1` (both `set.000` and `set.001`) + `data_2`, and shares no frame with the validation set, so neither code trained on the other's test frames.

---

## Appendix: where the numbers come from

| artefact | produced by | contents |
|---|---|---|
| `analysis/data/valid_sweep.tsv` | `deepmd/eval_campaign.py` | one row per (arm, checkpoint): full-split `rmse_e_peratom`, `rmse_f` |
| `analysis/data/valid_sweep_120k.tsv` | `deepmd/eval_campaign.py` | the same for the 40k-120k extension, one row per (arm, step) so any window or endpoint is computable; its 40k-80k rows reproduce `valid_sweep.tsv` exactly |
| `analysis/data/energy_decomp_{valid,train}.tsv` | `deepmd/decompose_energy.py` | per-frame energy error split into bias and spread, the calibrated residual, the fitted slope and the label span |
| `analysis/data/lr_mechanism_les.tsv` | `deepmd/lr_force_share.py` | per-arm long-range and kept-self force shares, each obtained by a flag flip rather than by differencing forces |
| `analysis/data/cace_valid.tsv` | `cace/score_cace.py`, `cace/collect_sweep.py` | the cace arms on the same split, with the same mechanism columns |
| `analysis/data/cace_epoch_metrics.tsv` | `cace/collect_sweep.py` | both cace arms' training traces, the source for FIG 8 |
| `analysis/data/les_metrics.tsv` | `analysis/extract_les_log.py` | parsed verbose `les.log`: latent charges per species, E_lr, charge-net weight norms |
| `analysis/data/lcurve/<arm>/lcurve.out` | `dp --pt train` | the training display log (3-frame validation columns; not used for ranking) |
| `analysis/fastlearn_campaign.ipynb` | - | the figures and tables, executed |

Figure map: FIG 1-5 are the DeepMD arms (learning curves, the full-split ranking, the level/shape split, the charge diagnostics, and the long-range energy scale).
FIG 6-9 are the cross-code comparison (the same split with cace, convergence on one epoch axis, cace's own loss trace, and the two dimensionless mechanism panels).
FIG 6-9 run only when `analysis/data/cace_valid.tsv` exists; without it each one prints a skip line and the rest of the notebook is unaffected.

The notebook is the presentation layer and this report is the argument.
Where they disagree, the notebook's printed numbers are the raw ones.
