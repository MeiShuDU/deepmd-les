# LES performance on the DeepMD-kit-FastLearn minimal dataset

Summary of how the LES (Latent Ewald Summation) long-range extension to DeePMD-kit performs on the small water dataset shipped in `DeePMD-kit-FastLearn/data/`.
All numbers below come from matched training runs in `01.train/rerun/extended/`.
Model code: `hybrid_ener` in `deepmd/pt/model/model/hybridles_model.py`.
Figures and the full per-checkpoint data are in `01.train/rerun/LES_analysis.ipynb`.

## Verdict

The fixed-charge LES variant (`use_fixed_atomic_charges: true`) beats an ordinary DeePMD model with the same short-range network on **force accuracy**, reproducibly across two seeds and across every late checkpoint.
It does **not** beat it on energy accuracy.

The learnable-charge variant (`use_fixed_atomic_charges: false`) also beats the ordinary model on force, but only by about 1-2%, and it shows no energy advantage.

The force gain is specific to the schedule it was measured under, not a property of the architecture alone.
Everything above uses `decay_steps 5000`, and the latent charges are still in transit at step 5000, so the gain depends on the long low-rate tail that follows.
The delayed-cliff probe later in this document shortens that tail and widens the LES arms' replicate spread from about 4% to 26 - 27%; read that section before quoting the gain as a model property.

So the defensible recommendation is narrower than "LES is better": if you use `hybrid_ener` on this dataset, use the fixed-charge configuration and expect a force gain, not an energy gain.

## Dataset

Small, periodic water boxes; each frame is 64 H2O (64 O + 128 H = 192 atoms).

| system | role | frames | atoms/frame |
|---|---|---|---|
| `data/data_0` | train | 80 | 192 |
| `data/data_1` | train (2 sets) | 160 | 192 |
| `data/data_2` | train | 80 | 192 |
| `data/data_3` | validation | 80 | 192 |
| **total** | | **400** | |

- `type_map`: `["O", "H"]`.
- Source trajectory: VASP `OUTCAR` in `00.data/`.
- Train set is 320 frames, validation 80 frames.

## Protocol

Matched comparison, so the only difference between arms is the long-range treatment.

- Same data, same `se_a` short-range network (`sel [46, 92]`, `rcut 6.0`, `neuron [25, 50, 100]`, fitting `[240, 240, 240]`), same loss weights, same LR schedule.
- 10,000 steps, `decay_steps 5000`, `start_lr 1e-3`, `stop_lr 3.51e-8`, `batch_size auto`, `numb_btch 3`.
- Two replicates: seed A (net seeds 1/1, train seed 10) and seed B (net seeds 2/2, train seed 20).
- Three arms: `ordinary` (standard model), `hybrid_q` (LES, learnable charges), `hybrid_fixed` (LES, fixed charges).
- Configs: `01.train/rerun/gen_extended.py` generates them; logs in `01.train/rerun/extended/run_*/`.

### How accuracy is measured here

The `rmse_e_val` / `rmse_f_val` columns written to `lcurve_*.out` are **not** averages over the validation set.
`training.validation_data` uses `batch_size: 1` with `numb_btch: 3`, and the training loop draws only those three batches at each display step (`deepmd/pt/train/training.py`, `log_loss_valid`).
Every logged point is therefore the RMSE over 3 randomly drawn frames out of 80.

Measured against the exact full-set value of the same checkpoint, a single 3-frame draw is a wide but only mildly biased estimate: its median sits 2% to 12% below the true value, while its 5th-to-95th percentile span is 2x to 5x.
So the logged curve is noisy without being strongly skewed low, and the bias is not its main hazard.
The main hazard is that consecutive logged points come from different checkpoints, so a stretch of the logged curve cannot be compared against any single checkpoint's exact value.
The numbers in this document instead come from evaluating each **saved checkpoint over all 80 validation frames**, with no subsampling, using the official DeepMD test math (`test_ener` + `weighted_average`).
This is deterministic: re-running it reproduces the digits exactly.
The harness is `01.train/rerun/eval_full.py` driven by `01.train/rerun/sweep_valid.sh`.

Evaluating on the full set removes the sampling noise but exposes a second one, described in the next section, which is why the headline numbers average over three late checkpoints (steps 8000, 9000, 10000) rather than quoting a single one.

`dp --pt test` cannot be used on these checkpoints: it calls `torch.jit.script` on the reconstructed model, and the hybrid forward path is not scriptable. The harness passes `no_jit=True` instead.

## Headline results

Energy RMSE per atom (eV). Lower is better.

| replicate | model | step 8000 | step 9000 | step 10000 |
|---|---|---|---|---|
| A | ordinary | 1.0658e-3 | 1.1971e-3 | 1.1636e-3 |
| A | hybrid_q | 1.0435e-3 | 1.1659e-3 | 9.7123e-4 |
| A | hybrid_fixed | 1.0649e-3 | 1.1398e-3 | 7.0353e-4 |
| B | ordinary | 8.8740e-4 | 1.2551e-3 | 9.2139e-4 |
| B | hybrid_q | 1.2146e-3 | 8.2252e-4 | 9.7991e-4 |
| B | hybrid_fixed | 9.9104e-4 | 6.7811e-4 | 1.5468e-3 |

Force RMSE (eV/A). Lower is better.

| replicate | model | step 8000 | step 9000 | step 10000 |
|---|---|---|---|---|
| A | ordinary | 8.4135e-2 | 8.3945e-2 | 8.3804e-2 |
| A | hybrid_q | 8.2085e-2 | 8.1892e-2 | 8.1749e-2 |
| A | hybrid_fixed | 7.6701e-2 | 7.6593e-2 | 7.6477e-2 |
| B | ordinary | 7.9574e-2 | 7.9057e-2 | 7.8743e-2 |
| B | hybrid_q | 7.8618e-2 | 7.8383e-2 | 7.8234e-2 |
| B | hybrid_fixed | 7.3670e-2 | 7.3488e-2 | 7.3354e-2 |

Means and the observed min-max range across those three checkpoints:

Means and the observed min-max range across those three checkpoints:

| replicate | model | rmse_e mean | range | rmse_f mean | range |
|---|---|---|---|---|---|
| A | ordinary | 1.1422e-3 | 1.0658 - 1.1971e-3 | 8.3962e-2 | 8.3804 - 8.4135e-2 |
| A | hybrid_q | 1.0602e-3 | 0.9712 - 1.1659e-3 | 8.1909e-2 | 8.1749 - 8.2085e-2 |
| A | **hybrid_fixed** | 9.6939e-4 | 0.7035 - 1.1398e-3 | **7.6590e-2** | 7.6477 - 7.6701e-2 |
| B | ordinary | 1.0213e-3 | 0.8874 - 1.2551e-3 | 7.9125e-2 | 7.8743 - 7.9574e-2 |
| B | hybrid_q | 1.0057e-3 | 0.8225 - 1.2146e-3 | 7.8412e-2 | 7.8234 - 7.8618e-2 |
| B | **hybrid_fixed** | 1.0720e-3 | 0.6781 - 1.5468e-3 | **7.3504e-2** | 7.3354 - 7.3670e-2 |

Change relative to the ordinary model in the same replicate, from those means (negative = LES better):

| replicate | model | d rmse_e | d rmse_f |
|---|---|---|---|
| A | hybrid_q | +7.2% | +2.4% |
| A | hybrid_fixed | +15.1% | +8.8% |
| B | hybrid_q | +1.5% | +0.9% |
| B | hybrid_fixed | -5.0% | +7.1% |

## Numerical stability: the energy metric moves by up to 2.3x between checkpoints

This is the single most important thing to know before reading any LES accuracy number from this dataset.

The energy RMSE swings widely between adjacent late checkpoints, in **every** arm including the ordinary one.
The largest swing is `hybrid_fixed` replicate B: 6.7811e-4 at step 9000, then 1.5468e-3 at step 10000, a factor of 2.28.
`ordinary` replicate B swings by 1.41x over the same interval.
The force RMSE, by contrast, falls monotonically over these checkpoints and varies by at most 1.1%.

Both facts are properties of the model, not of the measurement: these are exact full-set evaluations, and the LES weights are verified frozen over this window.
The practical consequence is that no single checkpoint ranks the arms, which is why everything above is a three-checkpoint mean with the range shown.

Because of the swing, a single-point energy comparison can be made to say almost anything.
At step 10000 alone, `hybrid_fixed` looks 40% better than `ordinary` in replicate A (7.0353e-4 vs 1.1636e-3) and 68% worse in replicate B (1.5468e-3 vs 9.2139e-4).

### Reconciling the logged curve with the exact values

The logged curve looks inconsistent with the exact values, and it is worth being explicit that this is not a contradiction.
For `hybrid_fixed` replicate B:

| step | exact, full set | median of a 3-frame draw | logged at that step |
|---|---|---|---|
| 8000 | 9.9104e-4 | 9.243e-4 | 1.100e-3 |
| 9000 | 6.7811e-4 | 6.208e-4 | 7.080e-4 |
| 10000 | 1.5468e-3 | 1.498e-3 | 1.050e-3 |

Every logged value is an unremarkable draw from its own checkpoint's sampling distribution, so the logged curve is not lying.
What cannot be done is to take the median of the logged values over steps 8000-10000, which is 7.08e-4, and read it against the step-10000 exact value of 1.5468e-3.
Those are different models.
The 2.2x gap is the checkpoint movement, not the sampling: a 3-frame draw from the step-10000 checkpoint has a median of 1.498e-3, only 3% below its exact value.

### Why the number moves: the energy zero, not the physics

The exact value is an L2 norm over per-frame errors, and that construction is worth spelling out because it is where the movement comes from.
Let `e_i = (E_pred,i - E_label,i) / natoms` be the signed energy error of validation frame `i`, in eV per atom.
Then the reported number is

```
rmse_e = sqrt( (1/N) * sum_i e_i^2 ),   N = 80.
```

Squared, and split into its two moments, this is

```
rmse_e^2 = mean(e)^2 + var(e).
```

The first term is the squared mean of the per-frame errors, a systematic offset the checkpoint carries on every frame; the second is their variance, how far the individual frames scatter around that offset.
Call them `bias` and `spread`.

This is the sample form of the bias-variance decomposition.
In its usual machine-learning statement, the expected squared error of a learner splits into a bias term (the learner is systematically wrong in one direction) and a variance term (the learner is sensitive to which data it saw), and the two move for unrelated reasons.
Here the "data" is the fixed set of 80 frames, so the same algebra says that a metric which mixes a per-frame constant with the scatter of the frames will jump whenever either component moves, and that a single scalar conceals which one did.
Reading the two components apart is therefore not cosmetic; it is the difference between measuring the model and measuring an unconstrained constant.

For an energy model the bias component has a specific physical meaning that makes the split decisive.
Forces are the negative gradient of the energy with respect to atomic positions, so a constant added to a whole frame's energy has zero gradient and cannot change any force: `rmse_f` is blind to the bias term by construction.
The absolute energy zero is in turn not fixed by any physical interaction; it is set only by the mean of the training energies, and it is a nearly flat direction of the loss.
A drift of that zero is exactly a change in `bias` alone, moving `rmse_e` while leaving `rmse_f` untouched.

That is what the data show.
Evaluating the same 80 frames at each late checkpoint of `hybrid_fixed` replicate B and splitting the error:

| step | rmse_e | bias | spread | bias^2 / rmse_e^2 |
|---|---|---|---|---|
| 8000 | 9.9104e-4 | -7.6733e-4 | 6.2719e-4 | 0.60 |
| 9000 | 6.7811e-4 | -2.6535e-4 | 6.2404e-4 | 0.15 |
| 10000 | 1.5468e-3 | -1.4160e-3 | 6.2246e-4 | 0.84 |

and the LES-free `ordinary` replicate B, which behaves the same way:

| step | rmse_e | bias | spread | bias^2 / rmse_e^2 |
|---|---|---|---|---|
| 8000 | 8.8740e-4 | +4.1551e-4 | 7.8411e-4 | 0.22 |
| 9000 | 1.2551e-3 | -9.8805e-4 | 7.7405e-4 | 0.62 |
| 10000 | 9.2139e-4 | +5.0519e-4 | 7.7055e-4 | 0.30 |

The `spread` column is constant to better than 1% within each run, while `bias` moves by up to a factor of 5 and even flips sign from one checkpoint to the next in the ordinary arm.
The 2.28x swing between steps 9000 and 10000 is the bias term alone.
A direct check of the same claim: regressing the step-10000 per-frame errors on the step-9000 ones gives

```
e(10000) = 0.995 * e(9000) - 1.152e-3,
```

with a residual standard deviation of 3.9e-5 against a signal (per-frame error) standard deviation of 6.2e-4, about 16x smaller.
A slope of 1.00 with a residual far below the signal means the two checkpoints make the *same* per-frame errors, frame for frame, offset by a single constant: 8 of the 10 worst frames are shared, and the top-10 share of the variance actually falls (0.505 to 0.321) as the constant grows, because the bulk of the frames move with it.

The offset is carried by the model, not by the validation frames.
Re-measuring the *train* split (320 frames) at the same three checkpoints and comparing its bias against the validation bias:

| run | step | train bias | valid bias | valid - train |
|---|---|---|---|---|
| hybrid_fixed_sB | 8000 | -6.7657e-4 | -7.6733e-4 | -9.08e-5 |
| hybrid_fixed_sB | 9000 | -1.7490e-4 | -2.6535e-4 | -9.05e-5 |
| hybrid_fixed_sB | 10000 | -1.3252e-3 | -1.4160e-3 | -9.08e-5 |
| ordinary_sB | 8000 | +4.6797e-4 | +4.1551e-4 | -5.25e-5 |
| ordinary_sB | 9000 | -9.3188e-4 | -9.8805e-4 | -5.62e-5 |
| ordinary_sB | 10000 | +5.6462e-4 | +5.0519e-4 | -5.94e-5 |

The two biases move together at every checkpoint, and the gap between them is a constant to four digits (-9.07e-5 across the hybrid_fixed window, about -5.6e-5 for ordinary), which is the train-versus-validation mean-energy mismatch.
The train spread is flat across the same window too (6.649, 6.640, 6.650e-4).
So the wandering quantity is a single number added to every frame of both splits, which is what an energy zero is, and not a property of the 80 validation frames.

Why this is not a physical mismatch:

- A physical error changes the shape of the energy surface, so it perturbs `spread`, the per-frame slope between checkpoints, and the forces. None of those move: the spread is flat, the regression slope is 1.00, and force RMSE is monotone and varies by at most 1.1% over the identical checkpoints.
- The same offset appears on the train split, measured over 320 frames, and tracks the validation offset to a constant, so it is not an artifact of which frames are in the validation set.
- The LES internals are frozen over this window (charges and charge-net weights move by less than 3 parts in 1e5), and the LES-free `ordinary` arm shows the same energy swing with no LES in the model at all. So the movement is not in the long-range physics.
- The magnitude is wrong for a physics error. The whole effect is about `1.15e-3 eV/atom * 192 = 0.22 eV` on frames whose total energy is near `-29000 eV`, that is 7.6 parts per million. A genuine electrostatic error (missing screening, double counting) would be percent-to-tens-of-percent of the interaction energy, orders of magnitude larger.
- The offset changes sign in the ordinary arm, which a systematic physical defect would not do.

So the honest reading is that `rmse_e` on this dataset reports the energy zero, and the energy zero is a training convention rather than a physical quantity.
The energy ranking between arms is unresolved because that convention wanders by more than the arm-to-arm differences; it is not unresolved because the physics disagrees.
The force ranking is unaffected, and the force metric is the one the bias term cannot touch.

## Is the effect real, or noise?

Applying a disjoint-range test, in which the arm's whole three-checkpoint range must sit below the ordinary model's range in the same replicate:

- **Force, `hybrid_fixed`: passes, in both replicates.**
  A: 7.6477-7.6701e-2 against ordinary's 8.3804-8.4135e-2.
  B: 7.3354-7.3670e-2 against ordinary's 7.8743-7.9574e-2.
  The gaps are 8.8% and 7.1%, and neither range comes close to overlapping.
- **Force, `hybrid_q`: passes technically, but by a thin margin.**
  A: 8.1749-8.2085e-2, disjoint by 2.4%.
  B: 7.8234-7.8618e-2 against ordinary's 7.8743-7.9574e-2, disjoint only because the edges differ by 0.16%.
- **Energy, `hybrid_fixed`: fails.**
  The replicate A range still overlaps ordinary's (7.0353e-4-1.1398e-3 vs 1.0658e-3-1.1971e-3), and in replicate B the mean is 5.0% *worse* than ordinary.

For calibration, the ordinary model's own two replicates differ by 12% on energy and 6% on force, while its consecutive checkpoints differ by up to 41% on energy.
Checkpoint-to-checkpoint variation dominates seed-to-seed variation here, so the earlier version of this document, which quoted single logged steps and reported an 11-23% fixed-charge energy win, was reading sampling and checkpoint noise as signal.

## Why: the LES decomposition

Late-training means over the last 20 verbose log entries (steps 8400-10300), parsed from `train.log`:

| run | E_SR (eV) | E_LR (eV) | \|E_LR/E_SR\| | RMS F_SR | RMS F_LR | \|F_LR\|/\|F_SR\| |
|---|---|---|---|---|---|---|
| hybrid_q_sA | -29922.4 | -21.1 | 0.0007 | 0.8353 | 0.2053 | 0.271 |
| hybrid_q_sB | -29917.9 | -25.8 | 0.0009 | 0.8101 | 0.1646 | 0.224 |
| hybrid_fixed_sA | -29267.9 | -675.6 | 0.0231 | 0.7187 | 0.4817 | 0.677 |
| hybrid_fixed_sB | -29246.6 | -697.1 | 0.0238 | 0.7864 | 0.5377 | 0.729 |

The fixed-charge channel binds about 700 eV and carries roughly 70% of the short-range force magnitude as long-range electrostatics.
The short-range network then models the residual, which is consistent with the force gain.
It is worth noting that both RMS force columns are an order of magnitude larger than the final force RMSE, so the two channels largely cancel at convergence.

The learnable-charge channel contributes under 0.1% of the total energy and about 22-27% of the force.

The reason is visible in the latent charges themselves, from the LES log at step 10000:

| run | H charge (e) | O charge (e) | spread \|q\| |
|---|---|---|---|
| hybrid_q_sA | +0.1133 | -0.1799 | 0.1399 |
| hybrid_fixed_sA | +0.6119 | -1.0669 | 0.7936 |

`hybrid_fixed` holds charges near the physical water values, so its long-range channel is fully engaged.
`hybrid_q` is free to learn any charges and learns nearly none: its charges collapse toward zero, leaving a long-range channel with roughly 30x less energy than the fixed-charge arm.
That is why it delivers no accuracy benefit, and it is a more likely explanation than "extra free parameters add variance".

## Convergence behavior (important)

Do not judge this comparison before convergence.
At 2000 steps both LES variants appeared to beat the ordinary model on both metrics; that reading was pre-convergence.

The ordinary model's validation RMSE, replicate A:

| step | rmse_e | rmse_f | source |
|---|---|---|---|
| 2000 | 8.62e-3 | 1.18e-1 | logged (3-frame) |
| 4000 | 5.28e-3 | 1.00e-1 | logged (3-frame) |
| 6000 | 1.1230e-3 | 8.5695e-2 | full set |
| 8000 | 1.0658e-3 | 8.4135e-2 | full set |
| 9000 | 1.1971e-3 | 8.3945e-2 | full set |
| 10000 | 1.1636e-3 | 8.3804e-2 | full set |

Checkpoints are only kept from step 6000 onward, so the first two rows can only come from the log.
Note what this shows: force error is still creeping down at 10,000 steps, about 2% over the last 4,000 steps, while energy error stopped improving after step 6000 and now merely wanders.
Training is not fully converged, and extension beyond 10,000 steps would be needed before treating either metric as final.

## Is the latent charge underfitted? A convergence probe with a delayed cliff

**Read this section as a probe, not as an accuracy comparison.**
The two schedules differ only in `decay_steps`, and neither is a schedule worth shipping: one ends the run at `5.93e-6`, the other at `4.56e-7`, and the second is so small that the final 2500 steps barely move the model at all.
The question here is narrow and diagnostic - were the latent charges still learning when the rate collapsed, that is underfitted, or had they genuinely converged? - and it is answered from the charge trajectories and from how much the model can still move, not from which arm scores better.
No accuracy conclusion follows from this section, in either direction.

### The schedule is a step function, not a ramp

Everything above was trained with `decay_steps: 5000` out of `numb_steps: 10000`.
That reads like a smooth decay, and it is not one.
`LearningRateExp` (`deepmd/dpmodel/utils/learning_rate.py`) is **piecewise constant**:

    decay_rate = exp( log(stop_lr / start_lr) / (stop_steps / decay_steps) )
    lr(step)   = max( start_lr * decay_rate ** (step // decay_steps), stop_lr )

So `decay_steps` is not a decay rate: it is the **length of the full-rate phase**, and with `start_lr 1e-3`, `stop_lr 3.51e-8` and `stop_steps 10000` the two schedules are:

| `decay_steps` | full rate `1.0e-3` | then, to the end | final 2500 steps |
|---|---|---|---|
| 5000 (everything above) | steps 0 - 4999 | `5.93e-6` | at `5.93e-6` |
| 7500 (this probe) | steps 0 - 7499 | `4.56e-7` | at `4.56e-7` |

The runs' own `lr` column confirms it rather than trusting the formula: the d5000 curves print `1.00e-3` through step 5000 then `5.9e-6` from step 5100, and the d7500 curves print `1.00e-3` through step 7500 then `4.6e-7` from step 7600.

Six further runs were trained with `decay_steps: 7500` and nothing else changed: same arms, same two seeds, same network, same data order, same loss.
Moving the cliff to 7500 buys the charges 2500 more full-rate steps.
It is not a clean single-variable change, because the same move also drops the post-cliff rate 13x; that is dealt with under confounds.
The confound limits what the *metrics* can say.
It does not limit the charge-trajectory evidence, which compares the two arms over a window in which they differ only in rate.

### Is the comparison matched?

Not automatically, and it was checked rather than assumed.
The model source also changed between the two run sets, so "same config" was not by itself evidence of "same code".
The log header cannot settle this: every run prints the same `source commit:` line, which comes from a `_version.py` frozen at install time and is never regenerated.

Four checks settle it, all on the logs themselves:

| check | result |
|---|---|
| the two `input.yaml` files, diffed | `decay_steps` only, plus the `disp_file` name; every other setting byte-identical |
| charge-network initial weight norms, printed at forward 1 | identical to the last digit for every arm pair (`outnet.0.linear.weight`: `3.260910e+00` for both `hybrid_q_sA` schedules, `3.265586e+00` for both `hybrid_fixed_sB` schedules) |
| the step-1 `[HybridLES]` decomposition | identical (`E_SR(frm0) = -29855.635874`, `RMS F_SR = 1.4504e-02`) |
| `lcurve` rows while both schedules share the learning rate | bit-identical, all 51 rows from step 1 to step 5000, every column, `max\|diff\| = 0.0` |

So the two schedules start from the same weights and follow the same trajectory until their cliffs separate them.
The long-range virial is not in this loss either (`start_pref_v: 0.0`, `limit_pref_v: 0.0`), so its addition between the two run sets cannot have mattered here.

### What the charges did across each cliff

Within every run the charges collapse once the cliff passes.
Ranges below are `max - min` of the per-atom charge over the phase:

| arm | `decay_steps` | q_H before cliff | after | q_O before cliff | after |
|---|---|---|---|---|---|
| q A | `5000` | 0.1581 | 0.0059 | 0.2597 | 0.0145 |
| q A | `7500` | 0.1844 | 0.0050 | 0.3143 | 0.0100 |
| q B | `5000` | 0.2171 | 0.0040 | 0.1737 | 0.0127 |
| q B | `7500` | 0.2171 | 0.0040 | 0.2149 | 0.0094 |
| fx A | `5000` | 0.1177 | 0.0031 | 0.0718 | 0.0070 |
| fx A | `7500` | 0.1267 | 0.0027 | 0.1046 | 0.0057 |
| fx B | `5000` | 0.1366 | 0.0025 | 0.0735 | 0.0058 |
| fx B | `7500` | 0.1366 | 0.0021 | 0.0939 | 0.0040 |

That table cannot settle the question on its own, because the two phases are not the same length: "before" is 5000 steps for one schedule and 7500 for the other, and "after" is 5000 against 2500.
Drift accumulates with steps, so the comparison is confounded by construction.

The clean test is a single window, steps 5001 - 7500, over which both schedules started from the identical state at step 5000 and differ only in learning rate.
There the d5000 arm is already below its cliff while the d7500 arm is still at the full rate:

| arm | charge | median \|grad\|, decay 5000 | decay 7500 | ratio | drift, decay 5000 | drift, decay 7500 |
|---|---|---|---|---|---|---|
| q A | q_H | 1.038e-01 | 2.092e+01 | 202x | 0.0050 | 0.0311 |
| q A | q_O | 1.038e-01 | 2.092e+01 | 202x | 0.0135 | 0.0513 |
| q B | q_H | 7.812e-02 | 1.171e+01 | 150x | 0.0040 | 0.0162 |
| q B | q_O | 7.812e-02 | 1.171e+01 | 150x | 0.0127 | 0.0423 |
| fx A | q_H | 3.391e-01 | 5.085e+01 | 150x | 0.0031 | 0.0095 |
| fx A | q_O | 3.391e-01 | 5.085e+01 | 150x | 0.0070 | 0.0310 |
| fx B | q_H | 2.744e-01 | 3.853e+01 | 140x | 0.0025 | 0.0099 |
| fx B | q_O | 2.744e-01 | 3.853e+01 | 140x | 0.0058 | 0.0197 |

Over the same 2500 steps from the same starting point, the charge-network gradient is **140x to 202x larger** under the later cliff, and the charges travel **3x to 7x further**.
So the answer to the question as posed is yes: the charges were being held still by the schedule, not by having converged.

The gradient magnitude tells the same story in phase terms.
The median total `|grad|` of the charge network is `1.27e+01` (q A) and `7.28e+01` (fx A) while the rate is full, then falls to `7.96e-02` and `2.74e-01` after the cliff at 5000 - a collapse of roughly 160x to 265x in a single step.
By step 5000 the charge network is nowhere near a stationary point; the cliff lands on a network that is still moving fast.

### The cliff freezes the model in place

The diagnostic signature of the delayed cliff is not the metric average but the metric's *stability*.
Over steps 8000 - 10000, by which point both schedules are past their cliff, per run:

| run | energy max/min, `5000` | energy max/min, `7500` | force change, `5000` | force change, `7500` |
|---|---|---|---|---|
| ordinary A | 1.12x | 1.01x | -0.39% | -0.25% |
| ordinary B | 1.41x | 1.05x | -1.04% | -0.15% |
| hybrid_q A | 1.20x | 1.01x | -0.41% | -3.27% |
| hybrid_q B | 1.48x | 1.02x | -0.49% | -0.09% |
| hybrid_fixed A | 1.62x | 1.03x | -0.29% | -2.94% |
| hybrid_fixed B | 2.28x | 1.03x | -0.43% | -0.11% |

`energy max/min` is the largest over the smallest of the three checkpoints' energy RMSE; `force change` is the last checkpoint's force RMSE relative to the first.

Under `decay_steps 5000` the energy error still wanders by up to 2.28x from checkpoint to checkpoint, which is the energy-zero drift described earlier in this document.
Under `7500` it is flat to within 5%, and the force to within 3.3%.
The rate `4.56e-7` is small enough that the model stops moving: wherever it stood at step 7500 is where it stays.

Two consequences follow, and they point in different directions.

The first is a limit on the rest of this document: neither schedule produces a converged model.
The d7500 "final checkpoint" is a snapshot of the step-7500 state, and the d5000 final checkpoint is one arbitrary phase of a series still drifting by up to 2.28x.
Neither is a fixed point, so neither should be read as the model's converged accuracy.

The second is what makes the probe useful.
Because the post-cliff phase is nearly a no-op under d7500, any difference between the two replicates that exists when the rate collapses survives to the end.
Under d5000 the 5000 post-cliff steps at `5.93e-6` still do real annealing and pull the replicates together; under d7500 nothing does.

### What that does to the replicates

The informative statistic is therefore the spread between replicates, not the mean over them.
Force RMSE at the final checkpoint, with replicate B expressed relative to A (`|B / A - 1|`):

| arm | spread, `5000` | spread, `7500` |
|---|---|---|
| ordinary | 6.0% | 6.6% |
| hybrid_q | 4.3% | 27.4% |
| hybrid_fixed | 4.1% | 26.2% |

The ordinary model's replicates land 6 - 7% apart under either schedule, so freezing costs it nothing.
Both LES arms land about 4% apart under d5000 and 26 - 27% apart under d7500.

The per-replicate deltas show the same shape, and show that the arm matters less than the seed:

| arm | rep | energy, `5000` | energy, `7500` | delta | force, `5000` | force, `7500` | delta |
|---|---|---|---|---|---|---|---|
| hybrid_fixed | A | 7.0353e-04 | 7.1836e-04 | +2.1% | 7.6477e-02 | 9.1040e-02 | +19.0% |
| hybrid_fixed | B | 1.5468e-03 | 5.5889e-04 | -63.9% | 7.3354e-02 | 6.7216e-02 | -8.4% |
| hybrid_q | A | 9.7123e-04 | 9.1626e-04 | -5.7% | 8.1749e-02 | 9.7333e-02 | +19.1% |
| hybrid_q | B | 9.7991e-04 | 7.0357e-04 | -28.2% | 7.8234e-02 | 7.0697e-02 | -9.6% |
| ordinary | A | 1.1636e-03 | 8.6664e-04 | -25.5% | 8.3804e-02 | 7.7136e-02 | -8.0% |
| ordinary | B | 9.2139e-04 | 8.0869e-04 | -12.2% | 7.8743e-02 | 7.2078e-02 | -8.5% |

Both LES arms' replicate A is about 19% worse on force and both arms' replicate B is 8 - 10% better, to within a percentage point.
So the two hybrid variants behave as one population, and under the frozen schedule the replicate is the dominant factor while the arm is close to irrelevant.
This also means the arm-level force change - +5.9% (`hybrid_q`) and +6.4% (`hybrid_fixed`) over the matched window - is not a resolvable effect: it is a two-sample mean of a quantity whose spread between the two replicates is 26%.

Where the separation appears:
at step 7000, the last full-rate checkpoint, the d7500 LES arms are 4.3% (`hybrid_q`) and 13.3% (`hybrid_fixed`) apart; at the first post-cliff checkpoint, step 8000, they are 42% and 39% apart.
The divergence therefore appears across the cliff rather than long before it.
With a single pre-cliff checkpoint and two seeds it cannot be said whether the cliff created the separation or merely first sampled it there.

Matched-window means, steps 8000 - 10000, the only window in which both schedules are past their cliff:

| arm | energy, `5000` | energy, `7500` | delta | force, `5000` | force, `7500` | delta |
|---|---|---|---|---|---|---|
| ordinary | 1.0817e-03 | 8.2602e-04 | -23.6% | 8.1543e-02 | 7.4688e-02 | -8.4% |
| hybrid_q | 1.0330e-03 | 8.1534e-04 | -21.1% | 8.0160e-02 | 8.4915e-02 | +5.9% |
| hybrid_fixed | 1.0207e-03 | 6.4016e-04 | -37.3% | 7.5047e-02 | 7.9885e-02 | +6.4% |

This replaces the earlier revision's comparison at the single final checkpoint, which was invalid twice over: it read a drifting series (`hybrid_fixed` B's final value, `1.5468e-03`, is that run's own window maximum, and most of its apparent -63.9%), and it read a frozen snapshot as if it were converged.
Over the matched window the energy deltas are -23.6% (ordinary), -21.1% (`hybrid_q`) and -37.3% (`hybrid_fixed`).
The ordinary model gains about as much as `hybrid_q`, so most of the energy movement is a property of the schedule rather than of the LES channel.
Averaging over 6000 - 10000 instead would be worse, not better: checkpoints 6000 and 7000 are still at full rate for the d7500 arms, where their energy error is 30x to 100x larger, so that average describes the transient.

### The terms of the LES output

Where the change lands, averaging over steps 8000 - 10000, by which point both schedules have decayed:

| arm | `decay_steps` | E_SR | E_LR | \|E_LR/E_SR\| | RMS F_SR | RMS F_LR | RMS F_LR / RMS F_SR |
|---|---|---|---|---|---|---|---|
| fx A | `5000` | -2.9268e+04 | -6.7559e+02 | 2.31e-02 | 7.2023e-01 | 4.8215e-01 | 0.670 |
| fx A | `7500` | -2.9244e+04 | -6.9988e+02 | 2.39e-02 | 7.5831e-01 | 5.0323e-01 | 0.665 |
| fx B | `5000` | -2.9247e+04 | -6.9706e+02 | 2.38e-02 | 7.8703e-01 | 5.3888e-01 | 0.687 |
| fx B | `7500` | -2.9237e+04 | -7.0690e+02 | 2.42e-02 | 7.6489e-01 | 5.1976e-01 | 0.681 |
| q A | `5000` | -2.9922e+04 | -2.1123e+01 | 7.06e-04 | 8.3811e-01 | 2.0558e-01 | 0.246 |
| q A | `7500` | -2.9911e+04 | -3.3026e+01 | 1.10e-03 | 8.4520e-01 | 1.9499e-01 | 0.232 |
| q B | `5000` | -2.9918e+04 | -2.5791e+01 | 8.62e-04 | 8.0998e-01 | 1.6477e-01 | 0.204 |
| q B | `7500` | -2.9915e+04 | -2.8683e+01 | 9.59e-04 | 8.0336e-01 | 1.3426e-01 | 0.168 |

The long-range energy grows in magnitude under the later cliff in all four LES arms: `q A` by 56% (`-2.11e+01` to `-3.30e+01`), `q B` by 11%, `fx A` by 3.6%, `fx B` by 1.4%, and `|E_LR/E_SR|` rises with it in every case.
So the extra full-rate steps do make the long-range channel carry more of the energy, which is the mechanism behind the energy movement in the table above.
It is a statement about what the charges do, not about whether the model is better.

The force side of the same split does not follow.
`RMS F_LR / RMS F_SR` is flat or slightly lower under the later cliff in all four arms (0.246 to 0.232, 0.204 to 0.168, 0.670 to 0.665, 0.687 to 0.681), and `RMS F_SR` itself rises for `q A` and `fx A`.
More full-rate steps shift the long-range channel's share of the energy without improving its share of the force.
That is consistent with the force fit being the seed-sensitive part: the channel that the extra steps grow is not the one that would need to improve to tighten the force.

### Verdict

The probe answers its question in the affirmative, and that answer is solid.
The charges were not converged at step 5000; the cliff froze them.
Over the identical window 5001 - 7500 the charge network's gradient is 140x to 202x larger under the later cliff and the charges move 3x to 7x further, so the plateau after step 5000 was imposed by the schedule rather than earned.

What follows from it is a change in the model's character, not in its accuracy.
Under the delayed cliff the post-cliff phase is a near no-op: the energy zero stops drifting (stability 1.01 - 1.05x against 1.12 - 2.28x), the force settles to within 3.3%, and the LES arms' replicate spread widens from about 4% to 26 - 27% while the ordinary model's stays at 6 - 7%.
A schedule that freezes the run freezes the seed too, and the LES long-range channel is the part that is still seed-dependent at the cliff.

The tempting reading - that more full-rate steps for the charges means a more accurate model - is not supported and cannot be tested here.
The force deltas between the schedules are +5.9% and +6.4% on the mean of two replicates whose spread is 26%, so they are not resolvable, and the energy movement is shared with the ordinary model, which has no LES channel at all.
The disputed ranking (the LES arms beating the ordinary model on force in all four arm-replicate comparisons under `decay_steps 5000`, and in only two of four under `7500`) is best read the same way: under the frozen schedule one LES seed lands badly and there is no annealing left to recover it.

None of this is an accuracy verdict, because the delayed schedule is not a usable one: its last 2500 steps run at an effective standstill.
It is a diagnostic that shows the charge network is not converged at step 5000 and that the LES channel's force fit is seed-sensitive when the anneal is removed.
Whether a schedule designed to converge both - a longer full-rate phase plus a genuinely decaying tail - would keep the force advantage is not known, and the caveats below say what it would take to find out.

### Confounds

The two effects are not separable in this experiment.
Moving the cliff to 7500 gives 2500 more steps at `1.0e-3` **and** ends the run at `4.56e-7` instead of `5.93e-6`.
Every metric comparison between the two schedules therefore measures the pair of changes together, and since the second change freezes the run, the metrics mostly measure the freeze.
Disentangling them needs a schedule that keeps the final rate equal, either a third arm with `decay_steps 7500` and `stop_lr` raised 13x, or a constant-rate control that never decays; neither was run here.
The charge-trajectory evidence in "What the charges did across each cliff" is unaffected, because that window holds the rate comparison on one side only.

Two seeds.
The 26% replicate spread is itself a two-sample estimate, so its magnitude is uncertain; the direction (the LES arms widen, the ordinary model does not) is consistent across both LES arms and both schedules.
Likewise the pre-cliff spread at step 7000 rests on a single checkpoint taken while the rate is still full, where the metrics are noisy.

Neither schedule is a production schedule, so nothing in this section should be quoted as an accuracy result.

### Reproduce

```bash
cd 01.train/rerun
python gen_extended.py --decay 7500      # writes extended/run_*_d7500/input.yaml
LOGTAG=_d7500 bash run_extended.sh run_ordinary_sA_d7500 run_hybrid_q_sA_d7500 \
    run_hybrid_fixed_sA_d7500 run_ordinary_sB_d7500 run_hybrid_q_sB_d7500 \
    run_hybrid_fixed_sB_d7500 > run_extended_d7500.log 2>&1

bash sweep_valid.sh $(cd extended && ls -d run_*_d7500)   # writes sweep_valid_d7500.tsv
python diag_lrphase.py                   # writes lrphase_q.tsv, lrphase_grad.tsv, lrphase_srlr.tsv
jupyter nbconvert --execute --inplace LES_analysis.ipynb
```


## Correctness prerequisite: three bugs had to be fixed first

Before these fixes the model produced energy-force **inconsistent** results, and any performance comparison was meaningless.
All three are in `hybridles_model.py`:

1. `coord.requires_grad_(True)` must be set **before** the descriptor is recomputed, otherwise the descriptor branch is treated as a constant and the long-range force loses its charge-response term.
2. The long-range force must be `autograd.grad(E_lr_total.sum(), coord)` with respect to the **full** `coord`, not the per-frame slice `coord[i]`, otherwise the descriptor-response term `dE_LR/dq * dq/ddesc * ddesc/dr` is silently dropped.
3. The long-range energy must be restored to rank `[nframes, 1]` before it is added to the short-range energy.
`Ewald` returns one scalar per frame, so the vectorized (formerly per-frame) call yields a 1-D `[nframes]` tensor, while the parent's `energy_redu` is `[nframes, 1]`.
Adding those two broadcasts to an **outer sum** `energy[i, j] = energy_redu[i] + E_LR[j]`.
Training never noticed, because `batch_size: auto` resolves to a single frame and a 1x1 matrix is accidentally correct; evaluation at batch 5 produced a `[5, 5]` energy and failed with `shape '[5, 1, 1]' is invalid for input of size 25`.
Fixed with an explicit `reshape(nframes, 1)`, which is value-preserving.
That the fix is semantically inert was verified two ways: the regenerated `decay_steps 5000` sweep reproduces the 2026-09-12 sweep bit-exactly (`max|diff| = 0.0` on both metrics across all 26 comparable rows), and the `dp freeze` output matches the eager model on a 4-frame batch (energy `0.0`, force `1.1e-13`, virial `2.7e-12`).

After all three fixes all three arms are energy-force consistent to the finite-difference noise floor, verified by `01.train/rerun/check2.py`.
Max `|F_autograd - F_finite_difference|` is 1.9e-8 (ordinary), 2.0e-8 (hybrid_q) and 7.5e-9 (hybrid_fixed), which is 3e-8 or less relative to the largest force component.

The pre-fix symptom was a 10x energy error.
The historical pre-fix hybrid checkpoint (`01.train/lcurve_q.out`, 10k steps) had `rmse_e_val = 1.06e-2` versus `1.07e-3` for the ordinary model, which is the signature of that inconsistency.

## Sanity checks

- The ordinary rerun reproduces the historical `01.train/lcurve_nonles.out` step-10000 row digit for digit (`rmse_e_val 1.07e-3`, `rmse_f_val 8.33e-2`), confirming the harness and data pipeline are sound.
- Batched and per-frame evaluation agree to 1e-12 (`diag_frames.py`), so there is no cross-frame leak in the LES/Ewald path and the full-set numbers are trustworthy.
- The saved checkpoints carry the LES weights (all 8 keys) and their norms match the final `les.log` block, so the evaluated models are the trained ones.
- Full-set results on the train split independently agree with the validation split, so the comparison is not an artifact of the validation set.
- Historical baseline at 10k steps (see table above for the post-fix comparison):

| run | rmse_e_val | rmse_f_val | note |
|---|---|---|---|
| `lcurve_nonles.out` (ordinary) | 1.07e-3 | 8.33e-2 | matches rerun |
| `lcurve_q.out` (hybrid, pre-fix) | 1.06e-2 | 8.08e-2 | energy 10x wrong, physically invalid |

## Caveats

- One dataset only: small periodic water, 400 frames, 64 molecules per frame. Conclusions may not transfer to other chemistries or sizes.
- Two seeds and three late checkpoints per run. That is enough to see the checkpoint-to-checkpoint swing but not enough to put a tight error bar on the force gain.
- Schedule dependence. Every headline number here uses `decay_steps 5000`. The force gain does not survive shortening the post-cliff anneal: see "Is the latent charge underfitted?".
- No converged model. Neither the `decay_steps 5000` runs nor the probe's `7500` runs reach a fixed point, so the late checkpoints are snapshots rather than converged values.
- The energy ranking is unresolved rather than resolved as a tie. A larger validation set or more checkpoints might settle it; this one cannot.
- Absolute differences are small. The fixed-charge force gain is 5.6e-3 to 7.4e-3 eV/A in absolute terms, which matters on this clean dataset but may be dwarfed by other errors at production scale.
- Accuracy only. See the open gaps below before treating this as deployable.

## Still-open gaps (independent of accuracy)

Two of the three gaps listed in earlier revisions of this document are now closed, and the third is partly closed.
Each claim below was re-verified against the current tree rather than carried over.

- **Closed: `serialize()` / `deserialize()`.** It now carries `les_params` and all 8 LES tensors, and the round trip reproduces every output bit-for-bit (`01.train/rerun/check_serialize.py`: `max|diff| = 0.0` on energy, force, virial, atom_energy and atom_virial).
- **Closed: the virial.** The long-range (Ewald) part is included, both the atomic term and the explicit cell term. On a sheared cell the full virial matches finite differences to `1.4e-05`, whereas the short-range part alone is off by `1.5e+02` (`01.train/rerun/check_virial.py`), so `pref_v > 0` training and NPT/stress evaluation are consistent and the one-time warning is gone.
- **Partly closed: scripting and freezing.** `torch.jit.script` and the real `dp --pt freeze` CLI both succeed, and the frozen model agrees with the eager model on a 4-frame batch (energy `0.0`, force `1.1e-13`, virial `2.7e-12`). But loading an *unfrozen* `hybrid_ener` `.pt` with TorchScript enabled still fails: `element_numbers` is registered `persistent=False` (it is derivable from `type_map`), and TorchScript discards that flag, so the scripted model expects a key the checkpoint does not contain - `Missing key(s) in state_dict: model.Default.atomic_model.element_numbers`. `dp --pt test` takes that path and so cannot be used on these checkpoints. The two workarounds that do work are `no_jit=True` (what the custom eval harness uses) and freezing first, then loading the `.pth`.

## Reproduce

```bash
cd 01.train/rerun
python gen_extended.py          # writes extended/run_*/input.yaml
bash run_extended.sh            # runs all 6 arms sequentially (~2h on one GPU)

# exact metrics: one (run, step) per process; see the note in sweep_valid.sh
bash sweep_valid.sh             # writes sweep_valid.tsv (steps 6000-10000, 6 runs)
python diag_frames.py run_ordinary_sB   # per-frame distribution and sampling check
python diag_tail.py run_hybrid_fixed_sB # logged curve vs each checkpoint's own distribution

# per-frame bias/spread; `valid` for all six arms, then `train` for the lockstep proof
for r in run_ordinary_sA run_hybrid_q_sA run_hybrid_fixed_sA \
         run_ordinary_sB run_hybrid_q_sB run_hybrid_fixed_sB; do
  python diag_bias.py valid $r     # writes bias_frames_valid.tsv
done
python diag_bias.py train run_hybrid_fixed_sB   # writes bias_frames_train.tsv
python diag_bias.py train run_ordinary_sB

jupyter nbconvert --execute --inplace LES_analysis.ipynb   # figures
```

Environment note: on this host (single 8 GB GPU under WSL2) the CUDA caching allocator must be told to release memory or training deadlocks partway through.
`run_extended.sh` exports `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for this reason.
Do not set that flag for the small evaluation runs, and load only one model per process: both are implicated in a `CUDACachingAllocator` internal assert on this host.
