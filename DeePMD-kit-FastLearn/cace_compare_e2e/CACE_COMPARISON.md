# deepmd `hybrid_ener` (LES) vs cace on the author's 64-H2O benchmark

Nine arms on one held-out split, so a difference between two rows is a difference in the model, not in the data or the metric.

**Bottom line.** Three findings, and the newest one qualifies the other two.

Among the four deepmd arms, a long-range term helps only when the charges are a physical assignment rather than learned.
The fixed SPC/E table gives 3.03% better force than the plain short-range model at *zero* extra parameters; the learned charge layer instead turns the long-range term into a co-equal cancelling channel that buys 0.6% of energy at the cost of 4.9% of force and 18% more parameters.

The author's **current** cace model is the best of all nine arms: **1.3721e-03 eV/atom and 6.2700e-02 eV/A**, 4.8x and 2.3x better than plain `sr`, with 34% fewer parameters.
Its long-range term has the signature of a *physical* correction (small, positively correlated with the short-range part) rather than of the cancelling channel the deepmd learned layer builds, and that measured difference is what separates the two learned charge layers.
But the author also publishes a short-range run of the same representation, and it scores 1.5327e-03 and 7.2147e-02 - so most of that 4.8x margin is the cace representation and training loop, not the long-range term.
Scored against that matched control the Ewald term itself is worth **10.5% of energy and 13.1% of force**, and it is the force gain that behaves like physics: it is present on 157 of 159 frames and it scales with the long-range force the model predicts (r = +0.55), while frame difficulty explains none of it (r = +0.02).

The pre-update cace checkpoint this document originally scored is a third thing again: inert, because its long-range weight is 0.01.
Details, and the caveats that bound all of this, are below.

The short-range network is identical across the four deepmd arms (same descriptor `sel [39,73]`, `rcut 5.5`, same fitting net `[64,64,64]`), so the only thing that varies is the long-range treatment and, for the learned arms, the charge network that produces q. The parameter counts are set out below, because they decide which comparisons are clean.

This was checked by diffing the four `input.json` files rather than assumed: `model.descriptor`, `model.fitting_net`, `model.type_map`, `learning_rate` (`exp`, `decay_steps 100000`, `1e-3` to `3.51e-8`), `loss` (`ener`, `pref_e 1.0`, `pref_f 1000`, `pref_v 0`), `numb_steps 100000`, `seed 10`, `disp_freq 1000`, `save_freq 5000`, train/valid batch size 4, and the train/valid system paths are **all byte-identical** across the four.
The only differences are `model.type` and `model.les_params`.
`sr` has no `les_params` at all (plain `ener` model, no LES module).

## The arms

| arm | code | charge layer | long-range weight |
|---|---|---|---|
| `sr` | deepmd | none (no LES at all) | - |
| `hyb_lrw1` | deepmd | `local_charge`: q = NN(descriptor) | 1.0 |
| `hyb_lrw001` | deepmd | `local_charge`: q = NN(descriptor) | 0.01 |
| `freeze_spce` | deepmd | `freeze_charge`: fixed SPC/E table | 1.0 |
| `cace_lrnew` | cace | author's learned q (current script) | 1.0 |
| `cace_timing_sr` | cace | none - the author's own short-range run of the same representation | - |
| `cace_lrnew_sr` | cace | same model, long-range term removed | - |
| `cace` | cace | author's learned q (pre-update script) | 0.01 |
| `cace_sr` | cace | same model, long-range term removed | - |

`hyb_lrw1` vs `freeze_spce` is the charge-layer benchmark this document is built around: both run the long-range term at full weight 1.0, so the only difference is whether q comes from a learned function of the local descriptor or from a fixed per-type table.

`freeze_spce` is the classical Ewald end of that axis, so it is the honest floor a learned charge layer has to beat; `hyb_lrw001` is the arm matched to the weight in the pre-update cace script, not that floor.

`cace_timing_sr` vs `cace_lrnew` is the cace equivalent of that same benchmark and it is the newest arm here: one published training run with the Ewald head, one without, everything else held fixed.
It matters because the two cace `_sr` rows are *not* short-range models - they are long-range checkpoints whose Ewald term has been subtracted, so their short-range half was only ever trained with the long-range term present - while `cace_timing_sr` is a model that had to fit the whole target itself.
The numbers below show those two controls are not interchangeable: the analytic subtraction leaves 1.0479e-02 eV/atom, the trained short-range model reaches 1.5327e-03, a 6.8x difference.

### The upstream repository has since been revised, and it changes what "the author's weight" means

`cace-lr-fit`'s root README describes an optimization of the LES path: the short-range and long-range energies are summed by a `FeatureAdd` module first, and a single autograd then produces the total force, whereas the earlier implementation ran two autograds (short-range and long-range forces separately) and added the two forces.
The claim is purely one of cost.
It is visible in the scripts: `fit-water/fit-water-mp0/fit-cace-nnp.py` combines two `NeuralNetworkPotential`s through `CombinePotential` with a `pot2` weight of `0.01`, while `fit-water-timing/fit-water-mp0-lr-FeatureAdd/fit_cace_new.py` builds one potential whose modules end in `e_add = FeatureAdd(['SR_energy', 'ewald_potential'], 'CACE_energy')`, and `FeatureAdd` sums its inputs at unit weight (`cace/modules/feature_mix.py`).
So the same revision that removes one autograd also raises the long-range weight from 0.01 to 1.0, a factor 100 in the term's contribution, and the README does not mention that part.
The two changes are not independent: with the old two-potential structure the long-range contribution was an explicitly weighted addend, and folding it into the energy sum at unit weight is what makes the single-autograd form work.

The same diff carries a third change that is not implied by either of those: the charge network's `n_out` goes 4 -> 1.
Everything else that could move the numbers is held fixed - the short-range network is untouched (`n_layers=3`, `n_hidden=[32,16]`, `add_linear_nn=True`), both scripts train the identical 40+100+100+100 epochs on the same 1593-frame dataset with the same `valid_fraction 0.1 / seed 1` split, and trainable parameters move only from 43,052 to 41,072.
So the pre-update vs updated comparison below is budget-matched, but it does change two things at once (the long-range weight and the charge channel count), and this document does not try to separate them.

The `cace` row scored in this document is the **pre-update** model: `cace/best_model.pth` is byte-identical to `fit-water/fit-water-mp0/best_model.pth` (sha256 `654f3456a821f6d5`, 198498 bytes), i.e. the `n_out=4`, weight-0.01, two-autograd script.
The `cace_lrnew` rows are the updated checkpoint, `fit-water-timing/fit-water-mp0-lr-FeatureAdd/best_model.pth` (sha256 `2345d4ad622c98a5`, 191394 bytes), scored as published.
Against the revision, `hyb_lrw1` (weight 1.0) is the arm that matches the author's current script and `hyb_lrw001` (weight 0.01) matches the script the pre-update checkpoint came from, so the weight-0.01 arm on the deepmd side is a match to a superseded configuration.

Scoring the updated checkpoint does require the branch it was built on, and this is not a matter of taste.
Its pickled module state stores the angular basis dictionary with TorchScript-safe string keys (`'0_0_0'` rather than `(0, 0, 0)`), and cace `main`'s `cace/modules/angular.py` indexes that dictionary with integer arithmetic, so under `main` the checkpoint loads and then dies in the first forward pass with `TypeError: unsupported operand type(s) for -: 'str' and 'int'`.
Each checkpoint was therefore run against the branch it was built on: `cace` under `main` (the default import, `/root/app/cace` at commit `3f1a66552684`) and `cace_lrnew` under `--cace-root /root/app/cace-ts`, a worktree of `origin/torchscript` (commit `6256b8e1f586`).
`compare/record_cace_valid.py` records which checkout ran which checkpoint in its JSON, and refuses to write a metric at all unless the loader's split matches `water/valid` frame-for-frame.
Every checkpoint in `fit-water-timing/` is a `torchscript`-branch artefact, the short-range control included: `fit-water-mp0-sr/best_model.pth` fails under `main` with the identical `TypeError`, so `cace_timing_sr` was scored with `--cace-root /root/app/cace-ts` as well.
This is a property of the checkpoints, verified by running them, and not something the repository's README states: the `cace-lr-fit` README carries only the cost claim about `FeatureAdd`, and the directory holds no README of its own.

### Mapping onto the four requested arms

The requested `deepmd` / `deepmd-les` / `cace` / `cace-lr` is a 2x2: each code contributes one arm with no long-range term and one with it.

| requested arm | arm here | long-range term |
|---|---|---|
| `deepmd` | `sr` | none |
| `deepmd-les` | `hyb_lrw1` | learned local charge, weight 1.0 |
| `cace` | `cace_timing_sr` | none (the author's own short-range run of the same representation) |
| `cace-lr` | `cace_lrnew` | author's learned q, weight 1.0 (current script) |

So the per-code long-range gain is `sr -> hyb_lrw1` for deepmd and `cace_timing_sr -> cace_lrnew` for cace, and those two gains are directly comparable because both are "same code, long-range term on vs off" against a model that had to fit the target without it.
An earlier version of this document used `cace_lrnew_sr -> cace_lrnew` for the cace row, which is a different and much weaker control: subtracting the Ewald term from a checkpoint whose short-range half was only ever trained alongside it asks how much of that prediction is the Ewald term, not what a short-range model reaches.
Both numbers are reported below, and they differ by 6.8x on energy.
The `cace` / `cace_sr` rows are kept as well, because they are what this document was originally built on and because the pre-update vs updated difference turns out to be the most informative single comparison in it.

## Protocol

- Held-out split: the 159 frames of `cace_compare_e2e/water/valid`, 192 atoms each, scored in full. No subsampling.
- Energy target is the residual (total minus the `atomic_ref.json` per-element reference), the same quantity both codes were trained on.
- Metrics are per-atom energy RMSE/MAE and force RMSE/MAE over all components, plus R^2 against the target's own variance (so 0 is "always predict the mean", 1 is perfect).
- Each deepmd arm is scored on its **three latest checkpoints** and the mean reported. The five cace rows come from three fixed files - the author's published `best_model.pth` (`cace` / `cace_sr`), the updated FeatureAdd checkpoint (`cace_lrnew` / `cace_lrnew_sr`) and the short-range run published next to it (`cace_timing_sr`) - so `--late` does not apply to them either.

Three checkpoints rather than one: on this run the late-checkpoint spread happens to be small (the per-atom energy RMSE of the three moves within about 1.2%, and force within 0.02%), but on earlier runs in this project a single checkpoint's exact energy RMSE moved by 1.1x to 2.3x between adjacent late checkpoints.
One checkpoint is therefore not a safe basis for a cross-arm ranking, the mean of three is cheap insurance, and `compare/metrics.py --late 3` prints the per-checkpoint values next to the mean so the reader can see the spread rather than take it on trust.

The logged `rmse_*_val` in `lcurve.out` is not used anywhere here: it is a 3-frame draw out of 159, so it is noisy and consecutive points come from different checkpoints.

Harness: `compare/metrics.py` (metrics), `compare/diag_les.py` (long-range magnitude and charge statistics), `compare/probe_fixed_table.py` (standalone kernel check of the fixed tables), `compare/record_cace_valid.py` (the cace rows plus their mechanism columns, with the checkpoint hash and cace-branch provenance recorded alongside), `compare/probe_lr_control.py` (the matched short-range control for the cace long-range term, from the two recorded predictions), `compare/probe_lr_charge.py` (what the learned charge encodes - the protonation ladder and the kernel amplitude test).

## Headline metrics

Per-atom energy RMSE is in eV/atom, force RMSE in eV/A.

| arm | E/atom RMSE | E/atom MAE | E R2 | F RMSE | F MAE | F R2 |
|---|---|---|---|---|---|---|
| `cace_lrnew` | **1.3721e-03** | **7.0967e-04** | **0.99966** | **6.2700e-02** | **3.3592e-02** | **0.99912** |
| `cace_timing_sr` | 1.5327e-03 | 8.9356e-04 | 0.99957 | 7.2147e-02 | 4.1869e-02 | 0.99883 |
| `cace_lrnew_sr` | 1.0479e-02 | 7.5095e-03 | 0.98011 | 1.1952e-01 | 6.5320e-02 | 0.99679 |
| `sr` | 6.6349e-03 | 4.8456e-03 | 0.99203 | 1.4555e-01 | 9.1273e-02 | 0.99524 |
| `hyb_lrw1` | 6.5884e-03 | 4.6423e-03 | 0.99214 | 1.4804e-01 | 9.2586e-02 | 0.99508 |
| `hyb_lrw001` | 6.0797e-03 | 4.7000e-03 | 0.99331 | 1.4652e-01 | 9.2171e-02 | 0.99518 |
| `freeze_spce` | 6.6280e-03 | 4.9547e-03 | 0.99204 | 1.4114e-01 | 8.9351e-02 | 0.99553 |
| `cace` | 4.3971e-02 | 4.1771e-02 | 0.64984 | 1.6567e-01 | 1.0904e-01 | 0.99384 |
| `cace_sr` | 4.3975e-02 | 4.1775e-02 | 0.64977 | 1.6569e-01 | 1.0905e-01 | 0.99384 |

Reference scale: the target's own per-atom energy spread is 7.431e-02 eV/atom, so an energy R^2 near 1 means the arm is tracking the frame-to-frame variation, not just the mean.

### What the numbers say

Long-range gain: the same short-range model with the long-range term on vs off (negative = the long-range term helped).

| comparison | energy | force |
|---|---|---|
| deepmd, `sr` -> `freeze_spce` (fixed table, weight 1.0) | -0.10% | **-3.03%** |
| deepmd, `sr` -> `hyb_lrw1` (learned q, weight 1.0) | -0.70% | **+1.71%** |
| deepmd, `sr` -> `hyb_lrw001` (learned q, weight 0.01) | **-8.37%** | +0.67% |
| cace, `cace_timing_sr` -> `cace_lrnew` (learned q, weight 1.0, matched short-range control) | **-10.48%** | **-13.09%** |
| cace, `cace_lrnew_sr` -> `cace_lrnew` (the same checkpoint with its Ewald term subtracted) | **-86.91%** | **-47.54%** |
| cace, `cace_sr` -> `cace` (learned q, weight 0.01, pre-update script) | -0.009% | -0.012% |

The first three rows all use the identical `sr` baseline, so they are directly comparable to one another.
The two `cace_lrnew` rows are the **same checkpoint** scored against two different controls, and the 8.3x gap between them on energy (86.91% against 10.48%) is exactly the difference between asking "how much of this prediction is the Ewald term" and asking "how much accuracy does the Ewald term add".
The article's claim is the second question, and only the matched control answers it.

Five things follow.
Items 1-3 are about the four deepmd arms and the pre-update cace checkpoint; item 4 is the author's current cace model, and item 5 is what the matched control does to item 4.

1. **The long-range term helps, but only when q is a physical charge assignment - and when it does, the benefit shows up in force, not energy.** `freeze_spce` is 3.03% better than `sr` on force at *zero* extra parameters, while both learned arms are worse than `sr` (1.71% and 0.67%). The force spread across each arm's three checkpoints is under 0.02% (freeze_spce's is 0.007%), so every one of these margins is far above the floor. Energy is the mirror image, and there the means are not enough on their own - read the ranges:
   - `freeze_spce` moves energy by 0.10%, but `sr` spans 6.5957 to 6.6615e-03 and `freeze_spce` spans 6.6108 to 6.6408e-03, so the ranges overlap and that 0.10% is **not established**.
   - the weight-0.01 learned arm's 8.37% energy gain *is* established (6.0506 to 6.1202e-03, disjoint from `sr`'s range), while `hyb_lrw1`'s 0.70% gain is not (6.5795 to 6.6034e-03, overlapping `sr`'s).
   - so the pattern is: a fixed physical charge adds force information the short-range net cannot represent, whereas a learned charge layer instead spends its freedom reallocating energy (see the magnitude finding below), which buys a little energy and costs a little force.
2. **In the author's pre-update cace model the long-range term changes nothing measurable** (0.009% on energy, 0.012% on force), so `cace` and `cace_sr` are the same model for practical purposes.
   Two independent factors multiply, and `cace/inspect_cace_les.py` measures both: the Ewald head's own output is tiny (per frame mean +0.081 eV, std 0.0264 eV over the split), and the combine weight 0.01 shrinks it by a further 100x.
   Against a target whose own per-frame spread is 8.44 eV rms, the raw term spans 0.31% of the range that would have to be reproduced and the weighted term 0.003%; on force the weighted term is 1.6e-05 eV/A against a 2.11 eV/A target, a ratio of 7.8e-06.
   The four learned charge channels are a learned feature vector rather than one charge per atom - each channel is near-zero-mean on its own (-0.013 to -0.043) with std up to 3.24 and max |q| 6.3 - but the learned q is not what makes the term inert: the per-element average over the four channels (O +1.58, H -0.83) is SPC/E-shaped in ratio, and the kernel's own scale is the main effect (the caveat below drives it with the physical SPC/E table on these frames and gets 0.0340 eV/frame against this checkpoint's 0.0264).
   So the -42 meV/atom offset sits entirely in the short-range network, and the pre-update cace checkpoint enters this comparison as a short-range model wearing a decorative Ewald head.
3. **On force the deepmd arms beat the pre-update cace model by ~14%; on energy the 6.63x must not be read at face value.** `sr`'s force RMSE is 12.1% lower than cace's (equivalently, cace's force is 13.8% higher than `sr`'s), and that one is a straight comparison.
   The energy row is not: 43.97 meV/atom against `sr`'s 6.63 is dominated by a **constant -42 meV/atom offset** in the cace prediction, not by the model scattering.
   Its Pearson r is 0.989 (r2 0.979) and the scatter sits below the 1:1 line; removing the offset leaves **13.73 meV/atom**, and a full linear calibration leaves 9.49 - about 2x the `sr` arm rather than 6.6x.
   `compare/probe_cace_offset.py` records that decomposition, and it also shows the offset belongs to the model rather than to the harness: the same forward pass over the frames the model was *trained* on gives the same picture (-47 meV/atom of bias, 15.6 meV/atom after removing it), so train and valid are indistinguishable and the checkpoint is under-fitting rather than over-fitting.
   A reference-convention mismatch is ruled out too - cace's loader and `prep_water.py` subtract identical reference energies.
   Read this as a statement about these two trained models, not about the two codes: deepmd `sr` has 62,434 trainable parameters against the pre-update cace checkpoint's 43,052, the architectures differ, so the comparison is not capacity-matched, and that cace checkpoint is a small published demo that under-fits its own training split (see caveats).
4. **The author's current cace model is the best arm here by a wide margin, and it is a learned-charge arm that works.** `cace_lrnew` scores **1.3721e-03 eV/atom and 6.2700e-02 eV/A** - 4.8x better than `sr` on energy and 2.3x better on force - and its Ewald term accounts for most of its own prediction: dropping it costs 86.91% of the energy error and 47.54% of the force error, against 0.009% and 0.012% for the pre-update pair in item 2.
   Item 5 gives the version of that number that answers "how much does the long-range term add", which is smaller.
   Its calibration is essentially exact (bias +0.06 meV/atom and slope 1.0016 against the target's own per-atom energy, where the pre-update model sits at -41.77 meV/atom and 0.866), and it achieves this with **41,072** trainable parameters, *fewer* than plain `sr`'s 62,434.
   So the "learned charges lose" conclusion above is a statement about the deepmd learned layer at this schedule and about the pre-update cace checkpoint, and **not** about learned charges in general.
   The mechanism is measurable and it is in the Task 1 table below: `cace_lrnew`'s long-range term is small (0.098x the target's frame-to-frame spread), positively correlated with the short-range part (+0.60), and carries 5.3% of the total force - the signature of the physical `freeze_spce` arm (0.209x, +0.54, 12.8%), not of the deepmd learned arm (1.035x, -0.61, 101%).
   Two learned charge layers, therefore, produced structurally opposite long-range terms here, and the one that behaves like a physical correction is the one that wins.
5. **The matched short-range control cuts the cace long-range gain down to size - and confirms that the term is doing physics, on force.** `cace_timing_sr` is the author's own short-range run of the same representation on the same 40+100+100+100 schedule with the Ewald head absent, so it had to fit the whole target itself; it reaches **1.5327e-03 eV/atom and 7.2147e-02 eV/A**, which is both the correct control and a far stronger one than the analytic subtraction (which leaves 1.0479e-02 / 1.1952e-01, 6.8x worse on energy).
   Against it the Ewald term is worth **10.48% of the energy error and 13.09% of the force error** (1.117x / 1.151x) - real, and eight times smaller than the 86.91% / 47.54% the subtraction implied.
   Two readings say the force part of that is physics rather than capacity.
   It is not a uniform improvement: the long-range model is better on **157 of 159 frames** for force, which is what a correction present in every frame looks like, but on only 91 of 159 for energy, which is what a weak effect looks like.
   And the per-frame force gain scales with the long-range force the model itself predicts - `corr(gain, max|F_lr|) = +0.547`, `corr(gain, rms|F_lr|) = +0.552` - while frame difficulty explains none of it (`corr(gain, max|F_ref|) = +0.016`).
   A larger network helps the hard frames; this helps exactly the frames where there is more long-range force to correct, which is what the term is for.
   So the implementation does work as the article claims, with two honest limits: the effect on this dataset is 1.12x / 1.15x rather than an order of magnitude, and it costs 67% more parameters (41,072 against 24,572); and on energy the gain is not separable from frame difficulty (`corr(gain, |E_lr|) = +0.202` against `corr(gain, |E_ref|) = +0.264`), so the energy half of the claim is not established here.
   `compare/probe_lr_control.py` records all of it.

**The parameter counts decide which of those gains are clean, so they are spelled out here.** All four deepmd arms share the same descriptor and fitting net; the learned arms additionally carry the Atomwise charge network:

| arm | trainable params | vs `sr` |
|---|---|---|
| `sr` | 62,434 | - |
| `freeze_spce` | 62,434 | **0** (exactly matched) |
| `hyb_lrw1` | 73,572 | +11,138 |
| `hyb_lrw001` | 73,572 | +11,138 |
| `cace_lrnew` | 41,072 | -21,362 |
| `cace_timing_sr` | 24,572 | -37,862 |
| `cace` | 43,052 | -19,382 |

The cace counts are for the whole model (representation plus its output networks). The pre-update to updated change is -1,980, which is the charge network's last layer going from four channels to one, and `cace_timing_sr` is the same representation with neither the charge network nor the Ewald head.

So:
- `sr` vs `freeze_spce` is **exactly** capacity-matched and differs only by a fixed physical long-range term. This is the only clean test here of whether the long-range interaction itself buys accuracy, and it answers yes - on force, by 3.03% - while leaving energy unchanged. That is why the `freeze_spce` row matters well beyond task 3.
- `sr` vs either learned arm is **not** matched: the learned arms have 18% more parameters, which is precisely the charge network. So their energy gain over `sr` cannot be attributed to long-range physics rather than to that added capacity on the evidence here - and indeed only the weight-0.01 arm shows an energy gain that clears the checkpoint spread, while both learned arms are *worse* than `sr` on force.
- `hyb_lrw1` vs `hyb_lrw001` **is** exactly matched (identical parameter counts, only `lr_weight` differs), so the weight comparison is clean.
- `cace_lrnew` vs `sr` is **not** matched either, but in the opposite direction: the cace model has 34% *fewer* parameters and is still 4.8x / 2.3x better, so its margin cannot be attributed to capacity. It is a cross-code comparison though - a different descriptor, a different short-range network, a different training loop - so it bounds what `sr` achieved here rather than isolating a single cause. `cace_timing_sr` makes that sharper: it is 4.3x / 2.0x better than `sr` with 61% fewer parameters still, so most of the cross-code margin is the cace representation and training loop rather than the long-range term (item 5).
- `cace_timing_sr` vs `cace_lrnew` is **not** matched, and unlike every other learned-charge pair here the extra capacity comes with an accuracy gain rather than a force loss: the charge network and Ewald head add 16,500 parameters (67% more) and buy 10.48% / 13.09%. So that gain bundles the long-range term with the added capacity and cannot be attributed to the physics alone on this evidence - but the per-frame scaling test in item 5 is what argues it is not merely capacity.

Read together with the magnitude finding below, the pattern within the four deepmd arms is coherent and it is what the exactly-matched pair measures: a fixed physical charge assignment adds a real interaction, and a real interaction shows up as better forces (3.03% here) at zero parameter cost. The learned charge layer in *this* deepmd model instead acts as a second learned energy channel - the two branches become large cancelling halves rather than one correcting the other (below) - and a channel that reallocates energy rather than adding an interaction buys a little energy (0.60% over the fixed table, established) while giving back force (4.89% worse than the fixed table, established).

So the answer to "learned or fixed charges" depends on which learned charge layer, and the two implementations here land on opposite sides:

- **Within our deepmd implementation, fixed charges win on force outright**, and the learned layer's small energy edge comes at a force cost and 18% more parameters.
- **Across codes, the author's current learned cace layer beats every fixed-charge arm here** - including `freeze_spce` - by 79% on energy and 56% on force (1.3721e-03 vs 6.6280e-03 eV/atom, 6.2700e-02 vs 1.4114e-01 eV/A), with 34% fewer parameters than `sr`. Most of that margin is the cace representation rather than the long-range term: the author's own short-range run of it is already 77% / 49% better than `freeze_spce` (1.5327e-03 / 7.2147e-02) at 61% fewer parameters than `sr`, and the Ewald head supplies the remaining 10.48% / 13.09% (item 5). Since both cace arms and the deepmd learned arms are all "a learned q feeding an Ewald sum at weight 1.0", the difference is not learned-vs-fixed at all; it is *what the learned q does to the long-range term*, which item 4 above and the Task 1 table below measure directly.

## Task 1: the reasonable magnitude of E_lr and F_lr

`compare/diag_les.py` reads E_lr and q by hooking the `Les` module inside the real forward, so the numbers come from the path training uses.
It also re-derives E_lr independently as `E_tot - E_sr` (forwarding the same frame with the weight forced to 0) and the two agree to **0.000e+00** on every arm, so the decomposition below is not an artifact of the hook.

Target std is 14.267 eV/frame.

| arm | E_lr mean | E_lr std | E_lr std / target std | E_sr std / target std | corr(E_sr, E_lr) | F_lr RMS / F_tot RMS |
|---|---|---|---|---|---|---|
| `freeze_spce` (physical) | -381.53 | 2.98 | 0.209 | 0.838 | **+0.539** | 0.128 |
| `hyb_lrw1` (weight 1.0) | -517.44 | 14.76 | 1.035 | 1.146 | **-0.608** | 1.013 |
| `hyb_lrw001` (weight 0.01) | -141.80 | 8.50 | 0.596 | 0.962 | **-0.305** | 1.066 |
| `cace_lrnew` (learned q, weight 1.0) | +1.453 * | 1.403 * | 0.098 | 0.940 | **+0.599** | 0.053 |
| `cace` (learned q, weight 0.01) | +0.001 * | 0.000 * | 0.000 | 0.876 | **+0.619** | 0.000 |

\* The two cace rows are produced by the author's Ewald kernel, which runs at `norm_factor = 1` (a convention the source documents as leaving charges scaled by `sqrt(90.0474)`), so the absolute level and std of their `E_lr` are on a different scale from the les rows above and must not be compared to them.
Their `E_lr mean` / `E_lr std` cells are the weighted values, as in the les rows above (weight 1.0 and 0.01 respectively; the unweighted checkpoint values are +1.4531 / 1.4030 for `cace_lrnew` and +0.0810 / 0.0264 for `cace`).
The four ratio columns are scale-free and are the ones to compare; the cace numbers come from `compare/record_cace_valid.py`'s `mechanism` block, which the metrics JSON carries.

`cace_timing_sr` does not appear in this table because it has no long-range term at all - `E_lr` is identically zero for it by construction, which is exactly what makes it the control in item 5.

**The physically motivated answer.** The frozen SPC/E table, run at full weight, gives a long-range term whose frame-to-frame variation is **21% of the target's**, and a long-range force that is **13% of the total force**.
It correlates *positively* with the short-range part (+0.54): the two channels reinforce each other, which is what a long-range correction to a short-range model should look like.
This is the number to use if the question is "how big should the long-range term be": **E_lr std around 0.2x the target std, F_lr around 0.13x F_tot**, at a physical weight of 1.0.

**What the learned charge layer does instead.** `hyb_lrw1` drives E_lr to **1.03x the target std** with the short-range part at **1.15x**, and the two are **anti-correlated at -0.61** with means that nearly cancel (+514.75 and -517.44).
So it is not adding a small long-range correction to a short-range model.
It is splitting one energy into two large opposing halves.
Its F_lr RMS is 101% of F_tot, i.e. the long-range force is as large as the total force rather than 13% of it.

At weight 0.01 (`hyb_lrw001`) the same thing happens one level down: the optimizer compensates for the 100x smaller weight by **growing the charges** (O +4.90, H -2.78, max |q| 7.59, against +0.97 / -0.50 in `hyb_lrw1`), and the weighted long-range term stays co-equal (E_lr std 0.60x target, E_sr 0.96x, still anti-correlated at -0.31, F_lr/F_tot 1.07).

So the learned arms' E_lr magnitude is roughly **5x the physical one**, and their charges are a fitting device rather than physical charges.
Whether that costs accuracy is a separate question, and it is the metrics table above that answers it, not the magnitude - a large long-range term is not by itself a worse fit.
The answer turns out to be: it costs *force* accuracy (both learned arms are worse than plain `sr` on force, and worse than the fixed table by 4.9% at weight 1.0) while buying a little *energy* accuracy, which is exactly what a reallocation rather than an added interaction would do.

**Where the author's current cace layer lands on the same columns.** `cace_lrnew` puts E_lr at **0.098x the target spread** with the short-range part at 0.940x, positively correlated with it at **+0.599**, and a long-range force at **5.3% of the total**.
That is the `freeze_spce` signature (0.209, 0.838, +0.539, 0.128), scaled down - a modest positive correction - and it is nothing like the deepmd learned arms (1.035, 1.146, -0.608, 1.013).
So the same column that shows the deepmd learned `q` behaving as a fitting coordinate shows this cace learned `q` behaving as a physical correction, and it is the correction-shaped one that wins the accuracy comparison.
Its charges match that reading: one channel, O +0.2436 / H -0.2760 - small, opposite-signed and near-equal in magnitude, which is the right shape for water, unlike `hyb_lrw1`'s O +0.97 / H -0.50 at 5x the scale.
The sign is inverted relative to SPC/E's O -0.85 / H +0.42, i.e. the learned assignment is the global negation of the physical one, and the Ewald energy is bilinear in the charges (`sum_ij q_i q_j phi_ij`), so a global sign flip is exactly unobservable in energy and forces.
So the two are the same kind of assignment up to a gauge that cannot matter.
The magnitudes differ: `cace_lrnew`'s O and H are nearly equal (0.244 against 0.276) where SPC/E's O is twice its H, so it is a weaker and more balanced split rather than a copy of the force field.

**What that single channel actually encodes: a protonation coordinate.**
All of this section is reproducible from `compare/probe_lr_charge.py` (E2E: `--with-kernel --cace-root /root/app/cace-ts`); the numbers below are its output.
The per-element means above hide a wide tail, and the tail is not the same element flickering - it is a different species.
96.0% of the O atom-frames in the valid split are ordinary water (two H within 1.3 A, no close O-O) and their charge is tight and single-signed: mean **+0.220 +- 0.349**, p1/p99 -0.60/+1.13.
Every value in the min/max sheet comes from the remaining 4%, and there `q` is strictly monotone in how many protons the oxygen carries:

| protons on the O | n | mean q |
|---|---|---|
| 0 H | 7 | **+3.34** |
| 1 H | 270 | **+1.96** |
| 2 H | 9774 | **+0.22** |
| 3 H | 125 | **-1.75** |

That is about 1.7 charge units per proton, i.e. the protonation ladder bare O / OH- / H2O / H3O+, and it is robust: the ordering holds at every H-cutoff tried (1.2, 1.3, 1.4, 1.5 A) and a cutoff-free smooth O-H coordination (r0 1.4 A, width 0.05-0.10 A) gives `corr(q, coordination) = -0.73` with binned means +4.00 / +2.27 / +0.23 / -1.47.
The H side agrees: H bonded to one O averages -0.270 (n=20149), one bridging two O +0.027 (n=22).
The two extremes of the whole dataset are one such pair in a single frame: an O with a genuine 1.61 A O-O contact at -4.97, beside an O whose nearest H is 1.63 A and which has no bonded H at 1.3 A, at +4.78.
Read with the physical global sign the ladder is textbook chemistry - the model's own convention is O +0.22 / H -0.28, and negating globally (exactly free, `E(-q) = E(q)` verified to 0.0000 on all 159 frames) gives H2O -0.22, OH- -1.96, bare O -3.34, H3O+ +1.75.
So the lesson of the paragraph above stands (the per-atom split is not identifiable) but the *variation* is not arbitrary: it tracks the local coordination, which is exactly what a descriptor-fed latent charge should do.
The residual sign changes on ordinary water are marginal rather than structural - 22.9% of regular O sit just below zero within a 0.35-wide distribution whose mean is only 0.22.
Amplitude check by driving the author's kernel directly with modified `q` (159 frames): clipping `|q| > 1` to +-1 moves E_lr by rms 0.999 eV against a 1.403 eV signal while the frame-to-frame pattern survives at **r = 0.967**; zeroing those atoms gives 1.461 eV and r = 0.693; zeroing a random 2.2% gives 0.131 eV.
So the anomalous atoms carry amplitude, while the cross-frame signal is carried by the regular majority.
Present the ladder, not the min/max sheets, when describing what the learned charge is.

**A note on the geometry used above.** The dataset's cell varies per frame (diagonal 9.257 to 16.027 A across the 159 valid frames), so every distance here is computed under minimum image with each frame's *own* cell.
Using one constant cell is what once produced a spurious "0.244 A near-contact"; with the correct cells the tightest interatomic contact anywhere is 0.752 A, 96% of O have exactly two H within 1.3 A, and O-H bonds run 0.752-1.299 A (median 0.974) - ordinary water.
The loader is faithful: all 159 valid frames match a raw `water.xyz` frame at 0.000 A (the split is 159 of 1593 raw frames).

`cace` (pre-update) sits at the far left of the same axis: E_lr std/tgt **0.000** and F_lr/F_tot **0.000** - the long-range term contributes essentially nothing, which is why `cace` and `cace_sr` are the same arm to five significant figures.
Its correlation is already positive (+0.619), so the pre-update checkpoint was not anti-correlated like the deepmd learned arms; it simply had no magnitude to correlate with.
Raising the weight from 0.01 to 1.0 is what gave the term its 0.098 magnitude, and the checkpoint that did that is the one that also narrowed the charge network to a single channel.

**Why the learned layer is free to do this.** The loss only ever sees `E_SR + E_LR`, and `E_SR` is a flexible network: for any target it can fit, there is a whole family of `(E_SR, E_LR)` pairs that sum to the same good total, and the loss cannot tell them apart.
So the magnitude of the learned long-range term is not set by the fit at all, it is set by optimization dynamics, and a large cancelling `E_LR` is one loss-equivalent solution among infinitely many rather than a bug.
The frozen table is the opposite: it has no free parameters, so its `E_LR` is fixed by the charge assignment and the short-range net has to fit around it.
That is the real difference between the two rows, and it means the learned `q` should be read as a fitting coordinate, not as a physical charge, unless something constrains it - which is what `claim_total_charge` / `claim_neutral` and the fixed table exist to do.
The amplitude of the deepmd learned `q` is set by optimization dynamics, then; the cace single-channel `q` above is the counter-example where the *variation* does track a physical coordinate (the oxygen's protonation state), even though its absolute values remain a gauge-free latent split.

**A hypothesis that measurement refuted.** The kernel has no jellium background (see `tests/hybrid_ener/check_ewald_reference.py`), so a drifting total charge Q adds a Q^2 self-energy, and `hyb_lrw001` does drift badly (net |Q| up to 50).
That looked like it could explain the whole large-E_lr effect, so `diag_les.py` now measures it directly: it re-feeds each frame's charges with the per-frame mean removed straight to the kernel and reports the share of E_lr variance that disappears.
The drift share is **0.3% (`hyb_lrw1`), 2.1% (`hyb_lrw001`), 0.0% (`freeze_spce`)**.
Q varies little from frame to frame, so the Q^2 term is a near-constant offset that the short-range bias absorbs.
The drift concern is real in principle - it is why `claim_neutral` exists - but it is **not** what produces these magnitudes.

**Side validation.** `freeze_spce`'s E_lr (mean -381.53, std 2.98) reproduces `probe_fixed_table.py`'s independent kernel prediction for the same SPC/E table (mean -380.74, std 2.91) to about 0.2% and 2%, its q is exactly -0.8476 / +0.4238 with std 0.0000, and its net charge is exactly 0.
The small gap is consistent with `dl 2.0` in the run against `dl 1.5` in the probe.
That validates the freeze path and the kernel end to end.

## Task 3: local_charge vs freeze_charge

The controlled comparison is `hyb_lrw1` against `freeze_spce`: identical short-range network, identical data, identical weight 1.0, so the only difference is the charge layer itself - a learned network of 11,138 parameters against a fixed table of none.
That difference is the thing being measured, so it is not a confound, but it does mean the learned arm has 18% more parameters and the comparison measures "learned charge layer" and "extra capacity" together.

- Accuracy: `hyb_lrw1` (6.5884e-03 E/atom, 1.4804e-01 F) against `freeze_spce` (6.6280e-03 E/atom, 1.4114e-01 F).
  The learned layer is **0.60% better on energy and 4.89% worse on force**.
  Both directions clear the checkpoint spread: `hyb_lrw1`'s three energy checkpoints span 6.5795 to 6.6034e-03, entirely below `freeze_spce`'s 6.6108 to 6.6408e-03, so the energy edge is real rather than spread; and `freeze_spce`'s three force values span 1.4113 to 1.4114e-01 against `hyb_lrw1`'s 1.4804e-01, so the force gap is real too.
  So the trade is real in both directions and it is lopsided: the learned layer buys 0.6% of energy for 4.9% of force, and it needs 18% more parameters to do it.
- Magnitude and structure: as above, the learned layer makes the long-range term a co-equal channel (anti-correlated at -0.61, ~1.03x the target spread), while the frozen table makes it a modest positive correction (+0.54, ~0.21x).
  This is the mechanism behind the trade: the fixed table adds an interaction, so it improves force; the learned layer reallocates energy between two cancelling halves, so it can nudge energy but does not add force information.

**Verdict within our deepmd implementation, on this dataset and schedule: the fixed physical charge layer is the better long-range treatment.** It has the best force of the four deepmd arms at exactly the parameter count of plain `sr`, and it beats the learned layer on force by 4.9%. The learned layer's only advantage is a 0.6% energy gain, which does not compensate for a 4.9% force loss and comes at 18% more parameters.
Whether that verdict survives a different schedule is an open question and a fair one: the LES force advantage on this model is known to be `decay_steps`-dependent (see the caveat below), so this is a statement about `decay_steps 100000`.

**This verdict is about our learned layer, not about learned charges in general.** The author's `cace_lrnew` is also a learned charge layer feeding an Ewald sum at weight 1.0, and it beats every arm here including `freeze_spce` - on energy by 79% and on force by 56% (item 4 above).
So "learned charges lose to fixed charges" does not generalize; it is false across codes.
And with the matched control applied (item 5), the cace long-range term still adds more than the fixed table does: 10.48% / 13.09% against `freeze_spce`'s 3.03% on force (its 0.10% energy figure is not established, see item 1), each measured against its own code's short-range baseline.
Those baselines are different models in different codes, so this is not a clean head-to-head either, but it does say the cace learned `q` extracts more long-range accuracy here than a fixed SPC/E assignment does - the opposite of the within-deepmd result.
What separates the two learned layers is the shape of the long-range term they produce, and the Task 1 table measures it: `hyb_lrw1` splits the energy into two large anti-correlated halves (1.035x the target spread, corr -0.608), while `cace_lrnew` produces a modest positive correction (0.098x, corr +0.599) - the `freeze_spce` shape.
The arm whose long-range term looks like a physical correction beats the arms whose long-range term looks like a second fitting channel, whether the charges behind it are learned or fixed.

## Caveats

- **The `<tag>_sr` rows are not short-range models.** `cace_lrnew_sr` and `cace_sr` are long-range checkpoints with the Ewald term subtracted *after* training, so their short-range half was only ever fit with the long-range term present. They answer "how much of this checkpoint's own prediction is the Ewald term" and not "what does a short-range model reach on this data". The trained control reaches 1.5327e-03 / 7.2147e-02 where the subtraction leaves 1.0479e-02 / 1.1952e-01 - 6.8x worse on energy - so the two are not interchangeable, and a long-range gain quoted against a subtraction (86.91% / 47.54%) overstates the term's accuracy contribution by 8.3x. Item 5 gives the matched-control version; `compare/probe_lr_control.py` records it.
- **One schedule only.** Every deepmd arm here uses `lr` exp decay with `decay_steps 100000` and 100000 steps, i.e. no decay cliff inside the run.
  The LES force advantage on this model is known to depend on the schedule, so the force comparison is a statement about this schedule and not a property of the architecture alone.
  The direction is at least consistent with the earlier FastLearn result in `LES_PERFORMANCE.md`, where the fixed-charge LES arm beat the ordinary model on force by about 8.8% (replicate A) and 7.1% (replicate B) on a different 400-frame dataset at `decay_steps 5000`: here the fixed-charge arm again beats `sr` on force, by 3.03%, at the other end of the schedule ladder.
  So the fixed-table force win is not a one-schedule accident, though its size clearly depends on the schedule; the learned-charge arms show no force win at this schedule.
- **The dataset is variable-volume** (cell diagonal 9.257 to 16.027 A, mean 12.86), so it is NPT-like and the long-range virial matters.
  The hybrid model includes both the short-range and the long-range virial for this reason (see `check_virial.py`).
- **The frozen table is an assignment, not a derivation.** SPC/E charges are a force-field choice, chosen here because they are a standard, physically motivated water model at a realistic magnitude, not because anything in this repo fitted them.
- **Pre-update `cace` and `cace_sr` differ only in the long-range term** and differ in the 5th significant figure on every metric, which is the metric-level statement that the author's 0.01-weighted E_lr moves nothing. Any `cace` vs deepmd conclusion is therefore really `cace_sr` vs deepmd.
  `cace/inspect_cace_les.py` shows why at the mechanism level: the Ewald head emits only mean +0.081 eV/frame (std 0.0264 eV) and the 0.01 combine weight removes another factor 100, leaving 1.6e-03 meV/atom of energy range and 1.6e-05 eV/A of force against targets of 8.44 eV/frame and 2.11 eV/A.
  `cace/probe_cace_ewald_scale.py` separates how much of that is the checkpoint and how much is the kernel, by driving the kernel with a known physical charge set (the same SPC/E table `freeze_spce` uses) on the same frames: it returns std 0.0340 eV/frame where the les kernel returns 2.98, a factor 87.6, essentially the `norm_factor = 1` convention that the source documents as leaving charges scaled by `sqrt(90.0474)`.
  So the 0.01 weight multiplies a quantity that is already ~90x below physical, which is why the arm reads as inert rather than merely weak; the trained q contributes a further mild reduction (0.0264 against that 0.0340) rather than the main effect.
  **This caveat is about the pre-update arm only.** `cace_lrnew` runs at weight 1.0 and its Ewald term carries 86.91% of the energy error and 47.54% of the force error of its own prediction - the matched-control version, against a short-range model rather than a subtraction, is 10.48% / 13.09% (item 5) - against 0.009% / 0.012% for `cace`.
- **Capacity is matched for some pairs and not others.** Trainable parameters: `sr` 62,434, `freeze_spce` 62,434, `hyb_lrw1` 73,572, `hyb_lrw001` 73,572, `cace` 43,052, `cace_lrnew` 41,072, `cace_timing_sr` 24,572.
  So `sr` vs `freeze_spce`, and `hyb_lrw1` vs `hyb_lrw001`, are exactly matched; `sr` vs either learned arm, and any deepmd arm vs cace, are not.
  The learned arms' extra 11,138 parameters are the charge network itself, so this is inherent to comparing a learned charge layer against a fixed table rather than an accident of setup - but it does bound what the energy gain can be attributed to.
  The cace mismatch runs the other way and is large enough to matter for the headline: `cace_lrnew` has 21,362 fewer parameters than `sr` (34% fewer) and still wins by 4.8x / 2.3x, so its margin cannot be a capacity effect.
  The one pair where the long-range model has *more* parameters is the cace matched control - 41,072 against 24,572, i.e. 67% more - so that 10.48% / 13.09% gain bundles the Ewald head and the charge network with their added capacity, and the per-frame scaling test in item 5 is what argues the term itself, not the capacity, is doing the work.
- **All three cace checkpoints are the author's published files, scored as-is; none was retrained here.**
  The pre-update `cace` is a small demo checkpoint (198 kB against the deepmd arms' 1.6 MB), and `compare/probe_cace_offset.py` shows it under-fits even its own training frames: train and valid score the same (-47 vs -42 meV/atom of bias, 15.6 vs 13.7 meV/atom after removing it).
  So the `cace` rows bound what *that published checkpoint* does, not what the cace code or the cace-LES idea can do - a retrained cace model on this split would be the fair arm.
  The updated `cace_lrnew` (`fit-water-timing/fit-water-mp0-lr-FeatureAdd/best_model.pth`, sha256 `2345d4ad622c98a5`) is the author's own revised long-range run and reaches 1.85% / 2.97% of target, so unlike `cace` it is not obviously convergence- or capacity-limited; but it is a separate training run with its own budget and sampler, so its margin over the deepmd arms is an existence proof that a learned charge layer can reach this level on this problem, not a like-for-like controlled arm.
  Its short-range counterpart (`fit-water-timing/fit-water-mp0-sr/best_model.pth`, sha256 `020dd981f4f44c93`) is the matched control for the long-range term, and it is the one arm here whose training script and data were checked file by file against the long-range one: identical script except for the `q` / `EwaldPotential` / `FeatureAdd` modules, identical `train_path` (byte-identical `water.xyz`, sha256 `a7a0d2ce9cb0663c`), identical `valid_fraction 0.1 / seed 1` split, identical 40+100+100+100 epochs.
  The control is a real model, not a subtraction, and that is what makes item 5 the strongest statement in this document about whether the long-range term earns its keep.
- **Every `fit-water-timing` checkpoint needs the `torchscript` branch; the pre-update one needs `main`.** Both `cace_lrnew` and `cace_timing_sr` will not load under `main`: their stored `lxlylz` keys are TorchScript-safe strings, so main's `angular.py` raises `TypeError: unsupported operand type(s) for -: 'str' and 'int'`. That was verified by running the short-range control under `main` and reproducing the identical error.
  Each arm was therefore scored with `--cace-root` pointing at the checkout that produced it - `cace` under `/root/app/cace` (main, commit `3f1a66552684`), `cace_lrnew` and `cace_timing_sr` under `/root/app/cace-ts` (torchscript, commit `6256b8e1f586`) - and `record_cace_valid.py` writes the imported path and commit into the JSON alongside the numbers, so which branch ran is on the record rather than assumed.
  A checkpoint pickles its module classes by name, so this is not optional: importing the wrong branch silently runs the wrong code.
- **The pre-update to updated change is two changes at once, and this document does not separate them.** The LR weight goes 0.01 -> 1.0 *and* the charge network's output width goes 4 -> 1, so the 32x energy improvement (43.97 -> 1.372 meV/atom) cannot be attributed to the weight, to the narrower charge channel, or to a split between them on this evidence.
  They are also separate published runs rather than one checkpoint retrained at a single changed knob, so even a matched-knob re-run would not decompose this pair.

## Reproduce

```bash
cd cace_compare_e2e
python compare/metrics.py --late 3 --device cuda     # the metrics table
python compare/diag_les.py --device cuda              # the E_lr / q diagnostics
python compare/probe_cace_offset.py                   # why the pre-update cace energy error looks like that
python compare/probe_lr_control.py                    # the matched short-range control for the cace
                                                      #   long-range term (item 5); reads the two npz files
python compare/probe_fixed_table.py                   # the standalone fixed-table kernel numbers
python cace/probe_cace_ewald_scale.py                 # what scale the cace Ewald kernel works at

# the cace rows, from the three published .pth files (see the branch caveat above:
# each checkpoint needs the cace checkout it was trained on)
python compare/record_cace_valid.py                   # cace / cace_sr, main cace
python compare/record_cace_valid.py --cace-root /root/app/cace-ts --lr-weight 1.0 \
  --model ../cace-lr-fit-datarepo/BingqingCheng-cace-lr-fit-0211150/fit-water-timing/fit-water-mp0-lr-FeatureAdd/best_model.pth \
  --arm-tag cace_lrnew \
  --out cace/cace_lrnew_valid_metrics.json --npz cace/cace_lrnew_valid_pred.npz
python compare/record_cace_valid.py --cace-root /root/app/cace-ts --no-long-range \
  --model ../cace-lr-fit-datarepo/BingqingCheng-cace-lr-fit-0211150/fit-water-timing/fit-water-mp0-sr/best_model.pth \
  --arm-tag cace_timing_sr \
  --out cace/cace_timing_sr_valid_metrics.json --npz cace/cace_timing_sr_valid_pred.npz

# the arms themselves (HPC pod, all at decay_steps 100000 / 100000 steps)
cd deepmd && dp --pt train runs100k/hyb_lrw1/input.json
cd deepmd && dp --pt train runs100k/hyb_lrw001/input.json
cd deepmd && dp --pt train runs100k/freeze_spce/input.json
```
