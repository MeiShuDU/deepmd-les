# deepmd-cace-eq - `cace-sea-lr` replicated through `deepmd_cace`, with latent-charge equalization

A replicate of `cace/runs/cace-sea-lr` trained through `desc_bridging/deepmd_cace` rather than through cace: a DeepMD `se_a` descriptor feeding CACE's `Atomwise` heads, JSON config, `python -m deepmd_cace input.json`.
The one deliberate addition is `long_range.charge_eq_latent` at `w = 10`.

This arm has now been trained: the full 80,000-step schedule ran on the pod on 2026-10-06 (8 of 8 blocks `ok`, `returncode 0`, 8,294 s wall at 0.104 s/batch, peak 303.8 MB of GPU memory).
`input.json` and `run.sh` were authored before the run and the smoke test in `../runs_smoke/deepmd-cace-eq_sA/` passed; `best_model.pth`, `model-*.pth`, `checkpoint.pt`, `blocks.json`, `timing.json`, `train.log` and `run_settings.json` are the pod's own files, rsynced back.

## Result

Scored on the 100-snapshot VASP BEC set (which it does not train on), dephased and at this arm's `sqrt(w*eps_inf) = 1.3342`: `Z*_O = -1.013`, `Z*_H = +0.520` against VASP's `-0.903` / `+0.451`, RMSE 0.115 e over all 57,600 diagonal elements (O 0.147, H 0.096), `R^2` 0.970 / 0.011 / 0.734.

For comparison at the same scale, `cace-sea-lr` scores 0.100 e / 0.124 / 0.086 with `R^2` 0.977 / 0.289 / 0.786.
The gap is a mean offset rather than a shape error: about each arm's own mean the residual scatters are 0.0966 e here and 0.0986 e there, while the O mean sits 0.110 e from VASP against `cace-sea-lr`'s 0.076 e.
It is also not attributable to anything in particular, because at one seed per arm it sits at the same scale as this campaign's own run-to-run spread.
The sibling pairs, which are the same recipe run twice at different seeds (descriptor/fitting seeds 1/1 against 2/2, training seed 10 against 20), end the 80,000-step schedule 1.01x apart on validation energy RMSE and 1.06x on force for `deepmd-cace-sched`, and 1.12x and 1.00x for `deepmd-les-cace-sched`, while this arm is 1.15x from `cace-sea-lr` on pooled BEC RMSE.
So the two BEC numbers are indistinguishable at n = 1 per arm, and the comparison is a single sample of a cross-framework replicate rather than a measured regression.

## The equalizer is not where any gap comes from

`desc_bridging/check_bec_eq_arm.py` scores the *same trained weights* twice on the same 100 frames, once as trained and once with `charge_eq_latent` disabled in the config.
`ChargeEqLatent` holds no parameters or buffers, so the state dict loads strictly into both and the pair differs only in whether `q_eq` or the head's raw `q` enters `Polarization`.
Turning the equalizer off makes the arm worse on every measure: pooled RMSE 0.1151 -> 0.1392 e, O RMSE 0.1467 -> 0.2184, O bias `-0.1104 -> -0.1964` e, `Z*_O` `-1.013 -> -1.099` against VASP's -0.903, and the acoustic sum rule `+1.705 -> -8.334` e per frame against VASP's `2e-06`.
The paired difference is very nearly a uniform shift in the diagonal Z*, which is what removing a charge drift does: the head's proposal carries `+7.60 e` of net charge per frame that the equalizer takes to `-0.0000`.
So the equalization is worth 17% of the pooled RMSE at these weights; it cannot be the source of the 15% gap, and deleting it would widen the gap.

The equalizer does to the charges what it promises (net charge per frame `-0.0000` against the head's `+62.3 e`, and no element moved by more than 4.6% of the largest proposal) and much less to the sum rule, which is a *response* property: with the kernel charges frozen, `sum_i Z*_i` is 8.7e-06, and with the response included it is 1.705 e per frame, against 2.361 e for `cace-sea-lr` and 3.556 e for `cace-lr`, all three at the same 1.3342 scale.

**Which checkpoint these numbers are from.** `best_model.pth` is cace's best-validation file *within one `fit()` call*, because `cace/tasks/train.py:190` resets `best_val_loss` at the top of each call. For this arm it is therefore the best epoch of the last block (timestamped 17:20:17, block 8 having started at 17:18:38), not the best over the whole schedule. The block-end `model-4.pth` scores `Z*_O = -1.012`, `Z*_H = +0.520`, O RMSE 0.119, H RMSE 0.076, i.e. within 0.002 e, so the distinction changes none of the numbers above.

`desc_bridging/bec_scatter_pred_dft.ipynb` puts this arm next to `cace-sea-lr` and the other arms, and `desc_bridging/vasp_bec_deepmd-cace-eq_sA.npz` is the `--dump` of the same scoring.

## The model

    DeepmdSeAInput -> Atomwise(SR_energy, [32,16])
                   -> Atomwise(tot_q / q, [24,12], n_out=1, bias=false, add_linear_nn=true)
                   -> ChargeEqLatent(q_eq)
                   -> FeatureAdd([SR_energy, ewald_potential] -> CACE_energy)
                   -> Forces

That is `cace-sea-lr`'s graph with the plain `EwaldPotential` swapped for the equalizer, which is what `deepmd_cace`'s `long_range` block builds.
The built smoke model has **119,478 trainable parameters**, exactly the `n_trainable_params` recorded for `cace-sea-lr` in `cace/runs/cace-sea-lr/run_settings.json` - the equalizer is parameter-free, so the count is unchanged by the swap.

## Provenance of every setting (parsed out of the cace source, not retyped)

| block | taken from |
|---|---|
| `se_a` descriptor | `SE_A` in `cace/sea_seam.py`: `rcut 5.5`, `rcut_smth 0.5`, `sel [39, 73]`, `neuron [25, 50, 100]`, `axis_neuron 16`, `resnet_dt false`, `seed 1`, `type_one_side false` |
| schedule, batches, weights | `gen_input_cace.py` imports `CUTOFF`, `TRAIN_BATCH`/`VALID_BATCH`, `ATOMIC_ENERGIES`, `ENV_PHASES`, `FORCE_WEIGHT`, `PHASE_CKPT` from `cace/fit_cace.py` |
| heads, Ewald, combine path | `cace/sea_seam.py:build_model(arm='lr')` |
| data split | `DeePMD-kit-FastLearn/data/{data_0,data_1,data_2}` train, `data_3` valid |
| seed | `training.seed = 10`, `fit_cace_sea.py`'s default and the seed the recorded run used |

`../gen_input_cace.py` regenerates both copies of `input.json` (this directory and `runs_smoke/`); `python gen_input_cace.py --check` reports drift, `--verify` also re-checks the data split.

## The five deliberate differences from `cace-sea-lr`

1. **`long_range.charge_eq_latent` is enabled at `w = 10`** (`total_charge: 0`).
   The point of the run: the head's proposal `q` is replaced by the KKT solution of `min_q 1/2 q^T A q + w ||q - q_r||^2 s.t. 1^T q = 0`, so the Ewald term sees an exactly neutral charge set that stays within `w` of the learned one.
2. **`combine_potentials: false`**, i.e. the `SR_energy + ewald_potential -> CACE_energy` `FeatureAdd` path inside a single model, which is structurally what `cace-sea-lr` is.
   The LR term is therefore added at unit weight with no `weight` key, matching the recorded `lr_mix_weight: 1.0`, and this arm's BEC scale is `sqrt(1.0 * eps_inf) = 1.3342` - not the water-interface arms' `sqrt(0.02 * 1.78) = 0.1887`.
3. **`ewald.remove_self_interaction: false`**, matching `cace-sea-lr`.
   It is written out explicitly rather than left to the default because the two cace checkouts disagree on that default (`cace-ts` false, `/root/app/cace` true), and the value has to stay pinned to `cace-sea-lr`'s for the two to be comparable.
   For the equalizer the setting is nearly irrelevant - on the diagonal it shifts `w` by `1/(sigma (2 pi)^1.5) = 0.063`, 0.6% at `w = 10` - but it is what the loss's `E_lr` sees.
4. **The descriptor keeps `precision: float64`.** `SE_A` omits the key, so `cace-sea-lr`'s `se_a` ran in double (`cace-sea-lr/run_settings.json:se_a` has no `precision`), and so do the campaign's native deepmd arms.
   The CACE heads stay float32, exactly as in `cace-sea-lr`, and `DeepmdSeAInput` casts its output back to `positions.dtype`, so this only changes the descriptor's internal matmuls.
   It means the arm must be run with `DP_INTERFACE_PREC=high` (deepmd's default) - `run.sh` and `smoke.sh` set it, and `run_deepmd_cace.py` needs `--precision float64`.
   Scoring it is likewise dtype-sensitive; see "Scoring" below.
5. **`training.seed = 10`.**

## Data

`DeePMD-kit-FastLearn/data`: `data_0` + `data_1` (two `set.*` dirs) + `data_2` = 320 training frames, `data_3` = 80 validation frames, 192 atoms each (64 O + 128 H = 64 H2O, cubic 12.445 A).
That is the same 320/80 split, in the same order, with the same total energies as `campaign_fastlearn/xyz/{train,valid}.xyz`, which is what the cace arms read: the two agree to `max abs diff 0.0`.
`gen_input_cace.py --verify` asserts the frame counts, the atom counts per element, and `type_map.raw == [O, H]`.

## Running

Local or on the pod, one process, in this directory:

    ./run.sh                      # full 80,000-step schedule, device from input.json (cuda)
    ./run.sh --check-only         # forward + backward + descriptor-grad check, no training

Prefer the campaign runner, which adds timing and refuses to start a block whose predecessor checkpoint is missing:

    cd DeePMD-kit-FastLearn/campaign_water_interface/deepmd
    python run_deepmd_cace.py --root ../../campaign_fastlearn/deepmd/runs \
        --arm deepmd-cace-eq --rep A --precision float64

(`--arm deepmd-cace-eq --rep A` resolves to this directory, `{arm}_s{rep}`.)

Smoke test, CPU, one epoch per fit block, artifacts kept out of this directory:

    ../runs_smoke/deepmd-cace-eq_sA/smoke.sh          # --check-only, then 8 x 1 epoch
    ../runs_smoke/deepmd-cace-eq_sA/smoke.sh --check  # --check-only only

The schedule is `cace/fit_cace.py`'s four phases as four blocks - five fresh 40-epoch tasks at E weight 0.1, then 1.0 / 10.0 / 1000.0 at 100 epochs each sharing the fifth task - i.e. 8 fits, 500 epochs, 160 steps/epoch, **80,000 steps**, exactly the recorded `total_steps`.

## Smoke test (passed 2026-10-06)

`--check-only` in 5.6 s: the descriptor builds, its statistics run over all 320 frames, a full forward (energy, force derivative, the KKT solve) and the gradient into `se_a` all check out.
`--smoke` in 4m27s: 8 fits at one epoch each, 1,280 steps, ~32 s/epoch on CPU, `fresh_task` true exactly for the five phase-0 fits and false for the three loss-swap fits.
The descriptor statistics match the ones recorded for `cace-sea-lr` (`mean_abs_max 5.222e-02`, `stddev_mean 1.001e-01` vs recorded `0.052223136663652206` / `0.10011521718486033`), and the built model's parameter count matches too.
At ~32 s/epoch the full 500-epoch schedule is roughly 4.4 h on this host's CPU; on the pod's GPU it is what the other arms cost there.

## Scoring the predicted BEC

    cd desc_bridging
    python eval_vasp_bec.py --ckpt ../DeePMD-kit-FastLearn/campaign_fastlearn/deepmd/runs/deepmd-cace-eq_sA/best_model.pth \
        --frames 100 --dump vasp_bec_deepmd-cace-eq_sA.npz
    python check_bec_arm.py --ckpt <same> --frames 5
    python check_bec_eq_arm.py --frames 100          # this arm, equalizer on against off

`eval_vasp_bec.py` scores against the `BEC/` VASP set (100 revPBE snapshots of bulk liquid water, 192 atoms, cubic 12.429 A), reporting both the dephased periodic and the direct-sum `Z*`, and reads the LR weight off the checkpoint's own config - this arm's `FeatureAdd` path has no weight key, so it reports `w = 1` and scales by 1.3342.
`--dump` writes the same npz schema as `check_bec_cace.py --dump` for the notebook.
`check_bec_arm.py` reports the head's proposal `q` and the kernel's `q_eq` side by side, which for this arm is the interesting pair: the equalizer is what the Ewald term sees.
`check_bec_eq_arm.py` is the controlled version of that pair: it scores the same weights twice on the same frames, with the equalizer switched off in the config rather than on the model, and writes `bec_eq_arm_compare.npz`.

**A dtype note that applies to this arm and not to the water-interface ones.** `check_bec.py` picks the dtype to give positions from the charge head, not from `next(model.parameters())`, because on this arm the first parameter belongs to the float64 descriptor while everything downstream is float32.
Feeding float64 positions makes `DeepmdSeAInput` hand float64 node features to a float32 head, which fails with `expected m1 and m2 to have the same dtype, but got: double != float`.
The water-interface `sea-lr-eq_sA` path is unaffected either way: its descriptor is float32, so both rules agree.

## The two cace checkouts give the same Ewald kernel here

`cace-sea-lr` was built on `/root/app/cace-ts`, while `deepmd_cace` resolves `cace` to the editable `/root/app/cace` - a different commit, and `ewald.py` differs between them (per-channel structure factor, the `static_Nk` CUDA-graph patch, a changed `add_external_field` sign, and the `remove_self_interaction` default).
At this arm's configuration (`n_out=1`, no external field, no field output, `remove_self_interaction` set explicitly) the two agree: driving both checkouts in separate processes on the same random charges gives bit-identical Ewald energies for a cubic and a sheared cell, and forces agreeing to `6e-8`, which is float32 summation order in the reciprocal term rather than a difference in the physics.
So scoring this arm and the cace arms in separate processes is a meaningful comparison.
