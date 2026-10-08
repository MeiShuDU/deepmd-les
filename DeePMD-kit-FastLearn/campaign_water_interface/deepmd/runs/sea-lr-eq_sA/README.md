# sea-lr-eq - sea-lr with latent-charge equalization

`sea-lr` (`DeepMD` `se_a` descriptor + CACE charge head + CACE `EwaldPotential`,
`combine_potentials` + `share_descriptor`, LR weight `0.02`) with
`ChargeEqLatent` swapped in for the plain `EwaldPotential`.

## What differs from `sea-lr_sA`

Three things under `model.long_range`:

```json
"charge_net": { ..., "n_out": 1 },                      // was 4
"ewald": { "dl": 2.0, "sigma": 1.0, "remove_self_interaction": true },   // was false
"charge_eq_latent": { "enabled": true, "regularization_weight": 10.0, "total_charge": 0.0 }
```

Everything else (descriptor, charge head `n_layers=3 n_hidden=[24,12] bias=false`,
the four-block cace schedule, seeds, split) is byte-identical to
`sea-lr_sA/input.json`.

**One charge channel (`n_out=1`).** The head gives one latent charge per atom, so
the equalization's single collective constraint is just the ordinary "the frame is
neutral" condition. This also removes the `n_out>1` confound where the plain
kernel's self-term correction is counted once per channel and the equalizing
module's energy diverges from it by `(n_out-1) sum q^2/(sigma (2 pi)^1.5)`; with
`n_out=1` the equalized energy and the plain kernel's energy are the same
quadratic form.

**`remove_self_interaction: true`.** Removes the Ewald self term, so `A` is the
physical Coulomb matrix and `E_lr = 0.5 q^T A q` is the physical Ewald energy.
This matches cace's `EwaldPotential` default and `gen_inputs.py`'s `LES_COMMON`
(the campaign's other `deepmd_cace` arms set `false`, which keeps a spurious
`+ sum_i q_i^2/(sigma (2 pi)^1.5)` term in `E_lr` - a penalty pulling charge
magnitudes toward zero). For the equalization solution itself the setting is
almost irrelevant: on the diagonal it is a constant, equivalent to shifting `w`
by `1/(sigma (2 pi)^1.5) = 0.063` (0.6% at `w=10`); what it changes is the `E_lr`
the loss sees.

`regularization_weight` is the `w` of

    min_q  1/2 q^T A q + w ||q - q_r||^2     s.t.  1^T q = Q_total

where `A` is the Ewald Coulomb matrix (CACE's own kernel) and `q_r` is the
charge head's proposal. `w = 10` is a strong trust region: applied at inference
to the trained `sea-lr_sA` head on the held-out valid frames, the equalized set
moves `||q_eq - q_r|| / ||q_r|| = 4.0%` (rms) away from the proposal, so this run
mostly **neutralizes** the charges (forces the per-frame total to zero) rather
than reshaping them. The module holds no parameters, so a plain `sea-lr`
checkpoint loads into it and back with no state-dict change.

Because the equalizer *projects out* the uniform offset rather than *penalizing*
it, it does not by itself push the charge magnitudes up: the trained head stays
free to drift to net charge and to small magnitudes, and the Ewald term simply
never sees the offset. Whether that helps is exactly what the run tests.

## The single channel

The charge head emits one channel (`n_out=1`), so the collective neutrality
constraint `1^T q = Q_total` over "channels and atoms" is simply the frame
carrying zero total charge - the condition the Ewald reciprocal sum needs, since
it drops the `k=0` term. (An earlier 4-channel revision of this directory
equalized all four against one shared Coulomb matrix with a single constraint on
their grand total; that config was revised to `n_out=1` before the run.)

## Running

    cd campaign_water_interface/deepmd
    python run_deepmd_cace.py --arm sea-lr-eq --rep A

which runs `python -m deepmd_cace input.json` in this directory (with
`PYTHONPATH=<repo>/desc_bridging` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`),
writes `blocks.json` / `timing.json` / `train.log` here, and both `best_model.pth`
and `water-model*.pth` per the schedule's `checkpoint` keys.

`run.sh` is the same thing for one replicate, without the timing wrapper.

## Scoring

    cd <repo>/desc_bridging
    python check_bec_arm.py --ckpt ../DeePMD-kit-FastLearn/campaign_water_interface/deepmd/runs/sea-lr-eq_sA/best_model.pth --frames 5

`check_bec_arm.py` reports both the head's proposal `q` and the kernel's charges
`q_eq` (the model carries the equalizer, so `q_eq` is what the Ewald term sees),
in physical electrons (`/9.48933`, the model's charge unit under
`norm_factor=1`). Since this arm's `n_out=1`, the channel sum is just `q` itself.

## Measured charges and BEC (finished 2026-10-04)

225,000 steps, 8/8 blocks, 6.62 h on the pod. Final `val_e/atom_rmse = 1.48e-4`,
`val_f_rmse = 0.0380` - essentially unchanged from `sea-lr_sA` (1.39e-4 /
0.0389). Charges and Born charges below are 5-frame means over the `valid` split
(1566 atoms).

| | sea-lr_sA | sea-lr-eq_sA | physical ref |
|---|---|---|---|
| `q_O` (e) | -0.114 | +1.426 | -0.85 |
| `q_H` (e) | +0.064 | -0.713 | +0.42 |
| net / frame (e) | +7.03 | 0.000 | 0 |

The charge magnitude grew ~12.5x to a physical size and the frame is now exactly
neutral. **The overall charge sign is a gauge**: `E_lr` is bit-invariant under
`q -> -q` (verified, difference exactly 0.0), so the eq arm landing on the
O-positive convention rather than `sea-lr_sA`'s O-negative one carries no
physics. Fixing the gauge to the physical O-negative convention:

| convention | O | H | max\|Z*\| | sum_i Z* |
|---|---|---|---|---|
| PBC dephased | +4.69 | -2.36 | 12.4 | 2.8e1 |
| direct sum | +2.46 | -1.23 | 46.8 | 1.2e-4 |
| direct sum, frozen | +1.43 | -0.71 | 2.2 | 4e-6 |

Reading, split into offset / scale / shape:

- **offset (sign)**: the BEC now has the correct sign (same sign as the arm's own
  charge; negative O in the physical convention). The wrong-sign pathology of
  `sea-lr_sA` is gone: there the response term flipped the sign (frozen -0.11 ->
  BEC +0.45), here it keeps it (frozen -1.43 -> BEC -2.46).
- **scale**: still ~2x too large (`|Z*_O| = 2.46` direct vs physical ~1.1-1.3).
- **shape**: `Z*_H / Z*_O = -0.5`, so the direct-sum acoustic sum rule holds
  (violation 1.2e-4, was 8.4e0) - because the charges are exactly neutral. The
  dephased periodic value still carries the ~`1/L^2` violation (2.8e1), and
  `max|Z*| = 47 e` is no better than `sea-lr_sA`'s 44.6, so individual atoms
  still have a jagged `dq/dr` that the isotropic mean hides.

**This supersedes the operator preview below.** The preview applied the
equalizer to the *untrained* `sea-lr_sA` head, whose proposal was still tiny, so
it showed the sign unchanged. Training with the term in the loss moved the head
somewhere else entirely.

## Operator-level BEC preview (from before the run)

`desc_bridging/check_bec_eq.py` applies `ChargeEqLatent(w=10)` to the **trained
`sea-lr_sA` head** at inference, so the mapping `BEC(q_r) -> BEC(q_eq)` can be
read off without retraining. It is a diagnostic of the operator at a fixed head,
and its headline result (that equalization alone does not change the sign) was
superseded by the run above.

Measured on 5 held-out valid frames (1566 atoms), `Z*_iso` in e:

| convention | arm | O | H | max abs Z* | sum_i Z* |
|---|---|---|---|---|---|
| PBC dephased | `q_r` | +1.778 | -0.776 | 3.0 | 1.2e2 |
| PBC dephased | `q_eq` | +1.456 | -0.655 | 4.6 | 1.0e2 |
| direct sum | `q_r` | +0.452 | -0.219 | 44.6 | 8.4e0 |
| direct sum | `q_eq` | +0.311 | -0.156 | 17.8 | 2.6e-5 |

Net charge goes `+7.50 e -> 0.000 e` per frame, exactly.

Reading: equalization (i) forces the frame neutral, which makes the direct-sum
BEC satisfy the acoustic sum rule to `2.6e-5` instead of `8.4e0`, and (ii) tames
the wildest individual atoms (`max|Z*|` 44.6 -> 17.8). It does **not** fix the
sign: O stays positive against the physical ~-1.1, so the wrong-sign problem is
in the learned `dP/dr`, not in the non-neutrality.
