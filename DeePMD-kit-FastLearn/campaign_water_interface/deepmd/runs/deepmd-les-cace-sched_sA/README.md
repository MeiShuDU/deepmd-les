# deepmd-les-cace-sched - deepmd's `hybrid_ener` on its sibling's own net and schedule

The campaign's fifth arm: `sea-lr_s*`'s descriptor, `sea-lr_s*`'s short-range net and
`sea-lr_s*`'s eight-block schedule, with the long-range term implemented by deepmd's
`hybrid_ener` model (the `les` package's Ewald summation) instead of cace's
`EwaldPotential`.

It exists to answer one question, and only that question: `deepmd-les_s*` scored far
worse than its sibling `sea-lr_s*`, and the cause was either the LES *implementation*
or the *settings* the legacy arm ran with.
This arm differs from `sea-lr_s*` in the long-range implementation and in nothing else
a config key can reach, so whatever gap remains is attributable to the implementation.

## What differs from `sea-lr_sA`

| knob | `sea-lr_sA/input.json` | this arm | reachable |
|---|---|---|---|
| descriptor | `se_a` rcut 5.5, rcut_smth 0.5, sel [43, 84], neuron [25, 50, 100], axis_neuron 16, `resnet_dt false`, `type_one_side false`, seed 1 | identical | yes |
| SR net | `Atomwise` `n_hidden [32, 16]`, silu | `ener` `neuron: [32, 16]`, `activation_function: silu`, `resnet_dt: false` | partly, see below |
| LR weight | `long_range.weight: 0.02` | `les_params.lr_weight: 0.02` | yes |
| Ewald | `dl 2.0`, `sigma 1.0`, `remove_self_interaction false` | identical | yes |
| charge head | `n_layers 3`, `n_hidden [24, 12]`, `n_out 4`, `bias false` | `n_hidden [24, 12]`, `n_layers 3`, `add_linear_nn true` | partly, see below |
| Ewald prefactor | `norm_factor 1.0` | les hardcodes 90.4756 | no |
| schedule | 4 phases, 8 blocks, 225,000 steps | identical | yes |
| optimiser | Adam lr 1e-2, `StepLR(20, 0.5)`, warmup 5 epochs, `max_grad_norm 10` | identical (`learning_rate: exp`, `decay_steps 9000`, `decay_rate 0.5`) | yes |
| loss | `w_E` 0.01 / 1 / 10 / 1000 by phase, force weight 1000 | `pref_e = 1566 x w_E` (15.66 / 1566 / 15660 / 1566000), `pref_f 1000` | yes |
| precision | `float32` | `float32` via `DP_INTERFACE_PREC=low` | yes |
| seeds | desc 1 (both reps), train 1 (A) / 2 (B) | desc 1, fit 1, train 10 (A); desc 2, fit 2, train 20 (B) | yes, but not transferable - see below |

`pref_e = natoms x w_E` because deepmd's energy loss is `pref_e * MSE(E) / natoms`
(`pt/loss/ener.py`) while cace's is `w_E * MSE(E)` on the frame energy; natoms is 1566
in this data, read from `type.raw` rather than assumed.

The seed row is the one place the two configs cannot be made equal, and the honest
statement is that they are not compared: a seed selects an RNG stream inside a code,
and these are two different codes, so "the same seed" has no meaning across them.
What the row records is each family's own replicate convention. The sea arms vary the
training seed between replicates and hold the descriptor seed at 1
(`sea-lr_sA/input.json:95` seed 1, `sea-lr_sB` seed 2; both `:16` seed 1). This arm
follows the deepmd family's convention instead - descriptor, fitting and training
seeds all move together, 1/1/10 for A and 2/2/20 for B - which is the same
convention `deepmd_s*` and the legacy `deepmd-les_s*` use, so the within-family
comparison is clean. The cross-family one carries this difference, which is one
extra arbitrary choice on top of the two replicates' own noise.

Both families do agree on the descriptor-sharing switch, and it is worth naming
here: `sea-lr_sA/input.json:27` sets `share_descriptor: true`, cace's own key for
reusing the short-range descriptor in the long-range branch. The pod's stale tree
was missing exactly that optimisation on the deepmd side (see below), so the fix
restores parity with a key the sibling declares outright.

The precision row is not a config key: `precision: float32` in `sea-lr_sA` pins cace's
own dtype, whereas deepmd reads its interface precision from `DP_INTERFACE_PREC` at
`import deepmd.env` time.
`run_chain.py` sets it to `low` for the child `dp`, which is what makes the arm fp32 like
its sibling; a bare `dp --pt train` on these configs would run them in deepmd's default
float64 instead, at roughly 1/64 the fp32 rate.

### The three keys cace has and deepmd does not

`gen_inputs.py`'s module docstring records these as declared deviations, and they are the
whole of the residual difference between "the same model" and "the same architecture the
two codes can both express":

- **`add_linear_nn`.** cace's `Atomwise.forward` computes
  `y = outnet(features) + linear_nn(features)`, where `linear_nn` is a parallel
  `Dense(n_in, n_out)` with no activation (`cace/modules/atomwise.py:159-161`), and
  `sea-lr_sA` sets `add_linear_nn: true`. deepmd's `ener` fitting has no equivalent, so
  this arm is cace's SR net minus that one linear term.
- **`bias`.** The charge head declares `bias: false`; les's `Atomwise` always has a bias.
  The SR net does not set it, so that one defaults to true in both codes and matches.
- **`n_out`.** cace's charge head emits 4 channels which its Ewald sums over; les has no
  `n_out` key and predicts one latent charge per atom.

All three change the *parameterisation* of the same functional form, and the charge-side
two are absorbed by a freely trained charge net.
The Ewald prefactor is in the same category: cace's `norm_factor 1.0` against les's
hardcoded 90.4756 is a constant scaling of `q`, absorbed exactly by rescaling the charge
head's output, so it cannot change what the model can represent.

`n_layers: 3` is written anyway for provenance but is inert here: cace's `build_mlp` uses
`n_neurons = [n_in] + n_hidden + [n_out]` and ignores `n_layers` whenever `n_hidden` is a
list (`cace/modules/blocks.py:54-58`), so `n_layers 3` with `n_hidden [32, 16]` is a
two-hidden-layer MLP, which is exactly deepmd's `neuron: [32, 16]`.
The model this arm trains has 169,692 parameters.

## Why the legacy arm scored badly: setting or coding?

The legacy arms that motivated this run, and what they actually did:

| | `campaign_fastlearn` `deepmd-les_s*` | `campaign_water_interface` `deepmd-les_s*` |
|---|---|---|
| SR net | `[240, 240, 240]` | `[240, 240, 240]` |
| rcut | 5.50 | 6.00 (not `sea-lr`'s 5.50) |
| `remove_self_interaction` | `false` | `true` (not `sea-lr`'s `false`) |
| energy weight | deepmd's stock ramp, `0.02 -> 1.0` | cace's phase weights, flat within each block (15.66 / 1566 / 15660 / 1566000) |
| force weight | deepmd's stock ramp, `1000 -> 1.0` | 1000 flat |

So of the two candidate causes the user proposed, the "constant energy loss weight" holds
for the FastLearn arm - which left deepmd's default `ener` schedule in place, ending at a
force:energy ratio of 1:1 rather than cace's 1000:1 - and does **not** hold for the
water-interface arm, whose `pref_e` already follows cace's phase schedule exactly.
What the water-interface arm got wrong was the model it bolted the long range onto: a
`[240, 240, 240]` fitter where `sea-lr_s*` runs `[32, 16]`, at a different cutoff, with the
Ewald self-term set the opposite way.
Two things therefore differed at once, which is precisely why that pair cannot attribute
its own gap - and why this arm exists.

### The implementation half is closed

Two independent pieces of evidence say the LES code is not the explanation:

1. **The Ewald kernel itself.** les's summation is bit-exact against cace's on the same
   charges and to 1e-14 against the textbook NaCl Madelung constant (1.747565).
2. **The pod's stale tree, A/B'd.** See below: the defect the pod was running cost
   computation, not accuracy - the stale and fixed trees produce bit-identical physics.

### The pod tree double-called the descriptor (found and fixed 2026-10-06)

The pod's `hybridles_model.py` predated the descriptor-reuse change, so
`HybridLESModel.forward` built the neighbour list and ran the descriptor a second time
instead of reusing the one the SR path had already produced.
`/tmp/les_ab.py` builds the arm's model from one config, monkeypatches
`DescrptSeA.forward` to count calls, and runs one forward; the two trees were measured on
the same frame with the same weights (the fixed tree's state dict loads into the stale one
with 0 missing and 0 unexpected keys):

| | stale pod tree | fixed tree |
|---|---|---|
| descriptor forward calls per model forward | **2** | **1** |
| energy (eV) | 195845.56081273785 | 195845.56081273785 |
| force norm (eV/A) | 3589088.6028092904 | 3589088.6028092904 |
| virial sum | 1535052.246083223 | 1535052.246083223 |
| parameters | 169692 | 169692 |
| descriptor state-dict keys | 58 | 58 |

Every physical number is bit-identical; the only difference is a duplicated computation.
Median s/step over 5 training-shaped iterations on the pod's GPU went 0.0894 -> 0.0718, so
the stale tree was about 20% slower per step - honest framing: the medians differ by that
much, the distributions overlap at the edges (stale min 0.0723 against fixed max 0.0738).

Worth recording, because the mission that produced this arm assumed the opposite: **no
parameter was ever registered twice.**
Neither revision defines `self.descriptor`, so the duplicate work could not duplicate
weights - the stale tree's `n_params` and descriptor key count are the same 169692 / 58,
and the fixed tree's state dict loads into it cleanly.
The cost was pure recomputation.

The fix is the local `descriptor-reuse` change (commit `b74a0bf`, 17 files: a `desc_out`
channel threaded through `base_atomic_model` / `dp_atomic_model` / `linear_atomic_model` /
`pairtab_atomic_model` / `make_model`, plus `forward_lower` and `need_lower_box`).
It was never pushed to origin, so the pod had to be patched file by file; the pre-patch
state of all 15 touched files is archived on the pod at
`/root/les_preupdate_backup_20261007/pod_sources_before.tar`, with the file list in
`FILES.txt` beside it.
The pod's tree now matches `b74a0bf` file for file, and this arm was launched from it.

### The setting half, measured

The run is finished; both replicates trained the full 225,000 steps and were scored
on the whole 50-frame validation split at their final checkpoint, by the same
formula the campaign uses for the cace/sea arms
(`sqrt(mean((dE/natoms)^2))` and `sqrt(mean(dF^2))`, `analysis/compare_pooled.py`):

| arm (final ckpt, pooled) | `rmse_e/atom` | `rmse_f` |
|---|---|---|
| this arm | **1.5801e-04** | **4.1934e-02** |
| legacy `deepmd-les_s*` | 1.8963e-04 | 4.5580e-02 |
| sibling `sea-lr_s*` | 1.4279e-04 | 3.8453e-02 |
| `sea-sr_s*` | 1.4985e-04 | 4.3375e-02 |
| `deepmd_s*` (short range only) | 1.8351e-04 | 4.9210e-02 |

The setting effect is the answer to the mission's question, and it is large and
clean: against the legacy `deepmd-les` arm this arm is **0.9200 in force and
0.8333 in energy**, and the two replicates' ranges are disjoint on both
(energy 1.544-1.616e-04 against 1.717-2.075e-04; force 4.149-4.238e-02 against
4.346-4.770e-02). The poor legacy score was the `[240, 240, 240]` fitter at rcut
6.0 with the self-term set the other way, not the LES code path - which is
consistent with the implementation half being bit-identical across the pod's stale
and fixed trees.

The residual **implementation effect against `sea-lr_s*` is 1.0905 in force and
1.1066 in energy**, ranges just disjoint on both. The honest reading at n=2:
replicate pairs in this campaign scatter by 1.00-1.12x on force (this arm's own
A/B spread is 1.021x/1.046x), so a ~10% gap is at the edge of what two replicates
can resolve - the data cannot call it a defect, and cannot rule out a ~10%
deficit either. The declared config differences that could produce one are all on
the training side: deepmd hardcodes Adam's betas to (0.9, 0.999) where cace sets
(0.99, 0.999), and the phase-0 blocks' LR schedule origin differs by 5 epochs (see
`gen_inputs.py`); the architecture-side deviations are parameterisations the
training absorbs (`add_linear_nn`, `bias`, `n_out`, the Ewald prefactor).

Two footnotes that change how these numbers should be quoted:

- **The `lcurve` energy column is not comparable across families.** deepmd logs
  the natoms-weighted mean of each validation batch's own metric
  (`pt/train/training.py:952-956`), and every arm validates with `batch_size: 1`,
  so a one-frame energy "RMSE" is that frame's absolute error and the logged mean
  is a mean **absolute** error, systematically below the pooled RMS
  (measured here: 1.3251e-04 against 1.6159e-04, ratio 0.820). The tail-window
  comparison in `analysis/compare_les_sched.py` therefore labels its energy
  column not cross-family comparable; the table above sidesteps the logging
  difference by evaluating every arm by one definition. On the `lcurve` footing
  this arm's energy looked best of all ten runs; on the pooled footing it is
  **worse than `sea-lr_s*`** - which is the corrected statement.
- **The campaign's worst force is the short-range `deepmd_s*` arm** (4.9210e-02),
  not the legacy `deepmd-les_s*` (4.5580e-02). The legacy les arm is the worst on
  energy. Both are beaten by this arm on force, and this arm also beats its
  short-range sibling `sea-sr_s*` on force (0.9668, disjoint) while being
  1.0544x its energy (overlapping ranges, i.e. within noise).

## Running

    cd campaign_water_interface/deepmd
    python run_chain.py --arm deepmd-les-cace-sched --rep A --rep B --dp ~/venv310/bin/dp

which runs the eight blocks in order, each in its own `sN/` directory, chaining with
`--init-model` from the previous block's last checkpoint.
`--init-model` rather than `--restart` is deliberate: it resets deepmd's step counter so
each block's `learning_rate` schedule really starts at local step 0, and rebuilds the
optimiser, which is what cace does at each of its five fresh phase-0 tasks.
cace does *not* rebuild the optimiser at the later phase boundaries, where it keeps it;
those Adam moment resets are the one place this chain is looser than cace.

`chain.sh` is the same eight commands without the timing wrapper.

The leftover deviations from cace's optimiser, both unavoidable: deepmd hardcodes Adam's
betas to (0.9, 0.999) where cace sets (0.99, 0.999), and `opt_type` is not reachable.
One schedule deviation is declared in `gen_inputs.py`: deepmd's scheduler origin is the end
of its warmup (`pt/train/training.py:664`'s `warm_up_linear` returns
`lr_exp.value(step - warmup_steps) / start_lr`), while cace's `StepLR` counts from the
task's own epoch 0, so a phase-0 block halves 5
epochs later than cace's does - 5 of 40 epochs, and the continued blocks carry
`warmup_steps: 0` and are exact.

## Scoring

`segments.json` carries each block's `global_offset`, which is what makes a chained
checkpoint comparable to a single-run arm: deepmd's checkpoint number counts batches, and
cace's epoch counter is the same count over 450.
The evaluator is `campaign_fastlearn/deepmd/score_sea_chain.py`, which scores each block on
its own and relabels its step to the global one.
For this campaign the validation data are `data/water-interface/valid` (the 50-frame
held-out split), and the metrics of record are the campaign's usual ones -
`rmse_e_peratom`, `rmse_f`, the long-range force decomposition, and parameter count.

The `hybrid_ener` arms cannot be scored by the `desc_bridging/` tools: `check_bec.load_model`
accepts only `deepmd_cace` checkpoints, and these chained checkpoints carry no `config`
key, which `eval_vasp_bec.lr_weight_of` requires.
Born effective charges for this arm therefore need the local `les` path, not the campaign's
cace-side BEC script.

## Status

Finished 2026-10-07, both replicates, 8 blocks / 225,000 steps each, fp32, on the
pod (`cpod-1vg1a4d7wa0n`) running the patched tree.
Whole-chain cost was **0.09377 s/batch (A) and 0.09369 s/batch (B)**, about 5.9 h
per replicate and 11.8 h for both.
That is ~6.6% over the sibling `sea-lr_sA`'s 0.08796 and ~7.3% over `sea-sr_sA`'s
0.08734 on the same data and budget - the residual cost of the long-range channel
once the descriptor is no longer computed twice, against the legacy
`deepmd-les_sB`'s 0.11421 s/batch on this data (a 17.5% premium, confounded by its
`[240, 240, 240]` fitter at rcut 6.0, so corroborating rather than isolating the
duplicate call).
`campaign_water_interface/pod/fix_evidence/ab_descriptor_reuse.log` records both
cost measurements and the bit-identical A/B.

Scores and where they live: the table above is `analysis/compare_pooled.py`
(output in `analysis/data/compare_pooled.txt`); its deepmd-family inputs are from
`analysis/decompose_final.py` (`analysis/data/valid_pooled_deepmd.tsv`), taken on
the final checkpoint of each arm; the tail-window view of the same arms is
`analysis/compare_les_sched.py` (`analysis/data/compare_les_sched.txt`), and it
agrees with the pooled one on every ratio. The step sweep for this arm is
`analysis/data/valid_sweep_les.tsv` and the LR decomposition
`analysis/data/lr_mechanism_les.tsv` (at the trained `lr_weight = 0.02` the
long-range share of force is 5.15%, inside the cace family's 5.1-5.5%; the raw
`w=1` branch is 0.9129, i.e. the Ewald force is large and the SR net nearly
cancels it).

The pod is Postpay and must be stopped once the run is home. On 2026-10-07 it was
still answering SSH (`Exceeded MaxStartups`, i.e. sshd up; `hpc_compshare.py`'s
API calls were rejected with `Signature VerifyAC Error` 3/3, so the stop had to go
through the console).
