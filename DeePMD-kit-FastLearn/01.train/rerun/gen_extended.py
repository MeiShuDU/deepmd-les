"""Generate the extended (10k-step, 2-seed) matched run configs.

Reproduce the comparison harness from run_all.sh, but at the old 10k-step
convergence level and with a second replicate so the ranking can be
distinguished from run-to-run noise.

Design of a replicate (identical across the three models, which is what makes
the comparison matched):
  - replicate A: descriptor/fitting seed 1/1, training seed 10  (reuses the
    2000-step runs verbatim, so those curves extend continuously)
  - replicate B: descriptor/fitting seed 2/2, training seed 20  (different NN
    init *and* different data order)

Each run gets its own subdirectory so checkpoints / lcurve / les.log never
clobber each other. Data paths are absolute so the configs are CWD-independent.

A second axis, the learning-rate schedule, is switched with ``--decay``. Only
``learning_rate.decay_steps`` changes; every other setting is held fixed, so a
decay pair is a matched comparison of "how long the learning rate stays high".
The default decay 5000 keeps the original run names (``run_*_s{A,B}``); any
other value appends a suffix (``run_*_s{A,B}_d7500``) so both schedules coexist.

Why decay_steps matters more than it looks (deepmd 3.1.2,
``dpmodel/utils/learning_rate.py:LearningRateExp``): the schedule is

    decay_rate = exp( log(stop_lr / start_lr) / (stop_steps / decay_steps) )
    lr(step)   = start_lr * decay_rate ** (step // decay_steps)

so the learning rate is *piecewise constant*, not smoothly decaying. With
start_lr 1e-3, stop_lr 3.51e-8 and stop_steps 10000 this gives

    decay_steps 5000 -> lr = 1e-3     for steps 0..4999
                       lr = 5.921e-6 for steps 5000..9999
    decay_steps 7500 -> lr = 1e-3     for steps 0..7499
                       lr = 4.563e-7 for steps 7500..9999

i.e. one cliff rather than a ramp, and moving the cliff to 7500 buys 2500 extra
steps at the full 1e-3 but ends the run at a 13x smaller learning rate.
"""
import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/root/app/deepmd-les/deepmd-les/DeePMD-kit-FastLearn/data"

# lr schedule of the original 10k run (parent input.yaml), kept so the
# converged numbers are directly comparable to the old lcurve_q.out.
LR = {"start_lr": "0.001", "stop_lr": "3.51e-8"}
DEFAULT_DECAY = 5000

MODELS = {
    "ordinary": None,  # standard model, no LES
    "hybrid_q": {"use_atomwise": True, "sigma": 1.0, "dl": 1.5,
                 "use_fixed_atomic_charges": False},
    "hybrid_fixed": {"use_atomwise": True, "sigma": 1.0, "dl": 1.5,
                     "use_fixed_atomic_charges": True},
}

REPLICATES = [
    {"tag": "A", "desc_seed": 1, "fit_seed": 1, "train_seed": 10},
    {"tag": "B", "desc_seed": 2, "fit_seed": 2, "train_seed": 20},
]

TEMPLATE = """\
model:
  type: {mtype}
  type_map: ["O", "H"]
  descriptor:
    type: se_a
    sel: [46, 92]
    rcut_smth: 0.50
    rcut: 6.00
    neuron: [25, 50, 100]
    axis_neuron: 16
    resnet_dt: false
    seed: {desc_seed}
  fitting_net:
    neuron: [240, 240, 240]
    resnet_dt: true
    seed: {fit_seed}
{les_block}
learning_rate:
  type: exp
  decay_steps: {decay_steps}
  start_lr: {start_lr}
  stop_lr: {stop_lr}

loss:
  type: ener
  start_pref_e: 0.02
  limit_pref_e: 1.0
  start_pref_f: 1000.0
  limit_pref_f: 1.0
  start_pref_v: 0.0
  limit_pref_v: 0.0

training:
  training_data:
    systems: ["{data}/data_0/", "{data}/data_1", "{data}/data_2/"]
    batch_size: auto
  validation_data:
    systems: [{data}/data_3]
    batch_size: 1
    numb_btch: 3
  numb_steps: 10000
  seed: {train_seed}
  disp_file: "lcurve_{tag}.out"
  disp_freq: 100
  save_freq: 1000
"""


def les_block(params):
    if params is None:
        return ""
    lines = ["  les_params:"]
    for k, v in params.items():
        lines.append(f"    {k}: {str(v).lower() if isinstance(v, bool) else v}")
    # verbose/log_freq are harness-only knobs, same for every LES run.
    lines.append("    verbose: true")
    lines.append("    log_freq: 100")
    return "\n".join(lines) + "\n"


def run_tag(name, rep_tag, decay_steps):
    """Run-directory basename. The default decay keeps the historical name."""
    suffix = "" if decay_steps == DEFAULT_DECAY else f"_d{decay_steps}"
    return f"{name}_s{rep_tag}{suffix}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--decay",
        type=int,
        default=DEFAULT_DECAY,
        help=f"learning_rate.decay_steps (default {DEFAULT_DECAY}; any other "
        "value writes suffixed run dirs and leaves the default set alone)",
    )
    args = ap.parse_args()
    decay_steps = args.decay
    for rep in REPLICATES:
        for name, les in MODELS.items():
            tag = run_tag(name, rep["tag"], decay_steps)
            rundir = os.path.join(HERE, "extended", f"run_{tag}")
            os.makedirs(rundir, exist_ok=True)
            body = TEMPLATE.format(
                mtype="hybrid_ener" if les else "standard",
                desc_seed=rep["desc_seed"],
                fit_seed=rep["fit_seed"],
                les_block=les_block(les),
                decay_steps=decay_steps,
                start_lr=LR["start_lr"],
                stop_lr=LR["stop_lr"],
                data=DATA,
                train_seed=rep["train_seed"],
                tag=tag,
            )
            path = os.path.join(rundir, "input.yaml")
            with open(path, "w") as f:
                f.write(body)
            print("wrote", path)


if __name__ == "__main__":
    main()
