"""The run-directory naming convention, shared by every analysis tool.

    run_<model>_s<rep>            decay_steps 5000  (the historical default)
    run_<model>_s<rep>_d<decay>   any other decay_steps

The default schedule carries no suffix on purpose: `sweep_valid.tsv`,
`bias_frames_*.tsv` and the first half of `LES_analysis.ipynb` were all written
against those names, so they keep working untouched while a second schedule is
added alongside them.

The second axis exists because `LearningRateExp` (deepmd 3.1.2,
`dpmodel/utils/learning_rate.py`) makes the learning rate *piecewise constant*
rather than smoothly decaying:

    decay_rate = exp( log(stop_lr / start_lr) / (stop_steps / decay_steps) )
    lr(step)   = start_lr * decay_rate ** (step // decay_steps)

With start_lr 1e-3, stop_lr 3.51e-8 and stop_steps 10000, decay_steps 5000
gives 1e-3 up to step 4999 and then 5.921e-6 for the whole second half - a
single cliff, not a ramp. `decay_steps` is therefore not a decay rate but the
*length of the high-learning-rate phase*.

Keep this module the single place that knows the convention: `gen_extended.py`
writes the names, and `eval_full.py` / `diag_bias.py` parse them back.
"""
import re

DEFAULT_DECAY = 5000

_RUN_RE = re.compile(r"^run_(?P<model>.+)_s(?P<rep>[A-Za-z])(?:_d(?P<decay>\d+))?$")


def parse_run(name: str) -> dict:
    """Split a run directory name into its model / rep / decay parts.

    Raises ValueError on anything that does not match, rather than returning a
    half-parsed result: a silent mismatch here would mislabel a whole arm.
    """
    m = _RUN_RE.match(name)
    if m is None:
        raise ValueError(
            f"run directory {name!r} does not match run_<model>_s<rep>[_d<decay>]"
        )
    return {
        "run": name,
        "model": m.group("model"),
        "rep": m.group("rep"),
        "decay": int(m.group("decay")) if m.group("decay") else DEFAULT_DECAY,
    }


def run_tag(model: str, rep: str, decay: int = DEFAULT_DECAY) -> str:
    """Inverse of ``parse_run``: the run directory name for a model/rep/decay."""
    suffix = "" if decay == DEFAULT_DECAY else f"_d{decay}"
    return f"{model}_s{rep}{suffix}"
