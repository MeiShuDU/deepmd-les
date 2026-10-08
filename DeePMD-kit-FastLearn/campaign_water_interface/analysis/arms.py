"""The eight arms of the water-interface campaign, as one list.

Four arms (short-range-only and long-range, each in two families) in two
replicates. The two families differ in *who trains*: the ``cace`` arms are cace's
own published scripts, the ``sea`` arms are the same cace training loop with
deepmd's ``se_a`` descriptor substituted at the input seam (``desc_bridging/``).

The two families agree on everything that makes their errors comparable, and
none of it is assumed here - ``score_arms.py`` checks it:

  * the same 500-frame benchmark and the same 90/10 split. ``data/water-interface``
    was written by calling cace's own ``random_train_valid_split(valid_fraction=
    0.1, seed=1)``, so the 50 validation frames are frame-for-frame the mobile
    frames the cace arms also validate on;
  * the same supervised target: the residual E_frame - sum_Z ref[Z], references
    {H: -187.42397696905275, O: -93.71198848452647}. The deepmd systems carry it
    in energy.npy directly; cace's AtomicData.from_atoms subtracts it. This is why
    ``val_e/atom_rmse`` means the same thing in all eight logs;
  * the same validation metric code path - the sea arms run cace's own
    ``TrainingTask`` and ``Metrics``, so the logs are directly stackable.

They do NOT agree on the step budget, and the arms were not run to correct for it:
cace's short-range arm stops after its 7th block (400 epochs) because that is the
script the authors published, while the sea runner was given the long-range
schedule for both arms (8 blocks, 500 epochs). So ``sea-sr`` saw 225,000 steps
against ``cace-sr``'s 180,000. Any sea-sr-versus-cace-sr comparison must carry
that, and the curves are cut at a common epoch wherever it would matter.
"""
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent

FAMILY_ROOT = {
    "cace": CAMPAIGN / "cace" / "runs",
    "sea": CAMPAIGN / "deepmd" / "runs",
}
# the run directories predate the arm naming: sea-sr lives in sea_sA/sea_sB and
# only the long-range arm spells out its topology in the directory name
RUN_DIR_NAME = {
    ("cace", "sr"): "cace-sr_s{}",
    ("cace", "lr"): "cace-lr_s{}",
    ("sea", "sr"): "sea-sr_s{}",
    ("sea", "lr"): "sea-lr_s{}",
}
TOPOLOGY = {"sr": "short-range only", "lr": "long-range"}
VALID_SYSTEM = CAMPAIGN / "data" / "water-interface" / "valid"
BENCHMARK_XYZ = CAMPAIGN / "cace" / "data" / "slab-fps-n-500.xyz"


@dataclass(frozen=True)
class Arm:
    family: str          # "cace" or "sea"
    topology: str        # "sr" or "lr"
    rep: str             # "A" or "B"

    @property
    def arm(self):
        """The arm without its replicate: ``cace-sr``, ``sea-lr``."""
        return f"{self.family}-{self.topology}"

    @property
    def arm_id(self):
        """The arm and its replicate: ``cace-sr_A``. Unique in the campaign."""
        return f"{self.arm}_{self.rep}"

    @property
    def run_dir(self):
        return FAMILY_ROOT[self.family] / RUN_DIR_NAME[(self.family, self.topology)].format(self.rep)

    @property
    def best_model(self):
        return self.run_dir / "best_model.pth"

    @property
    def train_log(self):
        return self.run_dir / "train.log"

    @property
    def timing(self):
        return self.run_dir / "timing.json"


ARMS = [
    Arm(family, topology, rep)
    for topology in ("sr", "lr")
    for family in ("cace", "sea")
    for rep in ("A", "B")
]

BY_ARM = {arm.arm_id: arm for arm in ARMS}


def missing():
    """Arms whose downloaded results are not all on disk yet."""
    return [arm.arm_id for arm in ARMS
            if not (arm.best_model.is_file() and arm.train_log.is_file())]


if __name__ == "__main__":
    for arm in ARMS:
        have = ",".join(name for name, path in
                        (("best_model.pth", arm.best_model), ("train.log", arm.train_log),
                         ("timing.json", arm.timing)) if path.is_file())
        print(f"{arm.arm_id:12s} {str(arm.run_dir).split('campaign_water_interface/')[-1]:28s} {have}")
