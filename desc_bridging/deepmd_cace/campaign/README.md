# Campaign runner

Run the local smoke test before sending a formal campaign to the HPC GPU:

```bash
cd deepmd-les
PYTHONPATH=desc_bridging CUDA_VISIBLE_DEVICES='' \
  python -m deepmd_cace.campaign.run \
  --mode smoke \
  --dataset DeePMD-kit-FastLearn/cace_compare_e2e/water \
  --output /tmp/deepmd-cace-smoke \
  --device cpu
```

The runner creates a deterministic eight-frame training set and two-frame
validation set, then runs check-only and one epoch per configured block for
the SR and LR bridge arms.  Smoke output is wiring validation, not a model
comparison.

On the HPC host, run the smoke command again after installing the environment
with `hpc_setup.sh`.  Only then launch the formal GPU run:

```bash
cd deepmd-les
PYTHON=/root/venv310/bin/python \
  desc_bridging/deepmd_cace/campaign/run_hpc.sh \
  DeePMD-kit-FastLearn/cace_compare_e2e/water \
  DeePMD-kit-FastLearn/campaign_hpc/formal
```

`run_hpc.sh` does not provision a pod or start a remote instance.  Use
`hpc_compshare.py ensure --yes` outside this script when the configured pod
needs to be started.  The formal runner keeps each arm in a separate output
directory so LES logs and checkpoints cannot collide.
