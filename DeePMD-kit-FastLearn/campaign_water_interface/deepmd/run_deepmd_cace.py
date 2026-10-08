"""Run DeepMD-descriptor/CACE fits and report per-block training time.

Each replicate is one Python process. Its CACE optimizer is rebuilt between
fresh-task blocks and retained across loss-only blocks, so the schedule cannot
be split into independent DeepMD-style checkpoint chains without changing it.
The trainer writes per-fit timing to ``blocks.json``; this runner adds process
wall time and a ``timing.json`` summary beside each replicate's ``input.json``.

Examples::

    python run_deepmd_cace.py --arm sea-lr --rep A --dry-run
    python run_deepmd_cace.py --arm sea-lr --rep A --cpu --smoke
    python run_deepmd_cace.py --arm sea-lr --rep A
    python run_deepmd_cace.py --precision float64 --arm sea-lr --rep B
"""

import argparse
import datetime as dt
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(HERE, "runs")
DESC_BRIDGING = os.path.abspath(os.path.join(HERE, "../../..", "desc_bridging"))


def stamp():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path):
    try:
        with open(path, encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return None


def write_json(path, record):
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def expected_final_checkpoint(config):
    blocks = config["training"].get("blocks", [])
    if not blocks:
        return "model.pth"
    return blocks[-1].get("checkpoint") or "model.pth"


def make_env(args):
    env = dict(os.environ)
    if args.cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["DEEPMD_CACE_DEVICE"] = "cpu"
    else:
        if env.get("CUDA_VISIBLE_DEVICES") == "":
            env.pop("CUDA_VISIBLE_DEVICES")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env.pop("DEEPMD_CACE_DEVICE", None)
    env["DP_INTERFACE_PREC"] = "low" if args.precision == "float32" else "high"
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = DESC_BRIDGING + (os.pathsep + pythonpath if pythonpath else "")
    return env


def adapt_block(record, returncode):
    out = []
    for block in record.get("blocks", []):
        out.append({
            "n": block.get("n"),
            "label": block.get("label"),
            "epochs": block.get("epochs"),
            "num_steps": block.get("num_steps"),
            "seconds": block.get("seconds"),
            "s_per_batch": block.get("s_per_batch"),
            "started": block.get("started"),
            "ended": block.get("ended"),
            "returncode": returncode,
            "ok": bool(block.get("ok", True)) and returncode == 0,
            "energy_weight": block.get("energy_weight"),
            "fresh_task": block.get("fresh_task"),
        })
    return out


def finish_timing(timing, out_path, dry_run=False):
    completed = [block for block in timing["blocks"] if block.get("seconds") is not None]
    if completed:
        timing["total_seconds"] = round(sum(block["seconds"] for block in completed), 2)
        timing["total_steps"] = sum(block.get("num_steps") or 0 for block in completed)
        if timing["total_steps"]:
            timing["s_per_batch"] = round(
                timing["total_seconds"] / timing["total_steps"], 6
            )
    timing["ended"] = stamp()
    if not dry_run:
        write_json(out_path, timing)
    return timing


def run_replicate(root, arm, tag, args, env):
    run_dir = os.path.join(root, f"{arm}_s{tag}")
    config_path = os.path.join(run_dir, "input.json")
    blocks_path = os.path.join(run_dir, "blocks.json")
    log_path = os.path.join(run_dir, "train.log")
    timing_path = os.path.join(run_dir, "timing.json")
    config = load_json(config_path)
    if config is None:
        print(f"{arm}_s{tag}: REFUSING - missing or invalid {config_path}", flush=True)
        return None

    final_name = expected_final_checkpoint(config)
    training_config = config.get("training", {})
    prior = load_json(blocks_path)
    if prior and prior.get("finished") and not prior.get("smoke"):
        if not os.path.isfile(os.path.join(run_dir, final_name)):
            print(f"{arm}_s{tag}: REFUSING - finished blocks but missing {final_name}",
                  flush=True)
            return None
        print(f"{arm}_s{tag}: already complete; reusing timing", flush=True)
        timing = load_json(timing_path) or {
            "arm": arm, "replicate": tag, "host": socket.gethostname(),
            "blocks": adapt_block(prior, 0), "device": prior.get("device"),
            "precision": prior.get("precision"), "smoke": False,
            "finished": True, "max_gpu_mb": prior.get("max_gpu_mb"),
            "steps_per_epoch": prior.get("steps_per_epoch"),
            "started": None, "wall_seconds": None, "aborted_at": None,
        }
        return finish_timing(timing, timing_path, args.dry_run)

    if prior and prior.get("blocks"):
        print(f"{arm}_s{tag}: WARNING - CACE optimizer state is not resumable; "
              "restarting the full schedule from block 1", flush=True)

    os.makedirs(run_dir, exist_ok=True)
    command = [args.python, "-m", "deepmd_cace", "input.json"]
    if args.smoke:
        command.append("--smoke")
    timing = {
        "arm": arm,
        "replicate": tag,
        "host": socket.gethostname(),
        "device": "cpu" if args.cpu else "gpu",
        "precision": args.precision,
        "smoke": bool(args.smoke),
        "started": stamp(),
        "ended": None,
        "wall_seconds": None,
        "blocks": [],
        "total_seconds": None,
        "total_steps": None,
        "s_per_batch": None,
        "steps_per_epoch": None,
        "max_gpu_mb": None,
        "finished": False,
        "aborted_at": None,
        "returncode": None,
        "preflight_error": None,
        "restarted_from_partial": bool(prior and prior.get("blocks")),
    }
    print(f"{arm}_s{tag}: {' '.join(shlex.quote(arg) for arg in command)} "
          f"[{timing['device']}, {args.precision}]", flush=True)
    if args.dry_run:
        return finish_timing(timing, timing_path, dry_run=True)

    configured_device = training_config.get("device", "auto")
    if not args.cpu and configured_device == "cuda":
        probe = subprocess.run(
            [args.python, "-c", "import torch; raise SystemExit(0 if torch.cuda.is_available() else 78)"],
            env=env,
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            details = (probe.stdout + probe.stderr).strip()
            timing["returncode"] = probe.returncode
            timing["preflight_error"] = "configured CUDA device is unavailable to PyTorch"
            timing["aborted_at"] = 0
            timing["cuda_probe_output"] = details[-4000:]
            print(f"{arm}_s{tag}: REFUSING - PyTorch CUDA preflight failed; "
                  "no training process was started", flush=True)
            if details:
                print(details, flush=True)
            return finish_timing(timing, timing_path)

    if prior and prior.get("blocks"):
        archived_blocks = os.path.join(
            run_dir, f"blocks.previous-{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}.json"
        )
        shutil.copy2(blocks_path, archived_blocks)
        print(f"{arm}_s{tag}: archived previous block timings to "
              f"{os.path.basename(archived_blocks)}", flush=True)

    write_json(timing_path, timing)

    started = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=run_dir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    timing["wall_seconds"] = round(time.perf_counter() - started, 2)
    timing["returncode"] = process.returncode
    result = load_json(blocks_path)
    if result is None:
        print(f"{arm}_s{tag}: trainer wrote no blocks.json; see {log_path}", flush=True)
        timing["aborted_at"] = 0
    else:
        timing["blocks"] = adapt_block(result, process.returncode)
        timing["steps_per_epoch"] = result.get("steps_per_epoch")
        timing["max_gpu_mb"] = result.get("max_gpu_mb")
        timing["finished"] = bool(result.get("finished")) and process.returncode == 0
        if process.returncode != 0:
            timing["aborted_at"] = len(timing["blocks"]) + 1
            print(f"{arm}_s{tag}: trainer exited {process.returncode}; see {log_path}",
                  flush=True)
        elif not args.smoke and timing["finished"]:
            final_path = os.path.join(run_dir, final_name)
            if not os.path.isfile(final_path):
                timing["finished"] = False
                timing["aborted_at"] = len(timing["blocks"])
                print(f"{arm}_s{tag}: final checkpoint {final_name} is missing", flush=True)
        elif args.smoke:
            print(f"{arm}_s{tag}: smoke complete; not a production model", flush=True)

    with open(log_path, encoding="utf-8") as log:
        tail = log.read().strip().splitlines()[-25:]
    if tail:
        print("\n".join("      | " + line for line in tail), flush=True)
    finish_timing(timing, timing_path)
    if timing["total_seconds"] is not None:
        print(f"  -> {timing['total_seconds'] / 60:.1f} min, "
              f"{timing['total_steps']} batches, "
              f"{timing['s_per_batch'] * 1000:.1f} ms/batch", flush=True)
    return timing


def summarize(records):
    done = [record for record in records if record and record.get("s_per_batch")]
    if not done:
        return
    print("\n" + "=" * 78)
    print(f"{'replicate':16s} {'blocks':>7s} {'batches':>9s} "
          f"{'minutes':>9s} {'ms/batch':>10s} {'peak GPU MB':>12s}")
    for record in done:
        print(f"{record['arm'] + '_s' + record['replicate']:16s} "
              f"{len(record['blocks']):7d} {record['total_steps']:9d} "
              f"{record['total_seconds'] / 60:9.1f} "
              f"{record['s_per_batch'] * 1000:10.1f} "
              f"{str(record.get('max_gpu_mb')):>12s}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help="directory containing sea-lr_sA/input.json and peers")
    parser.add_argument("--arm", action="append", default=None,
                        help="arm prefix, e.g. sea-lr; repeatable")
    parser.add_argument("--rep", action="append", default=None,
                        help="replicate tag, e.g. A or B; repeatable")
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter for deepmd_cace")
    parser.add_argument("--cpu", action="store_true",
                        help="hide CUDA and override the JSON device to CPU")
    parser.add_argument("--precision", choices=("float32", "float64"),
                        default="float32",
                        help="DeepMD interface precision; default matches CACE fp32")
    parser.add_argument("--smoke", action="store_true",
                        help="one epoch per fit block; plumbing only")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = os.path.abspath(args.root)
    env = make_env(args)
    arms = args.arm or ["sea-lr"]
    reps = args.rep or ["A", "B"]
    records = []
    for tag in reps:
        for arm in arms:
            record = run_replicate(root, arm, tag, args, env)
            records.append(record)
            if not args.dry_run and not args.smoke and (
                record is None
                or record.get("preflight_error")
                or record.get("returncode") not in (None, 0)
                or record.get("aborted_at") is not None
            ):
                print("stopping the chain after the first failed/aborted replicate",
                      flush=True)
                break
    summarize(records)
    if args.dry_run:
        return 0 if all(record is not None for record in records) else 1
    if args.smoke:
        return 0 if all(
            record and record.get("returncode") == 0 and record.get("blocks")
            for record in records
        ) else 1
    return 0 if all(record and record.get("finished") for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())