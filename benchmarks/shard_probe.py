"""Measure aggregate throughput when sharding envs across GPUs via processes.

This is the decision-relevant question for a multi-GPU box: does the second RTX
4090 buy anything, or is the machine host-bound on the CPU side of a step?

The batch kernel already runs every env on one device, so a second GPU can only
be used by running a second *process* with its own shard of envs and summing the
rates. That has two costs this script exists to measure honestly:

  * the per-step host work (action upload, reward, done handling) is repeated in
    each process instead of being shared, and
  * the aggregate only wins if the host has spare cores, so a bad env/shard
    ratio can make two processes *slower* than one.

Modes:
  step        nesle.make_vec(...).step() - the full SB3-facing path, including
              host-side action lookup and per-step reward/done bookkeeping
  step_device CudaBatch.step_device with a pre-allocated device action array and
              no observation copy - what a native PPO loop actually pays for
  step_reward NesleVecEnv.step_reward - no RGB rendering or copy

Example:
    python benchmarks/shard_probe.py --mode step_device --envs 2048 4096 --shards 1 2
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import numpy

import nesle

ROM = r"C:\games\nes\Contra (U) [T-Rus uBAH009 (12.11.2016)].nes"
STATE = r"C:\games\nes\fceux_rl\curriculum\dense100_s0\ckpt\s0f10100.fcs"
FRAMESKIP = 4


def build_env(envs: int, mode: str):
    """Return (runner, actions, batch). `batch` is only set for step_device."""
    env = nesle.make_vec(
        ROM,
        num_envs=envs,
        frameskip=FRAMESKIP,
        backend="cuda",
        device="cuda",
        reset_state_path=STATE,
        observation_mode="ram",
    )
    env.reset()
    if mode == "step_device":
        # step_device takes a device array, not a host one, so the benchmark has
        # to hold its action buffer on the GPU - which is the point: it removes
        # the per-step host upload an SB3-style loop pays.
        try:
            import torch
        except ImportError:
            raise SystemExit("--mode step_device needs torch for a device action array")
        if not torch.cuda.is_available():
            raise SystemExit("--mode step_device needs a working CUDA device")
        actions = torch.zeros(envs, dtype=torch.uint8, device="cuda")
        return env._cuda_batch, actions, env._cuda_batch
    return env, numpy.zeros(envs, dtype=numpy.int64), None


def run_loop(envs: int, mode: str, iters: int, warmup: int) -> dict:
    handle, actions, batch = build_env(envs, mode)

    def one_step(sync: bool = True) -> None:
        if mode == "step_device":
            batch.step_device(actions, auto_reset=False, synchronize=sync)
        elif mode == "step_reward":
            handle.step_reward(actions)
        else:
            handle.step(actions)

    for _ in range(warmup):
        one_step()
    if mode == "step_device":
        # Drain the warmup queue before timing anything.
        batch.step_device(actions, auto_reset=False, synchronize=True)

    if mode == "step_device":
        # step_device is asynchronous: timing each call separately would measure
        # kernel *queueing*, not execution, and report a meaningless rate. Queue
        # the whole run, then synchronize once and charge that wait to the loop.
        t0 = time.perf_counter()
        for _ in range(iters):
            batch.step_device(actions, auto_reset=False, synchronize=False)
        batch.step_device(actions, auto_reset=False, synchronize=True)
        total = time.perf_counter() - t0
        step_ms = total / iters * 1e3
        p10 = p90 = step_ms
    else:
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            one_step()
            samples.append(time.perf_counter() - t0)
        total = sum(samples)
        step_ms = statistics.median(samples) * 1e3
        ordered = sorted(samples)
        p10 = ordered[max(0, int(0.1 * iters) - 1)] * 1e3
        p90 = ordered[min(iters - 1, int(0.9 * iters))] * 1e3

    return {
        "envs": envs,
        "mode": mode,
        "iters": iters,
        "env_steps_per_s": round(envs * iters / total, 1),
        "step_ms": round(step_ms, 4),
        "step_ms_p10": round(p10, 4),
        "step_ms_p90": round(p90, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, nargs="+", required=True)
    ap.add_argument("--shards", type=int, nargs="+", required=True)
    ap.add_argument("--mode", default="step_device", choices=["step", "step_device", "step_reward"])
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--devices", default=None, help="comma-separated GPU ids; default 0..shards-1")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if os.environ.get("NESLE_SHARD_CHILD") != "1":
        return orchestrate(args)

    # Child: one shard, one pinned device.
    print(json.dumps(run_loop(args.envs[0], args.mode, args.iters, args.warmup)))
    return 0


def orchestrate(args) -> int:
    device_ids = (
        [d.strip() for d in args.devices.split(",")] if args.devices else None
    )
    rows = []
    baseline = {}
    for shards in args.shards:
        for envs in args.envs:
            total_envs = envs * shards
            devices = (
                device_ids[:shards] if device_ids else [str(i) for i in range(shards)]
            )
            procs = []
            for device in devices:
                env_vars = dict(os.environ)
                env_vars["CUDA_VISIBLE_DEVICES"] = device
                env_vars["NESLE_SHARD_CHILD"] = "1"
                procs.append(
                    (
                        device,
                        subprocess.Popen(
                            [sys.executable, __file__,
                             "--envs", str(envs), "--shards", "1",
                             "--mode", args.mode, "--iters", str(args.iters),
                             "--warmup", str(args.warmup)],
                            env=env_vars,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                        ),
                    )
                )
            rates = []
            per_device = []
            for device, proc in procs:
                out, err = proc.communicate(timeout=1800)
                if proc.returncode != 0:
                    sys.stderr.write(f"shard on GPU {device} failed:\n{err}\n")
                    return 1
                result = json.loads(out.strip().splitlines()[-1])
                rates.append(result["env_steps_per_s"])
                per_device.append({"device": device, **result})
            aggregate = round(sum(rates), 1)
            if shards == 1:
                # Key by *total* env count so a multi-shard run is only ever
                # compared against a single-GPU run of the same total size.
                baseline[total_envs] = aggregate
            row = {
                "shards": shards,
                "envs_per_shard": envs,
                "total_envs": total_envs,
                "devices": devices,
                "aggregate_env_steps_per_s": aggregate,
                "per_shard_env_steps_per_s": rates,
                "one_gpu_same_total_env_steps_per_s": baseline.get(total_envs),
                "speedup_vs_one_gpu_same_total": (
                    round(aggregate / baseline[total_envs], 3)
                    if total_envs in baseline and shards > 1
                    else None
                ),
            }
            rows.append(row)
            print(json.dumps(row))
            print(flush=True)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump({"mode": args.mode, "frameskip": FRAMESKIP, "rows": rows}, handle, indent=2)
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
