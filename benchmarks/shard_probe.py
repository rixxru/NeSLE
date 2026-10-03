"""Measure aggregate throughput when sharding envs across GPUs via processes.

Run one instance per GPU with CUDA_VISIBLE_DEVICES pinned; the parent sums the
per-process rates. This is the decision-relevant number: does a second RTX 4090
buy anything, or is the box host-bound?
"""
from __future__ import annotations

import argparse
import json
import time

import numpy

import nesle

ROM = r"C:\games\nes\Contra (U) [T-Rus uBAH009 (12.11.2016)].nes"
STATE = r"C:\games\nes\fceux_rl\curriculum\dense100_s0\ckpt\s0f10100.fcs"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, required=True)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    env = nesle.make_vec(
        ROM, num_envs=args.envs, frameskip=4, backend="cuda", device="cuda",
        reset_state_path=STATE, observation_mode="ram",
    )
    env.reset()
    actions = numpy.zeros(args.envs, dtype=numpy.int64)
    for _ in range(args.warmup):
        env.step(actions)

    t0 = time.perf_counter()
    for _ in range(args.iters):
        env.step(actions)
    dt = time.perf_counter() - t0

    print(json.dumps({
        "envs": args.envs,
        "seconds": round(dt, 4),
        "env_steps_per_s": round(args.envs * args.iters / dt, 1),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())