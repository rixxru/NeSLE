"""Can FCEUX 2.6 read a save state that the CUDA batch produced?

The same question was answered for the CPU console's writer. This is the one that
matters in practice, because curriculum checkpoints come out of GPU training runs:
the file has to be loadable by FCEUX, not just by NeSLE.

Method: run a batch, save env 0, hand the file to FCEUX through the Lua bridge, have
it re-save, and compare every field the two formats share. Then run forward from the
same file in both and compare RAM, against FCEUX's own state as a control - the point
is to show a GPU-written state is no worse than an FCEUX-written one.

Not part of the test suite; it needs fceux and the bridge. Run it directly.
"""

from __future__ import annotations

import glob
import os
import shutil
import sys
from pathlib import Path


sys.path.insert(0, os.environ.get("NESLE_SRC", "src"))
sys.path.insert(0, os.environ.get("NESLE_FCEUX", r"C:\games\nes\fceux_rl"))

import numpy as np  # noqa: E402
from fceux_env import FceuxEnv  # noqa: E402

from nesle import _core, _cuda_core  # noqa: E402

ROM_DIR = Path(r"C:\games\nes")
ROM = os.environ.get("NESLE_ROM", r"C:\games\nes\Contra (U) [T-Rus uBAH009 (12.11.2016)].nes")
OUT = Path(os.environ.get("NESLE_TMP", "."))
N = 60


def fields(blob):
    s = _cuda_core.parse_fcs_state(blob)
    return {k: (bytes(v) if isinstance(v, (bytes, bytearray)) else v) for k, v in s.items()}


def agreement(fe, ne, off):
    total = count = 0
    for i in range(len(fe)):
        j = i + off
        if j < 0 or j >= len(ne):
            continue
        for q in range(2048):
            if fe[i][q] == ne[j][q]:
                total += 1
        count += 2048
    return total / count


def main() -> int:
    rom_bytes = Path(ROM).read_bytes()
    real = sorted(glob.glob(os.path.join(os.environ.get("NESLE_FCEUX", r"C:\games\nes\fceux_rl"), "curriculum", "**", "*.fcs"), recursive=True))[0]
    real_bytes = Path(real).read_bytes()

    # A state the GPU produced, after some real stepping.
    batch = _cuda_core.CudaBatch(1, 1, rom_bytes, real_bytes)
    for _ in range(20):
        batch.step(np.zeros(1, dtype=np.uint8), render_frame=False, copy_obs=False)
    gpu_state = batch.save_state(0)
    gpu_path = OUT / "gpu_written.fcs"
    gpu_path.write_bytes(gpu_state)
    print(f"GPU wrote      : {len(gpu_state)} B  magic {gpu_state[:4]!r}")
    print(f"FCEUX's own   : {len(real_bytes)} B  magic {real_bytes[:4]!r}")

    # A CPU-written state for comparison, from the same starting point.
    console = _core.NativeConsole(rom_bytes)
    console.load_state(real_bytes)
    for _ in range(20):
        console.step(0, 1, 30000)
    cpu_path = OUT / "cpu_written.fcs"
    cpu_path.write_bytes(console.save_state())
    print(f"CPU wrote      : {cpu_path.stat().st_size} B")

    env = FceuxEnv(ROM, tag="gpushk", verbose=False)
    try:
        env.spawn(timeout=60)
        for tag, path in (
            ("gpu_state", gpu_path),
            ("cpu_state", cpu_path),
            ("fceux_orig", Path(real)),
        ):
            shutil.copyfile(path, os.path.join(env.ckpt, f"{tag}.fcs"))
            env.load_state(tag, ram2=False)
            resaved = OUT / f"{tag}_by_fceux.fcs"
            env.save_state(f"{tag}_rt")
            shutil.copyfile(os.path.join(env.ckpt, f"{tag}_rt.fcs"), resaved)
            got = resaved.read_bytes()
            want = fields(path.read_bytes())
            have = fields(got)
            bad = [k for k in want if want[k] != have.get(k, "<missing>")]
            # Note: the bridge runs RENDER_FRAMES=1 after a LOAD, so what FCEUX
            # re-saves is one frame past the file. Fields that do not change within a
            # frame therefore compare exactly, and the rest are expected to drift -
            # the behavioural RAM comparison below is the real check.
            stable = [k for k in ("chr_ram", "oam", "nametable_ram", "palette_ram", "prg_ram")]
            stable_ok = all(want[k] == have.get(k) for k in stable if k in want)
            print(
                f"\nFCEUX re-saved {tag}: {len(got)} B, "
                f"{len(want) - len(bad)}/{len(want)} fields identical, "
                f"per-frame memory blocks identical: {stable_ok}"
            )
            for k in bad:
                if k in stable:
                    print(f"    {k}: DIFFER  <-- this one should not have moved")

        # Behavioural: run N frames from each file in both engines.
        print(f"\n{N} idle frames from the same state, best frame offset:")
        results = {}
        for tag, path in (
            ("gpu_state", gpu_path),
            ("cpu_state", cpu_path),
            ("fceux_orig", Path(real)),
        ):
            blob = path.read_bytes()
            env.load_state(tag, ram2=False)
            fe = [env.ram()]
            for _ in range(N):
                env.idle(1, shot=0, ram=1, ram2=False)
                fe.append(env.ram())
            ne = []
            if tag == "gpu_state":
                b = _cuda_core.CudaBatch(1, 1, rom_bytes, blob)
                for _ in range(N + 1):
                    ne.append(bytes(b.ram()[0]))
                    b.step(np.zeros(1, dtype=np.uint8), render_frame=False, copy_obs=False)
            else:
                c = _core.NativeConsole(rom_bytes)
                c.load_state(blob)
                for _ in range(N + 1):
                    ne.append(bytes(c.ram()))
                    c.step(0, 1, 30000)
            scores = {o: agreement(fe, ne, o) for o in (-1, 0, 1)}
            best = max(scores, key=scores.get)
            results[tag] = scores[best]
            print(
                f"  {tag:<11} -1:{scores[-1]*100:5.1f}%  0:{scores[0]*100:5.1f}%  "
                f"+1:{scores[1]*100:5.1f}%   best {best:+d}"
            )
        print(
            f"\nGPU-written state vs FCEUX's own, RAM agreement: "
            f"{results['gpu_state']*100:.1f}% vs {results['fceux_orig']*100:.1f}%"
        )
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
