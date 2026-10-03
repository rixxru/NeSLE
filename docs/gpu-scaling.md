# Scaling beyond one-thread-per-env

Research notes toward a higher-throughput emulation core: what the current
design costs, what the literature says, and a ranked roadmap. Companion to
[design.md](design.md) ("the emulator inside the kernel").

## Where the design stands

One CUDA thread runs one complete NES (`cpp/src/cuda/kernels.cu:19-69`,
128 threads/block). CPU registers are cached in hardware registers for the
whole kernel (`load_cpu_state` once at entry). Measured throughput at the time
this note was written: ~29.5k env-steps/s at 4,096 envs on a GTX 1050 Ti;
~1.0M steps/s (no-copy) at 128 envs on an A100. **These are the pre-sprint
baseline.** The Tier 0 section below walks them to 180,437 on the same card;
current figures are in the [README](../README.md).

The same design choice has published precedent: NVIDIA's **CuLE** (CUDA Atari,
NeurIPS 2020) maps one thread to one Atari 2600 — whose 6507 CPU is nearly
the same silicon as the NES's 6502. Their assessment applies verbatim: a
1-to-1 thread↔emulator mapping "is not the most computationally efficient
way … but it makes the implementation relatively straightforward."

## The two costs, and which one is bigger

**Divergence.** Every lane runs an env at a different program counter, so the
151-entry decode table (`cpp/include/nesle/cpu.hpp:200-350`) and its
50-case operation switch (`cpu.hpp:590`) serializes over
the distinct opcodes present in each warp iteration, bus accesses take
different range branches (`batch_bus.cuh:148-292`), and per-frame events
(vblank snapshot copy, OAM DMA) hit lanes at different times as fully
serialized ~2.3 KB byte-loops. **However**: CuLE *measured* this on the same
ISA and found divergence costs ~30% in the worst case — painful, but nowhere
near the naive 32× serialization floor, because real game-state divergence
across a warp is partial. Anti-divergence work has a hard ceiling on its
payoff.

**Memory traffic.** This may be the larger cost and gets less attention:

- PPU bookkeeping scalars (`scanline`, `dot`, `status`, `ctrl`, `nmi_pending`,
  `frame`) are read-modify-written to **global memory ~6–10 times per emulated
  instruction** (`batch_ppu.cuh:126,193-198`, `batch_console.hpp:41`) — while
  CPU registers already show the fix: cache in registers, write back at kernel
  exit.
- Emulated CPU RAM is a contiguous 2 KB block per env (`state.cuh:146-153`),
  so warp lanes touching even the *same* NES address hit locations 2 KB apart:
  fully scattered, up to 32 transactions per warp access — and zero-page/stack
  (the 6502's hottest region) always takes this path.

## Ranked roadmap

### Tier 0 — quick wins, no redesign — **EXECUTED 2026-07-29, measured**

Three of the five items below were implemented and measured the same day
(GTX 1050 Ti, per-change A/B with the full test suite green at each step):

| Change | step @2048 envs | render @256 envs |
|---|---:|---:|
| (baseline, start of day) | 88.3 ms / 23.2k steps/s | 118.5 ms |
| + hybrid render dispatch (#2) | — | **41.4 ms (2.9×)** |
| + vectorized bulk copies (#3) | 80.6 ms / 25.4k (+9.5%) | — |
| + PPU hot-state register caching (#1) | **68.6 ms / 29.8k (+17%)** | — |
| **cumulative** | **+29% stepping** | **2.9× small-batch render** |

Effect on the official `benchmarks/gpu_vs_cpu.py` headline: 29,491 →
**35,488 env-steps/s at 4,096 envs (88× → 110× the CPU baseline)**. Honest
notes: the per-pixel render kernel loses to the serial one once the env count
alone saturates the GPU (321 ms vs 207 ms at 2,048 envs), so dispatch is
hybrid with the crossover at 512 envs; the refactored serial path also costs
~14% at 2,048 envs (207 → 237 ms) — accepted, as large-batch rendering serves
only RGB-observation training.

**Evening update — every remaining item resolved (all measured on the GTX
1050 Ti; behavior verified bit-exact via a fixed-seed training run whose
losses match pre-sprint to 4 decimals):**

| Item | Verdict | Stepping @2,048 envs after |
|---|---|---:|
| #5 hygiene (restrict, guard strip, merged frame_dot) | ✅ shipped, +10% | 92.8k steps/s |
| #4 zero page in shared memory | ❌ **rejected with data**: 1.9× regression on sm_61 — 32 KB/block collapses occupancy from 32 to 6 warps/SM; break-even needs ≤96 B/thread. Implementation preserved on its branch. | — |
| Tier 2: table-driven decode (+ forced render inlining) | ✅ shipped — the sprint's largest win (see below) | 84.4k steps/s |
| Tier 1: **lazy PPU settlement** (supersedes the kernel-split proposal below) | ✅ shipped, +9% | **100.8k steps/s** |

Official `benchmarks/gpu_vs_cpu.py` after the sprint: **180,437 env-steps/s
at 4,096 envs (576× CPU)** — 5.1× the morning's 35,488. A cautionary tale
worth keeping: the decode merge initially *slowed both render kernels 2.3×*
because the enlarged module blew nvcc's inlining heuristics — forcing
inlining on the render helper chain (`NESLE_CUDA_RENDER_INLINE`) fixed render
AND unlocked the decode change's full stepping value (57.6 → 24.3 ms in one
line). Watch for this whenever the module grows.

Original plan:

1. **Register-cache the hot PPU scalars** exactly like `CpuState` already is
   (verified single-writer per env within a kernel). Eliminates most
   per-instruction global RMWs. Likely the single largest safe win.
2. **Parallelize the render kernel** — today one thread paints all 61,440
   pixels of its env's frame (`batch_render.cuh`); block-per-env /
   thread-per-pixel is the natural shape. (This is also where warp-level
   parallelism genuinely belongs — not in the CPU loop.)
3. **Vectorize the byte-loop copies** (vblank presentation snapshot, resets,
   OAM DMA) with 16-byte chunks: 2048 iterations → 128.
4. **Zero page + stack in shared memory**: 256 B × 128 threads = 32 KB smem,
   converting the hottest scattered traffic to on-chip.
5. **Hygiene**: strip always-true nullptr guards from the hot bus path, add
   `__restrict__` to `BatchBuffers` pointers, merge `scanline`/`dot` into one
   stored word.

*Measure first*: the binding already ships an opcode/PC histogram profiler
(`step_profile`, `cuda_module.cu`) — quantify opcode fan-out and hot PCs
before and after each change.

### Tier 1 — CPU/PPU kernel split — **RESOLVED as lazy settlement (shipped)**

The original CuLE-style two-kernel split was superseded by the Tier-0 work:
with PPU scalars register-resident, events analytic, and rendering already a
separate kernel, a literal split would only add launch overhead and buffer
traffic. The surviving substance shipped as **lazy PPU settlement**: the step
kernel accumulates cycles and advances the PPU only when the span reaches the
precomputed next-event distance (PPU scalar state changes exclusively at the
three timing events, so between settles the register-resident state the bus
reads is exact). Measured: +9%, taking local 2,048-env stepping past 100k
env-steps/s. Timing stays instruction-granular, identical to before.

Original proposal (kept for the record): CuLE runs 6507 emulation in one
kernel (buffering graphics-chip register writes) and consumes the buffer in a
second — because register pressure and divergence profiles differ between CPU
and video work.

### Tier 2 — table-driven decode — **SHIPPED (the sprint's largest win)**

Shipped exactly as designed: a 256-entry constexpr decode table drives an
11-case effective-address switch and a 49-case operation switch (all eight
branch opcodes share one case). Verified by an 8M-random-instruction
differential harness against the old interpreter (zero divergence, identical
bus traffic and exception messages) and cycle-exact Klaus functional counts.
Combined with the forced-inlining fix it took local 2,048-env stepping from
29.8k to 84.4k env-steps/s — and it is also what makes Tier 3's explicit
binning redundant (see below).

### Tier 3 — wavefront / opcode-binned interpreter — **NO-GO (measured)**

Prototyped and measured (`benchmarks/wavefront_prototype/`, full write-up in
its RESULTS.md): the real table-driven interpreter over a synthetic NROM bus,
instruction streams sampled from the *measured* SMB opcode distribution
(535.7M profiled executions), wavefront variants validated byte-for-byte
against the baseline. Verdict: opcode-binned dispatch is **3.2× slower** than
thread-per-env on realistic decorrelated streams (0.31× at every scale from
4,096 to 131,072 instances) and merely break-even (+4–6%) on fully correlated
streams where there is no divergence to remove. The loss is group
serialization, not binning overhead: a decorrelated warp holds ~20.6 distinct
opcodes, and hardware SIMT reconvergence in the baseline already shares
fetch/table-lookup/addressing across differing opcodes — explicit binning
serializes what the hardware was sharing. Ceiling analysis: even *free*
divergence removal caps at ~1.9× on the CPU-only prototype (~1.4× diluted by
real PPU/bus work) — from a 3.2× hole. The original proposal (Laine & Karras
wavefront, Octax's CHIP-8 result) is kept in the sources; the pattern does
not transfer to a 6502 with a table-driven core.

Useful byproduct: the measured 1.9× correlated-vs-decorrelated gap is direct
empirical support for the cheap state-correlation scheduling lever below.

### Rejected: warp-per-env for the CPU loop

A serial 6502 instruction stream offers no work for 31 of 32 lanes —
throughput would drop roughly an order of magnitude. Lane-level parallelism
belongs to rendering and bulk copies (Tier 0, #2–3).

### Cheap orthogonal lever: state-correlation scheduling

CuLE measured throughput highest right after reset (envs share PC regions)
and decaying as states drift. NeSLE's snapshot resets already correlate the
population far more than ROM-boot Atari; grouping envs by reset state at the
rollout level (so warps tend to share game phase) is nearly free to prototype
and worth measuring, though its payoff is inferred, not published.

## Measured: does a second GPU help?

Measured 2026-10-03 on this box: two RTX 4090s, 24 physical cores, Contra at
frameskip 4 with RAM observations. Reproduce with
`benchmarks/shard_probe.py`; raw numbers in
[data/shard-2x4090-2026-10-03.json](data/shard-2x4090-2026-10-03.json).

The batch runs on one device, so a second GPU can only be used by running a
second *process* with its own shard of envs and summing the rates. A multi-shard
run is compared against a **single-GPU run of the same total env count**, which
is the only comparison that answers "is the second GPU making this faster".

`step_device`, i.e. what a native PPO loop actually pays for:

| total envs | 1 GPU | 2 GPUs | speedup |
| --- | --- | --- | --- |
| 2 048 | 56 895 | 63 227 | 1.11x |
| 4 096 | 120 764 | 119 778 | 0.99x |
| 8 192 | 228 777 | 256 045 | 1.12x |
| 16 384 | 433 381 | 491 510 | 1.13x |
| 32 768 | - | 907 359 | - |

Same picture on the two host-facing paths, `step` (the SB3 vec path, with host
action lookup and per-step reward/done bookkeeping) and `step_reward`: 1.07x to
1.13x at equal total env count.

**Two GPUs buy about 10%, not 2x.** The reason is visible in the single-GPU
column: throughput is close to linear in env count (1 024 -> 30 796, 16 384 ->
433 381, roughly 1.9x per doubling), so one 4090 is *not* saturated at batch
sizes that comfortably fit in memory. Splitting the same total work across two
GPUs cannot beat one GPU doing all of it. The ~10% that does show up comes from
two processes overlapping each other's per-step host and launch gaps, not from
parallel emulation.

What the second card is actually good for is running **more** envs than one card
would hold: 32 768 envs across two GPUs reaches ~907k env-steps/s against ~433k
at 16 384 on one. And memory is not what stops you:

| envs | VRAM | KiB/env |
| --- | --- | --- |
| 1 024 | 270 MiB | 270.0 |
| 4 096 | 878 MiB | 219.5 |
| 8 192 | 1 690 MiB | 211.3 |
| 16 384 | 3 310 MiB | 206.9 |
| 32 768 | 6 554 MiB | 204.8 |

Settling towards ~205 KiB/env, so 32 768 envs is 6.4 GiB of a 24 GiB card and
memory would only become the ceiling somewhere north of 100k envs.

Two measurement traps worth writing down, because both produce flattering
nonsense:

- `step_device` is asynchronous. Timing each call separately measures kernel
  *queueing* and reported 19.4M env-steps/s at 512 envs; queueing `iters` steps
  and charging one final synchronize to the loop gives 30.8k at 1 024.
- Comparing a 2-shard run against a 1-shard run of a *different* size answers
  "did the total grow", not "is the GPU helping". Compare at equal total envs.

Caveat on the numbers: GPU 0 also drives the desktop compositor and measured
~7% slower than GPU 1 at equal per-shard size (for example 236 252 vs 255 258
env-steps/s at 8 192). Per-shard rates were otherwise near-identical to their
single-GPU runs, so the two processes do not contend.

**Recommendation:** raise envs per GPU before reaching for a second one. Spend
effort on multi-process sharding only once a single card is genuinely
saturated, which on this hardware is past ~16k envs.

## Sources

- CuLE: [arXiv:1907.08467](https://arxiv.org/abs/1907.08467), code
  [NVlabs/cule](https://github.com/NVlabs/cule)
- Laine & Karras, "Megakernels Considered Harmful", HPG 2013
  ([PDF](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf))
- Octax (vectorized CHIP-8 RL env): [arXiv:2510.01764](https://arxiv.org/abs/2510.01764)
- Madrona batch-simulation engine: [madrona-engine.github.io](https://madrona-engine.github.io/)
- EnvPool (why general video-game envs resist GPU vectorization):
  [arXiv:2206.10558](https://arxiv.org/abs/2206.10558)
- GVM, GPU Java bytecode interpreter, ACM TOPLAS 2019
  ([PDF](https://users.ece.utexas.edu/~gligoric/papers/CelikETAL19GVM.pdf))
