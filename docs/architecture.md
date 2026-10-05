# Architecture

> **Status note (2026-07-28):** this document began as the design plan and some
> sections still use design-phase language ("the first GPU implementation
> shouldвЂ¦", "Benchmark Plan"). The described execution model вЂ” one CUDA thread
> per env, SoA state, snapshot reset, device-view API вЂ” **is** what's built and
> shipping; where the text lists plans (e.g. the full benchmark matrix), treat
> `benchmarks/` and the reports in `docs/` as the record of what actually ran.

## Goal

Run thousands of independent Super Mario Bros. environments on one NVIDIA A100,
with emulator state, observations, reward inputs, and reset state caches resident
on GPU.

## Modules

```text
nesle/
  python API: Gymnasium Env and SB3 VecEnv facade
  native bridge: pybind11 extension and CUDA console binding

cpp/
  core: ROM parsing, CPU/PPU/APU/input state, mapper interfaces
  cuda: batched state layout and kernels
  bindings: Python extension

benchmarks/
  nes-py comparison, FPS scaling, frame-skip, render/no-render modes,
  GPU-vs-CPU smoke, and correctness falsifiability checks
```

## Execution Model

The first GPU implementation should use one CUDA thread per environment for CPU
execution. This maps the sequential 6502 instruction stream naturally and keeps
debugging tractable. PPU rendering is separate because it writes many pixels and
has a different occupancy/register profile.

Per RL step:

1. Copy or write action masks into device input buffers.
2. Launch CPU/console kernel for `frameskip` raw frames.
3. During CPU execution, update CPU RAM, PPU registers, controller shift state,
   timers, NMI, OAMDMA, and minimal APU timing.
4. Render selected frames with a separate PPU kernel only when observation output
   requires pixels.
5. Launch reward/info kernel that reads the selected game's RAM addresses into
   compact arrays. `reward_kind` chooses the scraper: the Super Mario Bros. one
   by default, or the Contra one, which additionally reads player 2's fields
   when `$0022` says two players.
6. Auto-reset completed envs from cached initial states or FCEUX snapshot banks.
7. Return GPU tensors directly where possible; copy to NumPy only for Gym/SB3
   compatibility paths that require CPU arrays.

## State Layout

Use structure-of-arrays for hot batched state:

- CPU registers: `pc[n]`, `a[n]`, `x[n]`, `y[n]`, `sp[n]`, `p[n]`
- CPU timing: `cycles[n]`, `frame_cycles[n]`, `nmi_pending[n]`
- RAM: `cpu_ram[n][2048]`, env-major contiguous initially
- PPU registers: scalar arrays for control/mask/status/latches/scroll
- PPU memory: nametable RAM, palette RAM, OAM per env
- ROM: PRG and CHR in read-only device memory, shared by all envs
- Output: frame buffer `[n, 240, 256, 3]` or lower-resolution postprocessed
  buffers in device memory

The initial implementation favors correctness and debug visibility. Once the
CPU and PPU pass tests, profile alternatives: RAM tiling, grouping envs by PC,
multi-thread PPU rendering per env, and direct PyTorch tensor output.

## Mapper Strategy

Support mapper 0/NROM first:

- PRG ROM fixed at `$8000-$FFFF`
- NROM-128 mirrors 16 KB PRG into upper bank
- NROM-256 maps 32 KB PRG directly
- CHR ROM fixed at PPU `$0000-$1FFF`
- no mapper registers, no IRQs

After Mario throughput is proven, add mapper abstractions for other NES RL games.

## CPU Strategy

Build a single instruction implementation that can compile for CPU tests and
CUDA device code. Avoid divergent host-only behavior in the core instruction
functions. Test order:

1. Official opcode table metadata.
2. Unit tests for addressing modes and flags.
3. Klaus Dormann functional test on CPU.
4. Same test in a single CUDA thread.
5. Batched randomized differential tests against the CPU path.

## PPU Strategy

Correctness target is Mario, not every obscure PPU edge case on day one.

Required first:

- vblank/NMI timing
- `$2000-$2007` semantics used by Mario
- OAMDMA
- background rendering with scrolling
- 8x8 sprites, sprite priority, sprite 0 hit
- palette lookup into RGB output

Deferred until needed:

- rare open-bus decay behavior
- DMC/controller conflict
- tricky mid-scanline effects outside Mario
- non-NROM mapper scanline IRQs

## Reset Strategy

Follow CuLE's reset-cache idea, but the practical training path now uses
Stable Retro/FCEUX `.state` files as reset templates. Python loads raw or
gzip-wrapped state bytes, `cpp/include/nesle/fcs.hpp` parses them, and the CUDA
binding uploads one or more snapshot templates to device memory.

Both FCEUX formats are accepted and dispatched on the magic: legacy FCS
(`FCS\xff`, chunked into CPU / PPU / cartridge blocks) and FCSX, which FCEUX 2.6
writes by default. FCSX is `'FCSX'`, a u32 payload size, two opaque u32s, then
`[u8 id][u32 len][payload]` blocks. Its sub-chunk payload format is byte-for-byte
the legacy one, so both share the same appliers. The parser reads CPU registers
and CPU RAM, PPU registers, nametable RAM, palette RAM, OAM, cartridge RAM
(stored raw in FCSX) and cartridge CHR RAM.

CHR RAM matters more than it looks: on a CHR-RAM cartridge the pattern data is
the one thing the game does not re-upload, so restoring a snapshot without it
yields a completely black screen. Only FCSX carries it, and only a CHR-RAM
cartridge has any, so the device buffer is allocated per level only when the ROM
is CHR-RAM and the states actually contain a CHR block.

### Writing states

Reading a state was always possible; writing one is `Console::capture_state()` plus
`fcs::serialize_fcsx()`, exposed as `NativeConsole.save_state()` and through
`nesle.savestate.save()`. The writer is the parser's exact inverse for every field
the format carries, so a state written from a live console describes that same
machine when read back.

Verified against FCEUX 2.6.6 driving Contra: FCEUX loads a NeSLE-written state,
re-saves it, and every field survives byte for byte - CHR RAM, OAM, nametable,
palette, cartridge RAM, and all CPU and PPU registers. Run forward from a NeSLE
state, FCEUX agrees with NeSLE on 99.0% of RAM bytes over 60 frames at the
bridge's one-frame offset, against 98.6% from a state FCEUX wrote itself.

The CUDA batch can write states too, which is what makes a training run able to
seed a curriculum. `CudaBatch.save_state(env)` reads one environment's ~13 KB of
CPU, PPU and cartridge state out of the SoA device buffers and hands it to the same
writer. It is assembled from ~25 small `cudaMemcpy` calls rather than a staging
kernel: this is a once-per-checkpoint operation, ~13 microseconds of copies against
a kernel that would have to reproduce the SoA gather by hand. The PPU scalars are
register-cached inside a launch but stored back to global memory at kernel exit, so
reading them between steps is consistent. The device merges the frame position into
`frame_dot = scanline * 341 + dot`, so capture splits it again; neither half reaches
the file, but `ppu_dot` is what `require_frame_boundary` reads.

Measured on the same bridge, over 60 frames at the one-frame offset:

| state source | RAM agreement with FCEUX |
| --- | --- |
| written by the CUDA batch | 99.0% |
| written by the CPU console | 99.0% |
| written by FCEUX itself | 97.3% |

The reverse direction exists too. `CudaBatch.load_state(env, image)` parses the
state on the host, uploads it as a one-off single level, and reuses the **masked**
snapshot-reset kernel aimed at just that env - there is no second restore path to
keep in sync. `warm_reset_console_env` already does the right thing for everything
the format does not carry: it pins `frame_dot` and `frame` to the top of a frame
rather than leaving a stale mid-frame value next to the loaded `v`/`t`, resets the
controller shift registers so a half-read button cannot inject a phantom press,
clears any fault, puts the mapper back to power-on, and seeds the reward baselines
from the state's own RAM so the first reward after a load is not a synthetic delta
from zero.

A load is therefore also the way to recover an env that hit an unimplemented opcode
and was quarantined - the fault is cleared by the same reset path. It allocates
scratch device buffers for the upload and frees them through a scope guard, so a
throwing payload cannot leak them.

FCEUX also re-saves a GPU-written state with all 24 shared fields byte-identical
(`scripts/verify_fceux_gpu_state.py`, needs the bridge so it is not in the test
suite). NeSLE's two writers agreeing with each other exactly is the useful signal
there: the device capture is not losing anything the CPU one keeps.

Two things the format cannot carry, both properties of FCEUX's format rather than
of this implementation:

- **The PPU's position within the frame.** The in-memory snapshot carries
  `scanline`, `dot` and the frame counter, so `capture_state`/`apply_state` is
  exact even mid-frame. A *file* drops them, so a reloaded console rewinds the PPU
  to the top of the frame and a run continued from it diverges within a few dozen
  frames. `save_state(require_frame_boundary=True)` refuses such a save.
- **Mapper bank registers.** FCEUX's cartridge-RAM block runs to 60 KiB on a
  mapper 2 cartridge and its mapper-state encoding at the tail is not established,
  so the writer emits four bytes there - which FCEUX returns unchanged - but the
  parser does not read them back. A restored environment starts at the power-on
  bank, which is what the device reset path has always done for FCEUX-authored
  states. Reading a guessed byte would bank-switch a game into different code.

Neither limit matters for the reset-seeds-training use, which is how all 563
FCEUX states in this project are consumed; both matter for resuming one specific
run, which is why the Python API offers the stricter flag.

### Unimplemented opcodes are quarantined per env, not fatal

`cpu::step` used to compile to `asm("trap;")` under `__CUDA_ARCH__`, which aborts
the entire kernel. One env in a 16k batch reaching a byte the decode table rejects
therefore killed every other env with an opaque CUDA launch error, and nothing said
which env or which opcode. On the host the same case threw, which was fine, but the
message printed `std::to_string(opcode)` behind a `0x` prefix, so 0xFC was reported
as `opcode 0x252` - a value that does not exist.

`cpu::step` is now non-throwing and non-trapping on both sides: it returns
`StepResult::illegal`. Callers decide.

| caller | behaviour |
| --- | --- |
| `Console::step_cpu_instruction` | calls `cpu::step_or_throw`, so single-env and Python callers still get a `RuntimeError` naming the opcode in hex |
| `cpu_runner.hpp` `run_until_trap` | same wrapper, so `RunStatus::CpuException` still happens |
| `step_batch_cpu_env` | throws on the host (`run_batch_cpu` and the smoke tool already isolate per env with try/catch) |
| `console_step_kernel` | records `(pc << 8) \| opcode` in `cpu.fault[env]`, marks the env done with zero reward, and stops executing it. Every other env in the launch finishes the frame. |

`CudaBatch.faults()` returns `{env: (pc, opcode)}` for the quarantined envs and is
empty in normal operation. Every reset path clears it, since a reset env starts
from the reset vector and a fault recorded before it says nothing about the env
now.

Note that a snapshot's CPU registers are per *level*, not per env: the reset kernel
copies `snap.pc[level]` into every env assigned to that level. A state whose PC
points at an unimplemented opcode therefore faults every env using it, which is
correct but not isolation - to see isolation, put the faulted state on one level
and a healthy one on another.

NeSLE implements 152 of 256 opcodes. The 104 rejected ones are the unused-official
group (`SLO`, `RLA`, `SRE`, `RRA`, `SAX`, `LAX`, `DCP`, `ISC`, `ANC`, `ALR`,
`ARR`, `AXS`, `ASC`) plus the unofficial NOP family a 2A03 does execute (`0x1A`,
`0x3A`, `0x5A`, `0x7A`, `0xDA`, `0xFA`, `0x80`, `0x82`, `0x89`, `0xC2`, `0xE2`,
`0x04`, `0x44`, `0x64`, `0x0C`, `0x1C`, `0x3C`, `0x5C`, `0x7C`, `0xDC`, `0xFC`).
**No licensed NES game is affected by that set** - nothing in it is used by
commercial software, which is why they are rejected rather than implemented.

`0xEB`, the unofficial duplicate of `SBC` immediate, *was* the exception and is now
implemented. A handful of licensed titles use it, and on a 2A03 it is bit-identical
to `0xE9` because the Ricoh has no decimal mode. It sat next to `0xED` (duplicate
`SBC` absolute), which was already handled, so the pair had been half done.

For a single level, `reset_state_path` restores every env from the same
snapshot. For curriculum training, `reset_state_paths` uploads a snapshot bank
and `env_to_level[env]` selects the template used by each env. If no explicit
assignment is provided, Python assigns envs round-robin across the snapshots.

This avoids replaying fragile title-screen/start sequences and makes done-env
auto-reset cheap: the reset kernel copies the selected snapshot directly into
the env's device-resident emulator state.

## Python API

Expose two layers:

- `NesleEnv`: Gymnasium-compatible single environment for debugging and smoke
  tests.
- `NesleVecEnv`: SB3-compatible vector environment for training. It should return
  `obs, rewards, dones, infos`, auto-reset ended envs, and populate
  `terminal_observation`.

The vector API is the important performance path. The single env is a debugger.
For training, prefer `observation_mode="ram"` and `reset_state_path` or
`reset_state_paths`. RGB observations still work, but they copy full frames back
to host memory and should be reserved for debugging, videos, or visual-policy
experiments.

`backend="cuda"` controls NeSLE's emulator backend. SB3/PyTorch policy placement
is separate and is controlled by `--sb3-device` in `examples/sb3_train.py`.

## GPU-resident PPO path

For large-batch training where SB3's CPU rollout buffer is the bottleneck,
`nesle.native_ppo` provides a fully GPU-resident PPO loop. Observations, the
rollout buffer, the action sample, GAE, the policy/value loss, the optimizer
state, and the gradient update all stay on device. The only host roundtrip per
update is the small set of scalar log lines.

The bridge is provided by three methods on `nesle._cuda_core.CudaBatch`:

- `reset_device()` вЂ” runs the snapshot reset kernel and returns a
  `CudaDeviceArrayView` over the RAM observation buffer.
- `step_device(actions, auto_reset=True, synchronize=True)` вЂ” accepts a CUDA
  action tensor (uint8 mask or int64-encoded mask), launches the console step
  kernel, optionally fires the snapshot reset kernel for done envs, and returns
  a dict of `CudaDeviceArrayView` objects (`ram`, `rewards`, `dones`). When
  `auto_reset=True` the call always synchronizes (otherwise the next step's
  action copy would race the reset writes).
- `ram_device()`, `rewards_device()`, `last_done_device()` вЂ” direct device views
  for inspection between calls.

Each view implements both `__cuda_array_interface__` v3 and `__dlpack__`, so
PyTorch can build tensors directly via `torch.utils.dlpack.from_dlpack(view)`
without a host copy. The pybind layer uses `py::keep_alive<0, 1>()` to keep the
parent `CudaBatch` alive as long as any view (and any torch tensor built from
it) is reachable from Python вЂ” see the lifetime comment above
`CudaDeviceArrayView` in `cpp/bindings/cuda_module.cu`.

## Benchmark Plan

Benchmark modes:

- emulation only, random actions, no render
- render only where observations are requested
- full inference path with a small CNN policy on GPU
- SB3 PPO/A2C compatibility path

Compare against `nes-py`/`gym-super-mario-bros` at env counts:

```text
1, 8, 32, 128, 512, 1024, 2048, 4096, 8192
```

Report raw FPS, training-frame FPS, FPS/env, GPU utilization, memory footprint,
and reset rate.

The reproducible entrypoint is `benchmarks/phase5_benchmark.py`. Use the
`step`, `render`, and `inference` modes for NeSLE scaling runs, then rerun with
`--include-legacy` after installing `.[legacy-mario]` for CPU emulator
comparison rows using registered legacy env IDs such as `SuperMarioBros-v0`.
Use `scripts/benchmark_cuda_kernels.sh` for raw CUDA kernel scaling so benchmark
reports distinguish packaged Python backend throughput from lower-level GPU
reward/render capacity.
Use `scripts/build_cuda_extension.sh` to build the optional `nesle._cuda_core`
module; once present, `NesleVecEnv(..., backend="cuda")` runs the ROM-backed
CUDA batch CPU/PPU console loop through the public Python vector API. The
lower-level CUDA reward/render kernels remain available for calibration runs.

The local practical smoke is `benchmarks/gpu_vs_cpu.py`, which compares native
CPU single-env throughput against batched `cuda-console` stepping from the W1-1
snapshot. `benchmarks/verify_correctness.py` checks that the benchmark is doing
real per-env work by verifying action divergence, plausible instruction counts,
and independent RAM evolution.
