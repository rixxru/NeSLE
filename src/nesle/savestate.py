"""Reading and writing FCEUX save states.

NeSLE could load ``.fcs`` states from the start but had no way to write one, which
made a round trip through the CPU impossible: you could replay a curriculum state
but never export a position you had reached yourself.

The writer emits FCSX, the format FCEUX 2.6 produces and reads. It works from either
backend. The CPU console captures from ``Console::capture_state``; the CUDA batch
reads one environment's state out of the SoA device buffers with
``CudaBatch.save_state(env)``, which is what lets a training run seed a curriculum
from its own progress.

Verified against FCEUX 2.6.6 driving Contra: FCEUX loads a NeSLE-written state,
re-saves it, and every field comes back byte for byte - CHR RAM, OAM, the nametable,
the palette, cartridge RAM and all CPU and PPU registers. Over 60 frames from the
same file, FCEUX agrees with NeSLE on 99.0% of RAM bytes at the bridge's one-frame
offset, for a GPU-written state and a CPU-written one alike, against 97.3% from a
state FCEUX wrote itself.

One limitation is inherent to the format rather than to this implementation. An
FCEUX state records CPU registers, PPU registers and video memory, but **not**
where the PPU was inside the frame, nor the mapper's bank registers. A state
written mid-frame therefore reloads with the PPU rewound to the top of the frame,
and a run resumed from it diverges within a few dozen frames; a restored
environment likewise starts at the power-on mapper bank, which is what the CUDA
snapshot-reset path has always done for FCEUX-authored states. ``save`` takes
``require_frame_boundary`` for callers who would rather be told than lose that
silently. Neither limit applies to a reset that seeds training rather than
resuming a specific run, which is how every FCEUX state in this project is used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["load", "load_bytes", "save", "save_bytes"]


def _console(console: Any) -> Any:
    """Accept either a NativeConsole or anything exposing one as ``.console``."""
    inner = getattr(console, "console", None)
    return inner if hasattr(inner, "save_state") else console


def _is_gpu_batch(obj: Any) -> bool:
    """A CudaBatch, rather than a console.

    Both expose ``save_state`` and ``state_summary``, so neither of those
    discriminates. ``num_levels`` is the giveaway: it is a CudaBatch property, and a
    NativeConsole has no notion of levels. Checking ``state_summary`` instead would
    have classified every NativeConsole as a batch.
    """
    return not hasattr(obj, "console") and hasattr(obj, "num_levels")


def save_bytes(
    console: Any,
    *,
    env: int | None = None,
    require_frame_boundary: bool = False,
) -> bytes:
    """Serialize a console - or one environment of a CUDA batch - to FCSX bytes.

    Args:
        console: a ``NativeConsole``, anything with one on a ``.console`` attribute
            (the single-environment CPU backend), or a ``CudaBatch``.
        env: which environment to capture from a ``CudaBatch``. Required for a batch,
            and rejected for a console, which has only one environment.
        require_frame_boundary: refuse to save when the PPU is mid-frame, since the
            format cannot record that. Off by default: every FCEUX state this
            project already trains from has the same property, and a few dots of
            PPU phase does not matter for seeding a curriculum reset.
    """
    if _is_gpu_batch(console):
        if env is None:
            raise ValueError(
                "a CudaBatch has many environments; pass env=<index> to choose one"
            )
        return console.save_state(env, require_frame_boundary=require_frame_boundary)
    if env is not None:
        raise ValueError("env only applies to a CudaBatch, not to a single console")
    return _console(console).save_state(require_frame_boundary=require_frame_boundary)


def save(
    console: Any,
    path: str | Path,
    *,
    env: int | None = None,
    require_frame_boundary: bool = False,
) -> Path:
    """Write a console's state to ``path`` as FCSX. Returns the path written.

    ``env`` selects the environment when ``console`` is a ``CudaBatch``::

        nesle.savestate.save(batch, "checkpoint.fcs", env=1234)
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(
        save_bytes(console, env=env, require_frame_boundary=require_frame_boundary)
    )
    return target


def load_bytes(console: Any, image: bytes, *, env: int | None = None) -> None:
    """Restore a console - or one environment of a CUDA batch - from FCSX or FCS bytes."""
    if _is_gpu_batch(console):
        if env is None:
            raise ValueError(
                "a CudaBatch has many environments; pass env=<index> to choose one"
            )
        console.load_state(env, image)
        return
    if env is not None:
        raise ValueError("env only applies to a CudaBatch, not to a single console")
    _console(console).load_state(image)


def load(console: Any, path: str | Path, *, env: int | None = None) -> None:
    """Restore a console from an FCSX or legacy FCS file.

    ``env`` selects the environment when ``console`` is a ``CudaBatch``::

        nesle.savestate.load(batch, "checkpoint.fcs", env=1234)

    A GPU load goes through the same path as a snapshot reset, so the mapper returns
    to power-on and the PPU to the top of a frame - the format carries neither. It
    also seeds the reward baselines from the state's own RAM, so the first reward
    after a load is not a synthetic delta from zero.
    """
    load_bytes(console, Path(path).read_bytes(), env=env)
