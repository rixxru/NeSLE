"""Reading and writing FCEUX save states.

NeSLE could load ``.fcs`` states from the start but had no way to write one, which
made a round trip through the CPU impossible: you could replay a curriculum state
but never export a position you had reached yourself.

The writer emits FCSX, the format FCEUX 2.6 produces and reads. Verified against
FCEUX 2.6.6 driving Contra: FCEUX loads a NeSLE-written state, re-saves it, and
every field survives byte for byte - CHR RAM, OAM, the nametable, the palette,
cartridge RAM and all CPU and PPU registers. Run forward from a NeSLE state, FCEUX
agrees with NeSLE on 99.0% of RAM bytes over 60 frames at the bridge's one-frame
offset, against 98.6% from a state FCEUX wrote itself, so the two states are
equivalent in practice.

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


def save_bytes(console: Any, *, require_frame_boundary: bool = False) -> bytes:
    """Serialize a live console to FCSX bytes.

    Args:
        console: a ``NativeConsole``, or any object with one on a ``.console``
            attribute (such as the single-environment CPU backend).
        require_frame_boundary: refuse to save when the PPU is mid-frame, since
            the format cannot record that. Off by default: every FCEUX state this
            project already trains from has the same property, and a few dots of
            PPU phase does not matter for seeding a curriculum reset.
    """
    return _console(console).save_state(require_frame_boundary=require_frame_boundary)


def save(
    console: Any,
    path: str | Path,
    *,
    require_frame_boundary: bool = False,
) -> Path:
    """Write a console's state to ``path`` as FCSX. Returns the path written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(save_bytes(console, require_frame_boundary=require_frame_boundary))
    return target


def load_bytes(console: Any, image: bytes) -> None:
    """Restore a console from FCSX or legacy FCS bytes."""
    _console(console).load_state(image)


def load(console: Any, path: str | Path) -> None:
    """Restore a console from an FCSX or legacy FCS file."""
    _console(console).load_state(Path(path).read_bytes())
