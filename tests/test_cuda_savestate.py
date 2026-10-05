"""Saving an FCEUX state out of the CUDA batch.

The writer itself is shared with the CPU console (fcs::serialize_fcsx); what this
covers is the device-to-host direction, which did not exist at all until now. The
capture reads one environment's ~13 KB of CPU, PPU and cartridge state out of the
SoA device buffers and hands it to the same writer.

The strong check is a cross-backend round trip: take a state the GPU produced, load
it into the *CPU* console, and run it. That proves the two backends agree on what a
state means, which is the property that makes a GPU-saved checkpoint usable as a
curriculum reset.

Skips without a supported ROM, a state to start from, or a GPU.
"""

from __future__ import annotations

import glob
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROM_DIR = Path(r"C:\games\nes")
FCEUX_STATES = r"C:\games\nes\fceux_rl\curriculum\**\*.fcs"

# Fields the FCSX format carries, so a GPU-written state can be compared against the
# CPU console's own summary. ppu_dot / ppu_frame / cycles are deliberately absent: the
# format has nowhere to put them, so they cannot survive a file.
FILE_FIELDS = (
    "pc",
    "a",
    "x",
    "y",
    "sp",
    "p",
    "ppu_ctrl",
    "ppu_mask",
    "ppu_status",
    "ppu_oam_addr",
    "ppu_open_bus",
    "ppu_read_buffer",
    "ppu_x",
    "ppu_w",
    "ppu_v",
    "ppu_t",
    "has_chr_ram",
    "cpu_ram",
    "prg_ram",
    "nametable_ram",
    "palette_ram",
    "oam",
)


def _first_supported_rom(core: object) -> bytes | None:
    for path in sorted(glob.glob(str(ROM_DIR / "*.nes"))):
        data = Path(path).read_bytes()
        if core.parse_ines_metadata(data)["is_supported"]:
            return data
    return None


def _first_fceux_state() -> bytes | None:
    for path in sorted(glob.glob(FCEUX_STATES, recursive=True)):
        return Path(path).read_bytes()
    return None


class CudaSaveStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            from nesle import _core, _cuda_core  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - build dependent
            raise unittest.SkipTest(f"CUDA extension not available: {exc}") from exc
        cls._core = _core
        cls._cuda_core = _cuda_core
        cls.rom = _first_supported_rom(_core)
        if cls.rom is None:
            raise unittest.SkipTest("no supported .nes ROM in the games directory")
        cls.state = _first_fceux_state()
        if cls.state is None:
            raise unittest.SkipTest("no FCEUX state available")
        try:
            probe = _cuda_core.CudaBatch(2, 1, cls.rom, cls.state)
            del probe
        except Exception as exc:  # pragma: no cover - no GPU
            raise unittest.SkipTest(f"no usable CUDA device: {exc}") from exc

    def _batch(self, num_envs: int = 2):
        return self._cuda_core.CudaBatch(num_envs, 1, self.rom, self.state)

    def _step(self, batch, num_envs: int, frames: int = 4) -> None:
        for i in range(frames):
            batch.step(np.zeros(num_envs, dtype=np.uint8), render_frame=False, copy_obs=False)

    @staticmethod
    def _fields(summary: dict) -> dict:
        keys = list(FILE_FIELDS)
        if "chr_ram" in summary:
            keys.append("chr_ram")
        return {k: summary[k] for k in keys}

    # ---- format ----

    def test_gpu_written_state_is_fcsx(self) -> None:
        batch = self._batch()
        self._step(batch, 2)
        image = batch.save_state(0)
        self.assertEqual(image[:4], b"FCSX")
        declared = struct.unpack_from("<I", image, 4)[0]
        self.assertEqual(declared, len(image) - 16)
        self.assertEqual(struct.unpack_from("<II", image, 8), (0x0000507E, 0xFFFFFFFF))

    def test_blocks_tile_the_payload(self) -> None:
        batch = self._batch()
        self._step(batch, 2)
        image = batch.save_state(0)
        ids = []
        off = 16
        while off + 5 <= len(image):
            block_id = image[off]
            size = struct.unpack_from("<I", image, off + 1)[0]
            ids.append(block_id)
            off += 5 + size
        self.assertEqual(off, len(image))
        for required in (0x01, 0x03, 0x08):
            self.assertIn(required, ids)
        # CHR RAM only for a CHR-RAM cartridge.
        has_chr = self._core.parse_ines_metadata(self.rom)["chr_rom_banks"] == 0
        self.assertEqual(0x10 in ids, has_chr)

    def test_env_index_is_validated(self) -> None:
        batch = self._batch(2)
        with self.assertRaises(Exception):
            batch.save_state(7)
        with self.assertRaises(Exception):
            batch.state_summary(7)

    def test_each_env_saves_its_own_state(self) -> None:
        """Two envs that diverged must not save the same bytes."""
        batch = self._batch(2)
        batch.step(np.array([0x00, 0xFF], dtype=np.uint8), render_frame=False, copy_obs=False)
        self._step(batch, 2)
        self.assertNotEqual(batch.save_state(0), batch.save_state(1))

    # ---- the cross-backend round trip ----

    def test_gpu_state_reloads_into_the_cpu_console(self) -> None:
        """The point of the feature: a GPU-saved state is a usable FCEUX state."""
        batch = self._batch()
        self._step(batch, 2, frames=8)
        image = batch.save_state(0)

        console = self._core.NativeConsole(self.rom)
        console.load_state(image)
        self.assertEqual(
            self._fields(console.state_summary()),
            self._fields(batch.state_summary(0)),
            "the CPU console must see the same machine the GPU captured",
        )

    def test_gpu_state_survives_two_hops(self) -> None:
        """GPU -> file -> CPU console -> file again must stay well formed."""
        batch = self._batch()
        self._step(batch, 2, frames=6)
        first = batch.save_state(0)

        console = self._core.NativeConsole(self.rom)
        console.load_state(first)

        # Straight after the load, before running anything, the console must hold the
        # RAM the GPU had - that is what makes the file a faithful state.
        self.assertEqual(
            bytes(console.ram()),
            bytes(batch.state_summary(0)["cpu_ram"]),
            "loading a GPU state must reproduce the GPU's RAM exactly",
        )

        # Running on from there and re-saving must still produce a valid state, so the
        # GPU's output is not something only our own reader tolerates.
        self._step_console(console, 3)
        second = console.save_state()
        self.assertEqual(second[:4], b"FCSX")
        self.assertEqual(
            struct.unpack_from("<I", second, 4)[0],
            len(second) - 16,
        )
        third = self._core.NativeConsole(self.rom)
        third.load_state(second)
        self.assertEqual(bytes(third.ram()), bytes(console.ram()))

    @staticmethod
    def _step_console(console: object, frames: int) -> None:
        for i in range(frames):
            console.step((i * 7 + 3) & 0xFF, 1, 30000)

    # ---- the mid-frame caveat, same as the CPU path ----

    def test_mid_frame_save_is_refused_when_asked(self) -> None:
        batch = self._batch()
        self._step(batch, 2)
        if batch.state_summary(0)["at_frame_boundary"]:
            self.skipTest("this run landed exactly on a frame boundary")
        with self.assertRaises(RuntimeError) as ctx:
            batch.save_state(0, require_frame_boundary=True)
        self.assertIn("mid-frame", str(ctx.exception))

    def test_frame_position_is_reported(self) -> None:
        """ppu_dot is what require_frame_boundary reads, so it must be populated."""
        batch = self._batch()
        self._step(batch, 2)
        summary = batch.state_summary(0)
        self.assertIn("ppu_dot", summary)
        self.assertIn("ppu_scanline", summary)
        self.assertLess(summary["ppu_dot"], 341, "frame_dot must be split, not copied")
        self.assertEqual(
            batch.state_summary(0)["at_frame_boundary"],
            summary["ppu_dot"] == 0,
        )

    # ---- python helper ----

    def test_path_helper_writes_from_a_gpu_batch(self) -> None:
        from nesle import savestate

        batch = self._batch()
        self._step(batch, 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = savestate.save(batch, Path(tmp) / "gpu.fcs", env=1)
            self.assertTrue(path.exists())
            self.assertEqual(path.read_bytes(), batch.save_state(1))
            console = self._core.NativeConsole(self.rom)
            savestate.load(console, path)
        self.assertEqual(
            self._fields(console.state_summary()),
            self._fields(batch.state_summary(1)),
        )


if __name__ == "__main__":
    unittest.main()
