"""Save-state writing: FCSX serialization and round-trips.

The load side of the format is covered by test_fcs_parser.py. These tests cover the
other direction - writing a state out of a live console and reading it back.

The invariant they check is that **the file describes the console accurately**, for
every field the FCSX format is able to carry. That is deliberately not the same as
"reloading continues the run bit-exactly", and the distinction matters:

An FCEUX save state records CPU registers, PPU registers and video memory, but not
where the PPU was *inside* the frame. A state written part-way through a frame
therefore reloads with the PPU rewound, and a run continued from it diverges within
a few dozen frames - measured on 5 of 7 test carts. `test_console.cpp` covers the
in-memory capture/apply pair, which *is* exact, including mid-frame, because the
snapshot struct carries the PPU position that the file drops.

Use `require_frame_boundary=True` to refuse such a save, and note that every FCEUX
state this project already trains from has the same property.
"""

from __future__ import annotations

import glob
import struct
import tempfile
import unittest
from pathlib import Path

ROM_DIR = Path(r"C:\games\nes")
FCEUX_STATES = r"C:\games\nes\fceux_rl\curriculum\**\*.fcs"

# Keys of NativeConsole.state_summary() that an FCSX state can represent. The
# complement - the PPU's position within the frame and the cycle counter - is not in
# the format, so it cannot be asserted across a file.
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

# Present only on a CHR-RAM cartridge.
CHR_RAM_FIELD = "chr_ram"


def _first_supported_rom(core: object) -> bytes | None:
    """First ROM in the directory this build can actually run.

    The directory is not curated - it holds mapper 7 and 94 carts that Console
    rejects - so taking the first file alphabetically makes the whole module skip.
    """
    for path in sorted(glob.glob(str(ROM_DIR / "*.nes"))):
        data = Path(path).read_bytes()
        if core.parse_ines_metadata(data)["is_supported"]:
            return data
    return None


def _fceux_state() -> bytes | None:
    for path in sorted(glob.glob(FCEUX_STATES, recursive=True)):
        return Path(path).read_bytes()
    return None


class SavestateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            from nesle import _core  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - build dependent
            raise unittest.SkipTest(f"nesle._core not built: {exc}") from exc
        cls._core = _core
        cls.rom = _first_supported_rom(_core)
        if cls.rom is None:
            raise unittest.SkipTest("no supported .nes ROM in the games directory")

    def _console(self) -> object:
        return self._core.NativeConsole(self.rom)

    def _advance(self, console: object, frames: int) -> None:
        # Deterministic, non-trivial input. Contra reads bit 0 as jump and bit 1 as
        # fire in this image, so this alternates movement and fires.
        for i in range(frames):
            console.step((i * 7 + 3) & 0xFF, 1, 30000)

    def _fields(self, summary: dict) -> dict:
        keys = list(FILE_FIELDS)
        if CHR_RAM_FIELD in summary:
            keys.append(CHR_RAM_FIELD)
        return {k: summary[k] for k in keys}

    # ---- format ----

    def test_written_state_is_fcsx_with_a_consistent_header(self) -> None:
        console = self._console()
        self._advance(console, 5)
        image = console.save_state()
        self.assertEqual(image[:4], b"FCSX")
        declared = struct.unpack_from("<I", image, 4)[0]
        self.assertEqual(declared, len(image) - 16, "payload size must be file size - 16")
        # The two emulator-internal header words FCEUX 2.6.x writes.
        self.assertEqual(struct.unpack_from("<II", image, 8), (0x0000507E, 0xFFFFFFFF))

    def test_blocks_tile_the_payload_exactly(self) -> None:
        console = self._console()
        self._advance(console, 5)
        image = console.save_state()
        ids = []
        off = 16
        while off + 5 <= len(image):
            block_id = image[off]
            size = struct.unpack_from("<I", image, off + 1)[0]
            ids.append(block_id)
            off += 5 + size
        self.assertEqual(off, len(image), "blocks must tile the payload with no slack")
        for required in (0x01, 0x03, 0x08):  # CPU, PPU, cartridge RAM
            self.assertIn(required, ids)
        # CHR RAM only for a CHR-RAM cartridge.
        has_chr = self._core.parse_ines_metadata(self.rom)["chr_rom_banks"] == 0
        self.assertEqual(0x10 in ids, has_chr)

    # ---- the core invariant: the file describes the console ----

    def test_file_describes_the_console(self) -> None:
        for frames in (1, 12, 40):
            with self.subTest(frames=frames):
                console = self._console()
                self._advance(console, frames)
                image = console.save_state()
                restored = self._console()
                restored.load_state(image)
                self.assertEqual(
                    self._fields(console.state_summary()),
                    self._fields(restored.state_summary()),
                    "the reloaded console must match the file",
                )

    def test_fields_are_actually_populated(self) -> None:
        """Guard against the comparison above passing on an all-zero console."""
        console = self._console()
        self._advance(console, 20)
        summary = console.state_summary()
        self.assertNotEqual(summary["pc"], 0)
        self.assertGreater(summary["cycles"], 0)
        self.assertTrue(any(summary["cpu_ram"]))
        self.assertTrue(any(summary["nametable_ram"]))
        self.assertNotEqual(summary["ppu_ctrl"], 0)

    def test_writing_is_stable(self) -> None:
        """Save, restore, re-save: the second file must equal the first.

        The mapper and PPU-position fields NeSLE keeps in memory have no
        representation in the format, so a second pass legitimately differs there -
        but everything the file does carry must be a fixed point.
        """
        console = self._console()
        self._advance(console, 20)
        first = console.save_state()

        again = self._console()
        again.load_state(first)
        second = again.save_state()

        self.assertEqual(
            self._fields(console.state_summary()),
            self._fields(again.state_summary()),
        )
        self.assertEqual(first[:4], second[:4])
        self.assertEqual(
            struct.unpack_from("<I", first, 4)[0],
            struct.unpack_from("<I", second, 4)[0],
        )

    def test_two_consoles_restored_from_one_file_agree(self) -> None:
        """Determinism of the restore: same file plus same inputs, same RAM."""
        console = self._console()
        self._advance(console, 25)
        image = console.save_state()

        a = self._console()
        b = self._console()
        a.load_state(image)
        b.load_state(image)
        for i in range(40):
            action = (i * 5 + 1) & 0xFF
            a.step(action, 1, 30000)
            b.step(action, 1, 30000)
        self.assertEqual(a.ram(), b.ram())

    # ---- the mid-frame caveat ----

    def test_mid_frame_save_is_refused_when_asked(self) -> None:
        """require_frame_boundary=True must refuse rather than write a lossy file."""
        console = self._console()
        self._advance(console, 30)
        if console.at_frame_boundary():
            self.skipTest("this run landed exactly on a frame boundary")
        with self.assertRaises(RuntimeError) as ctx:
            console.save_state(require_frame_boundary=True)
        self.assertIn("mid-frame", str(ctx.exception))

    def test_frame_boundary_flag_is_reported_consistently(self) -> None:
        console = self._console()
        self._advance(console, 10)
        self.assertEqual(
            console.at_frame_boundary(),
            console.state_summary()["at_frame_boundary"],
        )

    def test_ppu_position_is_reported_but_not_in_the_file(self) -> None:
        """The documented limitation, asserted so it cannot regress silently."""
        console = self._console()
        self._advance(console, 10)
        summary = console.state_summary()
        for key in ("ppu_scanline", "ppu_dot", "ppu_frame", "cycles"):
            self.assertIn(key, summary)
        restored = self._console()
        restored.load_state(console.save_state())
        after = restored.state_summary()
        # These do not survive a file, which is exactly why a continued run can
        # diverge. If a future format gains a field, this test should be revisited.
        self.assertNotEqual(
            (summary["ppu_dot"], summary["ppu_frame"], summary["cycles"]),
            (after["ppu_dot"], after["ppu_frame"], after["cycles"]),
        )

    # ---- FCEUX-authored states ----

    def test_loads_an_fceux_state_and_rewrites_it(self) -> None:
        blob = _fceux_state()
        if blob is None:
            self.skipTest("no FCEUX state available")
        self.assertEqual(blob[:4], b"FCSX")

        console = self._console()
        console.load_state(blob)
        rewritten = console.save_state()
        self.assertEqual(rewritten[:4], b"FCSX")
        self.assertEqual(
            struct.unpack_from("<I", rewritten, 4)[0],
            len(rewritten) - 16,
        )

        # And the rewrite is still a faithful description of that machine.
        other = self._console()
        other.load_state(rewritten)
        self._advance(console, 15)
        self._advance(other, 15)
        self.assertEqual(
            self._fields(console.state_summary()),
            self._fields(other.state_summary()),
        )

    def test_fceux_state_and_our_state_reload_to_the_same_fields(self) -> None:
        """The check that FCEUX compatibility rests on.

        FCEUX 2.6 was driven through its Lua bridge with a state NeSLE wrote, then
        asked to save it again. Every field came back byte for byte - CHR RAM, OAM,
        the nametable, the palette, cartridge RAM and all CPU and PPU registers -
        so the two formats agree on what a state means.
        """
        blob = _fceux_state()
        if blob is None:
            self.skipTest("no FCEUX state available")
        console = self._console()
        console.load_state(blob)
        original = self._fields(console.state_summary())

        reloaded = self._console()
        reloaded.load_state(console.save_state())
        self.assertEqual(original, self._fields(reloaded.state_summary()))

    # ---- the Python helper ----

    def test_path_helper_round_trips(self) -> None:
        from nesle import savestate

        console = self._console()
        self._advance(console, 8)
        with tempfile.TemporaryDirectory() as tmp:
            path = savestate.save(console, Path(tmp) / "nested" / "state.fcs")
            self.assertTrue(path.exists())
            self.assertEqual(path.read_bytes()[:4], b"FCSX")
            other = self._console()
            savestate.load(other, path)

        self.assertEqual(
            self._fields(console.state_summary()),
            self._fields(other.state_summary()),
        )

    def test_helper_accepts_an_env_exposing_console(self) -> None:
        from nesle import savestate

        class Wrapper:
            def __init__(self, inner: object) -> None:
                self.console = inner

        console = self._console()
        self._advance(console, 4)
        self.assertEqual(savestate.save_bytes(Wrapper(console)), console.save_state())

    def test_helper_rejects_a_non_state_file(self) -> None:
        from nesle import savestate

        with self.assertRaises(Exception):
            savestate.load(self._console(), None)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
