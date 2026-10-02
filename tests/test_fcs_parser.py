"""Parse the bundled Stable Retro Level 1-1 FCS state and verify known field values.

The reference values come from inspecting the file with a hex dump. If the FCS
layout for FCEUX state.cpp changes upstream, update both this test and the
parser together.
"""
from __future__ import annotations

import gzip
import struct
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = REPO_ROOT / "docs" / "data" / "smb_level1_1.state"


def _load_decompressed_state() -> bytes:
    return gzip.decompress(STATE_PATH.read_bytes())


def _subchunk(name: str, payload: bytes) -> bytes:
    """One FCSX/FCS sub-chunk: 4-byte name, u32 length, payload."""
    assert len(name) <= 4
    return name.encode("ascii").ljust(4, b"\x00") + struct.pack("<I", len(payload)) + payload


def _block(block_id: int, payload: bytes) -> bytes:
    return bytes([block_id]) + struct.pack("<I", len(payload)) + payload


def _fcsx(*blocks: bytes, declared: int | None = None) -> bytes:
    """Assemble an FCSX image: 'FCSX', u32 payload size, two opaque u32s, blocks.

    The two middle header fields are emulator-internal and constant in every
    FCEUX 2.6 state; the parser must not depend on them, so the test leaves them
    at arbitrary values.
    """
    body = b"".join(blocks)
    if declared is None:
        declared = len(body)
    return b"FCSX" + struct.pack("<III", declared, 0x507E, 0xFFFFFFFF) + body


def _cpu_block(pc: int, a: int, x: int, y: int, sp: int, p: int, ram: bytes) -> bytes:
    return _block(
        0x01,
        _subchunk("PC", struct.pack("<H", pc))
        + _subchunk("A", bytes([a]))
        + _subchunk("X", bytes([x]))
        + _subchunk("Y", bytes([y]))
        + _subchunk("S", bytes([sp]))
        + _subchunk("P", bytes([p]))
        + _subchunk("RAM", ram),
    )


def _ppu_block(**kw: bytes) -> bytes:
    return _block(0x03, b"".join(_subchunk(k, v) for k, v in kw.items()))


class FcsParserTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from nesle._cuda_core import parse_fcs_state  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"_cuda_core not built with parse_fcs_state: {exc}")
        self.parse = parse_fcs_state
        self.bytes_in = _load_decompressed_state()

    def test_known_cpu_registers(self) -> None:
        snap = self.parse(self.bytes_in)
        self.assertEqual(snap["pc"], 0x8057)
        self.assertEqual(snap["a"], 0x90)
        self.assertEqual(snap["x"], 0x00)
        self.assertEqual(snap["y"], 0x01)
        self.assertEqual(snap["sp"], 0xFF)
        self.assertEqual(snap["p"], 0xA5)

    def test_known_ppu_registers(self) -> None:
        snap = self.parse(self.bytes_in)
        # PPUR = 90 1e 40 ?? -> ctrl, mask, status
        self.assertEqual(snap["ppu_ctrl"], 0x90)
        self.assertEqual(snap["ppu_mask"], 0x1E)
        self.assertEqual(snap["ppu_status"], 0x40)
        # RADD = 00 08 (LE) -> 0x0800
        self.assertEqual(snap["ppu_v"], 0x0800)
        # TADD = 00 00 (LE) -> 0x0000
        self.assertEqual(snap["ppu_t"], 0x0000)
        # XOFF = 0 (fine x; only low 3 bits kept)
        self.assertEqual(snap["ppu_x"], 0x00)
        # VTGL = 0 (write toggle)
        self.assertEqual(snap["ppu_w"], 0x00)
        # VBUF = 0xff (read buffer)
        self.assertEqual(snap["ppu_read_buffer"], 0xFF)
        # PGEN = 0x90 (open bus)
        self.assertEqual(snap["ppu_open_bus"], 0x90)

    def test_memory_buffer_sizes(self) -> None:
        snap = self.parse(self.bytes_in)
        self.assertEqual(len(snap["cpu_ram"]), 2048)
        self.assertEqual(len(snap["prg_ram"]), 8192)
        self.assertEqual(len(snap["nametable_ram"]), 2048)
        self.assertEqual(len(snap["palette_ram"]), 32)
        self.assertEqual(len(snap["oam"]), 256)

    def test_smb_ram_signals_are_in_gameplay(self) -> None:
        """The saved state was captured at the start of W1-1, so SMB's RAM should reflect
        the gameplay-ready scene (OperMode=1, World=1, Level/Area=1, full time)."""
        snap = self.parse(self.bytes_in)
        ram = snap["cpu_ram"]
        # OperMode at 0x0770 should be 1 (main game) — sidesteps the title-screen bug.
        self.assertEqual(ram[0x0770], 1, "OperMode should be 1 (main game) at this snapshot")
        # World index ($075F) = 0 means World 1; Level ($075C) = 0 means stage 1; Area = 0.
        self.assertEqual(ram[0x075F], 0, "World index should be 0 (World 1)")
        self.assertEqual(ram[0x075C], 0, "Level index should be 0 (stage 1)")
        # Time digits at $07F8-$07FA should sum to a non-zero total.
        time = ram[0x07F8] * 100 + ram[0x07F9] * 10 + ram[0x07FA]
        self.assertGreater(time, 0, "Game timer should be running")

    def test_palette_first_entry_is_background_color(self) -> None:
        """SMB W1-1 uses sky blue 0x22 as the universal background in PRAM[0]."""
        snap = self.parse(self.bytes_in)
        self.assertEqual(snap["palette_ram"][0], 0x22)

    def test_rejects_truncated_input(self) -> None:
        with self.assertRaises(Exception):
            self.parse(self.bytes_in[:32])

    def test_rejects_bad_magic(self) -> None:
        bad = b"XXXX" + self.bytes_in[4:]
        with self.assertRaises(Exception):
            self.parse(bad)


class FcsxParserTests(unittest.TestCase):
    """FCSX is FCEUX 2.6's replacement for FCS. The header is 'FCSX', a u32
    payload size, two opaque u32s, then [u8 id][u32 len][payload] blocks.

    Fixtures are synthesized rather than shipped, so the expectations here are
    readable next to the bytes. The block layout and sub-chunk names were taken
    from real FCEUX 2.6 states: 0x01 CPU, 0x02 CPU IRQ bookkeeping, 0x03 PPU,
    0x08 raw cartridge RAM, 0x10 CHR RAM, plus input/sound blocks we skip.
    """

    def setUp(self) -> None:
        try:
            from nesle._cuda_core import parse_fcs_state  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"_cuda_core not built with parse_fcs_state: {exc}")
        self.parse = parse_fcs_state

    def test_dispatches_on_magic_and_reads_cpu_block(self) -> None:
        ram = bytes(range(256)) * 8
        image = _fcsx(_cpu_block(0xC05B, 0x01, 0x7E, 0x42, 0xF0, 0xA5, ram))
        snap = self.parse(image)
        self.assertEqual(snap["pc"], 0xC05B)
        self.assertEqual(snap["a"], 0x01)
        self.assertEqual(snap["x"], 0x7E)
        self.assertEqual(snap["y"], 0x42)
        self.assertEqual(snap["sp"], 0xF0)
        self.assertEqual(snap["p"], 0xA5)
        self.assertEqual(snap["cpu_ram"], ram)

    def test_reads_ppu_block(self) -> None:
        image = _fcsx(
            _ppu_block(
                NTAR=bytes(2048),
                PRAM=bytes([0x0F, 0x19, 0x29]) + bytes(29),
                SPRA=bytes(256),
                PPUR=bytes([0xB0, 0x1E, 0x20, 0x00]),
                PSPL=bytes([0x07]),
                XOFF=bytes([0xFE]),  # only low 3 bits are kept
                VTGL=bytes([0x01]),
                RADD=struct.pack("<H", 0x0B87),
                TADD=struct.pack("<H", 0x0387),
                VBUF=bytes([0x5A]),
                PGEN=bytes([0xB0]),
            )
        )
        snap = self.parse(image)
        self.assertEqual(snap["ppu_ctrl"], 0xB0)
        self.assertEqual(snap["ppu_mask"], 0x1E)
        self.assertEqual(snap["ppu_status"], 0x20)
        self.assertEqual(snap["ppu_oam_addr"], 0x07)
        self.assertEqual(snap["ppu_x"], 0x06)
        self.assertEqual(snap["ppu_w"], 0x01)
        self.assertEqual(snap["ppu_v"], 0x0B87)
        self.assertEqual(snap["ppu_t"], 0x0387)
        self.assertEqual(snap["ppu_read_buffer"], 0x5A)
        self.assertEqual(snap["ppu_open_bus"], 0xB0)
        self.assertEqual(snap["palette_ram"][:3], bytes([0x0F, 0x19, 0x29]))

    def test_raw_cart_ram_block_is_truncated_to_8k(self) -> None:
        """FCEUX's NES cartridge-RAM block is 64 KiB; only $6000-$7FFF is mapped."""
        big = bytes((i * 7) & 0xFF for i in range(65536))
        snap = self.parse(_fcsx(_block(0x08, big)))
        self.assertEqual(len(snap["prg_ram"]), 8192)
        self.assertEqual(snap["prg_ram"], big[:8192])

    def test_short_cart_ram_block_zero_fills_the_tail(self) -> None:
        snap = self.parse(_fcsx(_block(0x08, b"\x01\x02\x03\x04")))
        self.assertEqual(snap["prg_ram"][:4], b"\x01\x02\x03\x04")
        self.assertEqual(snap["prg_ram"][4:], bytes(8192 - 4))

    def test_absent_blocks_leave_defaults(self) -> None:
        """An FCSX file with no usable block must still produce a sane snapshot
        rather than throwing - the stack pointer default matters, since a
        snapshot reset with sp=0 would fault."""
        snap = self.parse(_fcsx(_block(0x05, _subchunk("FCNT", b"\x00"))))
        self.assertEqual(snap["sp"], 0xFD)
        self.assertEqual(snap["p"], 0x24)
        self.assertEqual(snap["pc"], 0)

    def test_cpu_irq_block_does_not_disturb_registers(self) -> None:
        image = _fcsx(
            _cpu_block(0x8000, 0x11, 0x22, 0x33, 0x44, 0x55, bytes(2048)),
            _block(
                0x02,
                _subchunk("JAMM", b"\x00")
                + _subchunk("IQLB", struct.pack("<I", 0))
                + _subchunk("ICoa", struct.pack("<I", 0))
                + _subchunk("ICou", struct.pack("<i", -48))
                + _subchunk("TSBS", struct.pack("<Q", 0x0123456789ABCDEF))
                + _subchunk("MooP", b"\x24"),
            ),
        )
        snap = self.parse(image)
        self.assertEqual(snap["pc"], 0x8000)
        self.assertEqual(snap["a"], 0x11)
        self.assertEqual(snap["p"], 0x55)

    def test_unknown_blocks_are_skipped(self) -> None:
        image = _fcsx(
            _cpu_block(0x8123, 0x00, 0x00, 0x00, 0xFD, 0x24, bytes(2048)),
            _block(0x04, _subchunk("JYRB", b"\x08\x08")),   # input
            _block(0x10, _subchunk("CHRR", bytes(8192))),   # CHR RAM
            _block(0x1F, _subchunk("IDLS", b"\x00")),       # idle-loop detection
        )
        snap = self.parse(image)
        self.assertEqual(snap["pc"], 0x8123)

    def test_chr_ram_block_is_captured(self) -> None:
        """CHR RAM has to come from the state: a CHR-RAM cartridge that resets
        without its pattern data renders an entirely black screen, because the
        game never re-uploads the tiles by itself."""
        chr_ram = bytes((i * 3) & 0xFF for i in range(8192))
        snap = self.parse(_fcsx(_block(0x10, _subchunk("CHRR", chr_ram))))
        self.assertTrue(snap["has_chr_ram"])
        self.assertEqual(len(snap["chr_ram"]), 8192)
        self.assertEqual(snap["chr_ram"], chr_ram)

    def test_chr_ram_absent_from_legacy_fcs(self) -> None:
        """Legacy FCS predates the CHR block, so the flag must stay false rather
        than being inferred from the buffer happening to be zeroed."""
        snap = self.parse(_load_decompressed_state())
        self.assertFalse(snap["has_chr_ram"])
        self.assertEqual(snap["chr_ram"], bytes(8192))

    def test_chr_flag_distinguishes_absent_from_all_zero(self) -> None:
        snap = self.parse(_fcsx(_block(0x10, _subchunk("CHRR", bytes(8192)))))
        self.assertTrue(snap["has_chr_ram"])
        self.assertEqual(snap["chr_ram"], bytes(8192))

    def test_cpu_registers_and_chr_ram_together(self) -> None:
        """The shape the real corpus has: CPU block then a CHR block."""
        ram = bytes(range(256)) * 8
        chr_ram = bytes([0x7F] * 8192)
        image = _fcsx(
            _cpu_block(0xC05B, 0x01, 0x00, 0xFF, 0xFF, 0x25, ram),
            _block(0x10, _subchunk("CHRR", chr_ram)),
        )
        snap = self.parse(image)
        self.assertEqual(snap["pc"], 0xC05B)
        self.assertEqual(snap["cpu_ram"], ram)
        self.assertEqual(snap["chr_ram"], chr_ram)

    def test_rejects_mismatched_declared_size(self) -> None:
        image = _fcsx(_cpu_block(0x8000, 0, 0, 0, 0xFD, 0x24, bytes(2048)), declared=99)
        with self.assertRaises(Exception):
            self.parse(image)

    def test_rejects_block_overrunning_the_file(self) -> None:
        body = bytes([0x01]) + struct.pack("<I", 4096) + bytes(2048)
        image = b"FCSX" + struct.pack("<III", len(body), 0x507E, 0xFFFFFFFF) + body
        with self.assertRaises(Exception):
            self.parse(image)

    def test_rejects_header_shorter_than_16_bytes(self) -> None:
        with self.assertRaises(Exception):
            self.parse(b"FCSX" + bytes(8))

    def test_legacy_fcs_and_fcsx_are_both_accepted(self) -> None:
        legacy = _load_decompressed_state()
        self.assertEqual(self.parse(legacy)["pc"], 0x8057)
        self.assertEqual(
            self.parse(_fcsx(_cpu_block(0x1234, 0, 0, 0, 0xFD, 0x24, bytes(2048))))["pc"],
            0x1234,
        )


if __name__ == "__main__":
    unittest.main()
