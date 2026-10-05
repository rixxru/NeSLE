"""Shared helpers for the save-state tests: locating and patching an FCSX state.

Not a test module - no unittest cases live here. Both test_env_fault.py (which needs
a state whose PC points at an unimplemented opcode) and test_cuda_savestate.py (which
needs the same thing to check a load clears a quarantine) need this, and duplicating
the FCSX walker in two files would let them drift.
"""

from __future__ import annotations

import struct

# The decoder's Illegal set, mirrored from cpu.hpp's decode table. Duplicated rather
# than queried because the table is device-side and there is no host binding for it;
# test_cpu.cpp pins the host half of the same contract. 0xEB (duplicate SBC) was
# removed from this list when it was implemented - if the two ever drift, the tests
# that use it pick a byte that no longer faults and fail loudly.
ILLEGAL_OPCODES = frozenset(
    int(x, 16)
    for x in (
        "02 03 04 07 0B 0C 0F 12 13 14 17 1A 1B 1C 1F 22 23 27 2B 2F 32 33 34 37 3A 3B 3C 3F "
        "42 43 44 47 4B 4F 52 53 54 57 5A 5B 5C 5F 62 63 64 67 6B 6F 72 73 74 77 7A 7B 7C 7F "
        "80 82 83 87 89 8B 8F 92 93 97 9B 9C 9E 9F A3 A7 AB AF B2 B3 B7 BB BF C2 C3 C7 CB CF "
        "D2 D3 D4 D7 DA DB DC DF E2 E3 E7 EF F2 F3 F4 F7 FA FB FC FF"
    ).split()
)


def prg_rom_window(rom: bytes) -> bytes:
    """The fixed 16 KiB window of an NROM-family image, i.e. what $C000-$FFFF reads."""
    prg_banks = rom[4]
    prg = rom[16 : 16 + prg_banks * 16 * 1024]
    return prg[-16384:]


def find_illegal_pc(rom: bytes) -> int | None:
    """An address in the fixed window whose byte the decoder rejects.

    Found from the ROM rather than hard-coded: the filler and padding in an unused
    bank is full of such bytes, and which ones they are moves with the ROM.
    """
    window = prg_rom_window(rom)
    for offset, byte in enumerate(window):
        if byte in ILLEGAL_OPCODES:
            return 0xC000 + offset
    return None


def patch_pc(image: bytes, pc: int) -> bytes:
    """Return a copy of an FCSX state with CPU.PC replaced.

    Walks the FCSX block/sub-chunk structure and rewrites the two payload bytes of
    the "PC" sub-chunk in the CPU block. Raises if the layout is not what we expect,
    so a format change fails loudly instead of silently patching nothing.
    """
    out = bytearray(image)
    if bytes(out[:4]) != b"FCSX":
        raise AssertionError(f"expected FCSX, got {bytes(out[:4])!r}")
    off = 16
    while off + 5 <= len(out):
        block_id = out[off]
        size = struct.unpack_from("<I", out, off + 1)[0]
        off += 5
        if block_id in (0x01, 0x02):  # CPU / CPU2
            sub = off
            end = off + size
            while sub + 8 <= end:
                name = bytes(out[sub : sub + 4]).rstrip(b"\x00")
                length = struct.unpack_from("<I", out, sub + 4)[0]
                if name == b"PC":
                    if length != 2:
                        raise AssertionError(f"CPU.PC is {length} bytes, expected 2")
                    struct.pack_into("<H", out, sub + 8, pc)
                    return bytes(out)
                sub += 8 + length
        off += size
    raise AssertionError("no CPU.PC sub-chunk in the state")
