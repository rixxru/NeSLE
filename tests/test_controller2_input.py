"""Controller 2 input channel (Contra two-player mode).

Contra's two-player mode reads the standard controller on $4017, so the batch
needs a second, independent action channel rather than widening the single
`action_masks` byte. The test ROM below is hand-assembled: every frame it
strobes $4016 and then stores the A and B bits of *both* controllers into RAM,
which lets the assertions observe the two channels separately.

  $0300 P1 A    $0301 P2 A
  $0302 P1 B    $0303 P2 B
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import nesle
from nesle.actions import Button, encode_action
from nesle.rom import CHR_BANK_SIZE, PRG_BANK_SIZE

A = 1 << int(Button.A)
B = 1 << int(Button.B)

P1_A, P2_A, P1_B, P2_B = 0x0300, 0x0301, 0x0302, 0x0303


def _require_cuda() -> None:
    if not hasattr(np, "uint8"):
        raise unittest.SkipTest("complete numpy package is not available")
    try:
        import nesle._cuda_core  # noqa: F401
    except ImportError as exc:
        raise unittest.SkipTest(f"_cuda_core not available: {exc}")


def build_controller_probe_rom() -> bytes:
    """Assemble the two-controller probe program."""
    program = bytearray()
    # STA $4016 with A=1, then with A=0: strobe high, strobe low. The strobe
    # register is shared, so a single latch captures both controllers.
    program += bytes([0xA9, 0x01, 0x8D, 0x16, 0x40, 0xA9, 0x00, 0x8D, 0x16, 0x40])
    # A standard controller shift register returns the *next* bit in bit 0 on
    # every read, so each read is masked with $01 regardless of which button
    # that bit happens to correspond to.
    for port, target in (
        (0x4016, P1_A),
        (0x4017, P2_A),
        (0x4016, P1_B),
        (0x4017, P2_B),
    ):
        program += bytes([0xAD, port & 0xFF, port >> 8, 0x29, 0x01, 0x8D, target & 0xFF, target >> 8])
    loop = len(program)
    program += bytes([0x4C, loop & 0xFF, loop >> 8])  # JMP self

    # The 32 KB PRG bank maps linearly at $8000-$FFFF, so the program starts at
    # offset 0 and the reset vector must point at offset 0, not at the loop.
    prg = bytearray([0xEA] * (2 * PRG_BANK_SIZE))
    prg[0 : len(program)] = program
    prg[-4:-2] = bytes([0x00, 0x80])  # RESET -> $8000
    prg[-2:] = bytes([0x00, 0x80])  # NMI

    header = bytearray(b"NES\x1a")
    header.extend([2, 1, 0, 0])
    header.extend(b"\x00" * 8)
    return bytes(header + prg + bytearray(CHR_BANK_SIZE))


class Controller2InputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _require_cuda()
        from nesle._cuda_core import CudaBatch

        cls.batch = CudaBatch(1, 1, build_controller_probe_rom())
        cls.batch.reset()

    def ram(self) -> np.ndarray:
        return np.frombuffer(bytes(self.batch.ram()), dtype=np.uint8)

    def press(self, p1: int, p2: int | None) -> np.ndarray:
        self.batch.reset()
        self.batch.step(
            np.array([p1], dtype=np.uint8),
            render_frame=False,
            copy_obs=False,
            actions2=None if p2 is None else np.array([p2], dtype=np.uint8),
        )
        ram = self.ram()
        return np.array([ram[P1_A], ram[P2_A], ram[P1_B], ram[P2_B]], dtype=np.uint8)

    def test_controller_2_channel_reaches_port_4017(self) -> None:
        # $0301 P2 A, $0303 P2 B, with player 1 idle.
        self.assertEqual(list(self.press(0, A)), [0, 1, 0, 0])
        self.assertEqual(list(self.press(0, B)), [0, 0, 0, 1])
        self.assertEqual(list(self.press(0, A | B)), [0, 1, 0, 1])

    def test_controller_1_channel_unchanged(self) -> None:
        # The existing single-channel behaviour must survive untouched.
        self.assertEqual(list(self.press(A, None)), [1, 0, 0, 0])
        self.assertEqual(list(self.press(B, None)), [0, 0, 1, 0])
        self.assertEqual(list(self.press(A | B, None)), [1, 0, 1, 0])

    def test_channels_are_independent(self) -> None:
        # Pressing one controller must not leak into the other: this is what
        # makes the channel usable for a two-player action space.
        self.assertEqual(list(self.press(A, B)), [1, 0, 0, 1])
        self.assertEqual(list(self.press(B, A)), [0, 1, 1, 0])
        self.assertEqual(list(self.press(0, 0)), [0, 0, 0, 0])

    def test_omitted_second_channel_holds_no_buttons(self) -> None:
        # Omitting actions2 must be identical to sending zeros, so every
        # single-player cartridge keeps behaving exactly as before.
        self.assertEqual(list(self.press(0, None)), list(self.press(0, 0)))

    def test_shape_validation(self) -> None:
        self.batch.reset()
        for kwargs in (
            {"actions": np.zeros(2, dtype=np.uint8)},
            {"actions": np.zeros(1, dtype=np.uint8), "actions2": np.zeros(3, dtype=np.uint8)},
        ):
            with self.assertRaises(ValueError):
                self.batch.step(**kwargs)

    def test_vec_env_passes_second_channel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rom = Path(tmp) / "probe.nes"
            rom.write_bytes(build_controller_probe_rom())
            env = nesle.make_vec(rom_path=str(rom), num_envs=1, backend="cuda", observation_mode="ram")
            try:
                index = {mask: i for i, mask in enumerate(env.action_masks)}
                right = index[encode_action(["right"])]
                a = index[encode_action(["a"])]
                env.step([right], [a])
                ram = np.frombuffer(bytes(env._cuda_batch.ram()), dtype=np.uint8)
                # Player 1 holds RIGHT (bit 7), player 2 holds A (bit 0). Only
                # player 2's A bit may be set, which isolates the two channels.
                self.assertEqual(
                    (int(ram[P1_A]), int(ram[P2_A]), int(ram[P1_B]), int(ram[P2_B])),
                    (0, 1, 0, 0),
                )
            finally:
                env.close()

    def test_vec_env_rejects_second_channel_on_host_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rom = Path(tmp) / "probe.nes"
            rom.write_bytes(build_controller_probe_rom())
            env = nesle.make_vec(
                rom_path=str(rom), num_envs=1, backend="synthetic", observation_mode="ram"
            )
            try:
                index = {mask: i for i, mask in enumerate(env.action_masks)}
                a = index[encode_action(["a"])]
                with self.assertRaises(NotImplementedError):
                    env.step([a], [a])
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
