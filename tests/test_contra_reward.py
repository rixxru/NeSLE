"""On-device Contra reward: parity with the Python reference, and invariants.

The reward exists twice - once in `cpp/include/nesle/cuda/batch_step.cuh`, where it
runs inside the step kernel, and once in `nesle.contra`, where it runs on the
host. They have to agree exactly, so this module drives the real CUDA batch over
the real Contra ROM and compares every step's reward against the reference
computed from the same RAM.

It also pins the two properties that are easy to regress silently:

  * the first reward after any reset is exactly zero, because there is no
    previous step to diff against, and
  * a stage change, a perspective flip or a screen change contributes no
    progress term, instead of a large fake delta.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

from nesle import contra

REPO_ROOT = Path(__file__).resolve().parents[1]
# The ROM and the two-player state are not part of the repository. Point these at
# a local copy to run the CUDA parity tests; they skip when it is absent. Both are
# overridable so a checkout on another machine does not have to edit the file, and so
# no personal path ends up committed.
CONTRA_ROM = Path(
    os.environ.get(
        "NESLE_CONTRA_ROM", r"C:\games\nes\Contra (U) [T-Rus uBAH009 (12.11.2016)].nes"
    )
)
CONTRA_STATE = Path(os.environ.get("NESLE_CONTRA_STATE", REPO_ROOT / "local" / "contra_2p.fcs"))


def _require_cuda() -> None:
    if not hasattr(np, "uint8"):
        raise unittest.SkipTest("complete numpy package is not available")
    try:
        import nesle._cuda_core  # noqa: F401
    except ImportError as exc:
        raise unittest.SkipTest(f"_cuda_core not available: {exc}")


def _make_batch(reward_kind: str, state: Path | None = None):
    from nesle._cuda_core import CudaBatch

    rom = CONTRA_ROM.read_bytes()
    if state is None:
        return CudaBatch(1, 1, rom, reward_kind=reward_kind)
    return CudaBatch(1, 1, rom, state.read_bytes(), reward_kind=reward_kind)


def _ram(batch) -> np.ndarray:
    return np.frombuffer(bytes(batch.ram()), dtype=np.uint8)


class ContraOnDeviceRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _require_cuda()
        if not CONTRA_ROM.is_file():
            raise unittest.SkipTest(f"Contra ROM not found at {CONTRA_ROM}")

    def test_matches_python_reference_over_a_real_episode(self) -> None:
        """Step the real ROM and check every reward against nesle.contra."""
        if not CONTRA_STATE.is_file():
            self.skipTest(f"two-player state not found at {CONTRA_STATE}")
        batch = _make_batch("contra", CONTRA_STATE)
        batch.reset()

        previous = contra.read_ram(_ram(batch))
        checked = 0
        nonzero = 0
        # A deterministic, varied action pattern so the run covers movement,
        # jumps and probably a death or a screen change.
        masks = [0x80, 0x80, 0x81, 0x00, 0x40, 0x02, 0x82, 0x00, 0x00, 0x01] * 6
        for mask in masks:
            out = batch.step(np.array([mask], dtype=np.uint8), render_frame=False, copy_obs=False)
            current = contra.read_ram(_ram(batch))
            expected = contra.compute_reward(previous, current)
            self.assertEqual(
                float(out["rewards"][0]),
                float(expected.total),
                msg=(
                    f"reward mismatch at step {checked}: cuda={out['rewards'][0]} "
                    f"reference={expected.total}\n"
                    f"prev={previous}\ncur={current}"
                ),
            )
            if expected.total != 0:
                nonzero += 1
            previous = current
            checked += 1

        self.assertGreater(checked, 0)
        # A run that never produced a single non-zero reward would make the
        # comparison above vacuous, so require it to have exercised the path.
        self.assertGreater(nonzero, 0, "no non-zero reward in the reference run")

    def test_two_player_reward_matches_reference(self) -> None:
        """Same comparison in two-player mode, where player 2 also scores."""
        if not CONTRA_STATE.is_file():
            self.skipTest(f"two-player state not found at {CONTRA_STATE}")
        batch = _make_batch("contra", CONTRA_STATE)
        batch.reset()

        right = 0x80
        previous = contra.read_ram(_ram(batch))
        self.assertTrue(previous.two_player, "fixture should be a two-player state")

        for i in range(12):
            p1 = right if i % 3 else 0
            p2 = right if i % 2 else 0
            out = batch.step(
                np.array([p1], dtype=np.uint8),
                render_frame=False,
                copy_obs=False,
                actions2=np.array([p2], dtype=np.uint8),
            )
            current = contra.read_ram(_ram(batch))
            expected = contra.compute_reward(previous, current)
            self.assertEqual(
                float(out["rewards"][0]),
                float(expected.total),
                msg=f"2P reward mismatch at step {i}",
            )
            previous = current

    def test_first_reward_after_reset_is_zero(self) -> None:
        """The invariant, on the real ROM: reset then step must pay nothing."""
        batch = _make_batch("contra")
        batch.reset()
        # Walk away from the baseline first, so a stale carried baseline would
        # show up as a large first reward.
        for _ in range(8):
            batch.step(
                np.array([0x80], dtype=np.uint8), render_frame=False, copy_obs=False
            )
        for _ in range(3):
            batch.reset()
            out = batch.step(
                np.array([0x80], dtype=np.uint8), render_frame=False, copy_obs=False
            )
            self.assertEqual(float(out["rewards"][0]), 0.0)

    def test_reset_envs_first_reward_is_zero(self) -> None:
        """Same invariant for the per-env auto-reset path, not just reset()."""
        batch = _make_batch("contra")
        batch.reset()
        for _ in range(6):
            batch.step(np.array([0x80], dtype=np.uint8), render_frame=False, copy_obs=False)
        out = batch.step(
            np.zeros(1, dtype=np.uint8), render_frame=False, copy_obs=False
        )
        self.assertEqual(float(out["rewards"][0]), 0.0)
        batch.reset_envs(np.ones(1, dtype=np.uint8))
        out = batch.step(np.array([0x80], dtype=np.uint8), render_frame=False, copy_obs=False)
        self.assertEqual(float(out["rewards"][0]), 0.0)

    def test_reward_kind_none_and_auto_are_zero_for_contra(self) -> None:
        """Contra is not an SMB image, so 'auto' must keep paying nothing."""
        for kind in ("none", "auto"):
            batch = _make_batch(kind)
            batch.reset()
            for _ in range(4):
                out = batch.step(
                    np.array([0x80], dtype=np.uint8), render_frame=False, copy_obs=False
                )
                self.assertEqual(float(out["rewards"][0]), 0.0, f"reward_kind={kind}")
                self.assertEqual(int(out["dones"][0]), 0, f"reward_kind={kind}")

    def test_reward_kind_validation(self) -> None:
        from nesle._cuda_core import CudaBatch

        rom = CONTRA_ROM.read_bytes()
        with self.assertRaises(ValueError):
            CudaBatch(1, 1, rom, reward_kind="smb")
        with self.assertRaises(ValueError):
            CudaBatch(1, 1, rom, reward_kind="not-a-reward")
        with self.assertRaises(ValueError):
            CudaBatch(1, 1, rom, reward_kind=7)
        # 'auto' is the default and must stay accepted.
        CudaBatch(1, 1, rom)
        CudaBatch(1, 1, rom, reward_kind="auto")


class ContraRewardUnitTests(unittest.TestCase):
    """Host-side unit tests for the guards, mirroring the CUDA implementation."""

    @staticmethod
    def _ram(**overrides: int) -> bytearray:
        """Synthesize a mid-gameplay RAM image; override single addresses by name."""
        fields = {
            contra.ADDR_GAME_MODE: 0,
            contra.ADDR_SCREEN_TYPE: contra.SCREEN_NORMAL,
            contra.ADDR_STAGE: 1,
            contra.ADDR_LIVES: 3,
            contra.ADDR_LIVES_P2: 3,
            contra.ADDR_PLAYER_X: 25,
            contra.ADDR_PLAYER_Y: 45,
            contra.ADDR_PLAYER2_X: 32,
            contra.ADDR_PLAYER2_Y: 32,
        }
        ram = bytearray(contra.CPU_RAM_BYTES)
        for address, value in fields.items():
            ram[address] = value
        for name, value in overrides.items():
            ram[getattr(contra, name)] = value
        return ram

    @staticmethod
    def _set_score(ram: bytearray, address: int, value: int) -> bytearray:
        ram[address] = value & 0xFF
        ram[address + 1] = (value >> 8) & 0xFF
        return ram

    def test_stage_change_suppresses_progress(self) -> None:
        before = contra.read_ram(self._ram())
        after = contra.read_ram(self._ram(ADDR_STAGE=3, ADDR_PLAYER_X=200))
        reward = contra.compute_reward(before, after)
        self.assertEqual(reward.progress, 0)
        self.assertEqual(reward.total, 0)

    def test_perspective_flip_suppresses_progress(self) -> None:
        before = contra.read_ram(self._ram())
        # Flipping to the vertical axis makes the old X meaningless.
        after = contra.read_ram(self._ram(ADDR_PERSPECTIVE=1, ADDR_PLAYER_Y=200))
        self.assertEqual(contra.compute_reward(before, after).progress, 0)

    def test_screen_change_suppresses_progress(self) -> None:
        before = contra.read_ram(self._ram())
        after = contra.read_ram(
            self._ram(ADDR_SCREEN_TYPE=contra.SCREEN_BOSS_DEFEATED, ADDR_PLAYER_X=200)
        )
        self.assertEqual(contra.compute_reward(before, after).progress, 0)

    def test_teleport_is_not_progress(self) -> None:
        before = contra.read_ram(self._ram())
        after = contra.read_ram(self._ram(ADDR_PLAYER_X=25 + contra.MAX_PROGRESS_STEP + 1))
        self.assertEqual(contra.compute_reward(before, after).progress, 0)

    def test_score_decrease_is_clamped_to_zero(self) -> None:
        """A 16-bit wrap or a continue-screen reset must not pay a huge penalty."""
        before = self._ram()
        self._set_score(before, contra.ADDR_SCORE_P1, 300)
        after = self._set_score(bytearray(before), contra.ADDR_SCORE_P1, 10)
        reward = contra.compute_reward(contra.read_ram(before), contra.read_ram(after))
        self.assertEqual(reward.score, 0)
        # A 16-bit wraparound reads as a huge negative difference and is clamped
        # to zero, the same as a continue-screen reset.
        wrapped_before = self._ram()
        self._set_score(wrapped_before, contra.ADDR_SCORE_P1, 65500)
        wrapped_after = self._set_score(bytearray(wrapped_before), contra.ADDR_SCORE_P1, 36)
        wrapped = contra.compute_reward(
            contra.read_ram(wrapped_before), contra.read_ram(wrapped_after)
        )
        self.assertEqual(wrapped.score, 0)
        self.assertEqual(wrapped.total, 0)

    def test_death_suppresses_progress(self) -> None:
        """Losing a life respawns the sprite; that jump is not progress."""
        before = contra.read_ram(self._ram())
        after = contra.read_ram(self._ram(ADDR_LIVES=2, ADDR_PLAYER_X=30))
        reward = contra.compute_reward(before, after)
        self.assertEqual(reward.progress, 0)
        self.assertEqual(reward.death, -contra.DEATH_PENALTY)

    def test_player2_is_ignored_in_one_player_games(self) -> None:
        before = self._ram()
        # $0022 stays 0, but the P2 bytes hold the garbage a 1P game leaves.
        after = self._ram(ADDR_LIVES_P2=0x62)
        self._set_score(after, contra.ADDR_SCORE_P2, 0xFFFF)
        reward = contra.compute_reward(contra.read_ram(before), contra.read_ram(after))
        self.assertEqual(reward.p2_score, 0)
        self.assertEqual(reward.death, 0)
        self.assertEqual(reward.total, 0)

    def test_player2_scores_and_progress_are_counted(self) -> None:
        before = self._ram(ADDR_PLAYER_MODE=1)
        after = self._ram(ADDR_PLAYER_MODE=1, ADDR_PLAYER2_X=40)
        self._set_score(after, contra.ADDR_SCORE_P2, 7)
        reward = contra.compute_reward(contra.read_ram(before), contra.read_ram(after))
        self.assertEqual(reward.p2_score, 7)
        self.assertEqual(reward.p2_progress, 8)
        self.assertEqual(reward.total, 15)

    def test_player2_life_loss_is_charged(self) -> None:
        before = self._ram(ADDR_PLAYER_MODE=1)
        after = self._ram(ADDR_PLAYER_MODE=1, ADDR_LIVES_P2=2)
        reward = contra.compute_reward(contra.read_ram(before), contra.read_ram(after))
        self.assertEqual(reward.death, -contra.DEATH_PENALTY)


if __name__ == "__main__":
    unittest.main()
