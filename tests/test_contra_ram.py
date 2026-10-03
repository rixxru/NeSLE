"""Contra RAM decoding, checked against real save states and real play.

The states under C:\\games\\nes are not part of the repository, so these tests
synthesize the RAM they need and assert the exact bytes each field reads. The
values come from real captures: 428 FCEUX 2.6 states of
`Contra (U) [T-Rus uBAH009 (12.11.2016)].nes`, cross-checked against the HUD
values recorded alongside them (lives 257/257, sprite X 52/52, sprite Y 52/52,
stage 204/205) and against the published Contra RAM maps.
"""
from __future__ import annotations

import unittest

from nesle.contra import (
    ADDR_BOSS_DEFEATED,
    ADDR_CONTINUES,
    ADDR_GAME_MODE,
    ADDR_GAME_STATUS,
    ADDR_HI_SCORE,
    ADDR_LIVES,
    ADDR_PAUSED,
    ADDR_PERSPECTIVE,
    ADDR_PLAYER_FLAGS,
    ADDR_PLAYER_X,
    ADDR_PLAYER_Y,
    ADDR_SCREEN_TYPE,
    ADDR_STAGE,
    ADDR_WEAPON,
    ADDR_WEAPON_P2,
    ContraRamState,
    WEAPON_FLAG_BIT3,
    WEAPON_DEFAULT,
    WEAPON_FLAMETHROWER,
    WEAPON_INVALID,
    WEAPON_LASER,
    WEAPON_MACHINE_GUN,
    WEAPON_NAMES,
    WEAPON_SPREAD,
    WEAPON_SPEED_BONUS,
    WEAPON_TYPE_MASK,
    compute_reward,
    decode_weapon,
    read_ram,
)

CONTRA_RAM = 2048

# A mid-gameplay baseline: 2D stage 1, three lives, normal screen, player at
# (25, 45). Tests override single fields by address.
BASE_GAMEPLAY = {
    ADDR_LIVES: 3,
    ADDR_STAGE: 1,
    ADDR_SCREEN_TYPE: 0x04,
    ADDR_PERSPECTIVE: 0,
    ADDR_PLAYER_X: 25,
    ADDR_PLAYER_Y: 45,
}


def play_state(overrides: dict[int, int] | None = None, score: int = 0):
    fields = dict(BASE_GAMEPLAY)
    if overrides:
        fields.update(overrides)
    ram = bytearray(CONTRA_RAM)
    for address, value in fields.items():
        ram[address] = value
    ram[0x7E2] = score & 0xFF
    ram[0x7E3] = (score >> 8) & 0xFF
    return read_ram(ram)


class ContraRamTests(unittest.TestCase):
    def test_zeroed_ram_is_not_a_game_in_progress(self) -> None:
        """Power-on RAM must not look like play, or an agent gets dense reward
        for standing on the title screen. It is the menu, not a game over:
        $0032 == 0 means "last life" during real play, so lives alone cannot
        decide this."""
        state = read_ram(bytearray(CONTRA_RAM))
        self.assertEqual(state.score, 0)
        self.assertEqual(state.lives, 0)
        self.assertEqual(state.screen_type, 0x00)  # menu
        self.assertFalse(state.in_gameplay)
        self.assertFalse(state.is_game_over)
        self.assertFalse(state.is_alive)

    def test_reads_lives_stage_and_position(self) -> None:
        state = play_state()
        self.assertEqual(state.lives, 3)
        self.assertEqual(state.stage, 1)
        self.assertEqual(state.x_pos, 25)
        self.assertEqual(state.y_pos, 45)
        self.assertTrue(state.in_gameplay)
        self.assertTrue(state.is_alive)

    def test_score_is_raw_little_endian(self) -> None:
        self.assertEqual(play_state(score=194).score, 194)
        self.assertEqual(play_state(score=0x1234).score, 0x1234)

    def test_hud_shows_score_times_100(self) -> None:
        """Why the reward uses the raw value: the HUD multiplies it by 100."""
        self.assertEqual(play_state(score=194).score * 100, 19400)
        ram = bytearray(CONTRA_RAM)
        ram[ADDR_HI_SCORE] = 200
        self.assertEqual(read_ram(ram).hi_score * 100, 20000)

    def test_progress_follows_the_perspective_axis(self) -> None:
        side = play_state({ADDR_PERSPECTIVE: 0, ADDR_PLAYER_X: 100, ADDR_PLAYER_Y: 200})
        self.assertEqual(side.progress, 100)
        vertical = play_state({ADDR_PERSPECTIVE: 1, ADDR_PLAYER_X: 100, ADDR_PLAYER_Y: 200})
        self.assertEqual(vertical.progress, 200)

    def test_demo_and_pause_are_not_gameplay(self) -> None:
        demo = play_state({ADDR_GAME_MODE: 1})
        self.assertTrue(demo.is_demo)
        self.assertFalse(demo.in_gameplay)
        paused = play_state({ADDR_PAUSED: 1})
        self.assertTrue(paused.is_paused)

    def test_game_over_via_status_and_via_continue_screen(self) -> None:
        self.assertTrue(play_state({ADDR_GAME_STATUS: 1}).is_game_over)
        cont = play_state({ADDR_SCREEN_TYPE: 0x06, ADDR_CONTINUES: 2})
        self.assertTrue(cont.is_game_over)

    def test_player_flags_zero_is_not_a_death_signal(self) -> None:
        """$034E also carries facing and animation bits."""
        self.assertTrue(play_state({ADDR_PLAYER_FLAGS: 0}).is_alive)

    def test_rejects_short_ram(self) -> None:
        with self.assertRaises(ValueError):
            read_ram(bytearray(1024))


class ContraRewardTests(unittest.TestCase):
    def test_score_gain_is_the_main_term(self) -> None:
        reward = compute_reward(play_state(score=194), play_state(score=294))
        self.assertEqual(reward.score, 100)
        self.assertEqual(reward.progress, 0)
        self.assertEqual(reward.death, 0)
        self.assertEqual(reward.total, 100)

    def test_progress_reward_and_backwards_penalty(self) -> None:
        fwd = compute_reward(play_state(), play_state({ADDR_PLAYER_X: 28}))
        back = compute_reward(play_state(), play_state({ADDR_PLAYER_X: 20}))
        self.assertEqual(fwd.progress, 3)
        self.assertEqual(back.progress, -5)
        self.assertEqual(fwd.total, 3)

    def test_vertical_stage_progress_uses_y(self) -> None:
        before = play_state({ADDR_PERSPECTIVE: 1, ADDR_PLAYER_Y: 100})
        after = play_state({ADDR_PERSPECTIVE: 1, ADDR_PLAYER_Y: 140})
        self.assertEqual(compute_reward(before, after).progress, 40)

    def test_death_is_charged_once_per_lost_life(self) -> None:
        reward = compute_reward(play_state({ADDR_LIVES: 3}), play_state({ADDR_LIVES: 2}))
        self.assertEqual(reward.death, -500)
        self.assertEqual(reward.total, -500)

    def test_stage_clear_bonus_is_one_shot(self) -> None:
        before = play_state()
        cleared = play_state({ADDR_BOSS_DEFEATED: 1})
        self.assertEqual(compute_reward(before, cleared).stage_clear, 1000)
        self.assertEqual(compute_reward(cleared, cleared).stage_clear, 0)

    def test_no_reward_on_a_static_frame(self) -> None:
        state = play_state()
        self.assertEqual(compute_reward(state, state).total, 0)


class ContraPlayer2Tests(unittest.TestCase):
    """Addresses come from the annotated disassembly (vermiceli/nes-contra-us):
    $001A/$0334 are 10-byte arrays of "each player sprite" whose first two entries
    are the players, so +1 is player 2.

    Verified by patching $0022 to 1 inside a real FCEUX state: $0335/$031B go from
    a stale (0,0) to live, moving coordinates and the HUD grows a second set of
    life icons.
    """

    def two_player_state(self, overrides=None) -> ContraRamState:
        fields = dict(BASE_GAMEPLAY)
        fields[0x22] = 0x01  # PLAYER_MODE
        fields[0x33] = 0x03  # P2 lives
        fields[0x39] = 0x00  # P2 not game over
        fields[0x335] = 32  # P2 X
        fields[0x31B] = 32  # P2 Y
        if overrides:
            fields.update(overrides)
        ram = bytearray(CONTRA_RAM)
        for address, value in fields.items():
            ram[address] = value
        ram[0x7E2] = 194
        ram[0x7E3] = 0
        ram[0x7E4] = 7  # P2 score low
        ram[0x7E5] = 0
        return read_ram(ram)

    def test_player2_fields_are_read(self) -> None:
        state = self.two_player_state()
        self.assertTrue(state.two_player)
        self.assertTrue(state.p2_active)
        self.assertEqual(state.p2_lives, 3)
        self.assertEqual(state.p2_x_pos, 32)
        self.assertEqual(state.p2_y_pos, 32)
        self.assertEqual(state.p2_score, 7)
        self.assertFalse(state.p2_game_over)

    def test_player1_is_unaffected_by_player2(self) -> None:
        state = self.two_player_state()
        self.assertEqual(state.score, 194)
        self.assertEqual(state.lives, 3)  # from BASE_GAMEPLAY
        self.assertEqual(state.x_pos, 25)

    def test_one_player_flags_player2_inactive(self) -> None:
        state = play_state()
        self.assertFalse(state.two_player)
        self.assertFalse(state.p2_active)

    def test_player2_status_is_always_set_in_one_player_games(self) -> None:
        """$0039 reads 1 in 447 of 448 real one-player states, because the game
        uses it for "P2 game over *or P2 not playing*". It cannot be used to
        detect whether P2 exists."""
        fields = dict(BASE_GAMEPLAY)
        fields[0x22] = 0x00
        fields[0x39] = 0x01
        ram = bytearray(CONTRA_RAM)
        for address, value in fields.items():
            ram[address] = value
        state = read_ram(ram)
        self.assertTrue(state.p2_game_over)
        self.assertFalse(state.p2_active)

    def test_uninitialised_player2_bytes_are_preserved_not_sanitised(self) -> None:
        """In a one-player game these bytes hold leftovers. The decoder reports
        them as-is and gates on PLAYER_MODE rather than guessing."""
        fields = dict(BASE_GAMEPLAY)
        fields[0x33] = 0x62  # observed leftover
        fields[0x7E4] = 0xFF
        fields[0x7E5] = 0xFF
        ram = bytearray(CONTRA_RAM)
        for address, value in fields.items():
            ram[address] = value
        state = read_ram(ram)
        self.assertEqual(state.p2_lives, 0x62)
        self.assertEqual(state.p2_score, 0xFFFF)
        self.assertFalse(state.p2_active)


class ContraWeaponTests(unittest.TestCase):
    """Weapon bytes, identified by what actually leaves the muzzle.

    The addresses and the type/flag split were established empirically on real
    gameplay states: each value was poked into $00AA and read off the rendered
    frame at full NES resolution. $00AB was then confirmed for player 2 by
    firing through `actions2` in a two-player state.
    """

    @staticmethod
    def _state(weapon: int, weapon_p2: int | None = None):
        ram = bytearray(CONTRA_RAM)
        ram[ADDR_SCREEN_TYPE] = 0x04
        ram[ADDR_LIVES] = 3
        ram[ADDR_WEAPON] = weapon
        if weapon_p2 is not None:
            ram[ADDR_WEAPON_P2] = weapon_p2
        return read_ram(ram)

    def test_player1_and_player2_use_adjacent_bytes(self) -> None:
        self.assertEqual(ADDR_WEAPON, 0x00AA)
        self.assertEqual(ADDR_WEAPON_P2, 0x00AB)

    def test_every_weapon_type_decodes(self) -> None:
        for raw, expected, name in (
            (0x00, WEAPON_DEFAULT, "default"),
            (0x01, WEAPON_MACHINE_GUN, "M"),
            (0x02, WEAPON_FLAMETHROWER, "F"),
            (0x03, WEAPON_SPREAD, "S"),
            (0x04, WEAPON_LASER, "L"),
        ):
            state = self._state(raw)
            self.assertEqual(state.weapon, raw)
            self.assertEqual(state.weapon_type, expected)
            self.assertEqual(state.weapon_name, name)
            self.assertFalse(state.weapon_speed_bonus)

    def test_speed_bonus_is_a_separate_bit(self) -> None:
        # 0x13 came out of a real two-player state: spread, with the bonus.
        state = self._state(0x13)
        self.assertEqual(state.weapon_type, WEAPON_SPREAD)
        self.assertTrue(state.weapon_speed_bonus)
        self.assertEqual(state.weapon_name, "S")

        # The bonus must not disturb the type of any other weapon.
        for raw in (0x00, 0x01, 0x02, 0x03, 0x04):
            self.assertEqual(decode_weapon(raw | WEAPON_SPEED_BONUS), (raw, True))

    def test_type_is_the_low_three_bits(self) -> None:
        # The disassembly masks the type with &$07, and bit 4 is the speed
        # bonus, so the type cannot be the low *four* bits.
        self.assertEqual(WEAPON_TYPE_MASK, 0x07)
        self.assertEqual(decode_weapon(0x13), (WEAPON_SPREAD, True))
        self.assertEqual(decode_weapon(0x10), (WEAPON_DEFAULT, True))
        # Bit 3 is a separate unknown flag, so a raw 8 is type 0 plus bit 3 -
        # not a type 8. That also explains the duplicate-sprite glitch seen at
        # $AA=8: it is the bit, not the type.
        self.assertEqual(decode_weapon(0x08), (WEAPON_DEFAULT, False))
        self.assertEqual(decode_weapon(0x09), (WEAPON_MACHINE_GUN, False))
        self.assertEqual(self._state(0x08).weapon_type, WEAPON_DEFAULT)
        self.assertEqual(self._state(0x08).weapon, 0x08)
        self.assertTrue(self._state(0x08).weapon & WEAPON_FLAG_BIT3)

    def test_invalid_values_are_not_weapons(self) -> None:
        # 5 crashes the batch kernel, 6 paints a blue sprite over the agent,
        # 7 fires nothing. They are recorded so nothing treats them as types.
        for raw in sorted(WEAPON_INVALID):
            self.assertNotIn(raw, WEAPON_NAMES)
            self.assertEqual(self._state(raw).weapon_type, raw)
            self.assertTrue(self._state(raw).weapon_name.startswith("invalid"))
        # Type 4 is the last valid one.
        self.assertEqual(WEAPON_LASER, max(WEAPON_NAMES))
        self.assertFalse(WEAPON_LASER in WEAPON_INVALID)

    def test_player2_weapon_decodes_independently(self) -> None:
        state = self._state(0x01, weapon_p2=0x14)
        self.assertEqual(state.weapon_type, WEAPON_MACHINE_GUN)
        self.assertFalse(state.weapon_speed_bonus)
        self.assertEqual(state.p2_weapon, 0x14)
        self.assertEqual(state.p2_weapon_type, WEAPON_LASER)
        self.assertTrue(state.p2_weapon_speed_bonus)

    def test_player1_weapon_is_read_from_aa_not_ab(self) -> None:
        # Guards the mix-up that started this: $AB used to be labelled player 2
        # with no evidence, and player 1 had no weapon field at all.
        state = self._state(0x00, weapon_p2=0x02)
        self.assertEqual(state.weapon, 0x00)
        self.assertEqual(state.weapon_type, WEAPON_DEFAULT)
        self.assertEqual(state.p2_weapon, 0x02)
        self.assertEqual(state.p2_weapon_type, WEAPON_FLAMETHROWER)


if __name__ == "__main__":
    unittest.main()
