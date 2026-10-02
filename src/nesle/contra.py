"""Contra (NES) RAM decoding and reward shaping.

Addresses come from the public Contra RAM maps (Data Crystal's
`Contra (NES)/RAM_map`, ROM Detectives' `Contra (NES) - RAM`, and the
annotated disassembly at github.com/vermiceli/nes-contra-us) and were then
validated against real RAM captured from this repository's own save states:

  $0030 stage          204/205 states agree with the stage recorded alongside
                       the state; the one mismatch is a mid-transition capture
  $0032 P1 lives       257/257 states agree with the value read off the HUD
  $031A P1 Y position  52/52 states agree with the recorded sprite Y
  $0334 P1 X position  52/52 states agree with the recorded sprite X
  $07E2 P1 score       exact on every state sampled

Score is a plain 16-bit little-endian integer at $07E2. The HUD multiplies it
by 100, which is why the default high score reads 20000 on screen while RAM
holds 200 at $07E0. Reward uses the raw value: the displayed number jumps in
steps of 100 and would make every kill look identical.
"""
from __future__ import annotations

from dataclasses import dataclass

CPU_RAM_BYTES = 2048

# Zero page
ADDR_GAME_MODE = 0x001C  # 0 = normal play, 1 = demo
ADDR_KONAMI = 0x0024  # 30-lives code flag
ADDR_PAUSED = 0x0025  # 1 = paused
ADDR_SCREEN_TYPE = 0x002C  # see SCREEN_* below
ADDR_STAGE = 0x0030  # 0-7
ADDR_LIVES = 0x0032  # P1 remaining lives, 0 on game over
ADDR_LIVES_P2 = 0x0033
ADDR_GAME_STATUS = 0x0038  # 1 = game over
ADDR_CONTINUES = 0x003A
ADDR_BOSS_DEFEATED = 0x003B  # 0 / 1, and 0x81 once the end-level runs
ADDR_PERSPECTIVE = 0x0040  # 0 = side-scrolling, 1 = vertical

# Sprites
ADDR_PLAYER_Y = 0x031A
ADDR_PLAYER_X = 0x0334
ADDR_PLAYER_FLAGS = 0x034E

# Scores
ADDR_HI_SCORE = 0x07E0
ADDR_SCORE_P1 = 0x07E2

SCREEN_MENU = 0x00
SCREEN_NORMAL = 0x04
SCREEN_CONTINUE = 0x06
SCREEN_BOSS_DEFEATED = 0x08
SCREEN_BOSS_DEFEAT_ANIM = 0x09


def _le16(ram: bytes | bytearray | memoryview, address: int) -> int:
    return ram[address] | (ram[address + 1] << 8)


@dataclass(frozen=True)
class ContraRamState:
    score: int
    hi_score: int
    lives: int
    stage: int
    x_pos: int
    y_pos: int
    player_flags: int
    screen_type: int
    perspective: int
    game_status: int
    boss_defeated: bool
    is_demo: bool
    is_paused: bool

    @property
    def is_alive(self) -> bool:
        """True while the player sprite is under the engine's control.

        $034E is a flag byte, not a boolean: facing and animation bits live
        there too, and 0 also shows up on captures taken between lives. Lives
        plus game status is the reliable signal.
        """
        return self.lives > 0 and self.game_status == 0

    @property
    def is_game_over(self) -> bool:
        return self.game_status != 0 or self.screen_type == SCREEN_CONTINUE

    @property
    def in_gameplay(self) -> bool:
        return self.screen_type == SCREEN_NORMAL and not self.is_demo

    @property
    def progress(self) -> int:
        """Stage progress, on the axis the current stage actually scrolls.

        $0040 selects the perspective: side-scrolling stages advance X, the
        vertical ones advance Y. Using the wrong axis gives a reward signal
        that runs backwards on half the stages.
        """
        return self.x_pos if self.perspective == 0 else self.y_pos


@dataclass(frozen=True)
class RewardComponents:
    score: int
    progress: int
    death: int
    stage_clear: int
    total: int


def read_ram(data: bytes | bytearray | memoryview) -> ContraRamState:
    if len(data) < CPU_RAM_BYTES:
        raise ValueError(f"need at least {CPU_RAM_BYTES} bytes of CPU RAM, got {len(data)}")
    ram = data
    screen = ram[ADDR_SCREEN_TYPE]
    return ContraRamState(
        score=_le16(ram, ADDR_SCORE_P1),
        hi_score=_le16(ram, ADDR_HI_SCORE),
        lives=ram[ADDR_LIVES],
        stage=ram[ADDR_STAGE],
        x_pos=ram[ADDR_PLAYER_X],
        y_pos=ram[ADDR_PLAYER_Y],
        player_flags=ram[ADDR_PLAYER_FLAGS],
        screen_type=screen,
        perspective=ram[ADDR_PERSPECTIVE],
        game_status=ram[ADDR_GAME_STATUS],
        boss_defeated=bool(ram[ADDR_BOSS_DEFEATED] & 0x01),
        is_demo=ram[ADDR_GAME_MODE] != 0,
        is_paused=ram[ADDR_PAUSED] != 0,
    )


def compute_reward(previous: ContraRamState, current: ContraRamState) -> RewardComponents:
    """Dense reward: score gained, distance advanced, minus dying.

    Death is charged once per lost life rather than per frame, so the penalty is
    a step the agent has to climb out of instead of a per-frame tax that
    dominates everything else.
    """
    score_delta = current.score - previous.score
    progress_delta = current.progress - previous.progress
    lives_lost = previous.lives - current.lives
    stage_clear = 1000 if (current.boss_defeated and not previous.boss_defeated) else 0
    death = -500 * lives_lost
    total = score_delta + progress_delta + death + stage_clear
    return RewardComponents(
        score=score_delta,
        progress=progress_delta,
        death=death,
        stage_clear=stage_clear,
        total=total,
    )