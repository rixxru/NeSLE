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
ADDR_PLAYER_MODE = 0x0022  # 0 = single player, 1 = two players
ADDR_KONAMI = 0x0024  # 30-lives code flag
ADDR_PAUSED = 0x0025  # 1 = paused
ADDR_SCREEN_TYPE = 0x002C  # see SCREEN_* below
ADDR_STAGE = 0x0030  # 0-7
ADDR_LIVES = 0x0032  # P1 remaining lives, 0 on game over
ADDR_LIVES_P2 = 0x0033
ADDR_GAME_STATUS = 0x0038  # 1 = game over (P1)
ADDR_GAME_STATUS_P2 = 0x0039  # 1 = P2 game over *or P2 not playing*
ADDR_CONTINUES = 0x003A
ADDR_BOSS_DEFEATED = 0x003B  # 0 / 1, and 0x81 once the end-level runs
ADDR_PERSPECTIVE = 0x0040  # 0 = side-scrolling, 1 = vertical

# Weapon, one byte per player: $00AA is player 1, $00AB is player 2. The type
# lives in the low nibble and bit 4 is the bullet-speed bonus, so $13 reads as
# "spread, with the speed bonus".
#
# These were not taken from the published map. Each value was poked into a real
# gameplay state and identified from the rendered frame at full NES resolution,
# by what actually comes out of the muzzle. Verified for both players: firing
# player 2 through `actions2` in a two-player state and poking $AB to 3 produces
# the same diverging pair as $AA=3 does for player 1.
ADDR_WEAPON = 0x00AA
ADDR_WEAPON_P2 = 0x00AB

# Layout of the byte. The game can only ever produce bits 0-2 and bit 4; every
# write site in the disassembly masks or clears the rest.
#
#   bits 0-2  weapon type. The disassembly masks the type with &$07.
#   bit  3    UNDEFINED. The game never sets it and never tests it - there is no
#             and #$08, ora #$08 or bit #$08 against it anywhere. Forcing it is
#             not a weapon and not invincibility; it corrupts a table index,
#             because the one place that reads the byte masks with &$0f instead
#             of &$07 (bank6.asm:304) and then uses the value as a routine
#             index. Observed consequences, both reproduced here:
#               raw 8   -> index 8, the spread routine, whose bullet creation
#                          reads past the end of its 6-entry sprite table and
#                          picks up $08/$09, which are the player's own spin-jump
#                          tiles. That is the "second jumping agent".
#               raw 9   -> index 9, the laser routine. The laser is a beam held
#                          at the muzzle rather than a travelling projectile,
#                          which is why the shot appears to freeze.
#             Indoors an extra #$05 pushes the index to 13-16, past the end of
#             the 10-entry dispatch table entirely.
#   bit  4    the R pickup, called "rapid fire" in the game but implemented as
#             bullet velocity: measured +33% (M goes 6 -> 8 px/frame). It also
#             seeds the per-bullet F_RAPID/S_RAPID flags at creation and halves
#             the indoor delay between bullets ($2a -> $15).
WEAPON_TYPE_MASK = 0x07
WEAPON_SPEED_BONUS = 0x10  # bit 4: the R pickup, bullet velocity +33%
WEAPON_FLAG_BIT3 = 0x08    # bit 3: undefined; never set by the game, unsafe to force

# Invincibility lives in its own timers, not in the weapon byte. The B pickup is
# routed around the weapon code entirely: it compares the item attribute against
# #$05 and branches straight to INVINCIBILITY_TIMER, so "B gives invincibility"
# never touches $00AA/$00AB. Two timers per player, indexed like the weapon
# bytes ($ae/$af and $b0/$b1 for player 1/2):
#   NEW_LIFE_INVINCIBILITY  set to #$80 on respawn, protects during the drop-in
#   INVINCIBILITY           the B pickup, #$80 normally and #$90 on level 7,
#                           decremented every 8 frames
ADDR_NEW_LIFE_INVINCIBILITY = 0x00AE
ADDR_INVINCIBILITY = 0x00B0
# The B pickup protects from damage but not from falling into a pit, which is
# the documented behaviour of the item rather than a quirk of this image.

WEAPON_DEFAULT = 0  # small white spheres
WEAPON_MACHINE_GUN = 1  # red spheres, fired one after another
WEAPON_FLAMETHROWER = 2  # white spheres the size of M, flying in a spiral
WEAPON_SPREAD = 3  # red spheres flying in a fan
WEAPON_LASER = 4  # sustained beam about the agent's height, yellow-orange-red

WEAPON_NAMES = {
    WEAPON_DEFAULT: "default",
    WEAPON_MACHINE_GUN: "M",
    WEAPON_FLAMETHROWER: "F",
    WEAPON_SPREAD: "S",
    WEAPON_LASER: "L",
}

# Type values the game does not handle. 5 drives the batch kernel into an
# illegal launch failure, 6 draws a blue sprite over the agent with no bullet at
# all, and 7 fires nothing. Treating any of them as a weapon is not an option.
WEAPON_INVALID = frozenset({5, 6, 7})

# Rate of fire is *not* a property of this byte. The game counts fire presses,
# not the held level, so holding the button gives one press: F and S then fire
# about 2 times per 100 frames, while pulsing the button - an emulator's turbo -
# gives about 20, roughly ten times as much. The machine gun is unaffected
# because its inter-shot delay is short enough that holding already saturates
# it. The disassembly shows this as per-bullet flags named for the two weapons
# that consume them, PLAYER_BULLET_F_RAPID ($0458) and PLAYER_BULLET_S_RAPID
# ($0488), both of which read 0 for a held button and non-zero for a pulsed one.
#
# Consequence for training: a policy that holds fire gets roughly a tenth of the
# damage available to it on F and S, and with frameskip > 1 a single step holds
# the button for several frames, which counts as one press - the pulse has to be
# expressed by alternating steps.
FIRE_RATE_NOTE = (
    "rate of fire depends on how the fire button is used, and the gate is "
    "per-weapon: M and L are level-triggered (CONTROLLER_STATE) so they fire "
    "for as long as the button is held, while standard, F and S are "
    "edge-triggered (CONTROLLER_STATE_DIFF) so holding fires exactly once and "
    "firing again needs a release. Measured on F and S, holding is about 10x "
    "slower than pulsing. M is capped near 49 shots per 100 frames by its own "
    "frame counter, which throttles after six consecutive bullets"
)

# Sprites. $031A and $0334 are 10-byte arrays of "each player sprite"; the first
# two entries are the players, so +1 is player 2. These are on-screen positions,
# not world coordinates.
ADDR_PLAYER_Y = 0x031A
ADDR_PLAYER_X = 0x0334
ADDR_PLAYER2_Y = 0x031B
ADDR_PLAYER2_X = 0x0335
ADDR_PLAYER_FLAGS = 0x034E

# Scores. Hi score, then P1, then P2, each 16-bit little-endian.
ADDR_HI_SCORE = 0x07E0
ADDR_SCORE_P1 = 0x07E2
ADDR_SCORE_P2 = 0x07E4

TWO_PLAYER = 1

SCREEN_MENU = 0x00
SCREEN_NORMAL = 0x04
SCREEN_CONTINUE = 0x06
SCREEN_BOSS_DEFEATED = 0x08
SCREEN_BOSS_DEFEAT_ANIM = 0x09


def _le16(ram: bytes | bytearray | memoryview, address: int) -> int:
    # Coerce to int: with a numpy-backed buffer these are np.uint8, and the
    # subtraction in the reward then wraps around instead of going negative.
    return int(ram[address]) | (int(ram[address + 1]) << 8)


def decode_weapon(raw: int) -> tuple[int, bool]:
    """Split a raw weapon byte into (type, has_speed_bonus)."""
    return int(raw) & WEAPON_TYPE_MASK, bool(int(raw) & WEAPON_SPEED_BONUS)


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
    two_player: bool
    # Player 2. Only meaningful when `two_player` is set - in a one-player game
    # these bytes are never initialised and hold leftovers (observed: score
    # 0xFFFF, lives 0x62, stale sprite coordinates).
    p2_score: int
    p2_lives: int
    p2_x_pos: int
    p2_y_pos: int
    p2_game_over: bool
    # Raw weapon bytes: player 1 at $00AA, player 2 at $00AB. Use weapon_type /
    # weapon_speed_bonus rather than masking by hand.
    weapon: int
    p2_weapon: int
    # Invincibility timers, in frames; 0 when not invincible.
    invincibility: int
    new_life_invincibility: int
    p2_invincibility: int
    p2_new_life_invincibility: int

    @property
    def weapon_type(self) -> int:
        """Player 1's weapon, without the speed-bonus bit."""
        return decode_weapon(self.weapon)[0]

    @property
    def weapon_speed_bonus(self) -> bool:
        """True when player 1 has the bullet-speed bonus."""
        return decode_weapon(self.weapon)[1]

    @property
    def p2_weapon_type(self) -> int:
        return decode_weapon(self.p2_weapon)[0]

    @property
    def p2_weapon_speed_bonus(self) -> bool:
        return decode_weapon(self.p2_weapon)[1]

    @property
    def weapon_name(self) -> str:
        kind = self.weapon_type
        return WEAPON_NAMES.get(kind, f"invalid({kind})")

    @property
    def p2_active(self) -> bool:
        """Whether player 2's fields can be read at all.

        $0022 PLAYER_MODE is the only reliable presence test. $0039 cannot be used
        for it: the game sets P2_GAME_OVER_STATUS to 1 in a one-player game,
        which is documented as "game over or player 2 not playing".
        """
        return self.two_player

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
    p2_score: int
    p2_progress: int
    total: int


def read_ram(data: bytes | bytearray | memoryview) -> ContraRamState:
    if len(data) < CPU_RAM_BYTES:
        raise ValueError(f"need at least {CPU_RAM_BYTES} bytes of CPU RAM, got {len(data)}")
    # int() everywhere for the same reason as _le16: a numpy uint8 buffer would
    # otherwise make every position a uint8 and wrap on subtraction.
    ram = data
    screen = int(ram[ADDR_SCREEN_TYPE])
    return ContraRamState(
        score=_le16(ram, ADDR_SCORE_P1),
        hi_score=_le16(ram, ADDR_HI_SCORE),
        lives=int(ram[ADDR_LIVES]),
        stage=int(ram[ADDR_STAGE]),
        x_pos=int(ram[ADDR_PLAYER_X]),
        y_pos=int(ram[ADDR_PLAYER_Y]),
        player_flags=int(ram[ADDR_PLAYER_FLAGS]),
        screen_type=screen,
        perspective=int(ram[ADDR_PERSPECTIVE]),
        game_status=int(ram[ADDR_GAME_STATUS]),
        boss_defeated=bool(int(ram[ADDR_BOSS_DEFEATED]) & 0x01),
        is_demo=int(ram[ADDR_GAME_MODE]) != 0,
        is_paused=int(ram[ADDR_PAUSED]) != 0,
        two_player=int(ram[ADDR_PLAYER_MODE]) == TWO_PLAYER,
        p2_score=_le16(ram, ADDR_SCORE_P2),
        p2_lives=int(ram[ADDR_LIVES_P2]),
        p2_x_pos=int(ram[ADDR_PLAYER2_X]),
        p2_y_pos=int(ram[ADDR_PLAYER2_Y]),
        p2_game_over=int(ram[ADDR_GAME_STATUS_P2]) != 0,
        p2_weapon=int(ram[ADDR_WEAPON_P2]),
        weapon=int(ram[ADDR_WEAPON]),
        invincibility=int(ram[ADDR_INVINCIBILITY]),
        new_life_invincibility=int(ram[ADDR_NEW_LIFE_INVINCIBILITY]),
        p2_invincibility=int(ram[ADDR_INVINCIBILITY + 1]),
        p2_new_life_invincibility=int(ram[ADDR_NEW_LIFE_INVINCIBILITY + 1]),
    )


DEATH_PENALTY = 500
STAGE_CLEAR_BONUS = 1000
# No legal move covers this many pixels in one step, so a larger delta means the
# sprite was relocated (respawn, warp) rather than walked.
MAX_PROGRESS_STEP = 64


def _progress_delta(
    previous: ContraRamState, current: ContraRamState, player2: bool
) -> int:
    """Distance advanced along the stage's scrolling axis.

    Returns 0 whenever the two states are not comparable: a changed stage, a
    flipped perspective or a different screen all mean the coordinate is now
    measuring something else, and the raw difference would be a large fake
    reward or a large fake penalty.
    """
    if previous.stage != current.stage or previous.perspective != current.perspective:
        return 0
    if previous.screen_type != current.screen_type:
        return 0
    if player2:
        if previous.perspective == 0:
            delta = current.p2_x_pos - previous.p2_x_pos
        else:
            delta = current.p2_y_pos - previous.p2_y_pos
    else:
        delta = current.progress - previous.progress
    if delta > MAX_PROGRESS_STEP or delta < -MAX_PROGRESS_STEP:
        return 0
    return delta


def compute_reward(previous: ContraRamState, current: ContraRamState) -> RewardComponents:
    """Dense reward: score gained, distance advanced, minus dying.

    Death is charged once per lost life rather than per frame, so the penalty is
    a step the agent has to climb out of instead of a per-frame tax that
    dominates everything else.

    This mirrors `cpp/include/nesle/cuda/batch_step.cuh` exactly; the two are
    compared against each other in tests/test_contra_reward.py. The difference
    between them is only *where* it runs: this one is host-side and allocates,
    the CUDA one runs inside the step kernel.
    """
    score_delta = current.score - previous.score
    if score_delta < 0:
        # Score only rises during play. Clamping absorbs the 16-bit wrap and the
        # reset on a continue screen instead of paying out a huge negative.
        score_delta = 0

    progress_delta = _progress_delta(previous, current, player2=False)

    lives_lost = previous.lives - current.lives
    death = -DEATH_PENALTY * max(lives_lost, 0)
    if lives_lost > 0:
        # Losing a life teleports the sprite back to the side of the stage; that
        # movement is not progress and must not be paid out.
        progress_delta = 0

    stage_clear = STAGE_CLEAR_BONUS if (current.boss_defeated and not previous.boss_defeated) else 0

    # Player 2's bytes are never initialised in a one-player game (observed:
    # lives 0x62, score 0xFFFF), so they are only read when the mode says two
    # players are in. $0039 cannot decide this: the game sets P2_GAME_OVER to 1
    # in one-player games too.
    p2_score_delta = 0
    p2_progress_delta = 0
    if current.two_player and previous.two_player:
        p2_score_delta = max(current.p2_score - previous.p2_score, 0)
        p2_progress_delta = _progress_delta(previous, current, player2=True)
        p2_lives_lost = previous.p2_lives - current.p2_lives
        if p2_lives_lost > 0:
            death -= DEATH_PENALTY * p2_lives_lost
            p2_progress_delta = 0

    total = (
        score_delta
        + progress_delta
        + death
        + stage_clear
        + p2_score_delta
        + p2_progress_delta
    )
    return RewardComponents(
        score=score_delta,
        progress=progress_delta,
        death=death,
        stage_clear=stage_clear,
        p2_score=p2_score_delta,
        p2_progress=p2_progress_delta,
        total=total,
    )
