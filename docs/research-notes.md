# Research Notes

## CuLE Takeaways

CuLE demonstrates the thesis we want for NES: keep emulator state and rendered
frames resident on the GPU, run thousands of environments in parallel, and avoid
CPU/GPU observation transfer on the inference path. The paper reports up to
155M raw Atari frames per hour on one GPU and explicitly attributes the win to
GPU-side emulation, GPU-side rendering, and batching.

Important design lessons from CuLE:

- Use one logical emulator per GPU thread first. It is not theoretically optimal,
  but it is debuggable and already enough to beat CPU emulation at large env
  counts.
- Split execution kernels when their resource profiles differ. CuLE separates
  CPU/game execution from TIA rendering; NeSLE should separate 2A03 CPU/APU/PPU
  register execution from PPU frame rendering.
- Expect warp divergence after random policies decorrelate environments. Measure
  it and use reset-state caches, env grouping, and batch sizing to reduce harm.
- Render only when needed. RL commonly frame-skips and uses max-pooled or stacked
  frames, so many raw frames do not need full RGB output.

Sources:

- CuLE paper: https://arxiv.org/abs/1907.08467
- CuLE repo: https://github.com/NVlabs/cule

## NES Scope For Mario First

Super Mario Bros. is an NROM game, which is the right first target. NROM has no
bank switching, fixed PRG ROM, fixed CHR ROM, no mapper IRQs, and no cartridge
audio. That lets the first emulator avoid the hardest mapper problems while
still being a real NES emulator.

Core hardware needed for Mario:

- 2A03 CPU: NMOS 6502 core without decimal mode, plus NMI handling.
- CPU memory map: 2 KB internal RAM mirrored through `$1FFF`, PPU registers at
  `$2000-$2007` mirrored through `$3FFF`, APU/input around `$4000-$4017`, and
  cartridge space from `$4020-$FFFF`.
- PPU registers and enough cycle behavior for NMI, vblank, OAMDMA, scrolling,
  sprite 0 hit, background, and sprite rendering.
- Standard controller serial protocol through `$4016/$4017`.
- NROM mapper 0 with 16 KB or 32 KB PRG ROM and 8 KB CHR ROM.

Timing checkpoints for the initial NTSC model:

- NTSC PPU advances 3 dots per CPU cycle.
- Each scanline has 341 PPU dots and each no-rendering frame has 262 scanlines.
- Vblank flag is set at scanline 241, dot 1.
- Vblank flag is cleared at scanline 261, dot 1.
- The odd-frame skipped PPU dot and `$2002` vblank race behavior are deferred
  until the boot path needs that precision.

Sources:

- NESdev CPU memory map: https://www.nesdev.org/wiki/CPU_memory_map
- NESdev PPU registers: https://www.nesdev.org/wiki/PPU_registers
- NESdev PPU rendering: https://www.nesdev.org/wiki/PPU_rendering
- NESdev PPU frame timing: https://www.nesdev.org/wiki/PPU_frame_timing
- NESdev cycle reference: https://www.nesdev.org/wiki/Cycle_reference_chart
- NESdev NMI: https://www.nesdev.org/wiki/NMI
- NESdev controller reading: https://www.nesdev.org/wiki/Controller_reading
- NESdev iNES format: https://www.nesdev.org/wiki/INES
- NESdev NROM: https://www.nesdev.org/wiki/NROM
- 6502 opcode reference: https://www.nesdev.org/obelisk-6502-guide/reference.html
- Klaus Dormann tests: https://github.com/Klaus2m5/6502_65C02_functional_tests

## Mario RL Surface

Mario reward and info should match the existing learning ecosystem before we
optimize. The useful RAM values are already established by `gym-super-mario-bros`
and Data Crystal:

- x position: `ram[0x006D] * 256 + ram[0x0086]`
- time: decimal digits at `0x07F8..0x07FA`
- coins: decimal digits at `0x07ED..0x07EE`
- world/stage/area: `0x075F`, `0x075C`, `0x0760`
- status: `0x0756`
- player state/death: `0x000E`, `0x00B5`

The baseline reward is progress plus time delta plus death penalty:

```text
r = x_delta + time_delta + death_penalty
```

with reset/death x jumps filtered out.

Sources:

- Mario RAM map: https://datacrystal.tcrf.net/wiki/Super_Mario_Bros./RAM_map
- nes-py: https://github.com/Kautenja/nes-py
- gym-super-mario-bros reward/API: https://github.com/Kautenja/gym-super-mario-bros

## Contra

The second scored game. `nesle/contra.py` decodes its RAM and
`batch_step.cuh` scrapes the same addresses on the device; the two are compared
step-by-step against each other in `tests/test_contra_reward.py`.

Addresses were cross-checked against the published maps and then validated
against this repository's own FCEUX captures of
`Contra (U) [T-Rus uBAH009 (12.11.2016)].nes`: stage agreed on 204 of 205
states (the one mismatch is a mid-transition capture), lives on 257 of 257, and
sprite X and Y on 52 of 52 each.

The parts worth remembering because they are easy to get wrong:

- **Score is the raw 16-bit value at `$07E2`.** The HUD displays it times 100,
  so using the displayed number makes every kill worth exactly the same.
- **`$0040` selects the scrolling axis.** Side-scrolling stages advance X,
  vertical ones advance Y. Reading the wrong axis makes the reward run backwards
  on half the stages.
- **`$0022` is the only reliable player-2 presence test.** `$0039` is not: the
  game sets P2 game-over status to 1 in one-player games too, and the P2 fields
  themselves are never initialised in a 1P game (observed lives `0x62`, score
  `0xFFFF`).
- **`$034E` is a flag byte, not a boolean** - facing and animation bits live
  there too, and 0 also shows up between lives, so lives plus `$0038` is the
  reliable death signal.
- **Game mode `$001C` non-zero means the attract demo**, which is not a
  playable episode and must not be scored as one.
- **Fire is bit 1 and jump is bit 0** in this image, i.e. swapped relative to
  the usual NES convention where A is bit 0. Found by poking one button at a
  time: bit 0 produces a jump arc in `$031A` with lives unchanged, bit 1
  produces bullets. Worth knowing before writing a policy that assumes
  `actions[i] & 1` is fire.
- **In this ROM the buttons do nothing in some save states.** A state whose
  players have not finished spawning still accepts d-pad input but ignores
  fire, and no amount of holding the button changes anything. Let the state run
  ~60 idle frames before concluding that firing is broken.

### Weapons

`$00AA` is player 1's weapon and `$00AB` is player 2's: adjacent bytes, same
encoding for both. The type is the low nibble and bit 4 (`0x10`) is a separate
bullet-speed bonus, so a real two-player capture read `$13` = "spread, with the
bonus". Measured effect of the bonus on the machine gun: 6 -> 8 px/frame, +33%.

| type | weapon | what leaves the muzzle |
| --- | --- | --- |
| 0 | default | small white spheres, 4 at a time |
| 1 | M | red spheres, fired one after another |
| 2 | F | white spheres the size of M, flying in a spiral |
| 3 | S | red spheres flying in a fan |
| 4 | L | sustained beam about the agent's height, yellow-orange-red |

Weapon byte layout, established empirically and cross-checked against the
disassembly, which masks the type with `&$07`:

| bits | meaning |
| --- | --- |
| 0-2 | weapon type |
| 3 | unknown flag; **not** rapid fire. On M it stops the shot dead at the muzzle, and alone (raw 8) it spawns a duplicate jumping agent sprite |
| 4 | bullet-speed bonus, +33% (M measured 6 -> 8 px/frame) |

Types: 0 default, 1 M, 2 F, 3 S, 4 L. Type values the game does not handle:
**5** drives the batch kernel into an `unspecified launch failure`, 6 draws a
blue sprite over the agent with no bullet at all, and 7 fires nothing. Type 4 is
the last valid one.

Independent confirmation from the annotated disassembly, which has the canonical
table (`weapon_strength`, `bank7.asm`) in the same order the frames show:

```
.byte $00 ; Regular = Weak
.byte $02 ; M      = Strong
.byte $01 ; F      = Medium
.byte $03 ; S      = Very Strong
.byte $02 ; L      = Strong
```

Type index `0,1,2,3,4` = regular, M, F, S, L, matching what came out of the
muzzle.

These were identified empirically, not read off the published map: each value
was poked into `$00AA` in a real gameplay state and named from the rendered
frame at full NES resolution, by what actually comes out of the muzzle. `$00AB`
was then confirmed for player 2 by firing through `actions2` in a two-player
state - `$AB=0` gives a single small white sphere, `$AB=3` the same diverging
pair as `$AA=3` does for player 1.

That matters because the map had it wrong in a way that would have been easy to
ship: `$AB` was labelled player 2's weapon with no evidence behind it, and
player 1 had no weapon field at all.

### Rate of fire, and the rapid flags

Two separate mechanisms are easy to confuse, and the game's own naming does not
help - the R pickup is called "rapid fire" but is implemented as bullet velocity.

**Bit 4 is the R pickup.** `ram.asm:735` says "bit 4 set for rapid fire", the R
item stores `#$10` into the weapon byte (`bank7.asm:6925`), and `bank6.asm:349`
extracts it into `$09`, which selects `bullet_velocity_rapid` over the normal
one. Measured, that is the +33% (M goes 6 -> 8 px/frame). The same `$09` is
copied into `PLAYER_BULLET_F_RAPID` (`$0458`) and `PLAYER_BULLET_S_RAPID`
(`$0488`) when a bullet is created (`bank6.asm:490-491`, `524-525`), halves the
indoor delay between bullets (`$2a` -> `$15`), and alters the F spiral and the S
spread. `docs/Enemy Glossary.md` describes it as "Modifier that speeds up the
bullet velocity of all weapons except the laser rifle" - so "rapid fire" in the
game means speed, not rate.

**Pulsing the fire button is a different mechanism, and the bigger one for rate
of fire.** Contra counts fire presses rather than the held level, so holding the
button is one press:

| weapon | held | pulsed (turbo) |
| --- | --- | --- |
| M | 49.3 shots / 100 frames | 46.7 - unchanged |
| F | 2.0 | 18.0 |
| S | 2.0 | 19.3 |

Holding fire on F or S is about **10x slower** than pulsing it. M is unaffected
because its inter-shot delay is short enough that holding already saturates it.

The two are not fully independent: with bit 4 clear we still measured
`F_RAPID`/`S_RAPID` going non-zero under a pulsed button, where the disassembly
says they are seeded from bit 4. So `$09` has at least one further input we did
not isolate. The measurements above are unaffected - they separate velocity from
rate - but the reason is not closed.

One caveat on those two addresses: `ram.asm` lists each one **twice under
different names** - `$0458` is both `F_RAPID` and `S_INDOOR_ADJ`, `$0488` is both
`F_Y` and `S_RAPID` - so the disassembler was not sure which meaning applies
where. The values seen while pulsing do look like the other meaning bleeding
through (X positions 128-241 for F, a 0-35 counter for S).

Rapid fire cannot be emulated by pressing two buttons at once, as it is on the
original US cartridge: this image has bit 0 as jump and bit 1 as fire, so
pressing both is fire-and-jump. The pulse has to come from the single fire bit.

### Bit 3 is undefined, and forcing it corrupts a table index

Bit 3 is not a weapon, not rapid fire and **not invincibility**. The game never
sets it and never tests it: there is no `and #$08`, `ora #$08` or `bit #$08`
against the weapon byte anywhere in the disassembly. Only bits 0-2 and bit 4 can
ever be present, and every write site masks or clears the rest.

Forcing it nevertheless produces effects, and all of them come from one place.
`bank6.asm:304` reads the weapon with `and #$0f` - keeping bit 3, unlike every
other site which masks `&$07` - and passes the result as a routine index:

| raw | index | consequence |
| --- | --- | --- |
| 8 | 8, the spread routine | its bullet creation reads past the end of a 6-entry sprite table (`bank6.asm:655`) and picks up `$08`/`$09`, which are the player's own spin-jump tiles (`bank2.asm:1079`) - hence a second jumping agent drawn next to the real one |
| 9 | 9, `fire_weapon_routine_l` | the laser is a beam held at the muzzle, not a travelling projectile - hence the shot appearing to freeze |

The `cpy #$01` / `cpy #$04` tests above it also fail, so the hold-to-fire path
is skipped and only the edge-triggered one remains - which is the reduced shot
count. Indoors an extra `adc #$05` pushes the index to 13-16, past the end of the
10-entry dispatch table, and the routine pointer is read out of code bytes.

So the duplicate sprite is an out-of-bounds tile fetch, not a feature.

### Invincibility is a separate variable

The B pickup is routed *around* the weapon code: it compares the item attribute
against `#$05` and branches straight to the timer (`bank7.asm:6898-6900`), so
invincibility never touches `$00AA`/`$00AB`. Two timers per player:

- `$ae`/`$af` `NEW_LIFE_INVINCIBILITY_TIMER`, set to `#$80` on respawn
- `$b0`/`$b1` `INVINCIBILITY_TIMER`, the B pickup, `#$80` normally and `#$90`
  on level 7, decremented every 8 frames

Both are exposed in `nesle.contra`. "B protects from anything except falling into
a pit" is the documented behaviour of the item, not a quirk of this image.

### Weapon strength makes the level harder

`$2F` is `PLAYER_WEAPON_STRENGTH`, recomputed every frame from the weapon type,
and it drives *enemy* difficulty rather than player offence: enemy HP scaling in
the boss and guardian routines, faster enemy attack delays, more double-shots
from soldiers, and aliens only spawning fetuses at strength 3. Player bullet
damage is a flat 1 per hit.

Worth knowing for reward design - a stronger weapon genuinely makes the level
harder, so a "pick up the powerup" reward can pay for itself in avoided
difficulty rather than in faster kills.


These were identified empirically, not read off the published map: each value
was poked into `$00AA` in a real gameplay state and named from the rendered
frame at full NES resolution, by what actually comes out of the muzzle. `$00AB`
was then confirmed for player 2 by firing through `actions2` in a two-player
state - `$AB=0` gives a single small white sphere, `$AB=3` the same diverging
pair as `$AA=3` does for player 1.

That matters because the map had it wrong in a way that would have been easy to
ship: `$AB` was labelled player 2's weapon with no evidence behind it, and
player 1 had no weapon field at all.

Two traps when detecting bullets in a frame: the jungle waterfall at the top of
the screen is white and will match any naive "white pixel" test, and the small
white default spheres sit in the same rows as the background. Restrict to the
playfield rows at the agent's own Y before counting.

Sources:

- Contra RAM map: https://datacrystal.tcrf.net/wiki/Contra_(NES)/RAM_map
- Annotated US disassembly: https://github.com/vermiceli/nes-contra-us

## RL API Compatibility

Gymnasium single-env API returns `(obs, reward, terminated, truncated, info)`.
SB3 vector environments intentionally use a Gym 0.21-like VecEnv API:
`reset()` returns only observations, and `step(actions)` returns
`obs, rewards, dones, infos`. SB3 also expects automatic reset behavior and
`terminal_observation` in `infos` when an episode ends.

Sources:

- Gymnasium Env API: https://gymnasium.farama.org/api/env/
- SB3 VecEnv API: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
