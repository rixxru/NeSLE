# CPU Validation

The current Phase 1 CPU gate is Klaus Dormann-style functional testing. The
runner is intentionally generic: it loads a flat 6502 binary into a 64 KB test
bus, starts at a configured PC, and stops when the program counter traps by
looping on itself.

## Runner

Build the runner:

```sh
c++ -std=c++20 -Icpp/include cpp/tools/run_6502_binary.cpp -o /tmp/run_6502_binary
```

Run a Klaus binary:

```sh
/tmp/run_6502_binary path/to/6502_functional_test.bin \
  --load 0x0000 \
  --start 0x0400 \
  --success 0x3469 \
  --variant mos6502
```

The stock Klaus `6502_functional_test.bin` is a plain NMOS 6502 test. For the
NES CPU profile, use `--variant 2a03` and a decimal-disabled build of the test,
because the Ricoh 2A03 keeps the decimal flag but does not implement BCD
arithmetic. The full MOS 6502 profile exists only as a validation aid.

## Current Result

With the upstream stock binary downloaded to `/tmp` and kept out of the repo:

```text
/tmp/run_6502_binary /tmp/6502_functional_test.bin \
  --load 0x0000 --start 0x0400 --success 0x3469 --variant mos6502
success pc=0x3469 opcode=0x4c instructions=30646177 cycles=96241367
```

The same stock binary under `--variant 2a03` traps before success, as expected,
because the stock configuration enables decimal ADC/SBC checks that the NES CPU
does not implement in hardware.

## Agreement with FCEUX (measured 2026-10-03)

The batch kernel was compared against FCEUX 2.6 driving the same ROM
(`Contra (U) [T-Rus uBAH009 (12.11.2016)]`) through the Lua bridge in
`C:\games\nes\fceux_rl`, comparing all 2048 bytes of CPU RAM.

**Our own determinism is exact.** Two identical runs of 240 frames with a varied
input program (walk, jump, fire, reverse) produce bit-identical RAM: 0 of 2048
bytes differ. This is the property training actually depends on - a
non-deterministic environment silently turns a rollout buffer into noise.

**Loading a save state is byte-exact.** Immediately after restoring an FCEUX
FCSX state, our RAM matches FCEUX's on all 2048 bytes, with no frames run.
Independently, our parsed score/lives/stage match the HUD values recorded in
the curriculum's `meta.json` on 24 of 24 states sampled.

**After that we are close but not bit-exact.** Over 240 frames of identical
input, everything `nesle.contra` decodes agrees: score, hi-score, lives, stage,
screen type, perspective, game status, boss-defeated, demo and pause flags,
two-player mode, every player-2 field, both weapon bytes, and the player flag
byte. Only the sprite position drifts, by 1-5 px, and it oscillates rather than
accumulating:

| frames | dx | dy | score | lives | weapon |
| ---: | ---: | ---: | --- | --- | --- |
| 30 | 0 | 0 | 194/194 | 2/2 | 19/19 |
| 60 | +2 | -4 | 194/194 | 2/2 | 19/19 |
| 120 | +2 | +5 | 194/194 | 2/2 | 19/19 |
| 240 | +2 | -2 | 194/194 | 2/2 | 19/19 |

A few dozen RAM bytes differ throughout, concentrated in the stack, OAM and
zero-page counters, which is what a small PPU sampling difference looks like
rather than a logic error.

### Two traps when repeating this

**FCEUX is one frame ahead after a load.** The Lua bridge runs
`RENDER_FRAMES = 1` frames with no input right after a LOAD
(`bridge_tmpl.lua`), so `load_state` leaves FCEUX one frame past the file while
our `reset` sits exactly on it. The alignment is `ours(N) == fceux(N-1)`.
Without correcting for this, every comparison shows a large divergence that is
purely an artifact of the harness.

**The bridge's `ram.bin` goes stale across a multi-state sequence.** Loading
four different save states in one FCEUX session reported the *same* score for
all four, and a different same-value constant on a re-run - while our own RAM
changed by ~500 bytes between those states, as it should. `ack.txt` said
`DONE=1` with no error, so the failure is silent. Treat the bridge as usable
for one load-and-compare, not as a batch oracle; verify that FCEUX's RAM
actually changed before believing any per-state comparison.

### The residual drift is sprite animation, not physics

Comparing frame by frame, the player flag byte `$034E` differs on many frames in
a way that is exactly one animation step out of phase - the mismatches are
confined to the animation and facing bits (`$08`, `$80`), e.g. `40`/`48`,
`80`/`40`, `88`/`48`, `08`/`48`. Scroll agrees on most frames but steps one
frame early or late on a handful of transitions. And `x_pos`/`y_pos` drift is
bounded, oscillating within 1-5 px and settling rather than accumulating.

The decisive evidence that the physics is faithful: score, lives, stage, weapon
and every player-2 field match *exactly* over 240 frames of live shooting. If
the player were being moved differently, kills - and therefore score - would
diverge. Only the position read out of a sprite being drawn disagrees.

### `$21`: closed as harmless, but a useful diagnostic

`$21` differs persistently, FCEUX `$01` against our `$00`. It is **not** a
player state flag. The annotated disassembly names it:

```
GRAPHICS_BUFFER_OFFSET:
; $21 - current write offset into CPU_GRAPHICS_BUFFER, which contains graphics
;       write commands that are written to the PPU
```

So it is the cursor into the queue of pending VRAM write commands, reset to 0
every frame by `write_cpu_graphics_buffer_to_ppu`. Its value says how many bytes
of graphics commands are queued this frame, nothing more.

Gameplay never reads it. Every access is in `bank7.asm` (nametable, palette and
text writers) and `bank4.asm` (the ending and credits graphics); `bank0.asm`,
which holds the enemy routines, the player-bullet collision entry, firing and
damage, contains no reference to it at all, nor does `bank6.asm`'s
`run_player_bullet_routines`. The only read that is not an index into the buffer
is a capacity guard in `write_palette_colors_to_ppu`: `cmp #$30 / bcs exit`,
i.e. skip this frame's palette upload if the queue is already 48 bytes deep. No
gameplay routine writes it either.

So the divergence is harmless for training - and worth keeping as a probe.
A persistent `$01` vs `$00` means our CPU has written one fewer byte into the
VRAM command queue by the end of the frame than FCEUX has, which puts a number
on the NMI/vblank phase difference behind the animation drift above. The `$22`
seen in the mine state is 34 queued bytes, a larger pending graphics chunk, and
it dropping to `$01` on a jump is a redraw resetting the queue.

Sources:

- Klaus Dormann tests: https://github.com/Klaus2m5/6502_65C02_functional_tests
- 6502 test program overview: https://www.nesdev.org/wiki/Visual6502wiki/6502TestPrograms
- 6502 opcode reference: https://www.nesdev.org/obelisk-6502-guide/reference.html
