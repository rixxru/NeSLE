#pragma once

#include <cstdint>

#include "nesle/cuda/mapper.cuh"
#include "nesle/cuda/state.cuh"

#ifdef __CUDACC__
#define NESLE_CUDA_HD __host__ __device__
#else
#define NESLE_CUDA_HD
#endif

namespace nesle::cuda {

constexpr std::uint32_t kMarioPlayerState = 0x000E;
constexpr std::uint32_t kMarioPlayerFloatState = 0x001D;
constexpr std::uint32_t kMarioEnemyTypeBase = 0x0016;
constexpr std::uint32_t kMarioXPage = 0x006D;
constexpr std::uint32_t kMarioXScreen = 0x0086;
constexpr std::uint32_t kMarioYViewport = 0x00B5;
constexpr std::uint32_t kMarioLives = 0x075A;
constexpr std::uint32_t kMarioGameMode = 0x0770;
constexpr std::uint32_t kMarioTimeDigits = 0x07F8;

struct MarioBatchSnapshot {
    int x_pos = 0;
    int time = 0;
    bool flag_get = false;
    bool is_dying = false;
    bool is_dead = false;
    bool is_game_over = false;
};

struct BatchReward {
    int x = 0;
    int time = 0;
    int death = 0;
    int total = 0;
};

NESLE_CUDA_HD inline int read_bcd_digits(const std::uint8_t* ram,
                                         std::uint32_t address,
                                         std::uint32_t length) {
    int value = 0;
    for (std::uint32_t i = 0; i < length; ++i) {
        value = value * 10 + static_cast<int>(ram[address + i] & 0x0F);
    }
    return value;
}

NESLE_CUDA_HD inline bool is_stage_over(const std::uint8_t* ram) {
    for (std::uint32_t i = 0; i < 5; ++i) {
        const auto enemy = ram[kMarioEnemyTypeBase + i];
        if ((enemy == 0x2D || enemy == 0x31) && ram[kMarioPlayerFloatState] == 3) {
            return true;
        }
    }
    return false;
}

NESLE_CUDA_HD inline MarioBatchSnapshot read_mario_snapshot(const std::uint8_t* ram) {
    MarioBatchSnapshot snapshot;
    snapshot.x_pos = static_cast<int>(ram[kMarioXPage]) * 0x100 +
                     static_cast<int>(ram[kMarioXScreen]);
    snapshot.time = read_bcd_digits(ram, kMarioTimeDigits, 3);
    snapshot.is_dying = ram[kMarioPlayerState] == 0x0B || ram[kMarioYViewport] > 1;
    snapshot.is_dead = ram[kMarioPlayerState] == 0x06;
    snapshot.is_game_over = ram[kMarioLives] == 0xFF;
    snapshot.flag_get = ram[kMarioGameMode] == 2 || is_stage_over(ram);
    return snapshot;
}

NESLE_CUDA_HD inline BatchReward compute_batch_reward(const MarioBatchSnapshot& previous,
                                                      const MarioBatchSnapshot& current) {
    BatchReward reward;
    reward.x = current.x_pos - previous.x_pos;
    if (reward.x < -5 || reward.x > 5) {
        reward.x = 0;
    }

    reward.time = current.time - previous.time;
    if (reward.time > 0) {
        reward.time = 0;
    }

    reward.death = (current.is_dying || current.is_dead) ? -25 : 0;
    reward.total = reward.x + reward.time + reward.death;
    return reward;
}

// ---------------------------------------------------------------------------
// Contra
//
// Addresses mirror src/nesle/contra.py, which is the reference implementation
// and the one the parity test compares against. Scores are raw 16-bit
// little-endian integers at $07E2/$07E4; the HUD multiplies them by 100, which
// would make every kill worth the same 100 points.
// ---------------------------------------------------------------------------
constexpr std::uint32_t kContraGameMode = 0x001C;  // 0 = play, non-zero = demo
constexpr std::uint32_t kContraPlayerMode = 0x0022;  // 1 = two players
constexpr std::uint32_t kContraScreenType = 0x002C;
constexpr std::uint32_t kContraStage = 0x0030;
constexpr std::uint32_t kContraLives = 0x0032;
constexpr std::uint32_t kContraLivesP2 = 0x0033;
constexpr std::uint32_t kContraGameStatus = 0x0038;
constexpr std::uint32_t kContraGameStatusP2 = 0x0039;
constexpr std::uint32_t kContraBossDefeated = 0x003B;
constexpr std::uint32_t kContraPerspective = 0x0040;  // 0 = side-scrolling, 1 = vertical
constexpr std::uint32_t kContraPlayerY = 0x031A;
constexpr std::uint32_t kContraPlayerYP2 = 0x031B;
constexpr std::uint32_t kContraPlayerX = 0x0334;
constexpr std::uint32_t kContraPlayerXP2 = 0x0335;
constexpr std::uint32_t kContraScoreP1 = 0x07E2;
constexpr std::uint32_t kContraScoreP2 = 0x07E4;

constexpr std::uint8_t kContraScreenContinue = 0x06;
constexpr int kContraDeathPenalty = 500;
constexpr int kContraStageClearBonus = 1000;
// No legal move covers this many pixels in one step, so a larger delta means the
// sprite was relocated (respawn, warp) rather than walked.
constexpr int kContraMaxProgressStep = 64;

struct ContraBatchSnapshot {
    int score = 0;
    int score_p2 = 0;
    // Raw on-screen sprite coordinates; `progress` below selects the axis.
    std::uint8_t x_pos = 0;
    std::uint8_t y_pos = 0;
    std::uint8_t x_pos_p2 = 0;
    std::uint8_t y_pos_p2 = 0;
    std::uint8_t lives = 0;
    std::uint8_t lives_p2 = 0;
    std::uint8_t stage = 0;
    std::uint8_t screen_type = 0;
    std::uint8_t perspective = 0;
    std::uint8_t game_status = 0;
    std::uint8_t boss_defeated = 0;
    std::uint8_t two_player = 0;
    std::uint8_t is_demo = 0;
};

NESLE_CUDA_HD inline int read_le16(const std::uint8_t* ram, std::uint32_t address) {
    return static_cast<int>(ram[address]) | (static_cast<int>(ram[address + 1]) << 8);
}

NESLE_CUDA_HD inline ContraBatchSnapshot read_contra_snapshot(const std::uint8_t* ram) {
    ContraBatchSnapshot s;
    s.score = read_le16(ram, kContraScoreP1);
    s.score_p2 = read_le16(ram, kContraScoreP2);
    s.x_pos = ram[kContraPlayerX];
    s.y_pos = ram[kContraPlayerY];
    s.x_pos_p2 = ram[kContraPlayerXP2];
    s.y_pos_p2 = ram[kContraPlayerYP2];
    s.lives = ram[kContraLives];
    s.lives_p2 = ram[kContraLivesP2];
    s.stage = ram[kContraStage];
    s.screen_type = ram[kContraScreenType];
    s.perspective = ram[kContraPerspective];
    s.game_status = ram[kContraGameStatus];
    s.boss_defeated = static_cast<std::uint8_t>(ram[kContraBossDefeated] & 0x01);
    s.two_player = ram[kContraPlayerMode] == 1 ? 1 : 0;
    s.is_demo = ram[kContraGameMode] != 0 ? 1 : 0;
    return s;
}

// A delta is only meaningful while the thing being measured stayed the same
// thing: the stage, the scrolling axis and the screen all have to match, or the
// agent gets paid for a teleport or taxed for a respawn.
NESLE_CUDA_HD inline bool contra_progress_comparable(const ContraBatchSnapshot& previous,
                                                      const ContraBatchSnapshot& current) {
    return previous.stage == current.stage && previous.perspective == current.perspective &&
           previous.screen_type == current.screen_type;
}

NESLE_CUDA_HD inline int contra_progress_delta(const ContraBatchSnapshot& previous,
                                               const ContraBatchSnapshot& current,
                                               bool player2) {
    if (!contra_progress_comparable(previous, current)) {
        return 0;
    }
    const int before = player2 ? (previous.perspective == 0 ? previous.x_pos_p2 : previous.y_pos_p2)
                               : (previous.perspective == 0 ? previous.x_pos : previous.y_pos);
    const int after = player2 ? (current.perspective == 0 ? current.x_pos_p2 : current.y_pos_p2)
                              : (current.perspective == 0 ? current.x_pos : current.y_pos);
    const int delta = after - before;
    if (delta > kContraMaxProgressStep || delta < -kContraMaxProgressStep) {
        return 0;
    }
    return delta;
}

struct ContraBatchReward {
    int score = 0;
    int progress = 0;
    int death = 0;
    int stage_clear = 0;
    int p2_score = 0;
    int p2_progress = 0;
    int total = 0;
};

NESLE_CUDA_HD inline ContraBatchReward compute_contra_reward(const ContraBatchSnapshot& previous,
                                                            const ContraBatchSnapshot& current) {
    ContraBatchReward reward;
    // Score only ever rises during play. Clamping at zero absorbs the 16-bit
    // wrap and the reset on a continue screen instead of paying out a huge
    // negative for it.
    reward.score = current.score - previous.score;
    if (reward.score < 0) {
        reward.score = 0;
    }
    reward.progress = contra_progress_delta(previous, current, /*player2=*/false);

    const int lives_lost = static_cast<int>(previous.lives) - static_cast<int>(current.lives);
    reward.death = -kContraDeathPenalty * (lives_lost > 0 ? lives_lost : 0);
    if (lives_lost > 0) {
        // Losing a life teleports the sprite back to the side of the stage;
        // that movement is not progress and must not be paid out.
        reward.progress = 0;
    }
    reward.stage_clear = (current.boss_defeated != 0 && previous.boss_defeated == 0)
                             ? kContraStageClearBonus
                             : 0;

    // Player 2's bytes are never initialised in a one-player game (observed:
    // lives 0x62, score 0xFFFF), so they are only read when the mode says two
    // players are in. $0039 cannot be used to decide this: the game sets
    // P2_GAME_OVER_STATUS to 1 in one-player games too.
    if (current.two_player != 0 && previous.two_player != 0) {
        reward.p2_score = current.score_p2 - previous.score_p2;
        if (reward.p2_score < 0) {
            reward.p2_score = 0;
        }
        reward.p2_progress = contra_progress_delta(previous, current, /*player2=*/true);
        const int p2_lives_lost =
            static_cast<int>(previous.lives_p2) - static_cast<int>(current.lives_p2);
        if (p2_lives_lost > 0) {
            reward.death -= kContraDeathPenalty * p2_lives_lost;
            reward.p2_progress = 0;
        }
    }

    reward.total = reward.score + reward.progress + reward.death + reward.stage_clear +
                   reward.p2_score + reward.p2_progress;
    return reward;
}

NESLE_CUDA_HD inline bool contra_is_done(const ContraBatchSnapshot& s) {
    // $0038 is set on game over and $002C becomes the continue screen; either
    // one ends the episode. The attract demo is not a playable episode either.
    if (s.game_status != 0 || s.screen_type == kContraScreenContinue) {
        return true;
    }
    return s.is_demo != 0;
}

NESLE_CUDA_HD inline void capture_contra_baseline(BatchBuffers& buffers,
                                                  std::uint32_t env,
                                                  const ContraBatchSnapshot& s) {
    auto& c = buffers.contra;
    c.has_previous[env] = 1;
    c.score[env] = s.score;
    c.score_p2[env] = s.score_p2;
    c.x_pos[env] = s.x_pos;
    c.y_pos[env] = s.y_pos;
    c.x_pos_p2[env] = s.x_pos_p2;
    c.y_pos_p2[env] = s.y_pos_p2;
    c.lives[env] = s.lives;
    c.lives_p2[env] = s.lives_p2;
    c.stage[env] = s.stage;
    c.screen_type[env] = s.screen_type;
    c.perspective[env] = s.perspective;
    c.game_status[env] = s.game_status;
    c.boss_defeated[env] = s.boss_defeated;
    c.two_player[env] = s.two_player;
}

NESLE_CUDA_HD inline void apply_contra_reward_env(BatchBuffers& buffers, std::uint32_t env) {
    auto& c = buffers.contra;
    const auto* ram = env_cpu_ram(buffers, env);
    const auto current = read_contra_snapshot(ram);

    if (c.has_previous[env] == 0) {
        // First step of an episode: there is no baseline to compare against,
        // so the reward is zero by construction rather than by hoping the seeded
        // values happen to line up. The baseline is captured here, so the step
        // that follows this one already rewards normally.
        buffers.rewards[env] = 0.0F;
        buffers.done[env] = 0;
        capture_contra_baseline(buffers, env, current);
        return;
    }

    ContraBatchSnapshot previous;
    previous.score = c.score[env];
    previous.score_p2 = c.score_p2[env];
    previous.x_pos = c.x_pos[env];
    previous.y_pos = c.y_pos[env];
    previous.x_pos_p2 = c.x_pos_p2[env];
    previous.y_pos_p2 = c.y_pos_p2[env];
    previous.lives = c.lives[env];
    previous.lives_p2 = c.lives_p2[env];
    previous.stage = c.stage[env];
    previous.screen_type = c.screen_type[env];
    previous.perspective = c.perspective[env];
    previous.game_status = c.game_status[env];
    previous.boss_defeated = c.boss_defeated[env];
    previous.two_player = c.two_player[env];
    previous.is_demo = 0;

    const auto reward = compute_contra_reward(previous, current);
    buffers.rewards[env] = static_cast<float>(reward.total);
    buffers.done[env] = contra_is_done(current) ? 1 : 0;
    capture_contra_baseline(buffers, env, current);
}

NESLE_CUDA_HD inline void apply_batch_reward_env(BatchBuffers& buffers, std::uint32_t env) {
    const auto kind = static_cast<RewardKind>(buffers.cart.reward_smb);
    if (kind == RewardKind::kContra) {
        apply_contra_reward_env(buffers, env);
        return;
    }
    if (kind == RewardKind::kNone) {
        buffers.rewards[env] = 0.0F;
        buffers.done[env] = 0;
        return;
    }
    if (buffers.cart.reward_smb == 0 && kind != RewardKind::kSmb) {
        // Not a Super Mario Bros. image: the RAM scraper below would read
        // unrelated bytes and flag every environment done within a few steps,
        // which looks exactly like "the ROM does not boot". Non-SMB ROMs run
        // reward-free until the env's own step limit truncates the episode.
        buffers.rewards[env] = 0.0F;
        buffers.done[env] = 0;
        return;
    }
    const auto* ram = env_cpu_ram(buffers, env);
    const auto current = read_mario_snapshot(ram);
    MarioBatchSnapshot previous;
    previous.x_pos = buffers.previous_mario_x[env];
    previous.time = buffers.previous_mario_time[env];

    const auto reward = compute_batch_reward(previous, current);
    buffers.rewards[env] = static_cast<float>(reward.total);
    buffers.done[env] = (current.flag_get || current.is_dying || current.is_dead ||
                         current.is_game_over)
                            ? 1
                            : 0;
    buffers.previous_mario_x[env] = current.x_pos;
    buffers.previous_mario_time[env] = current.time;
}


// Clear the "we have a previous step" flag for one env. Called by every reset
// path: with the flag down, the next step reports reward 0 and then captures a
// fresh baseline, so the first reward of an episode is zero whatever the
// previous episode happened to leave in the baseline slots.
NESLE_CUDA_HD inline void clear_contra_baseline(BatchBuffers& buffers, std::uint32_t env) {
    if (buffers.contra.has_previous != nullptr) {
        buffers.contra.has_previous[env] = 0;
    }
}

// Clear a quarantined-opcode fault. Every reset path calls this: a reset env starts
// from the reset vector, so a fault recorded before it says nothing about the env
// now, and leaving it set would keep the env skipped forever.
NESLE_CUDA_HD inline void clear_env_fault(BatchBuffers& buffers, std::uint32_t env) {
    if (buffers.cpu.fault != nullptr) {
        buffers.cpu.fault[env] = 0;
    }
}

NESLE_CUDA_HD inline void cold_reset_console_env(BatchBuffers& buffers, std::uint32_t env) {
    // Read reset vector from PRG ROM. $FFFC/$FFFD live in the fixed window, so
    // the vector is the last four bytes of the image for NROM (16/32 KB) and
    // for every UxROM board alike, whose top 16 KB is the final 16 KB page.
    std::uint16_t reset_pc = 0;
    if (buffers.cart.prg_rom != nullptr && buffers.cart.prg_rom_size >= 4u * 1024u) {
        const auto base = static_cast<std::uint32_t>(buffers.cart.prg_rom_size - 4u);
        reset_pc = static_cast<std::uint16_t>(
            buffers.cart.prg_rom[base] |
            (static_cast<std::uint16_t>(buffers.cart.prg_rom[base + 1]) << 8));
    }

    // CPU registers.
    buffers.cpu.pc[env] = reset_pc;
    buffers.cpu.a[env] = 0;
    buffers.cpu.x[env] = 0;
    buffers.cpu.y[env] = 0;
    buffers.cpu.sp[env] = 0xFD;
    buffers.cpu.p[env] = 0x24;
    buffers.cpu.cycles[env] = 7;
    buffers.cpu.nmi_pending[env] = 0;
    buffers.cpu.irq_pending[env] = 0;
    buffers.cpu.controller1_shift[env] = 0;
    buffers.cpu.controller1_shift_count[env] = 8;
    buffers.cpu.controller1_strobe[env] = 0;
    if (buffers.cpu.controller2_shift != nullptr) {
        buffers.cpu.controller2_shift[env] = 0;
        buffers.cpu.controller2_shift_count[env] = 8;
        buffers.cpu.controller2_strobe[env] = 0;
    }
    buffers.cpu.pending_dma_cycles[env] = 0;

    // CPU RAM.
    auto* ram = env_cpu_ram(buffers, env);
    zero_bytes_fast(ram, static_cast<std::uint32_t>(kCpuRamBytes));

    // PRG RAM.
    auto* prg_ram = buffers.cpu.prg_ram + static_cast<std::uint64_t>(env) * kPrgRamBytes;
    zero_bytes_fast(prg_ram, static_cast<std::uint32_t>(kPrgRamBytes));

    // CHR RAM (null on every CHR ROM cartridge).
    if (buffers.ppu.chr_ram != nullptr) {
        auto* chr_ram = buffers.ppu.chr_ram + static_cast<std::uint64_t>(env) * kChrRamBytes;
        zero_bytes_fast(chr_ram, static_cast<std::uint32_t>(kChrRamBytes));
    }

    // Mapper registers: power-on banks and header mirroring.
    reset_mapper_state(buffers, env);

    // PPU state.
    buffers.ppu.ctrl[env] = 0;
    buffers.ppu.mask[env] = 0;
    buffers.ppu.status[env] = 0;
    buffers.ppu.oam_addr[env] = 0;
    buffers.ppu.nmi_pending[env] = 0;
    buffers.ppu.frame_dot[env] = 0;
    buffers.ppu.frame[env] = 0;
    buffers.ppu.v[env] = 0;
    buffers.ppu.t[env] = 0;
    buffers.ppu.x[env] = 0;
    buffers.ppu.w[env] = 0;
    buffers.ppu.open_bus[env] = 0;
    buffers.ppu.read_buffer[env] = 0;
    buffers.ppu.scroll_x[env] = 0;
    buffers.ppu.scroll_y[env] = 0;

    // PPU memory.
    auto* nt = buffers.ppu.nametable_ram + static_cast<std::uint64_t>(env) * kNametableRamBytes;
    zero_bytes_fast(nt, static_cast<std::uint32_t>(kNametableRamBytes));
    auto* pal = buffers.ppu.palette_ram + static_cast<std::uint64_t>(env) * kPaletteRamBytes;
    zero_bytes_fast(pal, static_cast<std::uint32_t>(kPaletteRamBytes));
    auto* oam = env_oam(buffers, env);
    zero_bytes_fast(oam, static_cast<std::uint32_t>(kOamBytes));

    // Reward baselines.
    buffers.previous_mario_x[env] = 0;
    buffers.previous_mario_time[env] = 0;
    clear_contra_baseline(buffers, env);
    clear_env_fault(buffers, env);
    buffers.rewards[env] = 0.0F;
    buffers.done[env] = 0;
}

NESLE_CUDA_HD inline void warm_reset_console_env(BatchBuffers& buffers,
                                                 std::uint32_t env,
                                                 const SnapshotTemplate& snap) {
    // Pick this env's source level. For the single-snapshot case env_to_level is all zeros.
    const auto level = static_cast<std::uint32_t>(snap.env_to_level[env]);
    const auto cpu_ram_base = static_cast<std::uint64_t>(level) * kCpuRamBytes;
    const auto prg_ram_base = static_cast<std::uint64_t>(level) * kPrgRamBytes;
    const auto nt_base = static_cast<std::uint64_t>(level) * kNametableRamBytes;
    const auto pal_base = static_cast<std::uint64_t>(level) * kPaletteRamBytes;
    const auto oam_base = static_cast<std::uint64_t>(level) * kOamBytes;

    // CPU registers — restored verbatim from the snapshot.
    buffers.cpu.pc[env] = snap.pc[level];
    buffers.cpu.a[env] = snap.a[level];
    buffers.cpu.x[env] = snap.x[level];
    buffers.cpu.y[env] = snap.y[level];
    buffers.cpu.sp[env] = snap.sp[level];
    buffers.cpu.p[env] = snap.p[level];
    buffers.cpu.cycles[env] = snap.cycles[level];
    buffers.cpu.nmi_pending[env] = 0;
    buffers.cpu.irq_pending[env] = 0;
    buffers.cpu.controller1_shift[env] = 0;
    buffers.cpu.controller1_shift_count[env] = 8;
    buffers.cpu.controller1_strobe[env] = 0;
    if (buffers.cpu.controller2_shift != nullptr) {
        buffers.cpu.controller2_shift[env] = 0;
        buffers.cpu.controller2_shift_count[env] = 8;
        buffers.cpu.controller2_strobe[env] = 0;
    }
    buffers.cpu.pending_dma_cycles[env] = 0;
    clear_env_fault(buffers, env);

    auto* ram = env_cpu_ram(buffers, env);
    copy_bytes_fast(ram, snap.cpu_ram + cpu_ram_base, static_cast<std::uint32_t>(kCpuRamBytes));
    auto* prg_ram = buffers.cpu.prg_ram + static_cast<std::uint64_t>(env) * kPrgRamBytes;
    copy_bytes_fast(prg_ram, snap.prg_ram + prg_ram_base,
                    static_cast<std::uint32_t>(kPrgRamBytes));

    // FCEUX save states carry no mapper registers, so a restored environment
    // starts from power-on banks; the game's own init code reloads them.
    reset_mapper_state(buffers, env);

    // CHR RAM is part of the visible machine state and has to come from the
    // state too: on a CHR-RAM cartridge the pattern data is the one thing the
    // game does not re-upload, so leaving it at whatever the previous episode
    // left behind (or zero, after a cold reset) renders a black screen. FCSX
    // states carry it; legacy FCS states do not, hence the null check.
    if (buffers.ppu.chr_ram != nullptr && snap.chr_ram != nullptr) {
        auto* chr_ram = buffers.ppu.chr_ram + static_cast<std::uint64_t>(env) * kChrRamBytes;
        copy_bytes_fast(chr_ram, snap.chr_ram + static_cast<std::uint64_t>(level) * kChrRamBytes,
                        static_cast<std::uint32_t>(kChrRamBytes));
    }

    // PPU registers + memory.
    buffers.ppu.ctrl[env] = snap.ppu_ctrl[level];
    buffers.ppu.mask[env] = snap.ppu_mask[level];
    buffers.ppu.status[env] = snap.ppu_status[level];
    buffers.ppu.oam_addr[env] = snap.ppu_oam_addr[level];
    buffers.ppu.nmi_pending[env] = 0;
    buffers.ppu.frame_dot[env] = 0;
    buffers.ppu.frame[env] = 0;
    buffers.ppu.v[env] = snap.ppu_v[level];
    buffers.ppu.t[env] = snap.ppu_t[level];
    buffers.ppu.x[env] = snap.ppu_x[level];
    buffers.ppu.w[env] = snap.ppu_w[level];
    buffers.ppu.open_bus[env] = snap.ppu_open_bus[level];
    buffers.ppu.read_buffer[env] = snap.ppu_read_buffer[level];
    buffers.ppu.scroll_x[env] = 0;
    buffers.ppu.scroll_y[env] = 0;

    auto* nt = buffers.ppu.nametable_ram + static_cast<std::uint64_t>(env) * kNametableRamBytes;
    copy_bytes_fast(nt, snap.nametable_ram + nt_base,
                    static_cast<std::uint32_t>(kNametableRamBytes));
    auto* pal = buffers.ppu.palette_ram + static_cast<std::uint64_t>(env) * kPaletteRamBytes;
    copy_bytes_fast(pal, snap.palette_ram + pal_base,
                    static_cast<std::uint32_t>(kPaletteRamBytes));
    auto* oam = env_oam(buffers, env);
    copy_bytes_fast(oam, snap.oam + oam_base, static_cast<std::uint32_t>(kOamBytes));

    // Reward baselines — seed previous_x and previous_time from the snapshot's RAM so the
    // first step's reward calculation doesn't see a synthetic large delta from zero.
    const auto* level_ram = snap.cpu_ram + cpu_ram_base;
    const int x_pos = static_cast<int>(level_ram[kMarioXPage]) * 0x100 +
                      static_cast<int>(level_ram[kMarioXScreen]);
    const int time_val = read_bcd_digits(level_ram, kMarioTimeDigits, 3);
    buffers.previous_mario_x[env] = x_pos;
    buffers.previous_mario_time[env] = time_val;
    // The Contra baseline is deliberately *not* seeded from the snapshot: the
    // has_previous flag below is what guarantees a zero first reward, and
    // seeding would only create a second, weaker version of the same guarantee.
    clear_contra_baseline(buffers, env);
    clear_env_fault(buffers, env);
    buffers.rewards[env] = 0.0F;
    buffers.done[env] = 0;
}

NESLE_CUDA_HD inline void cold_reset_synthetic_env(BatchBuffers& buffers, std::uint32_t env) {
    auto* ram = env_cpu_ram(buffers, env);
    zero_bytes_fast(ram, static_cast<std::uint32_t>(kCpuRamBytes));
    ram[kMarioXPage] = 1;
    ram[kMarioXScreen] = 2;
    ram[kMarioYViewport] = 1;
    ram[kMarioLives] = 2;
    ram[kMarioPlayerState] = 0;
    ram[kMarioTimeDigits] = 4;
    ram[kMarioTimeDigits + 1] = 0;
    ram[kMarioTimeDigits + 2] = 0;

    buffers.previous_mario_x[env] = 0x100 + 2;
    buffers.previous_mario_time[env] = 400;
    clear_contra_baseline(buffers, env);
    clear_env_fault(buffers, env);
    buffers.rewards[env] = 0.0F;
    buffers.done[env] = 0;
}

}  // namespace nesle::cuda

#undef NESLE_CUDA_HD
