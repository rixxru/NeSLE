// Contra reward math, exercised as host C++ (the headers are host-compilable).
//
// The same formulas run on the device inside the step kernel and on the host in
// src/nesle/contra.py; tests/test_contra_reward.py compares the two against each
// other over a real ROM. This file pins the individual rules so a failure points
// at the rule that broke rather than at "the numbers differ".
#include <cassert>
#include <cstdint>
#include <vector>

#include "nesle/cuda/batch_step.cuh"

namespace {

using namespace nesle::cuda;

std::vector<std::uint8_t> make_gameplay_ram() {
    std::vector<std::uint8_t> ram(kCpuRamBytes, 0);
    ram[kContraGameMode] = 0;
    ram[kContraScreenType] = 0x04;  // SCREEN_NORMAL
    ram[kContraStage] = 1;
    ram[kContraLives] = 3;
    ram[kContraLivesP2] = 3;
    ram[kContraPerspective] = 0;
    ram[kContraPlayerX] = 25;
    ram[kContraPlayerY] = 45;
    ram[kContraPlayerXP2] = 32;
    ram[kContraPlayerYP2] = 32;
    return ram;
}

void test_reads_the_documented_addresses() {
    auto ram = make_gameplay_ram();
    ram[kContraScoreP1] = 0x34;
    ram[kContraScoreP1 + 1] = 0x12;  // 0x1234 little-endian
    ram[kContraScoreP2] = 0x07;
    ram[kContraPlayerMode] = 1;

    const auto s = read_contra_snapshot(ram.data());
    assert(s.score == 0x1234);
    assert(s.score_p2 == 7);
    assert(s.lives == 3);
    assert(s.lives_p2 == 3);
    assert(s.stage == 1);
    assert(s.screen_type == 0x04);
    assert(s.x_pos == 25);
    assert(s.y_pos == 45);
    assert(s.x_pos_p2 == 32);
    assert(s.two_player == 1);
    assert(s.is_demo == 0);
    assert(!contra_is_done(s));
}

void test_progress_follows_the_perspective() {
    auto before = make_gameplay_ram();
    auto after = make_gameplay_ram();
    after[kContraPlayerX] = 30;  // side-scrolling: X is the axis

    assert(contra_progress_delta(read_contra_snapshot(before.data()),
                                read_contra_snapshot(after.data()), false) == 5);

    // Flipping $0040 makes the vertical axis the one that matters.
    auto vertical_before = before;
    auto vertical_after = after;
    vertical_before[kContraPerspective] = 1;
    vertical_after[kContraPerspective] = 1;
    vertical_after[kContraPlayerY] = 60;
    assert(contra_progress_delta(read_contra_snapshot(vertical_before.data()),
                                 read_contra_snapshot(vertical_after.data()), false) == 15);
}

void test_stage_screen_and_perspective_changes_suppress_progress() {
    const auto before = read_contra_snapshot(make_gameplay_ram().data());

    auto stage_changed = make_gameplay_ram();
    stage_changed[kContraStage] = 4;
    stage_changed[kContraPlayerX] = 200;
    assert(compute_contra_reward(before, read_contra_snapshot(stage_changed.data())).progress == 0);

    auto screen_changed = make_gameplay_ram();
    screen_changed[kContraScreenType] = 0x06;  // continue screen
    screen_changed[kContraPlayerX] = 200;
    assert(compute_contra_reward(before, read_contra_snapshot(screen_changed.data())).progress == 0);

    auto perspective_changed = make_gameplay_ram();
    perspective_changed[kContraPerspective] = 1;
    perspective_changed[kContraPlayerX] = 200;
    assert(
        compute_contra_reward(before, read_contra_snapshot(perspective_changed.data())).progress == 0);
}

void test_teleport_is_not_progress() {
    const auto before = read_contra_snapshot(make_gameplay_ram().data());
    auto jumped = make_gameplay_ram();
    jumped[kContraPlayerX] = static_cast<std::uint8_t>(25 + kContraMaxProgressStep + 1);
    assert(compute_contra_reward(before, read_contra_snapshot(jumped.data())).progress == 0);

    // Just inside the limit still pays out.
    auto walked = make_gameplay_ram();
    walked[kContraPlayerX] = static_cast<std::uint8_t>(25 + kContraMaxProgressStep);
    assert(compute_contra_reward(before, read_contra_snapshot(walked.data())).progress ==
           kContraMaxProgressStep);
}

void set_le16(std::vector<std::uint8_t>& ram, std::uint32_t address, int value) {
    ram[address] = static_cast<std::uint8_t>(value & 0xFF);
    ram[address + 1] = static_cast<std::uint8_t>((value >> 8) & 0xFF);
}

void test_score_increase_is_the_main_term() {
    auto before = make_gameplay_ram();
    auto after = make_gameplay_ram();
    set_le16(before, kContraScoreP1, 194);
    set_le16(after, kContraScoreP1, 294);
    const auto reward = compute_contra_reward(read_contra_snapshot(before.data()),
                                              read_contra_snapshot(after.data()));
    assert(reward.score == 100);
    assert(reward.total == 100);
}

void test_score_decrease_is_clamped() {
    auto before = make_gameplay_ram();
    auto after = make_gameplay_ram();
    set_le16(before, kContraScoreP1, 300);
    set_le16(after, kContraScoreP1, 10);
    assert(compute_contra_reward(read_contra_snapshot(before.data()),
                                 read_contra_snapshot(after.data()))
               .score == 0);

    // A 16-bit wraparound (65500 -> 36) reads as a huge negative difference, and
    // is clamped to zero rather than reconstructed modularly: the alternative
    // would have to guess between "wrapped" and "the score was reset on a
    // continue screen", and both are worth losing one step of signal.
    auto wrapped_before = make_gameplay_ram();
    auto wrapped_after = make_gameplay_ram();
    set_le16(wrapped_before, kContraScoreP1, 65500);
    set_le16(wrapped_after, kContraScoreP1, 36);
    assert(compute_contra_reward(read_contra_snapshot(wrapped_before.data()),
                                 read_contra_snapshot(wrapped_after.data()))
               .score == 0);
    assert(compute_contra_reward(read_contra_snapshot(wrapped_before.data()),
                                 read_contra_snapshot(wrapped_after.data()))
               .total == 0);
}

void test_death_is_charged_once_and_suppresses_progress() {
    auto before = make_gameplay_ram();
    auto after = make_gameplay_ram();
    after[kContraLives] = 2;
    after[kContraPlayerX] = 30;  // respawn teleports the sprite
    const auto reward = compute_contra_reward(read_contra_snapshot(before.data()),
                                              read_contra_snapshot(after.data()));
    assert(reward.death == -kContraDeathPenalty);
    assert(reward.progress == 0);
    assert(reward.total == -kContraDeathPenalty);
}

void test_gaining_a_lives_is_not_a_reward() {
    auto before = make_gameplay_ram();
    auto after = make_gameplay_ram();
    before[kContraLives] = 0;
    after[kContraLives] = 3;
    assert(compute_contra_reward(read_contra_snapshot(before.data()),
                                 read_contra_snapshot(after.data()))
               .death == 0);
}

void test_stage_clear_is_one_shot() {
    auto before = make_gameplay_ram();
    auto cleared = make_gameplay_ram();
    cleared[kContraBossDefeated] = 1;
    const auto previous = read_contra_snapshot(before.data());
    const auto current = read_contra_snapshot(cleared.data());
    assert(compute_contra_reward(previous, current).stage_clear == kContraStageClearBonus);
    assert(compute_contra_reward(current, current).stage_clear == 0);
}

void test_player2_is_ignored_in_one_player_games() {
    // $0022 = 0 while the P2 bytes hold the leftovers a 1P game leaves
    // (observed: lives 0x62, score 0xFFFF).
    auto before = make_gameplay_ram();
    auto after = make_gameplay_ram();
    after[kContraLivesP2] = 0x62;
    after[kContraScoreP2] = 0xFF;
    after[kContraScoreP2 + 1] = 0xFF;
    const auto reward = compute_contra_reward(read_contra_snapshot(before.data()),
                                              read_contra_snapshot(after.data()));
    assert(reward.p2_score == 0);
    assert(reward.death == 0);
    assert(reward.total == 0);
}

void test_player2_score_and_progress_are_counted() {
    auto before = make_gameplay_ram();
    auto after = make_gameplay_ram();
    before[kContraPlayerMode] = 1;
    after[kContraPlayerMode] = 1;
    after[kContraScoreP2] = 7;
    after[kContraPlayerXP2] = 40;
    const auto reward = compute_contra_reward(read_contra_snapshot(before.data()),
                                              read_contra_snapshot(after.data()));
    assert(reward.p2_score == 7);
    assert(reward.p2_progress == 8);
    assert(reward.total == 15);
}

void test_player2_life_loss_is_charged() {
    auto before = make_gameplay_ram();
    auto after = make_gameplay_ram();
    before[kContraPlayerMode] = 1;
    after[kContraPlayerMode] = 1;
    after[kContraLivesP2] = 2;
    after[kContraPlayerXP2] = 40;
    const auto reward = compute_contra_reward(read_contra_snapshot(before.data()),
                                              read_contra_snapshot(after.data()));
    assert(reward.death == -kContraDeathPenalty);
    assert(reward.p2_progress == 0);
}

void test_player2_uses_its_own_axis() {
    // A regression guard: player 2's progress must come from its own
    // coordinates, not from player 1's.
    auto before = make_gameplay_ram();
    auto after = make_gameplay_ram();
    before[kContraPlayerMode] = 1;
    after[kContraPlayerMode] = 1;
    before[kContraPerspective] = 1;
    after[kContraPerspective] = 1;
    before[kContraPlayerYP2] = 10;
    after[kContraPlayerYP2] = 20;
    after[kContraPlayerY] = 200;  // player 1 moved; must not leak into p2
    assert(contra_progress_delta(read_contra_snapshot(before.data()),
                                 read_contra_snapshot(after.data()), true) == 10);
}

void test_game_over_and_continue_screen_end_the_episode() {
    auto status = make_gameplay_ram();
    status[kContraGameStatus] = 1;
    assert(contra_is_done(read_contra_snapshot(status.data())));

    auto cont = make_gameplay_ram();
    cont[kContraScreenType] = kContraScreenContinue;
    assert(contra_is_done(read_contra_snapshot(cont.data())));

    // The attract demo is not a playable episode either.
    auto demo = make_gameplay_ram();
    demo[kContraGameMode] = 1;
    assert(contra_is_done(read_contra_snapshot(demo.data())));
}

void test_static_frame_pays_nothing() {
    const auto s = read_contra_snapshot(make_gameplay_ram().data());
    assert(compute_contra_reward(s, s).total == 0);
}

}  // namespace

int main() {
    test_reads_the_documented_addresses();
    test_progress_follows_the_perspective();
    test_stage_screen_and_perspective_changes_suppress_progress();
    test_teleport_is_not_progress();
    test_score_increase_is_the_main_term();
    test_score_decrease_is_clamped();
    test_death_is_charged_once_and_suppresses_progress();
    test_gaining_a_lives_is_not_a_reward();
    test_stage_clear_is_one_shot();
    test_player2_is_ignored_in_one_player_games();
    test_player2_score_and_progress_are_counted();
    test_player2_life_loss_is_charged();
    test_player2_uses_its_own_axis();
    test_game_over_and_continue_screen_end_the_episode();
    test_static_frame_pays_nothing();
    return 0;
}