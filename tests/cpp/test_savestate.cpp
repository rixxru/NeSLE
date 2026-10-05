// Savestate capture, restore, and FCSX serialization.
//
// Two distinct properties are checked, because they fail for different reasons:
//
//   1. capture_state/apply_state must be exact *in memory*, including the PPU's
//      position inside the frame. That position has no representation in an FCSX
//      file, so if the pair dropped it, a restore would silently rewind the PPU and
//      the emulation would diverge - and no file-level test could see it.
//   2. serialize_fcsx must be the exact inverse of parse_fcsx for every field the
//      format carries, so a state written from a live console describes that same
//      machine when read back.
//
// Everything is heap-allocated. Console is ~23 KiB and render_rgb_frame() puts
// ~245 KiB on the stack per call, so two live consoles plus snapshots overflow the
// default 1 MiB Windows stack.

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "nesle/console.hpp"
#include "nesle/cpu.hpp"
#include "nesle/fcs.hpp"
#include "nesle/rom.hpp"

namespace {

std::vector<std::uint8_t> make_nrom_bytes(std::uint8_t prg_banks, std::uint8_t chr_banks) {
    std::vector<std::uint8_t> data = {'N', 'E', 'S', 0x1A, prg_banks, chr_banks, 0, 0};
    data.resize(16, 0);
    const auto prg_size = static_cast<std::size_t>(prg_banks) * 16 * 1024;
    for (std::size_t i = 0; i < prg_size; ++i) {
        const auto bank = static_cast<std::uint8_t>(i / (16 * 1024));
        data.push_back(static_cast<std::uint8_t>((bank << 7) | (i & 0x7F)));
    }
    // A CHR-ROM cartridge gets 8 KiB of filler. Zeroed rather than patterned: the
    // PPU reads these bytes as pattern data, and a nonzero pattern is enough to
    // reach core paths that are not what this test is about.
    if (chr_banks != 0) {
        data.insert(data.end(), 8 * 1024, 0);
    }
    return data;
}

// Byte-for-byte the NMI program test_console.cpp already exercises, plus the
// vectors. Reusing a known-good image matters: a ROM full of filler bytes sends the
// CPU into nonsense, and there are pre-existing crashes in the CPU/PPU core on some
// of those paths (reproducible with a plain NROM image and no savestate code
// involved). None of that belongs in a savestate test, so this stays on ground
// test_console.cpp has already proven safe.
nesle::RomImage make_savestate_rom(std::uint8_t chr_banks) {
    auto bytes = make_nrom_bytes(2, chr_banks);
    constexpr std::size_t kPrgOffset = 16;
    bytes[kPrgOffset + 0x0000] = 0xA9;  // LDA #$80
    bytes[kPrgOffset + 0x0001] = 0x80;
    bytes[kPrgOffset + 0x0002] = 0x8D;  // STA $2000
    bytes[kPrgOffset + 0x0003] = 0x00;
    bytes[kPrgOffset + 0x0004] = 0x20;  // JSR $9000  (run the NMI body inline)
    bytes[kPrgOffset + 0x0005] = 0x4C;
    bytes[kPrgOffset + 0x0006] = 0x90;
    bytes[kPrgOffset + 0x0007] = 0x4C;  // JMP $8005
    bytes[kPrgOffset + 0x0008] = 0x05;
    bytes[kPrgOffset + 0x0009] = 0x80;
    bytes[kPrgOffset + 0x1000] = 0xE6;  // INC $00
    bytes[kPrgOffset + 0x1001] = 0x00;
    bytes[kPrgOffset + 0x1002] = 0x40;  // RTI
    bytes[kPrgOffset + 0x7FFA] = 0x00;  // NMI vector -> $9000
    bytes[kPrgOffset + 0x7FFB] = 0x90;
    bytes[kPrgOffset + 0x7FFC] = 0x00;  // RESET vector -> $8000
    bytes[kPrgOffset + 0x7FFD] = 0x80;
    return nesle::parse_ines(bytes);
}


std::uint64_t hash_frame(const nesle::Ppu& ppu) {
    const auto rgb = ppu.render_rgb_frame();
    std::uint64_t h = 1469598103934665603ULL;
    for (const auto byte : rgb) {
        h = (h ^ byte) * 1099511628211ULL;
    }
    return h;
}

using ConsolePtr = std::unique_ptr<nesle::Console>;

ConsolePtr make_console(const nesle::RomImage& rom) {
    auto console = std::make_unique<nesle::Console>(rom);
    return console;
}

void dirty_ram(nesle::Console& console) {
    for (std::uint16_t addr = 0x0000; addr < 0x0800; addr += 7) {
        console.write(addr, static_cast<std::uint8_t>(addr * 3 + 1));
    }
}

// Fill OAM through a real $4014 DMA. Writing console.ppu().oam() directly is not
// possible - it is const - and asserting on an all-zero OAM is how a restore that
// forgets OAM entirely passes: the ROM below deliberately does a DMA so the sprite
// table is populated when the snapshot is taken.
void dirty_oam_via_dma(nesle::Console& console) {
    for (std::uint16_t i = 0; i < 256; ++i) {
        console.write(static_cast<std::uint16_t>(0x0200 + i),
                      static_cast<std::uint8_t>(0xFF - i));
    }
    console.write(0x4014, 0x02);  // OAMDMA from $0200
}


void run_frames(nesle::Console& console, nesle::cpu::CpuState& cpu, int frames, int seed) {
    for (int i = 0; i < frames; ++i) {
        console.controller1().set_buttons(static_cast<std::uint8_t>(i * seed + 3));
        (void)console.step_frame(cpu, 20000);
    }
}

// ---- 1. in-memory round trip is exact, mid-frame included ----

void test_memory_round_trip_is_exact() {
    const auto rom = make_savestate_rom(0);
    auto source = make_console(rom);
    auto target = make_console(rom);
    nesle::cpu::CpuState source_cpu;
    nesle::cpu::CpuState target_cpu;
    source->reset_cpu(source_cpu);
    target->reset_cpu(target_cpu);

    run_frames(*source, source_cpu, 40, 7);
    dirty_ram(*source);
    dirty_oam_via_dma(*source);

    auto snapshot = std::make_unique<nesle::fcs::StateSnapshot>(source->capture_state(source_cpu));
    target->apply_state(*snapshot, target_cpu);
    auto after = std::make_unique<nesle::fcs::StateSnapshot>(target->capture_state(target_cpu));

    assert(after->pc == snapshot->pc);
    assert(after->a == snapshot->a);
    assert(after->x == snapshot->x);
    assert(after->y == snapshot->y);
    assert(after->sp == snapshot->sp);
    assert(after->p == snapshot->p);
    assert(after->cycles == snapshot->cycles);
    assert(after->cpu_ram == snapshot->cpu_ram);
    assert(after->prg_ram == snapshot->prg_ram);
    assert(after->nametable_ram == snapshot->nametable_ram);
    assert(after->palette_ram == snapshot->palette_ram);
    assert(after->oam == snapshot->oam);
    // Non-vacuous: OAM must actually hold data here, or the line above proves nothing.
    const std::array<std::uint8_t, 256> zero_oam{};
    assert(snapshot->oam != zero_oam);
    assert(after->chr_ram == snapshot->chr_ram);
    assert(after->mapper_latch == snapshot->mapper_latch);
    assert(after->prg_bank == snapshot->prg_bank);
    assert(after->ppu_ctrl == snapshot->ppu_ctrl);
    assert(after->ppu_mask == snapshot->ppu_mask);
    assert(after->ppu_status == snapshot->ppu_status);
    assert(after->ppu_oam_addr == snapshot->ppu_oam_addr);
    assert(after->ppu_open_bus == snapshot->ppu_open_bus);
    assert(after->ppu_read_buffer == snapshot->ppu_read_buffer);
    assert(after->ppu_x == snapshot->ppu_x);
    assert(after->ppu_w == snapshot->ppu_w);
    assert(after->ppu_v == snapshot->ppu_v);
    assert(after->ppu_t == snapshot->ppu_t);
    assert(after->has_chr_ram == snapshot->has_chr_ram);
    // The part an FCSX file cannot hold. If this pair dropped it, every file-based
    // test would still pass while real runs silently diverged.
    assert(after->ppu_scanline == snapshot->ppu_scanline);
    assert(after->ppu_dot == snapshot->ppu_dot);
    assert(after->ppu_frame == snapshot->ppu_frame);
    assert(after->ppu_scroll_x == snapshot->ppu_scroll_x);
    assert(after->ppu_scroll_y == snapshot->ppu_scroll_y);
    assert(after->chr_ram == snapshot->chr_ram);

    // The property that actually matters: execution continues identically.
    run_frames(*source, source_cpu, 20, 11);
    run_frames(*target, target_cpu, 20, 11);
    assert(source->capture_state(source_cpu).cpu_ram ==
           target->capture_state(target_cpu).cpu_ram);
    assert(hash_frame(source->ppu()) == hash_frame(target->ppu()));
}

// ---- 2. the writer is the parser's inverse ----

void check_round_trip(const nesle::RomImage& rom, bool expect_chr_ram) {
    auto console = make_console(rom);
    nesle::cpu::CpuState cpu;
    console->reset_cpu(cpu);
    run_frames(*console, cpu, 25, 5);
    dirty_ram(*console);
    dirty_oam_via_dma(*console);
    dirty_oam_via_dma(*console);

    const auto live = console->capture_state(cpu);
    assert(live.has_chr_ram == expect_chr_ram);

    const auto image = nesle::fcs::serialize_fcsx(live);
    assert(image.size() > 16);
    assert(image[0] == 'F' && image[1] == 'C' && image[2] == 'S' && image[3] == 'X');

    // Header: payload size, then the two words FCEUX 2.6.x writes.
    std::uint32_t declared = 0;
    for (int i = 0; i < 4; ++i) {
        declared |= static_cast<std::uint32_t>(image[4 + static_cast<std::size_t>(i)])
                    << (8 * i);
    }
    assert(declared == image.size() - 16);

    std::uint32_t word8 = 0;
    std::uint32_t word12 = 0;
    for (int i = 0; i < 4; ++i) {
        word8 |= static_cast<std::uint32_t>(image[8 + static_cast<std::size_t>(i)]) << (8 * i);
        word12 |= static_cast<std::uint32_t>(image[12 + static_cast<std::size_t>(i)]) << (8 * i);
    }
    assert(word8 == 0x0000507E);
    assert(word12 == 0xFFFFFFFF);

    // Blocks must tile the payload exactly.
    std::size_t off = 16;
    bool saw_cpu = false;
    bool saw_ppu = false;
    bool saw_ram = false;
    bool saw_chr = false;
    while (off + 5 <= image.size()) {
        const auto id = image[off];
        std::uint32_t size = 0;
        for (int i = 0; i < 4; ++i) {
            size |= static_cast<std::uint32_t>(image[off + 1 + static_cast<std::size_t>(i)])
                    << (8 * i);
        }
        off += 5;
        assert(off + size <= image.size());
        saw_cpu = saw_cpu || id == 0x01;
        saw_ppu = saw_ppu || id == 0x03;
        saw_ram = saw_ram || id == 0x08;
        saw_chr = saw_chr || id == 0x10;
        off += size;
    }
    assert(off == image.size());
    assert(saw_cpu && saw_ppu && saw_ram);
    assert(saw_chr == expect_chr_ram);

    const auto parsed = nesle::fcs::parse(image);
    assert(parsed.pc == live.pc);
    assert(parsed.a == live.a);
    assert(parsed.x == live.x);
    assert(parsed.y == live.y);
    assert(parsed.sp == live.sp);
    assert(parsed.p == live.p);
    assert(parsed.cpu_ram == live.cpu_ram);
    assert(parsed.prg_ram == live.prg_ram);
    assert(parsed.nametable_ram == live.nametable_ram);
    assert(parsed.palette_ram == live.palette_ram);
    assert(parsed.oam == live.oam);
    const std::array<std::uint8_t, 256> zero_oam{};
    assert(live.oam != zero_oam);
    assert(parsed.ppu_ctrl == live.ppu_ctrl);
    assert(parsed.ppu_mask == live.ppu_mask);
    assert(parsed.ppu_status == live.ppu_status);
    assert(parsed.ppu_oam_addr == live.ppu_oam_addr);
    assert(parsed.ppu_open_bus == live.ppu_open_bus);
    assert(parsed.ppu_read_buffer == live.ppu_read_buffer);
    assert(parsed.ppu_x == live.ppu_x);
    assert(parsed.ppu_w == live.ppu_w);
    assert(parsed.ppu_v == live.ppu_v);
    assert(parsed.ppu_t == live.ppu_t);
    assert(parsed.has_chr_ram == live.has_chr_ram);
    if (live.has_chr_ram) {
        // CHR RAM surviving the file is the difference between a restored screen
        // and a black one on a CHR-RAM cartridge.
        assert(parsed.chr_ram == live.chr_ram);
    }
}

// A file round trip must describe the same machine: restoring a written state and
// re-reading it has to be a fixed point for everything the format carries.
void test_file_round_trip_is_a_fixed_point() {
    const auto rom = make_savestate_rom(0);
    auto console = make_console(rom);
    nesle::cpu::CpuState cpu;
    console->reset_cpu(cpu);
    run_frames(*console, cpu, 30, 13);

    const auto image = console->capture_state(cpu);
    const auto written = nesle::fcs::serialize_fcsx(image);
    const auto reparsed = nesle::fcs::parse(written);
    const auto rewritten = nesle::fcs::serialize_fcsx(reparsed);

    assert(written.size() == rewritten.size());
    for (std::size_t i = 0; i < written.size(); ++i) {
        assert(written[i] == rewritten[i]);
    }
}

// A legacy FCS file must still load; the writer only emits FCSX but the parser is
// shared, and loading a state FCEUX wrote long ago is the whole point of the path.
void test_legacy_fcs_still_parses() {
    const auto rom = make_savestate_rom(0);
    auto console = make_console(rom);
    nesle::cpu::CpuState cpu;
    console->reset_cpu(cpu);
    run_frames(*console, cpu, 10, 3);

    // Hand-build a minimal legacy FCS: 'FCS\xff', a 16-byte header, then CPU / PPU /
    // mapper chunks. The parser starts at offset 16 and uses tags 1, 3 and 16.
    const auto live = console->capture_state(cpu);
    std::vector<std::uint8_t> legacy = {'F', 'C', 'S', 0xFF};
    legacy.resize(16, 0);


    const auto push_chunk = [&legacy](std::uint8_t tag, const std::vector<std::uint8_t>& body) {
        legacy.push_back(tag);
        for (int i = 0; i < 4; ++i) {
            legacy.push_back(static_cast<std::uint8_t>(body.size() >> (8 * i)));
        }
        legacy.insert(legacy.end(), body.begin(), body.end());
    };
    const auto push_sub = [](std::vector<std::uint8_t>& body, const char* name,
                             const std::vector<std::uint8_t>& data) {
        // Bound by the name's length. A two-character name like "PC" is stored as
        // three bytes including the NUL, so indexing to 4 unconditionally reads past
        // the literal.
        const std::size_t name_len = std::strlen(name);
        for (std::size_t i = 0; i < 4; ++i) {
            body.push_back(i < name_len ? static_cast<std::uint8_t>(name[i]) : 0);
        }
        for (int i = 0; i < 4; ++i) {
            body.push_back(static_cast<std::uint8_t>(data.size() >> (8 * i)));
        }
        body.insert(body.end(), data.begin(), data.end());
    };

    std::vector<std::uint8_t> cpu_body;
    const std::vector<std::uint8_t> pc{static_cast<std::uint8_t>(live.pc & 0xFF),
                                       static_cast<std::uint8_t>(live.pc >> 8)};
    push_sub(cpu_body, "PC", pc);
    push_sub(cpu_body, "RAM", {live.cpu_ram.begin(), live.cpu_ram.end()});
    push_chunk(0x01, cpu_body);

    std::vector<std::uint8_t> ppu_body;
    push_sub(ppu_body, "NTAR", {live.nametable_ram.begin(), live.nametable_ram.end()});
    push_sub(ppu_body, "SPRA", {live.oam.begin(), live.oam.end()});
    push_chunk(3, ppu_body);
    push_chunk(16, {live.prg_ram.begin(), live.prg_ram.end()});


    const auto parsed = nesle::fcs::parse(legacy);
    assert(parsed.pc == live.pc);
    assert(parsed.cpu_ram == live.cpu_ram);
    assert(parsed.nametable_ram == live.nametable_ram);
    assert(parsed.oam == live.oam);
    assert(parsed.prg_ram == live.prg_ram);
}

}  // namespace

int main() {
    test_memory_round_trip_is_exact();
    test_file_round_trip_is_a_fixed_point();
    test_legacy_fcs_still_parses();
    check_round_trip(make_savestate_rom(0), /*expect_chr_ram=*/true);
    check_round_trip(make_savestate_rom(1), /*expect_chr_ram=*/false);
    std::printf("test_savestate: all checks passed\n");
    return 0;
}