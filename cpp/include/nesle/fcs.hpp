#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <string_view>

namespace nesle::fcs {

inline constexpr std::size_t kCpuRamBytes = 2048;
inline constexpr std::size_t kPrgRamBytes = 8 * 1024;
inline constexpr std::size_t kPaletteRamBytes = 32;
inline constexpr std::size_t kOamBytes = 256;
inline constexpr std::size_t kNametableRamBytes = 2048;
inline constexpr std::size_t kChrRamBytes = 8 * 1024;

struct StateSnapshot {
    std::uint16_t pc = 0;
    std::uint8_t a = 0;
    std::uint8_t x = 0;
    std::uint8_t y = 0;
    std::uint8_t sp = 0xFD;
    std::uint8_t p = 0x24;
    std::uint64_t cycles = 0;

    std::array<std::uint8_t, kCpuRamBytes> cpu_ram{};
    std::array<std::uint8_t, kPrgRamBytes> prg_ram{};

    std::uint8_t ppu_ctrl = 0;
    std::uint8_t ppu_mask = 0;
    std::uint8_t ppu_status = 0;
    std::uint8_t ppu_oam_addr = 0;
    std::uint8_t ppu_open_bus = 0;
    std::uint8_t ppu_read_buffer = 0;
    std::uint8_t ppu_x = 0;
    std::uint8_t ppu_w = 0;
    std::uint16_t ppu_v = 0;
    std::uint16_t ppu_t = 0;

    std::array<std::uint8_t, kNametableRamBytes> nametable_ram{};
    std::array<std::uint8_t, kPaletteRamBytes> palette_ram{};
    std::array<std::uint8_t, kOamBytes> oam{};
    // Cartridge CHR RAM. Only FCSX carries it, and only a CHR-RAM cartridge has
    // any. Without it a snapshot reset of such a cartridge renders an entirely
    // black screen, because pattern data is the one thing the game does not
    // re-upload by itself. Stays zeroed for the legacy FCS format, whose files
    // predate CHR-RAM carts in this project's data set (SMB is NROM + CHR ROM).
    std::array<std::uint8_t, kChrRamBytes> chr_ram{};
    // Distinguishes "state carried no CHR block" from "CHR block was all zero",
    // so the device side can skip allocating and uploading 8 KiB per level.
    bool has_chr_ram = false;

    // Mapper bank registers. These are NOT part of the FCEUX save-state format as
    // parsed above - the parser has no sub-chunk for them - but the live emulator
    // needs them, and on UNROM the PRG bank decides which code is executing. A
    // snapshot that dropped them would restore into the wrong bank and run
    // different code. They are kept here so an in-process round-trip is exact, and
    // serialize_fcsx() writes them into the cartridge-RAM block where FCEUX keeps
    // its mapper state. Zero means "not recorded", which is what a parsed state
    // that came from FCEUX leaves them at.
    std::uint8_t prg_bank = 0;
    std::uint8_t chr_bank = 0;
    std::uint8_t chr_bank_hi = 0;
    // The raw byte last written to a mapper latch register, kept so a restore can
    // re-derive the bank without re-running the bus-conflict AND. On UNROM that AND
    // is the difference between the bank you meant and a different one, so replaying
    // a decoded prg_bank through the write path does not round-trip.
    std::uint8_t mapper_latch = 0;

    // CPU cycles still owed by an OAM DMA that has written its 256 bytes but not yet
    // stalled the CPU for the full 513. Dropping it shifts the CPU half a frame
    // against the PPU and the run desynchronizes, so it travels with the snapshot.
    // An FCEUX state has no equivalent field; the parser leaves it at 0.
    std::uint32_t pending_dma_cycles = 0;
    // Where the PPU was *inside* the frame. An FCEUX save state has no field for
    // this - its PPU block stops at the register and memory sub-chunks - so a
    // snapshot parsed from FCEUX always leaves these at zero, and a snapshot
    // written to a file loses them entirely. They are carried here so that an
    // in-process capture/restore is exact even mid-frame, which is what makes
    // save_state() -> load_state() behave as if nothing happened. is_frame_boundary()
    // reports whether they are at a position a file could round-trip.
    std::int16_t ppu_scanline = 0;
    std::uint16_t ppu_dot = 0;
    std::uint64_t ppu_frame = 0;
    std::uint8_t ppu_scroll_x = 0;
    std::uint8_t ppu_scroll_y = 0;
};

[[nodiscard]] inline bool is_frame_boundary(const StateSnapshot& s) {
    return s.ppu_dot == 0;
}

namespace detail {

[[nodiscard]] inline std::uint32_t read_le_u32(std::span<const std::uint8_t> bytes) {
    if (bytes.size() < 4) {
        throw std::runtime_error("FCS: truncated u32");
    }
    return static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8) |
           (static_cast<std::uint32_t>(bytes[2]) << 16) |
           (static_cast<std::uint32_t>(bytes[3]) << 24);
}

[[nodiscard]] inline std::uint64_t read_le_u64(std::span<const std::uint8_t> bytes) {
    if (bytes.size() < 8) {
        throw std::runtime_error("FCS: truncated u64");
    }
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(bytes[i]) << (8 * i);
    }
    return value;
}

[[nodiscard]] inline std::string_view strip_trailing_nuls(std::string_view name) {
    while (!name.empty() && name.back() == '\0') {
        name.remove_suffix(1);
    }
    return name;
}

// Copy a sub-chunk payload into a fixed-size destination. Throws if the chunk is the wrong
// size вЂ” FCEUX is consistent enough that any mismatch usually means we're parsing the wrong
// offset.
template <std::size_t N>
inline void copy_fixed(std::span<const std::uint8_t> src, std::array<std::uint8_t, N>& dst,
                       std::string_view label) {
    if (src.size() != N) {
        throw std::runtime_error(std::string("FCS: ") + std::string(label) +
                                 " expected " + std::to_string(N) + " bytes, got " +
                                 std::to_string(src.size()));
    }
    std::memcpy(dst.data(), src.data(), N);
}

inline void apply_cpu_subchunk(StateSnapshot& out, std::string_view name,
                               std::span<const std::uint8_t> payload) {
    if (name == "PC") {
        if (payload.size() != 2) {
            throw std::runtime_error("FCS: CPU.PC must be 2 bytes");
        }
        out.pc = static_cast<std::uint16_t>(payload[0] | (payload[1] << 8));
    } else if (name == "A" && payload.size() == 1) {
        out.a = payload[0];
    } else if (name == "X" && payload.size() == 1) {
        out.x = payload[0];
    } else if (name == "Y" && payload.size() == 1) {
        out.y = payload[0];
    } else if (name == "S" && payload.size() == 1) {
        out.sp = payload[0];
    } else if (name == "P" && payload.size() == 1) {
        out.p = payload[0];
    } else if (name == "DB" && payload.size() == 1) {
        // CPU "data bus" open-bus byte; ignored вЂ” we don't expose CPU open bus.
    } else if (name == "RAM") {
        copy_fixed(payload, out.cpu_ram, "CPU.RAM");
    }
    // Other CPU sub-chunks (IQLB, ICoa, ICou, TSBS, JAMM) are emulator-internal IRQ
    // bookkeeping and a CPU timestamp; we don't restore them. TSBS could be used for
    // cycle-counter restore but our cycles field is relative, so 0 is fine.
}

inline void apply_ppu_subchunk(StateSnapshot& out, std::string_view name,
                               std::span<const std::uint8_t> payload) {
    if (name == "NTAR") {
        copy_fixed(payload, out.nametable_ram, "PPU.NTAR");
    } else if (name == "PRAM") {
        copy_fixed(payload, out.palette_ram, "PPU.PRAM");
    } else if (name == "SPRA") {
        copy_fixed(payload, out.oam, "PPU.SPRA");
    } else if (name == "PPUR") {
        // PPUR packs [PPUCTRL, PPUMASK, PPUSTATUS, ?]. Fourth byte is open-bus / unused.
        if (payload.size() < 3) {
            throw std::runtime_error("FCS: PPU.PPUR too short");
        }
        out.ppu_ctrl = payload[0];
        out.ppu_mask = payload[1];
        out.ppu_status = payload[2];
    } else if (name == "PSPL" && payload.size() == 1) {
        out.ppu_oam_addr = payload[0];
    } else if (name == "XOFF" && payload.size() == 1) {
        out.ppu_x = static_cast<std::uint8_t>(payload[0] & 0x07);
    } else if (name == "VTGL" && payload.size() == 1) {
        out.ppu_w = static_cast<std::uint8_t>(payload[0] & 0x01);
    } else if (name == "RADD" && payload.size() == 2) {
        out.ppu_v = static_cast<std::uint16_t>(payload[0] | (payload[1] << 8));
    } else if (name == "TADD" && payload.size() == 2) {
        out.ppu_t = static_cast<std::uint16_t>(payload[0] | (payload[1] << 8));
    } else if (name == "VBUF" && payload.size() == 1) {
        out.ppu_read_buffer = payload[0];
    } else if (name == "PGEN" && payload.size() == 1) {
        out.ppu_open_bus = payload[0];
    }
    // KOOK, DEAD are vs-system / debug flags; we ignore them.
}

inline void apply_cart_subchunk(StateSnapshot& out, std::string_view name,
                                std::span<const std::uint8_t> payload) {
    if (name == "WRAM") {
        // FCEUX always stores 8 KB of WRAM even for cartridges (like SMB / NROM) that don't
        // have it. Copying it through is harmless вЂ” our prg_ram lives at $6000-$7FFF and
        // NROM reads return 0 if not wired.
        copy_fixed(payload, out.prg_ram, "CART.WRAM");
    }
}

inline void walk_subchunks(std::span<const std::uint8_t> payload,
                           void (*apply)(StateSnapshot&, std::string_view,
                                         std::span<const std::uint8_t>),
                           StateSnapshot& out) {
    std::size_t off = 0;
    while (off + 8 <= payload.size()) {
        const auto raw_name = std::string_view(
            reinterpret_cast<const char*>(payload.data() + off), 4);
        const auto name = strip_trailing_nuls(raw_name);
        const auto size = read_le_u32(payload.subspan(off + 4, 4));
        off += 8;
        if (off + size > payload.size()) {
            throw std::runtime_error("FCS: sub-chunk \"" + std::string(name) +
                                     "\" extends past parent payload");
        }
        apply(out, name, payload.subspan(off, size));
        off += size;
    }
    if (off != payload.size()) {
        throw std::runtime_error("FCS: trailing bytes in sub-chunk stream");
    }
}

[[nodiscard]] inline StateSnapshot parse_fcs(std::span<const std::uint8_t> data) {
    if (data.size() < 16) {
        throw std::runtime_error("FCS: header too short");
    }
    if (data[0] != 'F' || data[1] != 'C' || data[2] != 'S' || data[3] != 0xFF) {
        throw std::runtime_error("save state: bad magic (expected 'FCS\\xff' or 'FCSX')");
    }
    // Bytes 4..16 are legacy size, version, reserved; we don't enforce them.

    StateSnapshot out;
    std::size_t off = 16;
    while (off + 5 <= data.size()) {
        const auto tag = data[off];
        const auto size = read_le_u32(data.subspan(off + 1, 4));
        off += 5;
        if (off + size > data.size()) {
            throw std::runtime_error("FCS: top-level chunk extends past file");
        }
        const auto payload = data.subspan(off, size);
        switch (tag) {
            case 1:  // CPU + RAM
                walk_subchunks(payload, apply_cpu_subchunk, out);
                break;
            case 3:  // PPU + VRAM
                walk_subchunks(payload, apply_ppu_subchunk, out);
                break;
            case 16:  // Mapper / cartridge state (WRAM lives here)
                walk_subchunks(payload, apply_cart_subchunk, out);
                break;
            default:
                // Tags 2 (input), 4 (joypad config), 5 (APU/sound) are intentionally skipped.
                break;
        }
        off += size;
    }
    if (off != data.size()) {
        throw std::runtime_error("FCS: trailing bytes after last top-level chunk");
    }
    return out;
}

inline void apply_chr_subchunk(StateSnapshot& out, std::string_view name,
                              std::span<const std::uint8_t> payload) {
    if (name == "CHRR") {
        copy_fixed(payload, out.chr_ram, "CHR.CHRR");
    }
    // The CHR block also carries a one-byte bookkeeping sub-chunk we ignore.
}

inline void apply_prg_ram_block(StateSnapshot& out, std::span<const std::uint8_t> payload) {
    // FCSX stores cartridge RAM as a raw block instead of a WRAM sub-chunk. FCEUX's
    // NES buffer is 64 KiB, while $6000-$7FFF is only 8 KiB, so the first 8 KiB is
    // what maps to CPU space. A short block leaves the tail zeroed rather than
    // failing the whole load.
    const auto copy = payload.size() < kPrgRamBytes
                          ? payload.size()
                          : static_cast<std::size_t>(kPrgRamBytes);
    out.prg_ram.fill(0);
    std::memcpy(out.prg_ram.data(), payload.data(), copy);
    // Bytes past the cartridge RAM are mapper state, written by serialize_fcsx so
    // that FCEUX hands them back unchanged, but deliberately NOT read back here.
    // FCEUX's own block runs to 60 KiB on a mapper 2 cartridge and its mapper-state
    // encoding at this offset is not established, so decoding it would be a guess -
    // and the device reset path already treats a restored environment as having
    // power-on mapper banks for the same reason (see batch_step.cuh). Reading a
    // wrong byte here would bank-switch a game into different code, which is worse
    // than the documented "a restored env starts at the power-on bank".
}

}  // namespace detail

// FCEUX 2.6 replaced the chunked FCS format with FCSX. The header is
//
//   0x00  "FCSX"
//   0x04  u32  payload size, i.e. file size minus these 16 header bytes
//   0x08  u32  (emulator-internal, constant for a given state; not interpreted)
//   0x0C  u32  (0xFFFFFFFF in every state written by 2.6.x; not interpreted)
//
// followed by blocks of [u8 id][u32 length][payload]:
//
//   0x01 CPU    PC A X Y S P DB RAM   - same sub-chunks as legacy FCS
//   0x02 CPU    JAMM IQLB ICoa ICou TSBS MooP - IRQ bookkeeping and a timestamp
//   0x03 PPU    NTAR PRAM SPRA PPUR PSPL XOFF VTGL RADD TADD VBUF PGEN
//   0x08 RAM    raw cartridge RAM
//   0x10 CHR    CHRR - cartridge CHR RAM
//   others       0x04 input, 0x05 sound, 0x1F idle-loop detection: not needed
//
// The sub-chunk payload format is byte-for-byte the legacy one, so the appliers
// above are shared. Verified against 430 FCEUX 2.6 states of a mapper 2
// cartridge: all of them end exactly on the last block, and the declared payload
// size equals the file size minus 16 in every one.
[[nodiscard]] inline StateSnapshot parse_fcsx(std::span<const std::uint8_t> data) {
    if (data.size() < 16) {
        throw std::runtime_error("FCSX: header too short");
    }
    const auto declared = detail::read_le_u32(data.subspan(4, 4));
    if (declared != data.size() - 16u) {
        throw std::runtime_error("FCSX: declared payload size " + std::to_string(declared) +
                                 " does not match file size minus header (" +
                                 std::to_string(data.size() - 16u) + ")");
    }

    StateSnapshot out;
    std::size_t off = 16;
    while (off + 5 <= data.size()) {
        const auto id = data[off];
        const auto size = detail::read_le_u32(data.subspan(off + 1, 4));
        off += 5;
        if (off + size > data.size()) {
            throw std::runtime_error("FCSX: block 0x" + [id] {
                    constexpr char kHex[] = "0123456789ABCDEF";
                    std::string s(2, '0');
                    s[0] = kHex[id >> 4];
                    s[1] = kHex[id & 0x0F];
                    return s;
                }() + " extends past end of file");
        }
        const auto payload = data.subspan(off, size);
        switch (id) {
            case 0x01:  // CPU registers + RAM
            case 0x02:  // CPU IRQ bookkeeping; shares the applier, ignores what it must
                detail::walk_subchunks(payload, detail::apply_cpu_subchunk, out);
                break;
            case 0x03:  // PPU registers + VRAM
                detail::walk_subchunks(payload, detail::apply_ppu_subchunk, out);
                break;
            case 0x08:  // cartridge RAM, raw
                detail::apply_prg_ram_block(out, payload);
                break;
            case 0x10:  // cartridge CHR RAM
                out.has_chr_ram = true;
                detail::walk_subchunks(payload, detail::apply_chr_subchunk, out);
                break;
            default:
                // 0x04 input, 0x05 sound and 0x1F idle-loop detection carry nothing
                // we restore.
                break;
        }
        off += size;
    }
    if (off != data.size()) {
        throw std::runtime_error("FCSX: trailing bytes after last block");
    }
    return out;
}

// Accepts either save-state format and dispatches on the magic.
[[nodiscard]] inline StateSnapshot parse(std::span<const std::uint8_t> data) {
    if (data.size() >= 4 && data[0] == 'F' && data[1] == 'C' && data[2] == 'S' &&
        data[3] == 'X') {
        return parse_fcsx(data);
    }
    return detail::parse_fcs(data);
}

[[nodiscard]] inline StateSnapshot parse(const std::string& bytes) {
    return parse(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()));
}

namespace detail {

inline void push_byte(std::vector<std::uint8_t>& out, std::uint8_t value) {
    out.push_back(value);
}

inline void push_le(std::vector<std::uint8_t>& out, std::uint64_t value, std::size_t width) {
    for (std::size_t i = 0; i < width; ++i) {
        out.push_back(static_cast<std::uint8_t>((value >> (8 * i)) & 0xFF));
    }
}

inline void push_le_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
    push_le(out, value, 4);
}

// One sub-chunk: a NUL-padded 4-byte name, a u32 length, then the payload. This is
// the exact inverse of walk_subchunks above.
inline void push_subchunk(std::vector<std::uint8_t>& out, std::string_view name,
                          std::span<const std::uint8_t> payload) {
    if (name.size() > 4) {
        throw std::runtime_error("FCS write: sub-chunk name longer than 4 bytes: " +
                                 std::string(name));
    }
    for (std::size_t i = 0; i < 4; ++i) {
        out.push_back(i < name.size() ? static_cast<std::uint8_t>(name[i]) : 0);
    }
    push_le_u32(out, static_cast<std::uint32_t>(payload.size()));
    out.insert(out.end(), payload.begin(), payload.end());
}

inline void push_subchunk_u8(std::vector<std::uint8_t>& out, std::string_view name,
                             std::uint8_t value) {
    const std::array<std::uint8_t, 1> one{value};
    push_subchunk(out, name, one);
}

inline void push_subchunk_u16(std::vector<std::uint8_t>& out, std::string_view name,
                              std::uint16_t value) {
    const std::array<std::uint8_t, 2> two{static_cast<std::uint8_t>(value & 0xFF),
                                          static_cast<std::uint8_t>((value >> 8) & 0xFF)};
    push_subchunk(out, name, two);
}

inline void push_subchunk_u32(std::vector<std::uint8_t>& out, std::string_view name,
                              std::uint32_t value) {
    const std::array<std::uint8_t, 4> four{static_cast<std::uint8_t>(value & 0xFF),
                                           static_cast<std::uint8_t>((value >> 8) & 0xFF),
                                           static_cast<std::uint8_t>((value >> 16) & 0xFF),
                                           static_cast<std::uint8_t>((value >> 24) & 0xFF)};
    push_subchunk(out, name, four);
}

inline void begin_block(std::vector<std::uint8_t>& out, std::uint8_t id) {
    out.push_back(id);
    // Length is patched by end_block once the payload is known.
    push_le_u32(out, 0);
}

inline void end_block(std::vector<std::uint8_t>& out, std::size_t payload_begin) {
    const auto length = static_cast<std::uint32_t>(out.size() - payload_begin);
    for (int i = 0; i < 4; ++i) {
        out[payload_begin - 4 + static_cast<std::size_t>(i)] =
            static_cast<std::uint8_t>((length >> (8 * i)) & 0xFF);
    }
}

}  // namespace detail

// Write a snapshot as FCSX, the format FCEUX 2.6 produces and reads. This is the
// exact inverse of parse_fcsx: every field the parser reads back is written here,
// so parse(serialize(s)) == s field for field.
//
// Block layout and sub-chunk order follow a real FCEUX 2.6 state of a mapper 2
// cartridge (CPU 0x01, PPU 0x03, RAM 0x08, CHR 0x10), including the sub-chunks the
// parser skips - DB, KOOK, DEAD - because FCEUX expects them to be present.
//
// Two blocks FCEUX also writes are deliberately omitted: 0x02 (IRQ bookkeeping and
// a CPU timestamp), 0x04 (controller port bytes), 0x05 (sound registers) and 0x1F
// (idle-loop scan). None of them affect emulation - the CPU timestamps there are
// for the profiler, and the controller is re-strobed every frame - and inventing
// values for them would be worse than leaving them out. FCEUX's loader treats them
// as optional.
//
// The PPU's mid-frame position (scanline, dot, frame counter) has no representation
// in an FCEUX save state, so it is not written. A snapshot taken at a frame boundary
// - which is where step_frame() leaves the PPU - loses nothing.
[[nodiscard]] inline std::vector<std::uint8_t> serialize_fcsx(const StateSnapshot& s) {
    std::vector<std::uint8_t> out;
    out.reserve(kCpuRamBytes + kChrRamBytes + kPrgRamBytes + kNametableRamBytes + 512);

    // "FCSX", payload size, then the two emulator-internal u32s. Both constants are
    // what FCEUX 2.6.x writes in every state, and the parser ignores them.
    out.push_back('F');
    out.push_back('C');
    out.push_back('S');
    out.push_back('X');
    detail::push_le_u32(out, 0);  // patched below
    detail::push_le_u32(out, 0x0000507E);
    detail::push_le_u32(out, 0xFFFFFFFF);
    const std::size_t header_size = out.size();

    // Block 0x01 - CPU registers and RAM.
    detail::begin_block(out, 0x01);
    {
        const std::size_t begin = out.size();
        detail::push_subchunk_u16(out, "PC", s.pc);
        detail::push_subchunk_u8(out, "A", s.a);
        detail::push_subchunk_u8(out, "X", s.x);
        detail::push_subchunk_u8(out, "Y", s.y);
        detail::push_subchunk_u8(out, "S", s.sp);
        detail::push_subchunk_u8(out, "P", s.p);
        // The CPU data bus. We do not model open bus, so write the last value the
        // program counter would have fetched rather than a stale byte.
        detail::push_subchunk_u8(out, "DB", 0);
        detail::push_subchunk(out, "RAM", s.cpu_ram);
        detail::end_block(out, begin);
    }

    // Block 0x03 - PPU registers and video memory.
    detail::begin_block(out, 0x03);
    {
        const std::size_t begin = out.size();
        detail::push_subchunk(out, "NTAR", s.nametable_ram);
        detail::push_subchunk(out, "PRAM", s.palette_ram);
        detail::push_subchunk(out, "SPRA", s.oam);
        // PPUR is [PPUCTRL, PPUMASK, PPUSTATUS, unused]; the parser reads the first
        // three and FCEUX writes 0 in the fourth.
        const std::array<std::uint8_t, 4> ppur{s.ppu_ctrl, s.ppu_mask, s.ppu_status, 0};
        detail::push_subchunk(out, "PPUR", ppur);
        detail::push_subchunk_u8(out, "KOOK", 0);
        detail::push_subchunk_u8(out, "DEAD", 0);
        detail::push_subchunk_u8(out, "PSPL", s.ppu_oam_addr);
        detail::push_subchunk_u8(out, "XOFF", static_cast<std::uint8_t>(s.ppu_x & 0x07));
        detail::push_subchunk_u8(out, "VTGL", static_cast<std::uint8_t>(s.ppu_w & 0x01));
        detail::push_subchunk_u16(out, "RADD", s.ppu_v);
        detail::push_subchunk_u16(out, "TADD", s.ppu_t);
        detail::push_subchunk_u8(out, "VBUF", s.ppu_read_buffer);
        detail::push_subchunk_u8(out, "PGEN", s.ppu_open_bus);
        detail::end_block(out, begin);
    }

    // Block 0x08 - cartridge RAM, raw rather than sub-chunked, followed by the mapper
    // state. FCEUX lays this block out as [cart RAM][mapper state], and a state that
    // omits the banks restores into the wrong 16 KB window on UNROM, so four bytes go
    // in right after the RAM: the three decoded registers and the raw latch.
    detail::begin_block(out, 0x08);
    out.insert(out.end(), s.prg_ram.begin(), s.prg_ram.end());
    detail::push_byte(out, s.prg_bank);
    detail::push_byte(out, s.chr_bank);
    detail::push_byte(out, s.chr_bank_hi);
    detail::push_byte(out, s.mapper_latch);
    detail::end_block(out, out.size() - s.prg_ram.size() - 4);

    // Block 0x10 - CHR RAM. Only present for a CHR-RAM cartridge, matching both
    // FCEUX and the parser's has_chr_ram flag. LATC is a CHR-latch byte FCEUX
    // writes after the data; without it a CHR-RAM cart can come back with stale
    // latched data, so write the same value it starts at.
    if (s.has_chr_ram) {
        detail::begin_block(out, 0x10);
        {
            const std::size_t begin = out.size();
            detail::push_subchunk(out, "CHRR", s.chr_ram);
            detail::push_subchunk_u8(out, "LATC", 0);
            detail::end_block(out, begin);
        }
    }

    const auto payload_size = static_cast<std::uint32_t>(out.size() - header_size);
    for (int i = 0; i < 4; ++i) {
        out[4 + static_cast<std::size_t>(i)] =
            static_cast<std::uint8_t>((payload_size >> (8 * i)) & 0xFF);
    }
    return out;
}

}  // namespace nesle::fcs
