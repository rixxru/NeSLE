#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include "nesle/controller.hpp"
#include "nesle/cpu.hpp"
#include "nesle/fcs.hpp"
#include "nesle/ppu.hpp"
#include "nesle/rom.hpp"

namespace nesle {

class Console {
public:
    static constexpr std::size_t kCpuRamBytes = 2048;
    static constexpr std::size_t kPrgRamBytes = 8 * 1024;
    static constexpr std::size_t kApuIoBytes = 0x18;
    static constexpr std::uint32_t kPpuCyclesPerCpuCycle = 3;

    struct StepResult {
        cpu::StepResult cpu;
        std::uint32_t cpu_cycles = 0;
        std::uint32_t ppu_cycles = 0;
        std::uint32_t frames_completed = 0;
        bool nmi_serviced = false;
        bool nmi_started = false;
    };

    struct FrameResult {
        std::uint64_t instructions = 0;
        std::uint64_t cpu_cycles = 0;
        std::uint32_t frames_completed = 0;
    };

    explicit Console(RomImage rom)
        : rom_(std::move(rom)),
          layout_(describe_mapper(rom_.metadata)) {
        if (!layout_.supported) {
            throw std::invalid_argument(unsupported_mapper_reason(rom_.metadata));
        }
        if (rom_.prg_rom.empty()) {
            throw std::invalid_argument("Console requires PRG ROM bytes");
        }
        ppu_.configure_cartridge(rom_.chr_rom, rom_.metadata.nametable_arrangement);
        if (layout_.bank_kind == kBankKindNina8k) {
            // Power-on NINA-001 state: both 4 KB windows sit at window 0, so
            // chr_rom[0..0xFFF] is mirrored across the whole pattern space.
            ppu_.set_chr_windows(0, 0);
        }
    }

    [[nodiscard]] std::uint8_t read(std::uint16_t address) noexcept {
        if (address < 0x2000) {
            return cpu_ram_[address & 0x07FF];
        }
        if (address < 0x4000) {
            return ppu_.read_register(static_cast<std::uint16_t>((address - 0x2000) & 0x0007));
        }
        if (address == 0x4016) {
            return controller1_.read();
        }
        if (address == 0x4017) {
            return controller2_.read();
        }
        if (address < 0x4018) {
            return apu_io_[address - 0x4000];
        }
        if (address >= 0x6000 && address < 0x8000) {
            return read_prg_ram(address);
        }
        if (address >= 0x8000) {
            return read_prg(address);
        }
        return 0;
    }

    void write(std::uint16_t address, std::uint8_t value) noexcept {
        if (address < 0x2000) {
            cpu_ram_[address & 0x07FF] = value;
            return;
        }
        if (address < 0x4000) {
            ppu_.write_register(static_cast<std::uint16_t>((address - 0x2000) & 0x0007), value);
            return;
        }
        if (address == 0x4014) {
            apu_io_[address - 0x4000] = value;
            run_oam_dma(value);
            return;
        }
        if (address == 0x4016) {
            apu_io_[address - 0x4000] = value;
            controller1_.write_strobe(value);
            controller2_.write_strobe(value);
            return;
        }
        if (address < 0x4018) {
            apu_io_[address - 0x4000] = value;
            return;
        }
        if (address >= 0x6000 && address < 0x8000) {
            // NINA-001 puts its three registers inside PRG RAM: the byte is
            // both stored and interpreted, so the RAM store comes first.
            prg_ram_[address - 0x6000] = value;
            if (layout_.bank_kind == kBankKindNina8k && address >= 0x7FFD) {
                write_mapper_register(address, value);
            }
            return;
        }
        if (address >= 0x8000) {
            write_mapper_register(address, value);
        }
    }

    void reset_cpu(cpu::CpuState& state) noexcept {
        state.variant = cpu::CpuVariant::Ricoh2A03;
        cpu::reset(state, *this);
    }

    [[nodiscard]] StepResult step_cpu_instruction(cpu::CpuState& state) {
        const auto cycles_before = state.cycles;
        bool nmi_serviced = false;
        if (ppu_.nmi_pending()) {
            ppu_.clear_nmi_pending();
            cpu::nmi(state, *this);
            nmi_serviced = true;
        }

        auto cpu_step = cpu::step(state, *this);
        if (pending_dma_cycles_ != 0) {
            state.cycles += pending_dma_cycles_;
            pending_dma_cycles_ = 0;
        }

        const auto cpu_cycles = static_cast<std::uint32_t>(state.cycles - cycles_before);
        const auto ppu_cycles = cpu_cycles * kPpuCyclesPerCpuCycle;
        const auto ppu_step = ppu_.step(ppu_cycles);

        return StepResult{
            cpu_step,
            cpu_cycles,
            ppu_cycles,
            ppu_step.frames_completed,
            nmi_serviced,
            ppu_step.nmi_started,
        };
    }

    [[nodiscard]] FrameResult step_frame(cpu::CpuState& state, std::uint64_t max_instructions) {
        FrameResult result;
        const auto cycles_before = state.cycles;
        while (result.instructions < max_instructions) {
            const auto step = step_cpu_instruction(state);
            ++result.instructions;
            result.frames_completed += step.frames_completed;
            if (result.frames_completed != 0) {
                break;
            }
        }
        result.cpu_cycles = state.cycles - cycles_before;
        return result;
    }

    [[nodiscard]] const RomImage& rom() const noexcept {
        return rom_;
    }

    [[nodiscard]] Ppu& ppu() noexcept {
        return ppu_;
    }

    [[nodiscard]] const Ppu& ppu() const noexcept {
        return ppu_;
    }

    [[nodiscard]] StandardController& controller1() noexcept {
        return controller1_;
    }

    [[nodiscard]] StandardController& controller2() noexcept {
        return controller2_;
    }

    [[nodiscard]] const std::array<std::uint8_t, kCpuRamBytes>& cpu_ram() const noexcept {
        return cpu_ram_;
    }

    [[nodiscard]] std::array<std::uint8_t, kCpuRamBytes>& cpu_ram() noexcept {
        return cpu_ram_;
    }

    // ---- Savestate support ----
    // Capture is deliberately a plain copy of live members rather than a replay of
    // bus writes: the snapshot format records the derived PPU and mapper registers
    // explicitly, and reconstructing them by writing to $2000/$2005/$2006 would
    // re-derive v/t and clear the write latch, which is the opposite of restoring.

    [[nodiscard]] bool has_chr_ram() const noexcept {
        // A cartridge with no CHR ROM has CHR RAM, and that RAM is the only place
        // its pattern data lives - so a state without it restores to a black screen.
        return rom_.metadata.chr_rom_banks == 0 && rom_.metadata.chr_rom_size == 0;
    }

    [[nodiscard]] fcs::StateSnapshot capture_state(const cpu::CpuState& state) const {
        const auto ppu_state = ppu_.save_state();
        fcs::StateSnapshot out;
        out.pc = state.pc;
        out.a = state.a;
        out.x = state.x;
        out.y = state.y;
        out.sp = state.sp;
        out.p = state.p;
        out.cycles = state.cycles;
        out.cpu_ram = cpu_ram_;
        out.prg_ram = prg_ram_;
        out.ppu_ctrl = ppu_state.ctrl;
        out.ppu_mask = ppu_state.mask;
        out.ppu_status = ppu_state.status;
        out.ppu_oam_addr = ppu_state.oam_addr;
        out.ppu_open_bus = ppu_state.open_bus;
        out.ppu_read_buffer = ppu_state.read_buffer;
        out.ppu_x = ppu_state.fine_x;
        out.ppu_w = ppu_state.write_latch ? 1 : 0;
        out.ppu_v = ppu_state.v;
        out.ppu_t = ppu_state.t;
        // The PPU backs 4 KiB but mirrors on decode, so only the first
        // fcs::kNametableRamBytes are canonical - and that is exactly what an FCEUX
        // state carries for a two-screen cart. The rest is never addressed.
        std::copy_n(ppu_state.nametable_ram.begin(), fcs::kNametableRamBytes,
                    out.nametable_ram.begin());
        out.palette_ram = ppu_state.palette_ram;
        out.oam = ppu_state.oam;
        out.has_chr_ram = has_chr_ram();
        if (out.has_chr_ram) {
            out.chr_ram = ppu_state.chr_ram;
        }
        out.prg_bank = prg_bank_;
        out.chr_bank = chr_bank_;
        out.chr_bank_hi = chr_bank_hi_;
        out.mapper_latch = mapper_latch_;
        out.pending_dma_cycles = pending_dma_cycles_;
        out.ppu_scanline = ppu_state.scanline;
        out.ppu_dot = ppu_state.dot;
        out.ppu_frame = ppu_state.frame;
        out.ppu_scroll_x = ppu_state.scroll_x;
        out.ppu_scroll_y = ppu_state.scroll_y;
        return out;
    }

    void apply_state(const fcs::StateSnapshot& state, cpu::CpuState& cpu_state) noexcept {
        cpu_state.pc = state.pc;
        cpu_state.a = state.a;
        cpu_state.x = state.x;
        cpu_state.y = state.y;
        cpu_state.sp = state.sp;
        cpu_state.p = state.p;
        cpu_state.cycles = state.cycles;
        cpu_state.variant = cpu::CpuVariant::Ricoh2A03;
        cpu_ram_ = state.cpu_ram;
        prg_ram_ = state.prg_ram;

        Ppu::Savestate ppu_state;
        ppu_state.ctrl = state.ppu_ctrl;
        ppu_state.mask = state.ppu_mask;
        ppu_state.status = state.ppu_status;
        ppu_state.oam_addr = state.ppu_oam_addr;
        ppu_state.open_bus = state.ppu_open_bus;
        ppu_state.read_buffer = state.ppu_read_buffer;
        ppu_state.fine_x = static_cast<std::uint8_t>(state.ppu_x & 0x07);
        ppu_state.write_latch = (state.ppu_w & 0x01) != 0;
        ppu_state.v = state.ppu_v;
        ppu_state.t = state.ppu_t;
        std::copy_n(state.nametable_ram.begin(), fcs::kNametableRamBytes,
                    ppu_state.nametable_ram.begin());
        ppu_state.palette_ram = state.palette_ram;
        ppu_state.oam = state.oam;
        ppu_state.scanline = state.ppu_scanline;
        ppu_state.dot = state.ppu_dot;
        ppu_state.frame = state.ppu_frame;
        ppu_state.scroll_x = state.ppu_scroll_x;
        ppu_state.scroll_y = state.ppu_scroll_y;
        if (state.has_chr_ram) {
            ppu_state.chr_ram = state.chr_ram;
            ppu_state.chr_banked = true;
        }
        ppu_.apply_state(ppu_state);

        // An OAM DMA writes its 256 bytes immediately and defers only the CPU stall,
        // so this is the whole of the DMA's remaining state.
        pending_dma_cycles_ = state.pending_dma_cycles;

        // Restore the mapper from the raw latch byte. Going back through
        // write_mapper_register would be wrong: its bus-conflict AND exists so that a
        // CPU write only lands when the value matches the ROM, and re-latching an
        // already-decoded bank through it lands on a different bank than the one being
        // restored - on UNROM that silently runs the wrong 16 KB window.
        if (layout_.bank_kind == kBankKindNina8k) {
            // NINA-001 has three separate registers rather than one latch.
            prg_bank_ = state.prg_bank;
            chr_bank_ = state.chr_bank;
            chr_bank_hi_ = state.chr_bank_hi;
            ppu_.set_chr_windows(static_cast<std::uint32_t>(chr_bank_ & 0x0F) << 12,
                                 static_cast<std::uint32_t>(chr_bank_hi_ & 0x0F) << 12);
        } else if (state.mapper_latch != 0) {
            apply_mapper_latch(state.mapper_latch);
        } else if (state.prg_bank != 0) {
            // Only the decoded bank survived, which is what an FCEUX-written state
            // carries. Assign it directly: bus conflicts gate CPU writes, not state,
            // so replaying it through the write path would land on a different bank.
            prg_bank_ = state.prg_bank;
        }
    }

    // Decode a mapper latch exactly as write_mapper_register does, minus bus
    // conflicts. Split out so a restore can reuse the decoding without inheriting
    // the write-time AND.
    void apply_mapper_latch(std::uint8_t value) noexcept {
        if (layout_.window_bytes == 0) {
            return;  // NROM: no registers
        }
        prg_bank_ = static_cast<std::uint8_t>((value >> layout_.bank_shift) & layout_.bank_mask);
        if (rom_.metadata.mapper == kMapperUnrom512) {
            chr_bank_ = static_cast<std::uint8_t>((value >> 5) & layout_.chr_page_mask);
            if (layout_.runtime_mirroring) {
                ppu_.set_nametable_arrangement(
                    (value & 0x80) != 0 ? NametableArrangement::SingleScreenUpper
                                        : NametableArrangement::SingleScreenLower);
            }
        }
    }

private:
    [[nodiscard]] std::uint8_t read_prg_ram(std::uint16_t address) const noexcept {
        if (layout_.bank_kind == kBankKindNina8k && address >= 0x7FFD) {
            // Reading a NINA-001 register returns the register, not the RAM
            // cell underneath it.
            switch (address) {
                case 0x7FFD:
                    return prg_bank_;
                case 0x7FFE:
                    return chr_bank_;
                case 0x7FFF:
                    return chr_bank_hi_;
                default:
                    break;
            }
        }
        return prg_ram_[address - 0x6000];
    }

    void write_mapper_register(std::uint16_t address, std::uint8_t value) noexcept {
        if (layout_.window_bytes == 0) {
            return;  // NROM: no registers, the write goes nowhere
        }
        std::uint8_t latched = value;
        if (layout_.bus_conflicts) {
            latched = static_cast<std::uint8_t>(value & read_prg(address));
        }
        mapper_latch_ = value;

        if (layout_.bank_kind == kBankKindNina8k) {
            switch (address) {
                case 0x7FFD:
                    prg_bank_ = latched;
                    break;
                case 0x7FFE:
                    chr_bank_ = latched;
                    break;
                case 0x7FFF:
                    chr_bank_hi_ = latched;
                    break;
                default:
                    break;
            }
            // Two independent 4 KB CHR ROM windows.
            ppu_.set_chr_windows(static_cast<std::uint32_t>(chr_bank_ & 0x0F) << 12,
                                 static_cast<std::uint32_t>(chr_bank_hi_ & 0x0F) << 12);
            return;
        }

        prg_bank_ = static_cast<std::uint8_t>((latched >> layout_.bank_shift) & layout_.bank_mask);

        if (rom_.metadata.mapper == kMapperUnrom512) {
            // Bits 5-6 pick an 8 KB CHR-RAM page, bit 7 picks the CIRAM page
            // when the header asked for one-screen mirroring.
            chr_bank_ = static_cast<std::uint8_t>((latched >> 5) & layout_.chr_page_mask);
            if (layout_.runtime_mirroring) {
                ppu_.set_nametable_arrangement(
                    (latched & 0x80) != 0 ? NametableArrangement::SingleScreenUpper
                                          : NametableArrangement::SingleScreenLower);
            }
        }
    }

    [[nodiscard]] std::uint8_t read_prg(std::uint16_t address) const noexcept {
        if (layout_.window_bytes == 0) {
            auto index = static_cast<std::size_t>(address - 0x8000);
            if (rom_.prg_rom.size() == 16 * 1024) {
                index &= 0x3FFF;
            } else {
                index &= 0x7FFF;
            }
            return rom_.prg_rom[index];
        }
        std::size_t index;
        const std::uint32_t window_start =
            layout_.window_at_top ? (0x10000u - layout_.window_bytes) : 0x8000u;
        const std::uint32_t offset = static_cast<std::uint32_t>(address) - window_start;
        if (offset <= layout_.window_bytes - 1) {
            index = (static_cast<std::size_t>(prg_bank_ & layout_.bank_mask) * layout_.window_bytes) +
                    offset;
        } else {
            // The fixed window is the last part of the image, except on a board
            // that fixes bank 0 at the bottom (mapper 180).
            const std::size_t fixed_base =
                layout_.window_at_top ? 0 : (rom_.prg_rom.size() - layout_.fixed_bytes);
            index = fixed_base + (address & (layout_.fixed_bytes - 1));
        }
        return rom_.prg_rom[index % rom_.prg_rom.size()];
    }

    void run_oam_dma(std::uint8_t page) noexcept {
        const auto base = static_cast<std::uint16_t>(page << 8);
        for (std::uint16_t i = 0; i < 256; ++i) {
            ppu_.write_oam_dma(read(static_cast<std::uint16_t>(base + i)));
        }
        pending_dma_cycles_ += 513;
    }

    RomImage rom_;
    MapperLayout layout_;
    std::uint8_t prg_bank_ = 0;
    std::uint8_t chr_bank_ = 0;
    std::uint8_t chr_bank_hi_ = 0;
    // Last value written to a mapper latch, before bus conflicts are applied. Kept so
    // a savestore can put the mapper back exactly.
    std::uint8_t mapper_latch_ = 0;
    std::array<std::uint8_t, kCpuRamBytes> cpu_ram_{};
    std::array<std::uint8_t, kPrgRamBytes> prg_ram_{};
    std::array<std::uint8_t, kApuIoBytes> apu_io_{};
    Ppu ppu_;
    StandardController controller1_;
    StandardController controller2_;
    std::uint32_t pending_dma_cycles_ = 0;
};

}  // namespace nesle
