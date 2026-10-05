#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <string>
#include <vector>

#include "nesle/console.hpp"
#include "nesle/cpu.hpp"
#include "nesle/fcs.hpp"
#include "nesle/rom.hpp"
#include "nesle/smb.hpp"

namespace py = pybind11;

namespace {

std::vector<std::uint8_t> bytes_to_vector(const py::bytes& bytes) {
    const std::string raw = bytes;
    return {raw.begin(), raw.end()};
}

py::dict rom_metadata_to_dict(const nesle::RomMetadata& metadata) {
    py::dict out;
    out["prg_rom_banks"] = metadata.prg_rom_banks;
    out["chr_rom_banks"] = metadata.chr_rom_banks;
    out["mapper"] = metadata.mapper;
    out["submapper"] = metadata.submapper;
    out["has_trainer"] = metadata.has_trainer;
    out["has_battery"] = metadata.has_battery;
    out["is_nes2"] = metadata.is_nes2;
    out["nametable_arrangement"] = nesle::to_string(metadata.nametable_arrangement);
    out["prg_rom_size"] = metadata.prg_rom_size;
    out["chr_rom_size"] = metadata.chr_rom_size;
    out["is_nrom"] = metadata.is_nrom();
    out["is_uxrom"] = metadata.is_uxrom();
    out["is_supported"] = nesle::describe_mapper(metadata).supported;
    out["is_supported_mario_target"] = nesle::is_supported_mario_target(metadata);
    return out;
}

py::dict mario_state_to_dict(const nesle::smb::MarioRamState& state) {
    py::dict out;
    out["x_pos"] = state.x_pos;
    out["y_pos"] = state.y_pos;
    out["time"] = state.time;
    out["coins"] = state.coins;
    out["score"] = state.score;
    out["life"] = state.lives;
    out["world"] = state.world;
    out["stage"] = state.stage;
    out["area"] = state.area;
    out["status"] = nesle::smb::status_name(state.status_code);
    out["status_code"] = state.status_code;
    out["player_state"] = state.player_state;
    out["flag_get"] = state.flag_get;
    out["is_dying"] = state.is_dying;
    out["is_dead"] = state.is_dead;
    out["is_game_over"] = state.is_game_over;
    return out;
}

class NativeConsoleBinding {
public:
    explicit NativeConsoleBinding(const py::bytes& bytes)
        : console_(nesle::parse_ines(bytes_to_vector(bytes))) {
        reset();
    }

    void reset() {
        state_ = nesle::cpu::CpuState{};
        console_.reset_cpu(state_);
    }

    py::dict step(std::uint8_t action_mask,
                  std::uint32_t frameskip,
                  std::uint64_t max_instructions_per_frame) {
        console_.controller1().set_buttons(action_mask);
        std::uint64_t instructions = 0;
        std::uint64_t cpu_cycles = 0;
        std::uint32_t frames_completed = 0;
        for (std::uint32_t frame = 0; frame < frameskip; ++frame) {
            const auto result = console_.step_frame(state_, max_instructions_per_frame);
            instructions += result.instructions;
            cpu_cycles += result.cpu_cycles;
            frames_completed += result.frames_completed;
        }

        py::dict out;
        out["instructions"] = instructions;
        out["cpu_cycles"] = cpu_cycles;
        out["frames_completed"] = frames_completed;
        out["pc"] = state_.pc;
        return out;
    }

    py::bytes ram() const {
        const auto& ram = console_.cpu_ram();
        return py::bytes(reinterpret_cast<const char*>(ram.data()), ram.size());
    }

    py::bytes frame() const {
        const auto frame = console_.ppu().render_rgb_frame();
        return py::bytes(reinterpret_cast<const char*>(frame.data()), frame.size());
    }

    // Serialize the live console as an FCSX image, the format FCEUX 2.6 reads.
    //
    // An FCSX state has nowhere to record where the PPU was inside the frame, so this
    // is not a bit-exact checkpoint of a run: reloading rewinds the PPU to the top of
    // the frame, and measured over 200 frames that diverges on most carts. Pass
    // require_frame_boundary=True to refuse instead, which is what you want if the
    // file is meant to resume a specific run rather than seed a curriculum reset.
    //
    // That second use is why the default is off. Every FCEUX state this project
    // already trains from has the same property - 563 of them, loaded through the
    // snapshot reset path - and a handful of dots of PPU phase does not matter when
    // the game re-establishes frame alignment within a frame or two of running on.
    py::bytes save_state(bool require_frame_boundary = false) const {
        const auto snapshot = console_.capture_state(state_);
        if (require_frame_boundary && !nesle::fcs::is_frame_boundary(snapshot)) {
            throw std::runtime_error(
                "save_state: the PPU is mid-frame (scanline " +
                std::to_string(snapshot.ppu_scanline) + ", dot " +
                std::to_string(snapshot.ppu_dot) +
                "). An FCSX state cannot store that, so reloading this file would "
                "rewind the PPU and diverge.");
        }
        const auto image = nesle::fcs::serialize_fcsx(snapshot);
        return py::bytes(reinterpret_cast<const char*>(image.data()), image.size());
    }

    // Restore from either FCEUX format. Throws if the file is not a state, which
    // pybind surfaces as a ValueError.
    void load_state(const py::bytes& bytes) {
        const auto raw = bytes_to_vector(bytes);
        console_.apply_state(nesle::fcs::parse(raw), state_);
    }

    // Exposed so callers can check what they are about to write, and so tests can
    // assert round-trip equality field by field instead of comparing opaque bytes.
    py::dict state_summary() const {
        const auto s = console_.capture_state(state_);
        py::dict out;
        out["pc"] = s.pc;
        out["a"] = s.a;
        out["x"] = s.x;
        out["y"] = s.y;
        out["sp"] = s.sp;
        out["p"] = s.p;
        out["cycles"] = s.cycles;
        out["ppu_v"] = s.ppu_v;
        out["ppu_t"] = s.ppu_t;
        out["ppu_x"] = s.ppu_x;
        out["ppu_w"] = s.ppu_w;
        out["ppu_ctrl"] = s.ppu_ctrl;
        out["ppu_mask"] = s.ppu_mask;
        out["ppu_status"] = s.ppu_status;
        out["ppu_oam_addr"] = s.ppu_oam_addr;
        out["ppu_open_bus"] = s.ppu_open_bus;
        out["ppu_read_buffer"] = s.ppu_read_buffer;
        out["has_chr_ram"] = s.has_chr_ram;
        out["prg_bank"] = s.prg_bank;
        out["chr_bank"] = s.chr_bank;
        out["chr_bank_hi"] = s.chr_bank_hi;
        out["ppu_scanline"] = s.ppu_scanline;
        out["ppu_dot"] = s.ppu_dot;
        out["ppu_frame"] = s.ppu_frame;
        out["ppu_scroll_x"] = s.ppu_scroll_x;
        out["ppu_scroll_y"] = s.ppu_scroll_y;
        out["at_frame_boundary"] = nesle::fcs::is_frame_boundary(s);
        out["cpu_ram"] = py::bytes(reinterpret_cast<const char*>(s.cpu_ram.data()),
                                   s.cpu_ram.size());
        out["prg_ram"] = py::bytes(reinterpret_cast<const char*>(s.prg_ram.data()),
                                   s.prg_ram.size());
        out["nametable_ram"] = py::bytes(reinterpret_cast<const char*>(s.nametable_ram.data()),
                                         s.nametable_ram.size());
        out["palette_ram"] = py::bytes(reinterpret_cast<const char*>(s.palette_ram.data()),
                                       s.palette_ram.size());
        out["oam"] = py::bytes(reinterpret_cast<const char*>(s.oam.data()), s.oam.size());
        if (s.has_chr_ram) {
            out["chr_ram"] = py::bytes(reinterpret_cast<const char*>(s.chr_ram.data()),
                                       s.chr_ram.size());
        }
        return out;
    }

    // True when the PPU sits exactly on a frame boundary, which is the only place a
    // save is lossless: an FCEUX state has nowhere to put the mid-frame scanline and
    // dot counters, so saving part-way through a frame would silently drop them.
    bool at_frame_boundary() const {
        return nesle::fcs::is_frame_boundary(console_.capture_state(state_));
    }

private:
    nesle::Console console_;
    nesle::cpu::CpuState state_;
};

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "Native NeSLE core helpers";

    m.def("parse_ines_metadata", [](const py::bytes& bytes) {
        const auto data = bytes_to_vector(bytes);
        return rom_metadata_to_dict(nesle::parse_ines(data).metadata);
    });

    m.def("read_mario_ram", [](const py::bytes& bytes) {
        const auto data = bytes_to_vector(bytes);
        return mario_state_to_dict(nesle::smb::read_ram(data));
    });

    py::class_<NativeConsoleBinding>(m, "NativeConsole")
        .def(py::init<const py::bytes&>())
        .def("reset", &NativeConsoleBinding::reset)
        .def("step", &NativeConsoleBinding::step)
        .def("ram", &NativeConsoleBinding::ram)
        .def("frame", &NativeConsoleBinding::frame)
        .def("save_state", &NativeConsoleBinding::save_state,
             py::arg("require_frame_boundary") = false,
             "Serialize the console as FCSX. Not a bit-exact checkpoint: the format "
             "cannot store the PPU's position within a frame, so reloading rewinds "
             "it. Pass require_frame_boundary=True to refuse a mid-frame save.")
        .def("load_state", &NativeConsoleBinding::load_state)
        .def("state_summary", &NativeConsoleBinding::state_summary)
        .def("at_frame_boundary", &NativeConsoleBinding::at_frame_boundary);
}
