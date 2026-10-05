"""NeSLE Python API."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("nesle")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.0.1"

from .actions import (
    COMPLEX_MOVEMENT,
    COMPLEX_MOVEMENT_MASKS,
    MARIO_MOVEMENT,
    MARIO_MOVEMENT_MASKS,
    RIGHT_ONLY,
    RIGHT_ONLY_MASKS,
    SIMPLE_MOVEMENT,
    SIMPLE_MOVEMENT_MASKS,
    SIMPLE_MOVEMENT_WITH_START,
    SIMPLE_MOVEMENT_WITH_START_MASKS,
    Button,
    encode_action,
)
from .rom import INESRom, parse_ines
from .savestate import load as load_state
from .savestate import save as save_state
from .smb import MarioRamState, RewardComponents, compute_reward, read_ram

__all__ = [
    "__version__",
    "Button",
    "COMPLEX_MOVEMENT",
    "COMPLEX_MOVEMENT_MASKS",
    "INESRom",
    "MarioRamState",
    "MARIO_MOVEMENT",
    "MARIO_MOVEMENT_MASKS",
    "RIGHT_ONLY",
    "RIGHT_ONLY_MASKS",
    "RewardComponents",
    "SIMPLE_MOVEMENT",
    "SIMPLE_MOVEMENT_MASKS",
    "SIMPLE_MOVEMENT_WITH_START",
    "SIMPLE_MOVEMENT_WITH_START_MASKS",
    "Button",
    "compute_reward",
    "encode_action",
    "load_state",
    "make",
    "make_vec",
    "parse_ines",
    "read_ram",
    "save_state",
]



def make(*args, **kwargs):
    from .env import NesleEnv

    return NesleEnv(*args, **kwargs)


def make_vec(*args, **kwargs):
    from .env import NesleVecEnv

    return NesleVecEnv(*args, **kwargs)
