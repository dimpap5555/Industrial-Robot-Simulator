"""Predefined robot configurations and ready-to-run demos."""

from .rx90l import create_robot, rx90l_config
from .rx90l_demo import OCPResult, Rx90LDemo

__all__ = [
    "create_robot",
    "rx90l_config",
    "Rx90LDemo",
    "OCPResult",
]