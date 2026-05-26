"""Ability enum — names persisted in the save file and referenced in level JSON.

Add a new member here to introduce a new ability. The string value is the
canonical name used on disk, in chunk parameters, and in code paths.
"""

from __future__ import annotations

from enum import StrEnum


class Ability(StrEnum):
    DOUBLE_JUMP = "double_jump"
    # Future: WALL_JUMP = "wall_jump", GROUND_POUND = "ground_pound"
