"""Level loader — reads a JSON file and instantiates chunks into a World."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

# Importing the chunks package registers every chunk type
from . import chunks  # noqa: F401
from .chunks.base import CHUNK_REGISTRY


@dataclass(frozen=True)
class LevelMeta:
    name: str
    spawn: tuple[float, float]
    background: tuple[int, int, int]
    ground: tuple[int, int, int]
    total_width: float


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    s = hex_str.lstrip("#")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def load_level(source: Union[str, Path, dict], world) -> LevelMeta:
    if isinstance(source, dict):
        data = source
    else:
        data = json.loads(Path(source).read_text())
    chunks_list = data["chunks"]

    x = 0.0
    for entry in chunks_list:
        type_name = entry["type"]
        kwargs = {k: v for k, v in entry.items() if k != "type"}
        if type_name not in CHUNK_REGISTRY:
            available = ", ".join(sorted(CHUNK_REGISTRY))
            raise ValueError(f"Unknown chunk type {type_name!r}. Available: {available}")
        chunk = CHUNK_REGISTRY[type_name](**kwargs)
        width = chunk.build(world, x_offset=x)
        x += width

    spawn = tuple(data["spawn"])
    return LevelMeta(
        name=data["name"],
        spawn=spawn,
        background=_hex_to_rgb(data["background"]),
        ground=_hex_to_rgb(data["ground"]),
        total_width=x,
    )
