"""Level loader — reads a JSON file and instantiates chunks into a World."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

# Importing the chunks package registers every chunk type
from . import chunks  # noqa: F401
from ..entities.lava import Lava
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

    # Optional level-feature: rising lava.
    if "lava" in data:
        lava_cfg = data["lava"]
        start_y = float(lava_cfg.get("start_y", 1100))
        rise_speed = float(lava_cfg.get("rise_speed", 20))
        overflow = float(lava_cfg.get("width_overflow", 200))
        lava_width = x + 2 * overflow
        world.add_entity(Lava(
            world,
            position=(x / 2, start_y),
            width=lava_width,
            rise_speed=rise_speed,
        ))

    # Optional level-feature: periodic projectile cannons.
    if "cannons" in data:
        from ..entities.cannon import Cannon
        from .chunks.flat import GROUND_Y
        for c in data["cannons"]:
            imin = c.get("interval_min_s")
            imax = c.get("interval_max_s")
            world.add_entity(Cannon(
                world,
                position=(float(c["x"]), GROUND_Y - float(c.get("y_offset", 0))),
                direction=c.get("dir", "right"),
                interval_s=float(c.get("interval_s", 2.0)),
                interval_min_s=float(imin) if imin is not None else None,
                interval_max_s=float(imax) if imax is not None else None,
                speed=float(c.get("speed", 220.0)),
                pulse_period_s=float(c.get("pulse_period_s", 0.6)),
                max_travel=float(c.get("max_travel", 200.0)),
                projectile_radius=int(c.get("radius", 10)),
                phase_s=float(c.get("phase_s", 0.0)),
            ))

    spawn = tuple(data["spawn"])
    return LevelMeta(
        name=data["name"],
        spawn=spawn,
        background=_hex_to_rgb(data["background"]),
        ground=_hex_to_rgb(data["ground"]),
        total_width=x,
    )
