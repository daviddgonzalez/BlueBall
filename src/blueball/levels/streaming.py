"""Infinite Run terrain streaming — pygame-free.

`TerrainStream` is the chunk-streaming state machine for Infinite Run: it
lazily materializes chunks from a deterministic `ChunkSampler` ahead of the
ball and culls chunks that fall far enough behind. It operates purely on a
`World` (pymunk) and imports nothing from pygame or the scene layer, so the
headless GA trainer can stream the *same* terrain a human sees for a given
seed. `PlayScene` delegates to it.

Extracted from PlayScene so the live game and the trainer share one
implementation — otherwise the two would drift and the agent would train on
different terrain than it plays, defeating the set-seed reference contract.
"""

from __future__ import annotations

from .chunks.base import CHUNK_REGISTRY
# Importing the chunks package registers every chunk type in CHUNK_REGISTRY.
from . import chunks  # noqa: F401
from .chunks.flat import Flat, GROUND_Y
from .sampler import ChunkSampler

# Streaming distances for Infinite Run (px in world coords).
LOAD_AHEAD = 2000.0
LOAD_BEHIND = 800.0
# How many chunks to materialize at level start, before the player has moved.
INITIAL_BUILD_CHUNKS = 6
# Max height the running ground may rise above the baseline before stairs are
# biased back down (keeps elevation in a sane band, never below the baseline).
MAX_GROUND_ELEV = 280.0


class TerrainStream:
    """Streams Infinite Run chunks into *world* off a seeded sampler.

    Construction lays a guaranteed Flat at x=0 (so a floating first chunk can't
    drop the player into the void) and materializes `initial_chunks` chunks.
    Call `maintain(player_x)` once per tick to extend ahead and cull behind.
    """

    def __init__(
        self,
        world,
        sampler_seed: int,
        *,
        load_ahead: float = LOAD_AHEAD,
        load_behind: float = LOAD_BEHIND,
        initial_chunks: int = INITIAL_BUILD_CHUNKS,
        max_ground_elev: float = MAX_GROUND_ELEV,
        abilities=frozenset(),
    ) -> None:
        self.world = world
        self.load_ahead = load_ahead
        self.load_behind = load_behind
        self.max_ground_elev = max_ground_elev

        # Infinite Run has no checkpoints — death re-randomizes the whole run,
        # so a mid-run respawn anchor would be meaningless. `abilities` gates the
        # double-jump-only chunks: a run granted DOUBLE_JUMP surfaces them, a
        # plain single-jump run never does.
        self.sampler = ChunkSampler(
            seed=int(sampler_seed), emit_checkpoints=False, abilities=abilities
        )
        self._chunk_iter = iter(self.sampler)
        self.built_chunks: list[dict] = []
        self.build_x: float = 0.0
        # Running ground height at the current seam; carried across chunks so
        # surfaces connect (stairs raise/lower it, everything else keeps it).
        self.base_y: float = GROUND_Y

        # Guarantee a ground segment at the spawn point.
        self.materialize_chunk(Flat(width_tiles=4))
        for _ in range(initial_chunks):
            if not self.build_next_chunk():
                break

    def materialize_chunk(self, chunk) -> float:
        """Build *chunk* at the current cursor and append a tracking record to
        ``built_chunks``. Returns the chunk's width.

        Threads the running ground height: the chunk's left edge meets the
        current seam, and the seam advances by the chunk's net elevation change
        so the next chunk connects.
        """
        entry_dy = getattr(chunk, "entry_dy", 0.0)
        exit_dy = getattr(chunk, "exit_dy", 0.0)
        chunk_base = self.base_y - entry_dy

        pre_shapes = set(self.world.space.shapes)
        pre_bodies = set(self.world.space.bodies)
        pre_entities = set(self.world.entities)
        pre_constraints = set(self.world.space.constraints)

        width = chunk.build(self.world, x_offset=self.build_x, base_y=chunk_base)
        self.base_y = chunk_base + exit_dy

        new_shapes = set(self.world.space.shapes) - pre_shapes
        new_bodies = set(self.world.space.bodies) - pre_bodies
        new_entities = set(self.world.entities) - pre_entities
        new_constraints = set(self.world.space.constraints) - pre_constraints

        self.built_chunks.append({
            "x_start": self.build_x,
            "x_end": self.build_x + width,
            "shapes": new_shapes,
            "bodies": new_bodies,
            "entities": new_entities,
            "constraints": new_constraints,
        })
        self.build_x += width
        return width

    def build_next_chunk(self) -> bool:
        """Pop the next chunk dict from the sampler and materialize it,
        tracking exactly what got added so we can remove it later. Returns
        False if the sampler is exhausted or yields an unknown chunk type.
        """
        chunk_dict = next(self._chunk_iter, None)
        if chunk_dict is None:
            return False
        chunk_dict = self._bias_stairs(chunk_dict)
        type_name = chunk_dict["type"]
        kwargs = {k: v for k, v in chunk_dict.items() if k != "type"}
        chunk_cls = CHUNK_REGISTRY.get(type_name)
        if chunk_cls is None:
            return False
        self.materialize_chunk(chunk_cls(**kwargs))
        return True

    def _bias_stairs(self, chunk_dict: dict) -> dict:
        """Keep the running ground height in [GROUND_Y - max_ground_elev,
        GROUND_Y] by flipping a staircase that would climb too high or descend
        below the baseline. Non-stairs chunks pass through unchanged."""
        t = chunk_dict["type"]
        if t not in ("stairs_up", "stairs_down"):
            return chunk_dict
        rise = chunk_dict.get("steps", 3) * chunk_dict.get("step_height", 32)
        if t == "stairs_up" and self.base_y - rise < GROUND_Y - self.max_ground_elev:
            return {**chunk_dict, "type": "stairs_down"}
        if t == "stairs_down" and self.base_y + rise > GROUND_Y:
            return {**chunk_dict, "type": "stairs_up"}
        return chunk_dict

    def maintain(self, player_x: float) -> None:
        """Per-tick: build more chunks ahead, cull old chunks behind."""
        # Build ahead — keep at least load_ahead px of built world to the right.
        while self.build_x < player_x + self.load_ahead:
            if not self.build_next_chunk():
                break
        # Cull behind — drop chunks fully behind the player by load_behind.
        cutoff = player_x - self.load_behind
        while self.built_chunks and self.built_chunks[0]["x_end"] < cutoff:
            info = self.built_chunks.pop(0)
            for shape in info["shapes"]:
                if shape in self.world.space.shapes:
                    self.world.space.remove(shape)
                # Bare chunk segments were never indexed (pop is a no-op);
                # entity-owned shapes (Lava, PushableBox, ...) must be purged
                # here since the cull bypasses Entity._remove_from_space.
                self.world._shape_to_entity.pop(shape, None)
            for constraint in info["constraints"]:
                if constraint in self.world.space.constraints:
                    self.world.space.remove(constraint)
            for body in info["bodies"]:
                # Skip the shared static body — chunks add segments to it but
                # never the body itself.
                if body is self.world.space.static_body:
                    continue
                if body in self.world.space.bodies:
                    self.world.space.remove(body)
            for entity in info["entities"]:
                if entity in self.world.entities:
                    self.world.entities.remove(entity)
