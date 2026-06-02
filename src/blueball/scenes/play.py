"""PlayScene — gameplay loop. Accepts either a level path or in-memory data,
and (for Infinite Run) lazily streams chunks from a ChunkSampler instead of
materializing all of them at level load.
"""

from __future__ import annotations

import random
from pathlib import Path

import pygame

from .. import config, save
from ..abilities import Ability
from ..agent import HumanAgent
from ..camera import FollowCamera
from ..collision import register as register_collisions
from ..entities.player import Player
from ..levels.chunks.base import CHUNK_REGISTRY
# Importing the chunks package registers every chunk type
from ..levels import chunks  # noqa: F401
from ..levels.loader import load_level
from ..levels.sampler import ChunkSampler
from ..render.renderer import Renderer
from ..world import World
from .base import Scene


# Streaming distances for Infinite Run (px in world coords).
_LOAD_AHEAD = 2000.0
_LOAD_BEHIND = 800.0
# How many chunks to materialize at level start, before the player has moved.
_INITIAL_BUILD_CHUNKS = 6
# Max height the running ground may rise above the baseline before stairs are
# biased back down (keeps elevation in a sane band, never below the baseline).
_MAX_GROUND_ELEV = 280.0


class PlayScene(Scene):
    def __init__(
        self,
        screen: pygame.Surface,
        level_path: Path | None = None,
        level_data: dict | None = None,
        sampler_seed: int | None = None,
    ) -> None:
        if (level_path is None) == (level_data is None):
            raise ValueError("PlayScene requires exactly one of level_path or level_data")
        self.screen = screen
        self.level_path = level_path
        self.level_data = level_data
        self.sampler_seed = sampler_seed
        # Streaming is opted into by providing sampler_seed. Regular hand-built
        # levels still load eagerly via load_level().
        self._streaming: bool = sampler_seed is not None
        self.camera = FollowCamera(screen.get_width(), screen.get_height())
        self.renderer = Renderer(screen, self.camera)
        self._last_respawn_xy: tuple[float, float] | None = None
        self._exit_to_menu: bool = False
        # Infinite Run score = 10 * furthest x reached this run; best persists
        # per the local user's save across runs.
        self._best_score: int = save.get_best_score() if self._streaming else 0
        self._run_max_x: float = 0.0
        self._score: int = 0
        self._reset()

    def _reset(self) -> None:
        self.world = World()
        register_collisions(self.world.space, world_ref=self.world)

        if self._streaming:
            self._init_streaming_level()
        else:
            source = self.level_path if self.level_path is not None else self.level_data
            self.level_meta = load_level(source, self.world)

        unlocked_names = save.load()
        valid_names = {a.value for a in Ability}
        unlocked = {Ability(name) for name in unlocked_names if name in valid_names}
        self.player = Player(
            agent=HumanAgent(),
            spawn_xy=tuple(self.level_meta.spawn),
            abilities=unlocked,
        )
        if self._last_respawn_xy is not None:
            self.player.body.position = self._last_respawn_xy
        self.world.add_entity(self.player)
        self.camera.position = (self.player.body.position.x, self.player.body.position.y)

    def _init_streaming_level(self) -> None:
        """Set up the per-tick chunk pipeline for Infinite Run.

        - Construct a deterministic sampler keyed off sampler_seed.
        - Track the build cursor and a list of materialized chunks with the
          pymunk shapes/bodies/entities they added, so we can remove them when
          the player has moved past _LOAD_BEHIND.
        """
        from ..levels.loader import LevelMeta, _hex_to_rgb

        meta_dict = self.level_data or {}
        background = _hex_to_rgb(meta_dict.get("background", "#202028"))
        ground = _hex_to_rgb(meta_dict.get("ground", "#666c70"))
        spawn = tuple(meta_dict.get("spawn", [80, 540]))
        # total_width is unknowable for a streaming level; report 0.0.
        self.level_meta = LevelMeta(
            name=meta_dict.get("name", f"Infinite Run (seed={self.sampler_seed})"),
            spawn=spawn,
            background=background,
            ground=ground,
            total_width=0.0,
        )

        # Infinite Run has no checkpoints — death re-randomizes the whole run
        # (see update()), so a mid-run respawn anchor would be meaningless.
        self._sampler = ChunkSampler(seed=int(self.sampler_seed), emit_checkpoints=False)
        self._chunk_iter = iter(self._sampler)
        self._built_chunks: list[dict] = []
        self._build_x: float = 0.0
        # Running ground height at the current seam; carried across chunks so
        # surfaces connect (stairs raise/lower it, everything else keeps it).
        from ..levels.chunks.flat import GROUND_Y
        self._base_y: float = GROUND_Y
        # Guarantee a ground segment at the spawn point. The sampler can pick
        # a chunk like `platform` (floating, no ground) as its first emission,
        # which would drop the player into the void on frame 1.
        from ..levels.chunks.flat import Flat
        self._materialize_chunk(Flat(width_tiles=4))
        for _ in range(_INITIAL_BUILD_CHUNKS):
            if not self._build_next_chunk():
                break

    def _materialize_chunk(self, chunk) -> float:
        """Build *chunk* at the current cursor and append a tracking record
        to ``_built_chunks``. Returns the chunk's width.

        Threads the running ground height: the chunk is placed so its left edge
        meets the current seam, and the seam advances by the chunk's net
        elevation change so the next chunk connects.
        """
        entry_dy = getattr(chunk, "entry_dy", 0.0)
        exit_dy = getattr(chunk, "exit_dy", 0.0)
        chunk_base = self._base_y - entry_dy

        pre_shapes = set(self.world.space.shapes)
        pre_bodies = set(self.world.space.bodies)
        pre_entities = set(self.world.entities)
        pre_constraints = set(self.world.space.constraints)

        width = chunk.build(self.world, x_offset=self._build_x, base_y=chunk_base)
        self._base_y = chunk_base + exit_dy

        new_shapes = set(self.world.space.shapes) - pre_shapes
        new_bodies = set(self.world.space.bodies) - pre_bodies
        new_entities = set(self.world.entities) - pre_entities
        new_constraints = set(self.world.space.constraints) - pre_constraints

        self._built_chunks.append({
            "x_start": self._build_x,
            "x_end": self._build_x + width,
            "shapes": new_shapes,
            "bodies": new_bodies,
            "entities": new_entities,
            "constraints": new_constraints,
        })
        self._build_x += width
        return width

    def _build_next_chunk(self) -> bool:
        """Pop the next chunk dict from the sampler and materialize it,
        tracking exactly what got added to the world so we can remove it
        later. Returns False if the sampler is exhausted.
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
        self._materialize_chunk(chunk_cls(**kwargs))
        return True

    def _bias_stairs(self, chunk_dict: dict) -> dict:
        """Keep the running ground height in [GROUND_Y - _MAX_GROUND_ELEV,
        GROUND_Y] by flipping a staircase that would climb too high or descend
        below the baseline. Non-stairs chunks pass through unchanged."""
        from ..levels.chunks.flat import GROUND_Y
        t = chunk_dict["type"]
        if t not in ("stairs_up", "stairs_down"):
            return chunk_dict
        rise = chunk_dict.get("steps", 3) * chunk_dict.get("step_height", 32)
        if t == "stairs_up" and self._base_y - rise < GROUND_Y - _MAX_GROUND_ELEV:
            return {**chunk_dict, "type": "stairs_down"}
        if t == "stairs_down" and self._base_y + rise > GROUND_Y:
            return {**chunk_dict, "type": "stairs_up"}
        return chunk_dict

    def _maintain_streaming(self, player_x: float) -> None:
        """Per-tick: build more chunks ahead, cull old chunks behind."""
        # Build ahead — keep at least LOAD_AHEAD px of built world to the right.
        while self._build_x < player_x + _LOAD_AHEAD:
            if not self._build_next_chunk():
                break
        # Cull behind — drop chunks fully behind the player by LOAD_BEHIND.
        cutoff = player_x - _LOAD_BEHIND
        while self._built_chunks and self._built_chunks[0]["x_end"] < cutoff:
            info = self._built_chunks.pop(0)
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

    def handle_events(self, events):
        if self._exit_to_menu:
            from .menu import MenuScene
            return MenuScene(self.screen)
        for event in events:
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                from .menu import MenuScene
                return MenuScene(self.screen)
        return self

    def update(self, frame_dt: float) -> None:
        self.renderer.begin_frame(self.world)
        if self._streaming:
            self._maintain_streaming(self.player.body.position.x)
        self.world.step(frame_dt)
        if self._streaming:
            self._run_max_x = max(self._run_max_x, self.player.body.position.x)
            self._score = int(10 * self._run_max_x)
        if self.player.dead:
            if self._streaming:
                # Bank the run's score, then re-randomize the run on death
                # instead of replaying the same deterministic layout; no
                # checkpoint respawn in Infinite Run.
                if self._score > self._best_score:
                    self._best_score = self._score
                    save.set_best_score(self._score)
                self._run_max_x = 0.0
                self._score = 0
                self.sampler_seed = random.randrange(1 << 32)
                self._last_respawn_xy = None
            else:
                self._last_respawn_xy = self.player.respawn_xy
            self._reset()
            return
        if self.world.level_complete:
            for ability in self.player.abilities:
                save.add_ability(ability.value)
            self._last_respawn_xy = None
            self._exit_to_menu = True
            return
        self.camera.update(
            target=(self.player.body.position.x, self.player.body.position.y),
            dt=frame_dt,
        )

    def draw(self) -> None:
        self.renderer.draw_background(self.level_meta.background)
        self.renderer.draw_static_segments(self.world.space, color=self.level_meta.ground)
        alpha = self.world.alpha
        for entity in self.world.entities:
            entity.draw(self.renderer, alpha)
        if self._streaming:
            self.renderer.draw_score(self._score, max(self._best_score, self._score))
        pygame.display.flip()
