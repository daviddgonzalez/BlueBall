"""Playback — load a saved FTNN genome and watch it play, with a HUD that
shows the *same* fitness the headless trainer would score.

Two pieces:

* ``PlaybackSim`` — the display-free simulation core. It reconstructs exactly
  the world/loop the trainer's ``evaluate*`` functions use (static level,
  Infinite Run stream, or Completion-Gym stream), so its live ``fitness`` is
  faithful to a headless evaluation of the same genome/seed/steps. It needs no
  window or rendering — fully deterministic and CI-testable headless.
* ``PlaybackScene`` — a thin pygame ``Scene`` that wraps a ``PlaybackSim``,
  drives it at a fixed substep cadence, and draws the world plus a HUD overlay.

`watch-best` (see cli.py) builds a sim from CLI args, wraps it in a scene, and
runs the standard pygame loop.
"""

from __future__ import annotations

import bisect
from pathlib import Path

import numpy as np

from .. import config
from ..abilities import Ability
from ..agent import FTNNAgent
from ..ai.fitness import FitnessInputs, fitness
from ..collision import register as register_collisions
from ..entities.player import Player
from ..levels.loader import load_level
from ..world import World
from .base import Scene

GENOME_FILENAME = "final_best.npy"


def resolve_genome_path(arg: str | Path) -> Path:
    """A run-dir argument resolves to its ``final_best.npy``; a file path (a
    specific ``.npy``) is returned unchanged."""
    p = Path(arg)
    if p.is_dir():
        return p / GENOME_FILENAME
    return p


def load_genome(arg: str | Path) -> np.ndarray:
    """Resolve `arg` to a genome file and load it as a float array."""
    return np.load(resolve_genome_path(arg))


def _abilities_set(abilities) -> set[Ability]:
    """Coerce a name/enum iterable into a set of Ability members."""
    return {a if isinstance(a, Ability) else Ability(a) for a in abilities}


def _parse_abilities(spec) -> tuple[str, ...]:
    """Parse an --abilities value: a comma-separated string or an iterable of
    names. Empty string -> no abilities."""
    if isinstance(spec, str):
        return tuple(a.strip() for a in spec.split(",") if a.strip())
    return tuple(str(a) for a in spec)


# Default static target when no mode flag is given.
DEFAULT_LEVEL = "tutorial_hill"


def build_playback_sim(
    target: str | Path,
    *,
    level: str | None = None,
    infinite: int | None = None,
    gym: int | None = None,
    abilities=None,
    max_steps: int | None = None,
    world_seed: int = config.DEFAULT_SEED,
) -> "PlaybackSim":
    """Resolve CLI-style args into a ready PlaybackSim. Picks the mode from the
    mode flags (default: a static level) and applies the per-mode default
    ability set unless `abilities` is given explicitly.

    The per-mode ability defaults mirror the headless evaluators so the HUD is
    faithful out of the box: static/infinite grant none; gym grants double_jump
    (the gym is trained with it). Pass `abilities` to override — e.g. to replay
    a double-jump genome on a static level.
    """
    genome = load_genome(target)
    explicit = abilities is not None
    parsed = _parse_abilities(abilities) if explicit else ()

    if infinite is not None:
        return PlaybackSim(genome, mode="infinite", seed=int(infinite),
                           world_seed=world_seed, max_steps=max_steps,
                           abilities=parsed)
    if gym is not None:
        default = () if explicit else (Ability.DOUBLE_JUMP.value,)
        return PlaybackSim(genome, mode="gym", seed=int(gym),
                           world_seed=world_seed, max_steps=max_steps,
                           abilities=parsed if explicit else default)

    from ..ai.episodes import resolve_level_paths
    level_path = resolve_level_paths([level or DEFAULT_LEVEL])[0]
    return PlaybackSim(genome, mode="static", level_path=level_path,
                       world_seed=world_seed, max_steps=max_steps,
                       abilities=parsed)


class PlaybackSim:
    """Display-free playback core for one genome (needs no window or renderer).

    Reconstructs the exact world + substep loop the trainer's ``evaluate*``
    functions use, so ``fitness`` reports — at any step — what a headless
    evaluation truncated there would score. ``step_once`` advances one
    deterministic PHYS_DT substep (the trainer's drift-free path); ``done`` is
    true once the episode would have ended (death, goal for static, or the
    step budget) and further ``step_once`` calls are no-ops.

    Modes mirror the three evaluators:
        "static"   -> trainer.evaluate            (a hand-built level; ends on goal)
        "infinite" -> trainer.evaluate_infinite   (streamed Infinite Run; no goal)
        "gym"      -> trainer.evaluate_gym         (streamed Completion Gym; no goal)
    """

    def __init__(
        self,
        genome: np.ndarray,
        *,
        mode: str = "static",
        level_path: str | Path | None = None,
        seed: int | None = None,
        world_seed: int = config.DEFAULT_SEED,
        max_steps: int | None = None,
        abilities=(),
    ) -> None:
        self.mode = mode
        self.world_seed = int(world_seed)
        self.abilities: set[Ability] = _abilities_set(abilities)
        self.seed: int | None = None  # set by the streamed modes

        self.world = World(seed=self.world_seed)
        register_collisions(self.world.space, world_ref=self.world)

        self.steps = 0
        self._terminated = False
        self.level_width = 0.0
        self.level_name = mode
        # Streaming source (TerrainStream / SegmentStream), maintained ahead of
        # the ball each step. None for a hand-built static level.
        self._stream = None
        # Static levels end on goal contact; the streamed modes have no goal.
        self._goal_terminates = False
        # Completion-Gym scoring state (counted only in gym mode; see
        # _track_gym / evaluate_gym). keys_held is cleared at each segment
        # boundary, so keys are accumulated incrementally rather than read off
        # the player at the end.
        self._gym = mode == "gym"
        self._cleared = 0
        self._cumulative_keys = 0
        self._prev_keys_popcount = 0

        if mode == "static":
            if level_path is None:
                raise ValueError("static mode requires level_path")
            self.max_steps = int(max_steps) if max_steps is not None else config.MAX_STEPS
            self._goal_terminates = True
            self._setup_static(genome, str(level_path))
        elif mode == "infinite":
            self.max_steps = int(max_steps) if max_steps is not None else config.MAX_STEPS
            self._setup_infinite(genome, int(seed) if seed is not None else config.INFINITE_RUN_SEED)
        elif mode == "gym":
            self.max_steps = int(max_steps) if max_steps is not None else config.GYM_MAX_STEPS
            self._setup_gym(genome, int(seed) if seed is not None else config.GYM_SEED)
        else:
            raise ValueError(f"unknown playback mode {mode!r}")

        self.max_x = self.spawn_x

    # -- per-mode setup --------------------------------------------------- #
    def _setup_static(self, genome: np.ndarray, level_path: str) -> None:
        meta = load_level(level_path, self.world)
        self.level_width = float(meta.total_width)
        self.level_name = meta.name
        self.spawn_x, spawn_y = float(meta.spawn[0]), float(meta.spawn[1])
        self._spawn_player(genome, (self.spawn_x, spawn_y))

    def _setup_infinite(self, genome: np.ndarray, seed: int) -> None:
        from ..ai.trainer import INFINITE_SPAWN
        from ..levels.streaming import TerrainStream

        self.seed = seed
        self.level_name = f"Infinite Run (seed={seed})"
        # Grant abilities to the terrain sampler too (not just the player), so the
        # ability-gated double-jump chunks surface in the replay exactly as in
        # evaluate_infinite. Mirrors the gym path's SegmentStream(..., granted).
        self._stream = TerrainStream(self.world, seed,
                                     abilities=frozenset(self.abilities))
        self.spawn_x, spawn_y = float(INFINITE_SPAWN[0]), float(INFINITE_SPAWN[1])
        self._spawn_player(genome, (self.spawn_x, spawn_y))

    def _setup_gym(self, genome: np.ndarray, seed: int) -> None:
        from ..levels.segment_stream import SegmentStream

        self.seed = seed
        self.level_name = f"Completion Gym (seed={seed})"
        granted = frozenset(self.abilities)
        self._stream = SegmentStream(self.world, seed, granted)
        self.spawn_x, spawn_y = float(config.GYM_SPAWN[0]), float(config.GYM_SPAWN[1])
        self._spawn_player(genome, (self.spawn_x, spawn_y))

    def _spawn_player(self, genome: np.ndarray, spawn_xy: tuple[float, float]) -> None:
        self.player = Player(agent=FTNNAgent(genome), spawn_xy=spawn_xy,
                             abilities=set(self.abilities))
        self.world.add_entity(self.player)

    # -- stepping --------------------------------------------------------- #
    def step_once(self) -> None:
        """Advance one PHYS_DT substep, mirroring the matching evaluator's loop
        (maintain the stream ahead of the ball, then substep). No-op once
        ``done``."""
        if self.done:
            return
        if self._stream is not None:
            self._stream.maintain(self.player.body.position.x)
        self.world.substep()
        self.steps += 1
        if self.player.body.position.x > self.max_x:
            self.max_x = self.player.body.position.x
        if self._gym:
            self._track_gym()
        if self.player.dead or (self._goal_terminates and self.player.reached_goal):
            self._terminated = True

    def _track_gym(self) -> None:
        """Mirror evaluate_gym's per-step bookkeeping: accumulate keys before a
        boundary clears the scope, then count segment boundaries the ball's
        max_x has passed (resetting keys_held at each clear)."""
        cur = bin(self.player.keys_held).count("1")
        if cur > self._prev_keys_popcount:
            self._cumulative_keys += cur - self._prev_keys_popcount
        self._prev_keys_popcount = cur
        new_cleared = bisect.bisect_right(self._stream.segment_ends, self.max_x)
        if new_cleared > self._cleared:
            self._cleared = new_cleared
            self.player.keys_held = 0
            self._prev_keys_popcount = 0

    @property
    def done(self) -> bool:
        return self._terminated or self.steps >= self.max_steps

    @property
    def reached_goal(self) -> bool:
        return bool(self.player.reached_goal)

    @property
    def segments_cleared(self) -> int:
        """Completion-Gym segments the ball has passed (0 in the other modes)."""
        return self._cleared

    @property
    def fitness(self) -> float:
        """Live fitness, identical in form to the matching evaluator's score.

        Gym mode reads its incrementally-tracked key/segment counts (keys_held
        is cleared at each boundary, so it can't be read off the player); the
        goal-terminal/streamed modes read keys straight off keys_held with no
        segment bonus, matching evaluate / evaluate_infinite."""
        if self._gym:
            keys_collected = self._cumulative_keys
            segments_cleared = self._cleared
        else:
            keys_collected = bin(self.player.keys_held).count("1")
            segments_cleared = 0
        return fitness(FitnessInputs(
            progress_x=float(self.max_x - self.spawn_x),
            collectibles=int(self.player.collectibles_collected),
            reached_goal=bool(self.player.reached_goal),
            died=bool(self.player.dead),
            steps_taken=self.steps,
            keys_collected=int(keys_collected),
            level_width=float(self.level_width),
            segments_cleared=int(segments_cleared),
        ))


class PlaybackScene(Scene):
    """Pygame shell around a ``PlaybackSim``: drives it at a fixed substep
    cadence and draws the world plus a HUD showing the live (faithful) fitness.

    The sim is stepped a fixed number of substeps per frame rather than via the
    real-time accumulator, so the replayed trajectory — and the HUD fitness —
    stays bit-identical to a headless evaluation. At the capped frame rate that
    cadence is real-time; if rendering lags, the replay slows but stays
    faithful. The window/font calls are confined to this class (the sim core
    runs headless), and pygame is imported inside the methods that use it.
    """

    def __init__(self, screen, sim: PlaybackSim, *, substeps_per_frame: int | None = None) -> None:
        from ..camera import FollowCamera
        from ..render.core import RenderCore
        from ..render.particles import ParticleSystem
        from ..render.renderer import Renderer
        from ..render.theme import get_active_theme

        self.screen = screen
        self.sim = sim
        if substeps_per_frame is None:
            substeps_per_frame = max(1, round(config.PHYS_HZ / config.TARGET_FPS))
        self.substeps_per_frame = int(substeps_per_frame)

        self.core = RenderCore(screen)
        self.camera = FollowCamera(self.core.vw, self.core.vh)
        # Preserve the pre-overhaul visible-world span on the smaller surface
        # (mirrors PlayScene).
        self.camera.scale = 1.0 / self.core.scale
        self.renderer = Renderer(self.core, self.camera)
        self.particles = ParticleSystem(
            int(get_active_theme().params.get("particle_cap", 300)))
        self._hud_font = None
        # Snap the camera onto the ball for frame 0 (no lerp-in from the origin).
        self.camera.position = (sim.player.body.position.x, sim.player.body.position.y)

    def handle_events(self, events):
        import pygame
        for event in events:
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return None
        return self

    def update(self, frame_dt: float) -> None:
        self.renderer.begin_frame(self.sim.world)
        for _ in range(self.substeps_per_frame):
            if self.sim.done:
                break
            self.sim.step_once()
        self.core.update(frame_dt)
        self.particles.update(frame_dt)
        self.camera.update(
            target=(self.sim.player.body.position.x, self.sim.player.body.position.y),
            dt=frame_dt,
        )

    def draw(self) -> None:
        self.renderer.draw_parallax(self.camera)
        self.renderer.draw_static_segments(self.sim.world.space)
        alpha = self.sim.world.alpha
        for entity in self.sim.world.entities:
            entity.draw(self.renderer, alpha)
        self.particles.draw(self.renderer)
        self._draw_hud()
        self.core.present()

    def _draw_hud(self) -> None:
        """Top-left overlay drawn on the virtual surface (so the ×N upscale
        keeps it crisp), mirroring Renderer.draw_score's approach."""
        import pygame
        from ..render.theme import get_active_theme
        if self._hud_font is None:
            self._hud_font = pygame.font.SysFont(None, 16)
        sim = self.sim
        pal = get_active_theme().palette
        surf = self.renderer.screen
        lines = [
            sim.level_name,
            f"fitness: {sim.fitness:.1f}",
            f"x: {sim.player.body.position.x:.0f}  (max {sim.max_x:.0f})",
            f"step: {sim.steps}/{sim.max_steps}",
            f"dead: {int(sim.player.dead)}   goal: {int(sim.reached_goal)}"
            + (f"   cleared: {sim.segments_cleared}" if sim.mode == "gym" else ""),
        ]
        y = 4
        for i, text in enumerate(lines):
            color = pal["hud_best"] if i == 0 else pal["hud"]
            surf.blit(self._hud_font.render(text, True, color), (6, y))
            y += 11
