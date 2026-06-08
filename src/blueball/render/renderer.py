"""Renderer — draws the world by blitting theme sprites (with primitive
fallbacks for entities not yet themed)."""

from __future__ import annotations

import math

import pygame
import pymunk

from .. import config


def _lerp(a, b, t):
    return a + (b - a) * t


class Renderer:
    def __init__(self, screen, camera) -> None:
        # `screen` accepts a RenderCore (production) or a raw Surface (legacy tests).
        from .core import RenderCore
        if isinstance(screen, RenderCore):
            self.core = screen
            self.screen = screen.surface
        else:
            self.core = None
            self.screen = screen
        self.camera = camera
        # Track previous positions per body for interpolation
        self._prev_pos: dict[int, tuple[float, float]] = {}
        self._prev_angle: dict[int, float] = {}
        self._hud_font = None  # lazily created (needs pygame.font init)

    def begin_frame(self, world) -> None:
        """Snapshot previous positions for next frame's interpolation.

        Only moving (non-STATIC) bodies are snapshotted: static bodies never
        move, so _interp_body_pos's fallback (return current position when the id
        is absent) yields the identical result, and skipping them avoids
        re-snapshotting the shared static_body every frame plus unbounded growth
        of the snapshot dicts as streamed-in static geometry churns.
        """
        for body in world.space.bodies:
            if body.body_type == pymunk.Body.STATIC:
                continue
            self._prev_pos[id(body)] = (body.position.x, body.position.y)
            self._prev_angle[id(body)] = body.angle

    def reset_interpolation(self) -> None:
        """Clear cached per-body previous positions/angles.

        Call whenever the renderer's target World is swapped (e.g. TrainScene's
        generation rollover). Without the reset, these dicts accumulate stale
        entries keyed by ``id(body)`` of bodies that are now garbage-collected;
        CPython can reuse those ids for freshly allocated bodies and
        ``_interp_body_pos`` would then read a dead body's last position for an
        unrelated new body — a one-frame teleport.
        """
        self._prev_pos.clear()
        self._prev_angle.clear()

    def draw_background(self, color: tuple[int, int, int]) -> None:
        self.screen.fill(color)

    def _sky_surface(self) -> pygame.Surface:
        # Cached vertical gradient sized to the virtual surface.
        if getattr(self, "_sky", None) is None:
            theme = self._theme()
            w, h = self.screen.get_size()
            sky = pygame.Surface((w, h))
            top, bot = theme.palette["sky_top"], theme.palette["sky_bottom"]
            for y in range(h):
                t = y / max(1, h - 1)
                col = tuple(round(_lerp(top[i], bot[i], t)) for i in range(3))
                pygame.draw.line(sky, col, (0, y), (w, y))
            self._sky = sky
        return self._sky

    def draw_parallax(self, camera) -> None:
        from .parallax import layer_offset
        theme = self._theme()
        self.screen.blit(self._sky_surface(), (0, 0))
        vw = self.screen.get_width()
        cam_x = camera.position[0]
        for layer in theme.parallax:
            surf = theme.sprites[layer.sprite_key].bake(theme.palette)
            tw = surf.get_width()
            x = layer_offset(cam_x, layer.factor, tw)
            while x < vw:
                self.screen.blit(surf, (int(x), layer.y))
                x += tw

    def _w2s(self, world_xy):
        return self.camera.world_to_screen(world_xy)

    def _theme(self):
        from .theme import get_active_theme
        return get_active_theme()

    def _blit_sprite(self, world_xy, key, *, deg=0.0, frame=0, scale=None):
        theme = self._theme()
        surf = theme.sprites[key].frame(frame, theme.palette)
        if scale is not None:
            sx, sy = scale
            surf = pygame.transform.scale(
                surf, (max(1, round(surf.get_width() * sx)),
                       max(1, round(surf.get_height() * sy))))
        if deg:
            surf = pygame.transform.rotate(surf, deg)
        px, py = self.camera.world_to_screen(world_xy)
        ox, oy = self.core.shake_offset if self.core else (0.0, 0.0)
        self.screen.blit(surf, surf.get_rect(center=(round(px + ox), round(py + oy))))

    def _blit_point(self, world_xy, color, size):
        px, py = self.camera.world_to_screen(world_xy)
        ox, oy = self.core.shake_offset if self.core else (0.0, 0.0)
        s = max(1, int(size))
        x, y = round(px + ox) - s // 2, round(py + oy) - s // 2
        self.screen.fill(color, (x, y, s, s))

    def _interp_body_pos(self, body: pymunk.Body, alpha: float) -> tuple[float, float]:
        prev = self._prev_pos.get(id(body), (body.position.x, body.position.y))
        return (_lerp(prev[0], body.position.x, alpha), _lerp(prev[1], body.position.y, alpha))

    def _interp_body_angle(self, body: pymunk.Body, alpha: float) -> float:
        prev = self._prev_angle.get(id(body), body.angle)
        return _lerp(prev, body.angle, alpha)

    def _pulse(self, magnitude: float = 0.15) -> float:
        """Time-based scale factor oscillating around 1.0 for idle-pulse visuals."""
        t = pygame.time.get_ticks() / 1000.0
        return 1.0 + magnitude * math.sin(t * 2 * math.pi * config.COLLECTIBLE_PULSE_HZ)

    def draw_ball(self, body: pymunk.Body, alpha: float) -> None:
        from .animation import squash_stretch
        wx, wy = self._interp_body_pos(body, alpha)
        angle = self._interp_body_angle(body, alpha)
        sx, sy = squash_stretch(body.velocity.y,
                                self._theme().params.get("squash_max", 0.3))
        self._blit_sprite((wx, wy), "ball", deg=-math.degrees(angle), scale=(sx, sy))

    def draw_spike(self, pos, width, height, orientation: str = "up"):
        """Draw a spike sprite oriented to the given direction."""
        deg = {"up": 0, "right": -90, "down": 180, "left": 90}[orientation]
        self._blit_sprite(pos, "spike", deg=deg)

    def draw_collectible(self, pos):
        self._blit_sprite(pos, "collectible")

    def draw_ability_pickup(self, pos, radius, ability: str) -> None:
        key = f"ability_{ability}" if f"ability_{ability}" in self._theme().sprites else "ability"
        p = self._pulse()
        self._blit_sprite(pos, key, scale=(p, p))

    def draw_boost_pad(self, pos, width, direction: float = 1.0) -> None:
        """Boost-pad strip sprite; chevrons point along the launch arrow
        (flipped 180deg when the pad launches left)."""
        sx = width / self._sprite_w("boost_pad")
        self._blit_sprite(pos, "boost_pad", deg=180 if direction < 0 else 0, scale=(sx, 1.0))

    def draw_goal(self, pos, width, height):
        self._blit_sprite(pos, "goal")

    def draw_patroller(self, body, size, alpha):
        wx, wy = self._interp_body_pos(body, alpha)
        self._blit_sprite((wx, wy), "patroller")

    def draw_falling_hazard(self, body, radius, alpha):
        wx, wy = self._interp_body_pos(body, alpha)
        self._blit_sprite((wx, wy), "falling_hazard")

    def draw_static_segments(self, space: pymunk.Space, color=None) -> None:
        """Draw every static pymunk.Segment as a thick line plus a darker top edge."""
        if color is None:
            color = self._theme().palette["ground"]
        sw, sh = self.screen.get_size()
        for shape in space.shapes:
            if isinstance(shape, pymunk.Segment) and shape.body is space.static_body:
                a = self._w2s((shape.a.x, shape.a.y))
                b = self._w2s((shape.b.x, shape.b.y))
                # Skip segments whose screen-space bounding box is entirely off
                # the viewport — they contribute no on-screen pixels. Uses min/max
                # of both endpoints so a long segment crossing the screen with
                # both endpoints off-screen is still drawn.
                if (max(a[0], b[0]) < 0 or min(a[0], b[0]) > sw
                        or max(a[1], b[1]) < 0 or min(a[1], b[1]) > sh):
                    continue
                pygame.draw.line(self.screen, color, a, b, 6)
                pygame.draw.line(self.screen, self._theme().palette["ground_edge"], a, b, 2)

    # ------------------------------------------------------------------ #
    # Phase 3 entity renderers                                            #
    # ------------------------------------------------------------------ #

    def _sprite_w(self, key: str) -> int:
        """Native pixel width of a sprite's frame (for length-matching scale)."""
        theme = self._theme()
        return theme.sprites[key].frame(0, theme.palette).get_width()

    def _sprite_h(self, key: str) -> int:
        """Native pixel height of a sprite's frame (for height-matching scale)."""
        theme = self._theme()
        return theme.sprites[key].frame(0, theme.palette).get_height()

    def draw_moving_platform(self, body: pymunk.Body, alpha: float, length: float) -> None:
        """Interpolated horizontal platform, sprite scaled to the entity length."""
        wx, wy = self._interp_body_pos(body, alpha)
        sx = length / self._sprite_w("platform")
        self._blit_sprite((wx, wy), "platform", scale=(sx, 1.0))

    def draw_spring(self, pos, width: float, t: float) -> None:
        """Spring pad sprite with a gentle vertical squish pulse via `t`."""
        sx = width / self._sprite_w("spring")
        sy = 1.0 + 0.15 * abs(math.sin(t * 6))
        self._blit_sprite(pos, "spring", scale=(sx, sy))

    def draw_checkpoint(self, pos, radius: int, t: float, active: bool) -> None:
        """Flag sprite; bright `checkpoint_active` variant once activated."""
        self._blit_sprite(pos, "checkpoint_active" if active else "checkpoint")

    def draw_crumbling_platform(self, pos, alpha: float, width: float, progress: float) -> None:
        """Crumbling platform sprite scaled to width (progress darkening dropped
        for this pass — single sprite)."""
        sx = width / self._sprite_w("crumbling")
        self._blit_sprite(pos, "crumbling", scale=(sx, 1.0))

    def draw_key(self, pos, radius: int, key_id: int) -> None:
        """Key sprite."""
        self._blit_sprite(pos, "key")

    def draw_door(self, pos, height: int, open_: bool) -> None:
        """Door sprite; passable `door_open` variant when open. The entity
        position is the door's base; the sprite is centered on the door's span."""
        x, y = pos
        self._blit_sprite((x, y - height / 2), "door_open" if open_ else "door",
                          scale=(1.0, height / self._sprite_h("door")))

    def draw_pushable_box(self, body: pymunk.Body, alpha: float, size: float) -> None:
        """Interpolated crate sprite scaled to the box footprint."""
        wx, wy = self._interp_body_pos(body, alpha)
        s = size / self._sprite_w("box")
        self._blit_sprite((wx, wy), "box", scale=(s, s))

    def draw_swinging_hazard(
        self,
        anchor_body: pymunk.Body,
        bob_body: pymunk.Body,
        bob_radius: float,
        alpha: float,
    ) -> None:
        """Chain line from anchor to bob, plus a spiked-ball sprite for the bob."""
        ax, ay = anchor_body.position.x, anchor_body.position.y
        bx, by = self._interp_body_pos(bob_body, alpha)
        sa = self._w2s((ax, ay))
        sb = self._w2s((bx, by))
        pygame.draw.line(self.screen, self._theme().palette["door_hi"], sa, sb, 2)
        s = (2 * bob_radius) / self._sprite_w("swing_hazard")
        self._blit_sprite((bx, by), "swing_hazard", scale=(s, s))

    def draw_one_way_platform(self, pos, width: float) -> None:
        """One-way platform strip sprite (down-chevrons baked in)."""
        sx = width / self._sprite_w("one_way")
        self._blit_sprite(pos, "one_way", scale=(sx, 1.0))

    def draw_charger(self, body: pymunk.Body, alpha: float, state: str, radius: int = 12) -> None:
        """Interpolated charger; brighter `charger_charge` variant while charging."""
        wx, wy = self._interp_body_pos(body, alpha)
        self._blit_sprite((wx, wy), "charger_charge" if state == "charge" else "charger")

    def draw_lava(self, body: pymunk.Body, alpha: float, width: float, height: float) -> None:
        """Lava block sprite. body.position is the top-edge center, so the sprite
        is offset down by half its height to sit below the deadly surface."""
        wx, wy = self._interp_body_pos(body, alpha)
        sx = width / self._sprite_w("lava")
        sy = height / self._sprite_h("lava")
        self._blit_sprite((wx, wy + height / 2), "lava", scale=(sx, sy))

    def draw_projectile(self, body: pymunk.Body, alpha: float, radius: int = 10) -> None:
        """Interpolated fiery orb fired by a cannon."""
        wx, wy = self._interp_body_pos(body, alpha)
        s = (2 * radius) / self._sprite_w("projectile")
        self._blit_sprite((wx, wy), "projectile", scale=(s, s))

    def draw_cannon(self, position, direction: str) -> None:
        """Wall-mounted barrel. Sprite points right; flip 180deg for left.
        The cannon is fixed, so its position is drawn directly (no interp)."""
        self._blit_sprite(position, "cannon", deg=180 if direction == "left" else 0)

    def draw_score(self, score: int, best: int) -> None:
        """Top-left HUD rendered small to the virtual surface so the ×2 upscale
        makes it crisp/chunky-pixelated."""
        if self._hud_font is None:
            self._hud_font = pygame.font.SysFont(None, 16)
        pal = self._theme().palette
        score_surf = self._hud_font.render(f"Score: {score}", True, pal["hud"])
        self.screen.blit(score_surf, (6, 5))
        best_surf = self._hud_font.render(f"Best: {best}", True, pal["hud_best"])
        self.screen.blit(best_surf, (6, 18))
