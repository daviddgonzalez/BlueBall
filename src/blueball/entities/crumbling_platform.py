"""CrumblingPlatform — static segment that removes itself after a dynamic body
contacts it for crumble_delay_s seconds.

Removal is deferred to update() because pymunk forbids shape removal during a
collision callback / mid-step.  Contact detection uses an AABB-overlap probe:
each update tick, we iterate all shapes in the space and check whether any
dynamic body's shape AABB intersects the segment's AABB.
"""

from __future__ import annotations

import pymunk

from .base import Entity

# Match-everything filter for the broadphase contact probe (see update()).
_CRUMBLE_QUERY_FILTER = pymunk.ShapeFilter()


class CrumblingPlatform(Entity):
    def __init__(
        self,
        world,
        position: tuple[float, float],
        width: float,
        crumble_delay_s: float,
    ) -> None:
        super().__init__()
        cx, cy = position
        hw = width / 2
        self.shape = pymunk.Segment(
            world.space.static_body,
            (cx - hw, cy),
            (cx + hw, cy),
            5,
        )
        self.shape.friction = 1.0
        self.shapes.append(self.shape)
        self.position = position
        self.width = width
        self.crumble_delay_s = crumble_delay_s

        self._world = world
        self._contacted: bool = False
        self._contact_timer: float = 0.0
        self._removed: bool = False

    def update(self, dt: float) -> None:
        if self._removed:
            return

        if not self._contacted:
            # Probe (via pymunk's broadphase index) for any dynamic-body shape
            # whose AABB overlaps ours. bb_query returns the same overlap set as
            # scanning every shape and calling bb.intersects, but uses the C
            # spatial index (fresh after space.step, which runs before this loop)
            # — so the AABB-overlap trigger timing is unchanged.
            for shape in self._world.space.bb_query(self.shape.bb, _CRUMBLE_QUERY_FILTER):
                if shape is self.shape:
                    continue
                if shape.body.body_type == pymunk.Body.DYNAMIC:
                    self._contacted = True
                    break
            return  # first contact tick: start timer next tick

        # Timer running
        self._contact_timer += dt
        if self._contact_timer >= self.crumble_delay_s:
            self._remove_from_space()
            self._removed = True

    def draw(self, renderer, alpha: float) -> None:
        if self._removed:
            return
        progress = 0.0
        if self._contacted and self.crumble_delay_s > 0:
            progress = min(1.0, self._contact_timer / self.crumble_delay_s)
        renderer.draw_crumbling_platform(self.position, alpha, self.width, progress)
