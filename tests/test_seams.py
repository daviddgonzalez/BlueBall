"""Ground-seam welding: a fast ball must not catch on the endcaps where two
collinear chunk ground segments meet. Regression for the "seam hop"."""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from blueball.abilities import Ability
from blueball.agent import Action
from blueball.collision import register as register_collisions
from blueball.entities.player import Player
from blueball.levels.chunks.boost_pad import BoostPadChunk
from blueball.levels.chunks.flat import GROUND_Y, Flat
from blueball.levels.seams import weld_ground_seams
from blueball.world import World


class StubAgent:
    def __init__(self, a=Action.RIGHT):
        self._a = a
    def act(self, obs):
        return self._a


def _worst_seam_rise(weld: bool) -> float:
    """Boost to full speed, then measure the largest the ball rises above its
    resting height while crossing the Flat|Flat seams beyond the pad."""
    w = World(seed=0)
    register_collisions(w.space, world_ref=w)
    x = 0.0
    x += Flat(width_tiles=8).build(w, x_offset=x)
    x += BoostPadChunk(width_tiles=4, multiplier=2.0).build(w, x_offset=x)
    for _ in range(8):
        x += Flat(width_tiles=6).build(w, x_offset=x)
    if weld:
        weld_ground_seams(w.space)
    p = Player(agent=StubAgent(Action.RIGHT), spawn_xy=(40.0, GROUND_Y - 30.0),
               abilities={Ability.DOUBLE_JUMP})
    w.add_entity(p)
    rest_y = None
    worst = 0.0
    for i in range(700):
        w.substep()
        y = p.body.position[1]
        if i == 40:
            rest_y = y
        if rest_y is not None and p.body.position[0] > 300:  # past the pad, at speed
            worst = max(worst, rest_y - y)  # y-up is smaller
    return worst


def test_unwelded_ground_has_a_seam_hop():
    # sanity: the bug exists without welding (so the welded assertion is meaningful)
    assert _worst_seam_rise(weld=False) > 1.5


def test_welding_removes_the_seam_hop():
    assert _worst_seam_rise(weld=True) < 0.5


def test_weld_is_idempotent_and_counts_segments():
    w = World(seed=0)
    register_collisions(w.space, world_ref=w)
    x = 0.0
    for _ in range(4):
        x += Flat(width_tiles=4).build(w, x_offset=x)
    first = weld_ground_seams(w.space)
    second = weld_ground_seams(w.space)
    assert first >= 3       # the interior seams got neighbours
    assert second == first  # idempotent
