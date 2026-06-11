"""Shared scripted agents for tuning probes AND committed segment tests.

The same agent that FOUND a geometry's safe margin also GUARDS it, so the two
can never drift. The double-jump maneuver is the project's MAX-DISTANCE,
apex-fired double jump — the strongest "cheese" attempt — and it is a GENUINE
two-impulse arc: a jump fires only on a FRESH press (input_feel.JumpController),
so the air jump needs a release (one tick not holding) followed by a re-press.

Why fire at the apex: with AIR_CONTROL==0 horizontal speed is fixed at launch
for the entire flight, so maximum horizontal distance == maximum airtime. The
earliest moment we can release WITHOUT jump-cutting the first jump is the apex,
where the body has stopped ascending (vy >= 0) and the cut is a no-op (the
Player only applies the cut while vy < 0). Releasing one tick there, then
re-pressing, stacks the second full impulse as early as legally possible,
maximizing the combined arc. No reachable input sequence flies farther.
"""
from __future__ import annotations

from blueball.agent import Agent, Action
from blueball.world import World
from blueball.collision import register as register_collisions


def fresh_world(seed: int = 0) -> World:
    w = World(seed=seed)
    register_collisions(w.space, world_ref=w)
    return w


def find_entity(world, type_name: str):
    return next((e for e in world.entities
                if type(e).__name__ == type_name), None)


def run_segment(world, player, steps: int = 2500) -> str:
    for _ in range(steps):
        world.substep()
        if player.reached_goal:
            return "GOAL"
        if player.dead:
            return "DEAD"
    return "TIMEOUT"


class _Maneuver:
    """Mixin: one max-distance double jump to the right. Set self.player first."""
    def _start_jump(self):
        self._mj = "primary"

    def _maneuver(self):
        p = self.player
        grounded = p.grounded
        vy = p.body.velocity.y
        if self._mj == "primary":
            self._mj = "ascend"
            return Action.RIGHT_JUMP          # grounded fresh press -> primary fires
        if self._mj == "ascend":
            # Hold through the first jump's ascent (a release here would cut it),
            # then release the instant the ascent ends at the apex (vy >= 0,
            # where the cut is a no-op). Re-pressing immediately stacks the air
            # jump as early as legally possible — the max-distance arc.
            if (not grounded) and vy >= 0:
                self._mj = "release"
                return Action.RIGHT            # release at apex (cut is harmless)
            return Action.RIGHT_JUMP           # hold through ascent
        if self._mj == "release":
            self._mj = "air"
            return Action.RIGHT_JUMP           # fresh airborne press -> air jump
        if self._mj == "air":
            if (not grounded) and vy >= 0:
                self._mj = "done"
            return Action.RIGHT_JUMP if self._mj == "air" else Action.RIGHT
        return Action.RIGHT                    # drift right under air control


class DoubleJumpVaultAgent(Agent, _Maneuver):
    """Roll right to launch_x, then one competent double jump. The strongest
    'cheese' attempt for anti-cheese tests, and the boost-gap solver when a pad
    sits on the run-up (the boost comes from the world, not the agent)."""
    def __init__(self, launch_x: float):
        self.launch_x = launch_x
        self.player = None
        self._mj = None

    def act(self, observation):
        if self._mj is None:
            if self.player.body.position[0] < self.launch_x:
                return Action.RIGHT
            self._start_jump()
        return self._maneuver()


class SingleStepAgent(Agent):
    """Curriculum stage-1 solver: run to launch_x, a SINGLE jump ONTO a pre-placed
    box, then a single jump OFF it to the goal. A single jump (press, hold to
    apex, release, NO air jump) lands ~411px out — where stage 1 places the box —
    so no timing finesse is needed. Set self.player (self.box optional)."""
    def __init__(self, launch_x: float, on_box_run: int = 4):
        self.launch_x = launch_x
        self.on_box_run = on_box_run
        self.player = None
        self.box = None
        self._phase = "APPROACH"
        self._j = None
        self._run = 0

    def _single_jump(self):
        p = self.player
        vy = p.body.velocity.y
        gr = p.grounded
        if self._j == "press":
            self._j = "ascend"
            return Action.RIGHT_JUMP
        if self._j == "ascend":
            if (not gr) and vy >= 0:        # apex -> release, do NOT re-press
                self._j = "fall"
                return Action.RIGHT
            return Action.RIGHT_JUMP        # hold through ascent
        return Action.RIGHT                 # falling / drifting right

    def act(self, observation):
        p = self.player
        px = p.body.position[0]
        if self._phase == "APPROACH":
            if px < self.launch_x:
                return Action.RIGHT
            self._phase = "JUMP1"
            self._j = "press"
        if self._phase == "JUMP1":
            a = self._single_jump()
            if self._j == "fall" and p.grounded:   # landed on the box
                self._phase = "ONBOX"
                self._run = 0
            return a
        if self._phase == "ONBOX":
            if self._run < self.on_box_run:
                self._run += 1
                return Action.RIGHT
            self._phase = "JUMP2"
            self._j = "press"
        if self._phase == "JUMP2":
            return self._single_jump()
        return Action.RIGHT


class DoubleStepAgent(Agent, _Maneuver):
    """Curriculum stage-2 solver: run to launch_x, a max DOUBLE jump ONTO a bigger
    pre-placed box (placed where the natural max double-jump lands), then a max
    double jump OFF it to the goal. Reuses _Maneuver's corrected apex-fired double
    jump twice. Set self.player (self.box optional)."""
    def __init__(self, launch_x: float, on_box_run: int = 6):
        self.launch_x = launch_x
        self.on_box_run = on_box_run
        self.player = None
        self.box = None
        self._phase = "APPROACH"
        self._mj = None
        self._run = 0

    def act(self, observation):
        p = self.player
        px = p.body.position[0]
        if self._phase == "APPROACH":
            if px < self.launch_x:
                return Action.RIGHT
            self._phase = "JUMP1"
            self._start_jump()
        if self._phase == "JUMP1":
            a = self._maneuver()
            if self._mj == "done" and p.grounded:   # landed on the box
                self._phase = "ONBOX"
                self._run = 0
            return a
        if self._phase == "ONBOX":
            if self._run < self.on_box_run:
                self._run += 1
                return Action.RIGHT
            self._phase = "JUMP2"
            self._start_jump()
        if self._phase == "JUMP2":
            return self._maneuver()
        return Action.RIGHT


class BoxHopAgent(Agent, _Maneuver):
    """Box-lava solver: shove the box into the pit, brake on the near ledge, then
    double-jump near-ledge -> box-top -> far-ledge -> goal. Set player+box."""
    def __init__(self, push_steps: int, jump1_x: float, box_run: int = 0):
        self.push_steps = push_steps
        self.jump1_x = jump1_x
        self.box_run = box_run
        self.player = None
        self.box = None
        self.phase = "SHOVE"
        self._t = 0
        self._mj = None
        self._run = 0
        self._settle_t = 0  # set on entering SETTLE; init here so it is never unbound

    def act(self, observation):
        p, box = self.player, self.box
        self._t += 1
        px = p.body.position[0]
        bx = box.body.position[0]

        if self.phase == "SHOVE":
            if self._t <= self.push_steps:
                return Action.RIGHT
            self.phase = "BRAKE"
        if self.phase == "BRAKE":
            # Brake until slow and parked ~21px back from the near ledge edge
            # (x=256 in box-lava geometry) so the jump-off point is consistent.
            if p.body.velocity[0] > 8.0 or px > 235.0:
                return Action.LEFT
            self.phase = "SETTLE"
            self._settle_t = self._t
        if self.phase == "SETTLE":
            if abs(box.body.velocity[0]) > 3.0 and self._t - self._settle_t < 120:
                return Action.IDLE
            self.phase = "APPROACH"
        if self.phase == "APPROACH":
            if px < self.jump1_x:
                return Action.RIGHT
            self.phase = "JUMP1"
            self._start_jump()
        if self.phase == "JUMP1":
            a = self._maneuver()
            if self._mj == "done" and p.grounded and abs(px - bx) < 40.0:
                self.phase = "ONBOX"
                self._run = 0
            return a
        if self.phase == "ONBOX":
            if self._run < self.box_run:
                self._run += 1
                return Action.RIGHT
            self.phase = "JUMP2"
            self._start_jump()
        if self.phase == "JUMP2":
            return self._maneuver()
        return Action.RIGHT
