from blueball.world import World
from blueball.collision import register as register_collisions
from blueball.entities.player import Player
from blueball.agent import Agent, Action
from blueball.levels.chunks.flat import GROUND_Y
from blueball.levels.segments import GoalSegment, KeyDoorGoalSegment
from blueball.abilities import Ability
from blueball.levels.segments import (
    BoxLavaSegment, KeyDoorBoxLavaSegment, BoxStepSegment, BoxLeapSegment,
    BoostGapSegment,
    SEGMENT_TEMPLATES,
    _BOX_LAVA_PIT_TILES, _BOX_LAVA_DEPTH,
)
from tests.segment_maneuvers import (
    SingleStepAgent,
    DoubleStepAgent,
    DoubleJumpVaultAgent,
    fresh_world,
    find_entity,
    run_segment,
    remove_entity,
)


class _DelayedJumpAgent(Agent):
    """Rolls right to build speed, then spams RIGHT_JUMP — the strongest vault
    attempt (full jump + air-jump arc launched near the pit edge)."""
    def __init__(self, delay: int) -> None:
        self._delay = delay
        self._t = 0

    def act(self, observation):
        self._t += 1
        return Action.RIGHT if self._t <= self._delay else Action.RIGHT_JUMP


def _fresh_world():
    w = World(seed=0)
    register_collisions(w.space, world_ref=w)
    return w


def _names(world):
    return [type(e).__name__ for e in world.entities]


class _RightAgent(Agent):
    """Always rolls right — solves any flat, jump-free segment."""
    def act(self, observation):
        return Action.RIGHT


def _rolls_to_goal(world, player, max_steps=3000):
    for _ in range(max_steps):
        world.substep()
        if player.reached_goal:
            return True
    return False


def test_goal_segment_builds_a_goal_with_positive_width():
    w = _fresh_world()
    width = GoalSegment().build(w, x_offset=0.0)
    assert width > 0
    assert "Goal" in _names(w)
    assert GoalSegment.tier == 0
    assert GoalSegment.min_abilities == frozenset()


def test_keydoorgoal_builds_key_left_of_door_plus_goal():
    w = _fresh_world()
    width = KeyDoorGoalSegment().build(w, x_offset=0.0)
    names = _names(w)
    assert names.count("Key") == 1 and names.count("Door") == 1 and "Goal" in names
    key = next(e for e in w.entities if type(e).__name__ == "Key")
    door = next(e for e in w.entities if type(e).__name__ == "Door")
    assert key.position[0] < door.position[0]
    assert width > 0


def test_goal_segment_is_solvable_by_rolling_right():
    w = _fresh_world()
    GoalSegment().build(w, x_offset=0.0)
    p = Player(agent=_RightAgent(), spawn_xy=(40.0, GROUND_Y - 30.0))
    w.add_entity(p)
    assert _rolls_to_goal(w, p)


def test_keydoorgoal_is_solvable_by_rolling_right():
    w = _fresh_world()
    KeyDoorGoalSegment().build(w, x_offset=0.0)
    p = Player(agent=_RightAgent(), spawn_xy=(40.0, GROUND_Y - 30.0))
    w.add_entity(p)
    # Rolling right collects the low key, opens the door, reaches the goal.
    assert _rolls_to_goal(w, p)


def test_boxlava_segment_composition_and_requirements():
    w = _fresh_world()
    width = BoxLavaSegment(pit_tiles=6).build(w, x_offset=0.0)
    names = _names(w)
    assert "Lava" in names and "PushableBox" in names and "Goal" in names
    assert BoxLavaSegment.tier == 2
    assert Ability.DOUBLE_JUMP in BoxLavaSegment.min_abilities
    assert width > 0


def test_boxlava_random_is_the_tuned_fixed_geometry():
    # Stage 3 is fixed at the controller-validated vault-proof cell; random()
    # returns it unconditionally (no width randomization that could re-introduce
    # a vaultable, box-optional pit).
    import random
    for s in range(40):
        seg = BoxLavaSegment.random(random.Random(s))
        assert seg.pit_tiles == _BOX_LAVA_PIT_TILES
        assert seg.depth == _BOX_LAVA_DEPTH


def test_tier3_combo_composition():
    w = _fresh_world()
    KeyDoorBoxLavaSegment().build(w, x_offset=0.0)
    names = _names(w)
    for kind in ("Key", "Door", "Lava", "PushableBox", "Goal"):
        assert kind in names, kind
    assert KeyDoorBoxLavaSegment.tier == 3
    assert Ability.DOUBLE_JUMP in KeyDoorBoxLavaSegment.min_abilities


def test_all_four_tiers_registered():
    assert {t.tier for t in SEGMENT_TEMPLATES} == {0, 1, 2, 3}


def test_boxlava_pit_requires_the_box_not_vaultable():
    # The box must be NECESSARY: with the box removed, a DOUBLE_JUMP agent
    # cannot clear the pit by jumping. Guards against a narrow (vaultable) pit
    # that would train a jump instead of the box-push. Tests the low end (20)
    # of the gym's pit range across many jump-timing phases.
    for jump_delay in range(0, 64, 4):
        w = _fresh_world()
        BoxLavaSegment(pit_tiles=20).build(w, x_offset=0.0)
        box = next(e for e in w.entities if type(e).__name__ == "PushableBox")
        remove_entity(w, box)
        p = Player(agent=_DelayedJumpAgent(jump_delay),
                   spawn_xy=(40.0, GROUND_Y - 30.0),
                   abilities={Ability.DOUBLE_JUMP})
        w.add_entity(p)
        reached = False
        for _ in range(1500):
            w.substep()
            if p.reached_goal:
                reached = True
                break
            if p.dead:
                break
        assert not reached, f"vaulted the box-removed pit at jump_delay={jump_delay}"


def test_boxlava_stage3_pit_is_vault_proof():
    """At the tuned stage-3 cell (pit=24, depth=72), the strongest cheese — the
    apex-fired MAX double jump (DoubleJumpVaultAgent) — cannot clear the pit with
    the box removed, swept across launch_x. Proves the box-push is mandatory at
    the ACTUAL shipped geometry. NB: the weaker _DelayedJumpAgent test above
    (pit=20) gives false confidence — the max-distance maneuver vaults pit<=23
    (~990px reach), so 24 is the minimum vault-proof width."""
    for launch_x in range(220, 261, 8):  # 220, 228, 236, 244, 252, 260
        w = fresh_world()
        BoxLavaSegment(pit_tiles=_BOX_LAVA_PIT_TILES,
                       depth=_BOX_LAVA_DEPTH).build(w, x_offset=0.0)
        box = find_entity(w, "PushableBox")
        remove_entity(w, box)
        agent = DoubleJumpVaultAgent(launch_x=float(launch_x))
        p = Player(
            agent=agent,
            spawn_xy=(40.0, GROUND_Y - 30.0),
            abilities={Ability.DOUBLE_JUMP},
        )
        agent.player = p
        w.add_entity(p)
        result = run_segment(w, p, steps=1500)
        assert result != "GOAL", (
            f"DoubleJumpVaultAgent(launch_x={launch_x}) vaulted the "
            f"box-removed stage-3 pit (pit={_BOX_LAVA_PIT_TILES})"
        )


# ---------------------------------------------------------------------------
# BoxStepSegment — curriculum stage 1 (Task 2b)
# ---------------------------------------------------------------------------

def test_boxstep_is_solvable_by_single_step():
    """At least one (launch_x, on_box_run) combo from the sweep must reach GOAL."""
    sweep = [(lx, obr) for lx in (232, 238, 245) for obr in (3, 6)]
    solved = []
    for launch_x, on_box_run in sweep:
        w = fresh_world()
        BoxStepSegment().build(w, x_offset=0.0)
        p = Player(
            agent=None,
            spawn_xy=(40.0, GROUND_Y - 30.0),
            abilities={Ability.DOUBLE_JUMP},
        )
        w.add_entity(p)
        agent = SingleStepAgent(launch_x=launch_x, on_box_run=on_box_run)
        agent.player = p
        agent.box = find_entity(w, "PushableBox")
        p.agent = agent
        result = run_segment(w, p)
        if result == "GOAL":
            solved.append((launch_x, on_box_run))
    assert solved, f"No SingleStepAgent config reached GOAL; tried {sweep}"


def test_boxstep_requires_the_box_not_vaultable():
    """With the box removed, no DoubleJumpVaultAgent launch_x clears the pit."""
    for lx in (220, 240, 248, 254, 260):
        w = fresh_world()
        BoxStepSegment().build(w, x_offset=0.0)
        box = find_entity(w, "PushableBox")
        remove_entity(w, box)
        agent = DoubleJumpVaultAgent(launch_x=lx)
        p = Player(
            agent=agent,
            spawn_xy=(40.0, GROUND_Y - 30.0),
            abilities={Ability.DOUBLE_JUMP},
        )
        agent.player = p
        w.add_entity(p)
        result = run_segment(w, p, steps=1500)
        assert result != "GOAL", (
            f"DoubleJumpVaultAgent(launch_x={lx}) vaulted the box-removed pit"
        )


def test_boxstep_composition():
    """BoxStepSegment must include Lava, PushableBox, Goal; tier 2; DOUBLE_JUMP."""
    w = fresh_world()
    width = BoxStepSegment().build(w, x_offset=0.0)
    names = [type(e).__name__ for e in w.entities]
    for kind in ("Lava", "PushableBox", "Goal"):
        assert kind in names, f"missing entity type {kind!r} in {names}"
    assert BoxStepSegment.tier == 2
    assert Ability.DOUBLE_JUMP in BoxStepSegment.min_abilities
    assert width > 0


# ---------------------------------------------------------------------------
# BoxLeapSegment — curriculum stage 2 (Task 2c)
# ---------------------------------------------------------------------------

def test_boxleap_is_solvable_by_double_step():
    """A DoubleStepAgent must double-jump onto the box and off to the goal. The
    probe found this cell solves at 24/24 combos; this small in-sweep stays well
    inside that robust margin."""
    sweep = [(lx, obr) for lx in (232, 244, 256) for obr in (6, 10)]
    solved = []
    for launch_x, on_box_run in sweep:
        w = fresh_world()
        BoxLeapSegment().build(w, x_offset=0.0)
        p = Player(
            agent=None,
            spawn_xy=(40.0, GROUND_Y - 30.0),
            abilities={Ability.DOUBLE_JUMP},
        )
        w.add_entity(p)
        agent = DoubleStepAgent(launch_x=launch_x, on_box_run=on_box_run)
        agent.player = p
        agent.box = find_entity(w, "PushableBox")
        p.agent = agent
        result = run_segment(w, p)
        if result == "GOAL":
            solved.append((launch_x, on_box_run))
    assert solved, f"No DoubleStepAgent config reached GOAL; tried {sweep}"


def test_boxleap_requires_the_box_not_vaultable():
    """With the box removed, no DoubleJumpVaultAgent launch_x clears the pit."""
    for lx in (220, 232, 240, 248, 254, 260):
        w = fresh_world()
        BoxLeapSegment().build(w, x_offset=0.0)
        box = find_entity(w, "PushableBox")
        remove_entity(w, box)
        agent = DoubleJumpVaultAgent(launch_x=lx)
        p = Player(
            agent=agent,
            spawn_xy=(40.0, GROUND_Y - 30.0),
            abilities={Ability.DOUBLE_JUMP},
        )
        agent.player = p
        w.add_entity(p)
        result = run_segment(w, p, steps=1500)
        assert result != "GOAL", (
            f"DoubleJumpVaultAgent(launch_x={lx}) vaulted the box-removed pit"
        )


def test_boxleap_not_single_jumpable():
    """No SingleStepAgent combo reaches GOAL — a single jump cannot get onto or
    across the box. This is the stage-1 -> stage-2 discriminator."""
    for launch_x in (232, 244, 256):
        for on_box_run in (4, 8):
            w = fresh_world()
            BoxLeapSegment().build(w, x_offset=0.0)
            p = Player(
                agent=None,
                spawn_xy=(40.0, GROUND_Y - 30.0),
                abilities={Ability.DOUBLE_JUMP},
            )
            w.add_entity(p)
            agent = SingleStepAgent(launch_x=launch_x, on_box_run=on_box_run)
            agent.player = p
            agent.box = find_entity(w, "PushableBox")
            p.agent = agent
            result = run_segment(w, p)
            assert result != "GOAL", (
                f"SingleStepAgent(launch_x={launch_x}, on_box_run={on_box_run}) "
                f"reached GOAL — stage 2 must not be single-jumpable"
            )


def test_boxleap_composition():
    """BoxLeapSegment must include Lava, PushableBox, Goal; tier 3; DOUBLE_JUMP."""
    w = fresh_world()
    width = BoxLeapSegment().build(w, x_offset=0.0)
    names = [type(e).__name__ for e in w.entities]
    for kind in ("Lava", "PushableBox", "Goal"):
        assert kind in names, f"missing entity type {kind!r} in {names}"
    assert BoxLeapSegment.tier == 3
    assert Ability.DOUBLE_JUMP in BoxLeapSegment.min_abilities
    assert width > 0


# ---------------------------------------------------------------------------
# BoostGapSegment — boost-or-die lava gap (Task 3)
# ---------------------------------------------------------------------------

# Pit left edge sits at (runway 8 + pad 3 + approach 2) * 32 = 416px; the solver
# launches just before it. Same value the probe (tune_boost_gap.py) tuned to.
_BOOSTGAP_LAUNCH_X = (8 + 3 + 2) * 32 - 8  # 408


def test_boostgap_composition():
    """BoostGapSegment must include BoostPad, Lava, Goal; tier 2; DOUBLE_JUMP."""
    w = fresh_world()
    width = BoostGapSegment().build(w, x_offset=0.0)
    names = [type(e).__name__ for e in w.entities]
    for kind in ("BoostPad", "Lava", "Goal"):
        assert kind in names, f"missing entity type {kind!r} in {names}"
    assert BoostGapSegment.tier == 2
    assert Ability.DOUBLE_JUMP in BoostGapSegment.min_abilities
    assert width > 0


def test_boostgap_is_solvable_with_boost():
    """With the boost pad present, a max apex-fired double jump (boosted by the
    world's pad) clears the wide lava gap and reaches GOAL."""
    w = fresh_world()
    BoostGapSegment().build(w, x_offset=0.0)
    agent = DoubleJumpVaultAgent(launch_x=_BOOSTGAP_LAUNCH_X)
    p = Player(
        agent=agent,
        spawn_xy=(40.0, GROUND_Y - 30.0),
        abilities={Ability.DOUBLE_JUMP},
    )
    agent.player = p
    w.add_entity(p)
    result = run_segment(w, p, steps=2000)
    assert result == "GOAL", (
        f"boosted DoubleJumpVaultAgent did not reach GOAL (got {result})"
    )


def test_boostgap_requires_boost_not_double_jumpable():
    """Strip the boost pad and the SAME double jump falls short into the lava: it
    must NOT reach GOAL, proving the boost is mandatory (boost-or-die)."""
    w = fresh_world()
    BoostGapSegment().build(w, x_offset=0.0)
    remove_entity(w, find_entity(w, "BoostPad"))
    agent = DoubleJumpVaultAgent(launch_x=_BOOSTGAP_LAUNCH_X)
    p = Player(
        agent=agent,
        spawn_xy=(40.0, GROUND_Y - 30.0),
        abilities={Ability.DOUBLE_JUMP},
    )
    agent.player = p
    w.add_entity(p)
    result = run_segment(w, p, steps=2000)
    assert result != "GOAL", (
        "DoubleJumpVaultAgent cleared the gap WITHOUT the boost pad — the gap "
        "is not boost-or-die"
    )
