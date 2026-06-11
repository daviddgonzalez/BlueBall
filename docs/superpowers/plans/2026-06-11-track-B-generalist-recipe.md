# Track B — Generalist Recipe (KEYSTONE) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A combined multi-episode generalist trainer (`python main.py train generalist`) that mixes Infinite Run + the 5 static levels + the completion gym into one worst-case (`min`) selection objective — the harness that tests H2 (distribution) and produces "clears all 5 levels + moves well on Infinite Run."

**Architecture:** Compose the existing episode constructors (`infinite_episodes` / `static_episodes` / `gym_episodes`) into one `mixed_episodes` list, all double-jump-granted, each per-episode-normalized; feed it to the **existing** `train(episodes=…, aggregate="min")` path (no new GA code — `train` already supports an episode list and `min` aggregation). Add an optional warm-start genome to seed the initial population, and a `train generalist` CLI that records the mixed set + per-kind scores in `run.json`.

**Tech Stack:** Python, NumPy, pygame-ce (headless), multiprocessing GA, pytest. Interpreter: `python3` with `PYTHONPATH=src` (no `.venv` in this worktree).

**Base:** `feature/generalist-recipe` off `master` `556f309` (synced gym + double-jump infinite chunks + watch-best). Full suite green at 542 tests.

**Key facts discovered (2026-06-11, verified in this worktree):**
- `train()` (`ai/trainer.py:247`) already takes `episodes: Sequence[EpisodeSpec]`, `aggregate: str` (`"mean_std"`|`"min"`), `map_fn`, `save_dir`. **No GA/aggregation code is needed — B passes `aggregate="min"`.**
- `EpisodeSpec` (`ai/episodes.py:25`) has `norm: float = 1.0` and `abilities: tuple[str,…] = ()`. `evaluate_episodes` routes `ep.abilities` to gym; Task 1 adds static `evaluate()`. **INTEGRATION DEPENDENCY:** the infinite dispatch does NOT yet pass `ep.abilities` to `evaluate_infinite` — that's Track D's flag, which is **uncommitted on `feature/movement-floor`** (not on master). `mixed_episodes` sets abilities on infinite specs, but the infinite terrain stays single-jump until D's flag lands on master and B rebases. **Before the definitive generalist run, land D and confirm the infinite dispatch passes `ep.abilities`.**
- `compute_level_par` (`ai/episodes.py:64`) gives the static normalization divisor; `static_episodes` already sets `norm=compute_level_par(p)`.
- **`LevelMeta.starting_abilities` is ABSENT from master** — only `maze.json` would declare double-jump, and only after the foundation lands. Task 0 cherry-picks it (4 commits already on `feature/controller-capacity`).
- `train()` seeds the population at `ai/trainer.py:325`: `population = [random_genome(ga_rng) for _ in range(pop_size)]` — the single insertion point for warm-start.

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/blueball/levels/loader.py`, `levels/maze.json`, `ai/curriculum.py`, `ai/trainer.py`, `scenes/train.py`, `scenes/play.py` | Task 0 (cherry-pick) | `LevelMeta.starting_abilities` foundation — static `evaluate()` grants level abilities |
| `src/blueball/ai/episodes.py` | Modify | new `mixed_episodes(...)`; thread `abilities` into `static_episodes` |
| `src/blueball/ai/trainer.py` | Modify | static `evaluate()` unions `ep.abilities`; `train(init_genome=…)` warm-start |
| `src/blueball/ai/persistence.py` | Modify | `run_dir_name` generalist (`gen*`) variant |
| `src/blueball/cli.py` | Modify | `train generalist` subcommand + per-kind `run.json` |
| `src/blueball/config.py` | Modify | mix-count keys (distinct from Track C's net-version key) |
| `tests/test_generalist_recipe.py` | Create | unit + integration coverage for all of the above |

---

## Task 0: Foundation — land `LevelMeta.starting_abilities`

**Goal:** Make static/maze training episodes spawn the player WITH double-jump, so the generalist isn't graded as a single-jump player on double-jump levels (the false-verdict bug).

**Files:**
- Cherry-pick: `c830f92`, `0d76c7c`, `9e4080e`, `75f3778` (in chronological order) from `feature/controller-capacity`
- Touches: `levels/loader.py`, `levels/maze.json`, `ai/curriculum.py`, `ai/trainer.py` (`evaluate`), `scenes/train.py`, `scenes/play.py` (+ their tests)

**Acceptance Criteria:**
- [ ] `LevelMeta.starting_abilities` exists; `maze.json` declares `double_jump`.
- [ ] Static `evaluate()` builds the `Player` with the level's `starting_abilities`.
- [ ] Full suite green (no regressions from the cherry-pick merge resolution).

**Verify:** `PYTHONPATH=src python3 -m pytest -q` → all pass; `grep -rn starting_abilities src/blueball/levels/loader.py` → present.

**Steps:**

- [ ] **Step 1: Cherry-pick the four commits in order**

```bash
cd .claude/worktrees/generalist-recipe
git cherry-pick c830f92 0d76c7c 9e4080e 75f3778
```

- [ ] **Step 2: Resolve the known mild conflicts**

These four predate the gym base, so `ai/curriculum.py` and `scenes/play.py` may conflict. Resolution rule (from Track C's audit): **keep BOTH** the gym-side change and the new `starting_abilities` argument; **drop** any reference to box-push-shaping's `build_box_lava_curriculum` (absent from this base). After resolving:

```bash
git add -A && git cherry-pick --continue
```

- [ ] **Step 3: Run the full suite**

Run: `PYTHONPATH=src python3 -m pytest -q`
Expected: PASS (542+ tests; the cherry-picked commits bring their own tests).

- [ ] **Step 4: Commit** — already committed by cherry-pick; no extra commit needed.

---

## Task 1: `mixed_episodes` constructor + static ability grant

**Goal:** One constructor that returns the mixed, double-jump-granted, per-episode-normalized episode list; and make static `evaluate()` honor episode-level abilities so the grant is uniform across all 5 levels (not just maze's json).

**Files:**
- Modify: `src/blueball/ai/episodes.py` (add `mixed_episodes`; add `abilities` param to `static_episodes`)
- Modify: `src/blueball/ai/trainer.py` (`evaluate()` unions `ep.abilities` with the loaded level's `starting_abilities`)
- Test: `tests/test_generalist_recipe.py`

**Acceptance Criteria:**
- [ ] `mixed_episodes(infinite_seeds, level_names, gym_seeds, world_seed, max_steps, abilities)` returns infinite + static + gym `EpisodeSpec`s, in that order.
- [ ] Every static and gym episode carries `abilities=("double_jump",)`; infinite episodes too (D's path honors it).
- [ ] Static episodes keep `norm=compute_level_par(level)`; infinite/gym keep `norm=1.0`.
- [ ] Static `evaluate()` grants `ep.abilities ∪ level.starting_abilities` to the `Player`.
- [ ] Backward-compat: `static_episodes(paths, world_seed, max_steps)` with no `abilities` arg is unchanged (default `()`), so existing `train levels` is byte-identical.

**Verify:** `PYTHONPATH=src python3 -m pytest tests/test_generalist_recipe.py -q` → PASS

**Steps:**

- [ ] **Step 1: Write failing tests**

```python
# tests/test_generalist_recipe.py
import numpy as np
from blueball.abilities import Ability
from blueball.ai import trainer as tr
from blueball.ai.episodes import (mixed_episodes, static_episodes,
                                   resolve_level_paths, compute_level_par)
from blueball.ai.genome import random_genome


def test_mixed_episodes_composition():
    eps = mixed_episodes(infinite_seeds=[1, 2], level_names=["tutorial_hill", "maze"],
                         gym_seeds=[4242], world_seed=1, max_steps=500,
                         abilities=("double_jump",))
    kinds = [e.kind for e in eps]
    assert kinds == ["infinite", "infinite", "static", "static", "gym"]
    # static + gym carry double-jump
    for e in eps:
        if e.kind in ("static", "gym"):
            assert e.abilities == ("double_jump",)
    # static normalization preserved
    statics = [e for e in eps if e.kind == "static"]
    assert statics[0].norm == compute_level_par(resolve_level_paths(["tutorial_hill"])[0])


def test_static_episodes_backward_compat_no_abilities():
    eps = static_episodes(resolve_level_paths(["tutorial_hill"]), world_seed=1, max_steps=100)
    assert eps[0].abilities == ()


def test_static_evaluate_grants_episode_abilities(monkeypatch):
    captured = {}
    real_player = tr.Player
    def spy(*a, **kw):
        captured["abilities"] = kw.get("abilities")
        return real_player(*a, **kw)
    monkeypatch.setattr(tr, "Player", spy)
    path = resolve_level_paths(["tutorial_hill"])[0]
    g = random_genome(np.random.default_rng(0))
    # evaluate signature: (idx, genome, world_seed, level_path, max_steps[, abilities])
    tr.evaluate((0, g, 1, path, 60, ("double_jump",)))
    assert Ability.DOUBLE_JUMP in captured["abilities"]
```

- [ ] **Step 2: Run to confirm red**

Run: `PYTHONPATH=src python3 -m pytest tests/test_generalist_recipe.py -q`
Expected: FAIL (`mixed_episodes` undefined; `evaluate` ignores abilities).

- [ ] **Step 3: Add `abilities` to `static_episodes` and the `mixed_episodes` constructor**

```python
# ai/episodes.py
def static_episodes(level_paths, world_seed: int, max_steps: int,
                    abilities: Sequence[str] = ()) -> list[EpisodeSpec]:
    ab = tuple(str(a) for a in abilities)
    return [EpisodeSpec(kind="static", seed=0, level_path=str(p),
                        world_seed=world_seed, max_steps=max_steps,
                        norm=compute_level_par(p), abilities=ab)
            for p in level_paths]


def mixed_episodes(infinite_seeds, level_names, gym_seeds, world_seed: int,
                   max_steps: int, abilities: Sequence[str]) -> list[EpisodeSpec]:
    """Generalist objective: infinite + static + gym, in that order. Static and
    gym are granted `abilities` (double-jump per the global invariant); infinite
    is granted them too (Track D's path honors EpisodeSpec.abilities). Static
    keeps per-level par normalization; infinite/gym stay norm=1.0."""
    ab = tuple(str(a) for a in abilities)
    eps: list[EpisodeSpec] = []
    inf = infinite_episodes(infinite_seeds, world_seed=world_seed, max_steps=max_steps)
    eps += [replace(e, abilities=ab) for e in inf]          # from dataclasses import replace
    eps += static_episodes(resolve_level_paths(level_names),
                           world_seed=world_seed, max_steps=max_steps, abilities=ab)
    eps += gym_episodes(gym_seeds, world_seed=world_seed, max_steps=max_steps, abilities=ab)
    return eps
```

- [ ] **Step 4: Make static `evaluate()` honor `ep.abilities`**

In `ai/trainer.py`, `evaluate()` unpacks `(idx, genome, world_seed, level_path, max_steps)`. Make the 6th element optional and union it with the level's `starting_abilities` (present after Task 0):

```python
def evaluate(args: tuple) -> tuple[int, float]:
    idx, genome, world_seed, level_path, max_steps, *rest = args
    ep_abilities = frozenset(Ability(a) for a in (rest[0] if rest else ()))
    world = World(seed=int(world_seed))
    register_collisions(world.space, world_ref=world)
    level = load_level(level_path)
    granted = ep_abilities | frozenset(level.meta.starting_abilities)  # Task 0 field
    player = Player(agent=FTNNAgent(genome), spawn_xy=level.spawn,
                    abilities=set(granted))
    # … rest unchanged …
```

Also update the `evaluate_episodes` static-dispatch to pass `ep.abilities`:

```python
        else:  # static
            _, raw = evaluate(
                (idx, genome, ep.world_seed, ep.level_path, ep.max_steps, ep.abilities))
```

- [ ] **Step 5: Run tests to green**

Run: `PYTHONPATH=src python3 -m pytest tests/test_generalist_recipe.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/blueball/ai/episodes.py src/blueball/ai/trainer.py tests/test_generalist_recipe.py
git commit -m "feat(ai): mixed_episodes generalist objective + static episode ability grant"
```

---

## Task 2: Warm-start genome in `train()`

**Goal:** Optionally seed the initial GA population with a known genome (Track D's mover), so generation 0 already contains a strong individual.

**Files:**
- Modify: `src/blueball/ai/trainer.py` (`train()` signature + population seeding at line ~325)
- Test: `tests/test_generalist_recipe.py`

**Acceptance Criteria:**
- [ ] `train(..., init_genome=<np.ndarray>)` places that genome at `population[0]`; the rest stay random.
- [ ] `init_genome=None` (default) is byte-identical to today (same `ga_rng` draw sequence — seed it BEFORE drawing the random remainder so existing runs are unchanged).
- [ ] A wrong-length `init_genome` raises a clear `ValueError`.
- [ ] Gen-0 best fitness ≥ the init genome's own fitness on the same objective.

**Verify:** `PYTHONPATH=src python3 -m pytest tests/test_generalist_recipe.py -k warm_start -q` → PASS

**Steps:**

- [ ] **Step 1: Write failing tests**

```python
def test_warm_start_places_genome_first():
    from blueball.ai.trainer import train, GENOME_SIZE  # GENOME_SIZE re-exported via ftnn
    from blueball.ai.ftnn import GENOME_SIZE as GZ
    seed = np.zeros(GZ); seed[0] = 1.0
    captured = {}
    def grab(gen, best, pop): captured.setdefault("gen0_pop0", pop[0].copy())
    train(pop_size=4, generations=1, infinite_seed=1, ga_seed=0,
          max_steps=60, init_genome=seed, on_generation=grab)
    assert np.array_equal(captured["gen0_pop0"], seed)


def test_warm_start_none_is_unchanged():
    from blueball.ai.trainer import train
    a = train(pop_size=4, generations=1, infinite_seed=1, ga_seed=0, max_steps=60)
    b = train(pop_size=4, generations=1, infinite_seed=1, ga_seed=0, max_steps=60,
              init_genome=None)
    assert a.history[-1]["best"] == b.history[-1]["best"]


def test_warm_start_wrong_length_raises():
    from blueball.ai.trainer import train
    with __import__("pytest").raises(ValueError):
        train(pop_size=4, generations=1, infinite_seed=1, max_steps=60,
              init_genome=np.zeros(3))
```

- [ ] **Step 2: Run to confirm red**

Run: `PYTHONPATH=src python3 -m pytest tests/test_generalist_recipe.py -k warm_start -q`
Expected: FAIL (`init_genome` not a param).

- [ ] **Step 3: Implement**

Add `init_genome: np.ndarray | None = None` to `train()`'s keyword args. Replace the seeding line:

```python
population = [random_genome(ga_rng) for _ in range(pop_size)]
if init_genome is not None:
    init = np.asarray(init_genome, dtype=np.float64)
    if init.shape[0] != GENOME_SIZE:
        raise ValueError(f"init_genome length {init.shape[0]} != GENOME_SIZE {GENOME_SIZE}")
    population[0] = init.copy()
```

(Place AFTER the random draw so the `ga_rng` stream for the rest is identical to a no-warm-start run → `init_genome=None` byte-identical.)

- [ ] **Step 4: Run to green**

Run: `PYTHONPATH=src python3 -m pytest tests/test_generalist_recipe.py -k warm_start -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/blueball/ai/trainer.py tests/test_generalist_recipe.py
git commit -m "feat(train): optional warm-start genome seeds the initial population"
```

---

## Task 3: `train generalist` CLI + config + run.json per-kind scores

**Goal:** Wire everything into `python main.py train generalist`, with mix-count config keys, a generalist run-dir name, and a `run.json` that records the mixed set and per-episode-kind scores of the final best.

**Files:**
- Modify: `src/blueball/config.py` (mix-count keys)
- Modify: `src/blueball/ai/persistence.py` (`run_dir_name` generalist variant)
- Modify: `src/blueball/cli.py` (`cmd_train_generalist` + subparser; per-kind scoring)
- Test: `tests/test_generalist_recipe.py`

**Acceptance Criteria:**
- [ ] `train generalist --pop … --gens … [--init genome.npy]` runs end-to-end and writes `final_best.npy` + `run.json`.
- [ ] `run.json` includes the mixed episode set (kinds + seeds/levels) and `per_kind_scores` (infinite/static-per-level/gym) for the final best.
- [ ] Default objective is `aggregate="min"`; `--aggregate mean_std` selectable.
- [ ] `--abilities` defaults to `double_jump`; `--init` loads a `.npy` warm-start.
- [ ] Config keys (`GENERALIST_INFINITE_SEEDS`, `GENERALIST_GYM_SEEDS`, level list) are distinct from Track C's `NET_VERSION` key.

**Verify:** `PYTHONPATH=src python3 main.py train generalist --pop 6 --gens 2 --max-steps 300 --workers 4` → prints a run dir; `run.json` has `per_kind_scores`.

**Steps:**

- [ ] **Step 1: Config keys**

```python
# config.py (new block, distinct from Track C keys)
GENERALIST_INFINITE_SEEDS = 4    # how many infinite seeds in the mix
GENERALIST_GYM_SEEDS      = 4    # how many gym chain seeds in the mix
GENERALIST_LEVELS         = ("tutorial_hill", "speed_run", "lava_rising",
                             "vertical_climb", "maze")
```

- [ ] **Step 2: `run_dir_name` generalist variant**

Add a `generalist: bool = False` (or `num_generalist`) branch yielding e.g. `genL5_w1_<ts>`:

```python
if generalist:
    return f"gen{num_levels}_w{world_seed}_{timestamp}"
```

- [ ] **Step 3: Failing test for per-kind scoring helper**

```python
def test_per_kind_scores_groups_by_kind():
    from blueball.cli import _per_kind_scores
    from blueball.ai.episodes import mixed_episodes
    import numpy as np
    from blueball.ai.genome import random_genome
    eps = mixed_episodes([1], ["tutorial_hill"], [4242], world_seed=1, max_steps=120,
                         abilities=("double_jump",))
    g = random_genome(np.random.default_rng(0))
    scores = _per_kind_scores(g, eps)
    assert set(scores) >= {"infinite", "gym"} and "static:tutorial_hill" in scores
```

- [ ] **Step 4: Implement `_per_kind_scores` + `cmd_train_generalist`**

```python
# cli.py
def _per_kind_scores(genome, episodes) -> dict:
    """Score the genome per episode-kind (raw, un-normalized) for run.json."""
    from .ai.trainer import evaluate, evaluate_infinite, evaluate_gym
    out = {}
    for ep in episodes:
        if ep.kind == "infinite":
            _, r = evaluate_infinite((0, genome, ep.seed, ep.world_seed, ep.max_steps, ep.abilities))
            out.setdefault("infinite", []).append(r)
        elif ep.kind == "gym":
            _, r = evaluate_gym((0, genome, ep.seed, ep.world_seed, ep.max_steps, ep.abilities))
            out.setdefault("gym", []).append(r)
        else:
            _, r = evaluate((0, genome, ep.world_seed, ep.level_path, ep.max_steps, ep.abilities))
            name = Path(ep.level_path).stem
            out[f"static:{name}"] = r
    for k in ("infinite", "gym"):
        if k in out: out[k] = float(np.mean(out[k]))
    return out


def cmd_train_generalist(args) -> int:
    import numpy as np
    from .ai.episodes import generate_seeds, mixed_episodes
    from .ai.trainer import train
    abilities = tuple(a.strip() for a in args.abilities.split(",") if a.strip())
    inf_seeds = generate_seeds(args.infinite_seed, config.GENERALIST_INFINITE_SEEDS)
    gym_seeds = generate_seeds(args.gym_seed, config.GENERALIST_GYM_SEEDS)
    levels = list(config.GENERALIST_LEVELS)
    episodes = mixed_episodes(inf_seeds, levels, gym_seeds, world_seed=args.world_seed,
                              max_steps=args.max_steps, abilities=abilities)
    init = np.load(args.init) if args.init else None
    run_dir = _run_dir(world_seed=args.world_seed, num_levels=len(levels), generalist=True)
    print(f"Training {args.pop}x{args.gens} GENERALIST "
          f"(inf={len(inf_seeds)} static={len(levels)} gym={len(gym_seeds)}) "
          f"agg={args.aggregate} abilities={abilities} -> {run_dir}")
    with _pool(args.workers) as (_, map_fn):
        result = train(pop_size=args.pop, generations=args.gens, episodes=episodes,
                       aggregate=args.aggregate, ga_seed=args.ga_seed,
                       world_seed=args.world_seed, max_steps=args.max_steps,
                       map_fn=map_fn, save_dir=run_dir, init_genome=init)
    best = result.best_genome
    pk = _per_kind_scores(best, episodes)
    (run_dir / "per_kind_scores.json").write_text(__import__("json").dumps(pk, indent=2))
    final = result.history[-1]
    print(f"Done. gen {final['gen']}: best={final['best']:.1f}  per-kind={pk}")
    return 0
```

(If `TrainingRunWriter` is the single owner of `run.json`, write `per_kind_scores` there instead by passing `pk` into the writer; otherwise the sidecar `per_kind_scores.json` above is acceptable. Prefer folding into `run.json` if the writer API allows an `extra` dict.)

- [ ] **Step 5: Subparser**

```python
p_gen = tsub.add_parser("generalist", help="combined infinite+static+gym generalist")
_add_common_train_args(p_gen, max_steps_default=config.MAX_STEPS)
p_gen.add_argument("--infinite-seed", type=int, default=config.INFINITE_RUN_SEED)
p_gen.add_argument("--gym-seed", type=int, default=config.GYM_SEED)
p_gen.add_argument("--abilities", type=str, default="double_jump")
p_gen.add_argument("--aggregate", choices=["min", "mean_std"], default="min")
p_gen.add_argument("--init", type=str, default=None, help="warm-start genome .npy (Track D)")
p_gen.set_defaults(func=cmd_train_generalist)
```

- [ ] **Step 6: Run unit + a live smoke**

Run: `PYTHONPATH=src python3 -m pytest tests/test_generalist_recipe.py -q` → PASS
Run: `PYTHONPATH=src python3 main.py train generalist --pop 6 --gens 2 --max-steps 300 --workers 4`
Expected: prints a run dir; `genomes/genL5_*/final_best.npy` and per-kind scores exist.

- [ ] **Step 7: Commit**

```bash
git add src/blueball/config.py src/blueball/ai/persistence.py src/blueball/cli.py tests/test_generalist_recipe.py
git commit -m "feat(cli): train generalist (mixed min objective, warm-start, per-kind run.json)"
```

---

## Task 4: Acceptance — backward-compat + warm-start + end-to-end

**Goal:** Lock the acceptance criteria from the brief with explicit tests, and confirm the single-mode trainers are untouched.

**Files:**
- Test: `tests/test_generalist_recipe.py`

**Acceptance Criteria:**
- [ ] Backward-compat: `train infinite` / `train levels` / `train gym` produce byte-identical best fitness to a pre-Track-B baseline at a fixed tiny config (regression-pin the numbers).
- [ ] Warm-start verified: seeding from a known mover genome yields gen-0 best ≥ that mover's own infinite score.
- [ ] `train generalist` end-to-end writes `final_best.npy` + per-kind scores (asserted in a `tmp_path` run with `pop=4, gens=1`).

**Verify:** `PYTHONPATH=src python3 -m pytest tests/test_generalist_recipe.py -q` → PASS

**Steps:**

- [ ] **Step 1: Backward-compat pins**

```python
def test_train_infinite_unchanged_small():
    from blueball.ai.trainer import train
    r = train(pop_size=4, generations=2, infinite_seed=1234, ga_seed=0, max_steps=200)
    # Pin to the value observed on master BEFORE Task 0-3 (fill in once measured).
    assert r.history[-1]["best"] == EXPECTED_INFINITE_BEST
```

(Measure `EXPECTED_INFINITE_BEST` on the base commit first, then hard-code it. Repeat for `train levels` on one small level and `train gym` on one seed.)

- [ ] **Step 2: Warm-start ≥ mover**

```python
def test_warm_start_reproduces_mover_score():
    import numpy as np
    from blueball.ai.trainer import train, evaluate_infinite
    from blueball.ai.genome import random_genome
    mover = random_genome(np.random.default_rng(7))
    _, mover_score = evaluate_infinite((0, mover, 1234, 1, 300, ("double_jump",)))
    r = train(pop_size=6, generations=1, infinite_seed=1234, ga_seed=0, max_steps=300,
              init_genome=mover)
    assert r.history[0]["best"] >= mover_score - 1e-6
```

- [ ] **Step 3: End-to-end generalist writes artifacts** — run `train generalist` via the CLI entrypoint into a `tmp_path` `GENOMES_ROOT`, assert `final_best.npy` + per-kind scores exist.

- [ ] **Step 4: Full suite** — `PYTHONPATH=src python3 -m pytest -q` → all green.

- [ ] **Step 5: Commit**

```bash
git add tests/test_generalist_recipe.py
git commit -m "test(generalist): backward-compat pins + warm-start + e2e acceptance"
```

---

## Beyond the plan (compute, not code): the definitive run

Once Tasks 0–4 land, run the real generalist (the actual goal of the whole effort):

```bash
PYTHONPATH=src python3 main.py train generalist --pop 80 --gens 300 \
    --aggregate min --init <Track-D final_best.npy> --workers 12
```

**Success (the H2 verdict):** beats the 0/5 baseline — completes the easy static levels (tutorial_hill, speed_run) while holding a non-trivial Infinite distance. Watch the best genome with `python main.py watch-best <run-dir> --level maze` (Track A). This is also where Track C's stage-2 verdict runs (`BLUEBALL_NET_VERSION=v2` vs `v1` through this same `min` recipe).

## Open knobs (tune, don't block)
Mix ratios (`GENERALIST_INFINITE_SEEDS` / `GENERALIST_GYM_SEEDS` / level subset), whether to ramp the gym share over generations, and `max_steps` per kind. Start even-ish; iterate based on per-kind scores in `run.json`.
