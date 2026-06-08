"""Training-isolation guard.

The AI/GA *headless* training path must never construct a Renderer/RenderCore:
training runs without visuals, so pulling in rendering would waste work and
risk coupling the trainer to the (window-bound) render stack.

Discovered headless entrypoint: `blueball.ai.trainer.train(...)`, the GA
generation loop. It drives `evaluate_episodes` -> `evaluate_infinite` /
`evaluate`, all of which build a headless `World` + `Player(FTNNAgent(...))`
and step physics directly. None of them touch the renderer.

We monkeypatch both `Renderer.__init__` and `RenderCore.__init__` to record
any construction, then run a TINY real training step (pop_size=2, 1 gen, 10
steps, serial map) and assert nothing was recorded. This is robust even if
some module transitively imports the render package: we trip only on actual
instantiation, not import.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from blueball.ai.trainer import train
from blueball.render.core import RenderCore
from blueball.render.renderer import Renderer


def test_infinite_training_never_instantiates_renderer(monkeypatch):
    calls = []
    monkeypatch.setattr(Renderer, "__init__",
                        lambda self, *a, **k: calls.append("Renderer"))
    monkeypatch.setattr(RenderCore, "__init__",
                        lambda self, *a, **k: calls.append("RenderCore"))

    # Smallest possible streamed Infinite-Run training run.
    result = train(
        pop_size=2,
        generations=1,
        infinite_seed=1,
        ga_seed=0,
        max_steps=10,
    )

    assert result.history, "training produced no generation history"
    assert calls == [], f"training instantiated rendering: {calls}"


def test_static_level_training_never_instantiates_renderer(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(Renderer, "__init__",
                        lambda self, *a, **k: calls.append("Renderer"))
    monkeypatch.setattr(RenderCore, "__init__",
                        lambda self, *a, **k: calls.append("RenderCore"))

    # Tiny in-memory static level so the static evaluate() path is exercised
    # too. A single flat strip plus a spawn is enough to step physics.
    import json

    level = {
        "name": "isolation-probe",
        "background": "#202028",
        "ground": "#666c70",
        "spawn": [80, 540],
        "chunks": [
            {"type": "flat", "width_tiles": 8},
        ],
    }
    level_path = tmp_path / "probe.json"
    level_path.write_text(json.dumps(level))

    result = train(
        pop_size=2,
        generations=1,
        level_path=level_path,
        ga_seed=0,
        max_steps=10,
    )

    assert result.history, "training produced no generation history"
    assert calls == [], f"training instantiated rendering: {calls}"
