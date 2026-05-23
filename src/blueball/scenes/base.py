"""Scene base class. A scene owns one frame of the game's loop."""

from __future__ import annotations

import abc


class Scene(abc.ABC):
    @abc.abstractmethod
    def handle_events(self, events) -> "Scene | None":
        """Return self to continue, a new scene to switch, or None to exit."""

    @abc.abstractmethod
    def update(self, frame_dt: float) -> None: ...

    @abc.abstractmethod
    def draw(self) -> None: ...
