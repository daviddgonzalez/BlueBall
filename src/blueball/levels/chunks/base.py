"""Chunk base class and shared constants."""

from __future__ import annotations

import abc

TILE = 32  # world units per chunk tile

# Populated by chunk submodules at import time.
CHUNK_REGISTRY: dict[str, type["Chunk"]] = {}


def register_chunk(name: str):
    """Decorator: registers a chunk class under the given name."""

    def deco(cls):
        CHUNK_REGISTRY[name] = cls
        return cls

    return deco


class Chunk(abc.ABC):
    # Sampler integration. Default values keep existing chunks sampler-eligible
    # at trivial difficulty until concrete subclasses override.
    difficulty: int = 0
    sampler_include: bool = True

    @classmethod
    def random_params(cls, rng) -> dict:
        """Return a kwargs dict the sampler should pass to __init__.
        Default: use the chunk's __init__ defaults."""
        return {}

    @abc.abstractmethod
    def build(self, world, x_offset: float) -> float:
        """Materialize the chunk's bodies and entities into `world`, anchored at
        `x_offset` (left edge of the chunk). Returns the chunk's width in world
        units so the level loader can place the next chunk.
        """
