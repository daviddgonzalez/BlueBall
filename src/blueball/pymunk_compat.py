"""Lets the pymunk 7 collision API run on pymunk 6.x.

The browser build (pygbag) only ships pymunk 6.4, which has no
``Space.on_collision`` and no ``Arbiter.process_collision``. On 6.x this
module installs an ``on_collision`` built on ``add_collision_handler``, with
pymunk 7 semantics: a callback's return value is ignored, and setting
``arbiter.process_collision = False`` is what drops the contact. On pymunk 7
it does nothing.
"""

from __future__ import annotations

import pymunk


def _wrap(callback):
    def handler(arbiter, space, data):
        arbiter.process_collision = True
        callback(arbiter, space, data)
        return arbiter.process_collision
    return handler


def _on_collision(self, collision_type_a=None, collision_type_b=None,
                  begin=None, pre_solve=None, post_solve=None, separate=None,
                  data=None):
    if collision_type_a is None or collision_type_b is None:
        raise NotImplementedError("compat on_collision needs both collision types")
    handler = self.add_collision_handler(collision_type_a, collision_type_b)
    if begin is not None:
        handler.begin = _wrap(begin)
    if pre_solve is not None:
        handler.pre_solve = _wrap(pre_solve)
    if post_solve is not None:
        handler.post_solve = post_solve
    if separate is not None:
        handler.separate = separate
    if data:
        handler.data.update(data)


if not hasattr(pymunk.Space, "on_collision"):
    pymunk.Space.on_collision = _on_collision
