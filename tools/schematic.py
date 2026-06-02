"""Render a level's static geometry + entity markers to a PNG schematic.

Usage: .venv/bin/python tools/schematic.py <level_name> [xmax]
Saves /tmp/<level_name>_schematic.png (cropped to x<=xmax if given).
"""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import sys
import pathlib

import pygame
import pymunk

from blueball.levels.loader import load_level
from blueball.world import World

pygame.init()

name = sys.argv[1]
xmax = float(sys.argv[2]) if len(sys.argv) > 2 else float("inf")

w = World()
load_level(pathlib.Path("src/blueball/levels") / f"{name}.json", w)

segs = []
for s in w.space.shapes:
    if isinstance(s, pymunk.Segment):
        bx, by = s.body.position
        segs.append((bx + s.a.x, by + s.a.y, bx + s.b.x, by + s.b.y))

marks = []
for e in w.entities:
    t = type(e).__name__
    pos = getattr(e, "position", None)
    if pos is None and hasattr(e, "body"):
        pos = (e.body.position.x, e.body.position.y)
    if pos is not None:
        marks.append((t, pos[0], pos[1]))

segs = [s for s in segs if min(s[0], s[2]) <= xmax]
marks = [m for m in marks if m[1] <= xmax]

xs, ys = [], []
for ax, ay, bx, by in segs:
    xs += [ax, bx]; ys += [ay, by]
for _, x, y in marks:
    xs.append(x); ys.append(y)
minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
pad = 40
W_, H_ = maxx - minx + 2 * pad, maxy - miny + 2 * pad
scale = min(1500.0 / W_, 2200.0 / H_)
iw, ih = int(W_ * scale), int(H_ * scale)
surf = pygame.Surface((iw, ih))
surf.fill((20, 20, 28))

def tx(x): return int((x - minx + pad) * scale)
def ty(y): return int((y - miny + pad) * scale)

for ax, ay, bx, by in segs:
    pygame.draw.line(surf, (220, 220, 220), (tx(ax), ty(ay)), (tx(bx), ty(by)), 2)

colors = {
    "Spring": (120, 120, 255), "PushableBox": (255, 160, 60), "Goal": (255, 240, 120),
    "Door": (200, 140, 60), "Cannon": (255, 80, 80), "SwingingHazard": (255, 120, 200),
    "Lava": (255, 90, 30),
}
font = pygame.font.SysFont(None, 20)
for t, x, y in marks:
    c = colors.get(t, (150, 255, 150))
    pygame.draw.circle(surf, c, (tx(x), ty(y)), 6)
    surf.blit(font.render(t[:4], True, c), (tx(x) + 7, ty(y) - 7))

out = f"/tmp/{name}_schematic.png"
pygame.image.save(surf, out)
print("saved", out, iw, "x", ih, "| x-range", round(minx), round(maxx), "| y", round(miny), round(maxy))
