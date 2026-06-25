"""Bounce/efficiency behavior metrics for A/B comparison.

Pure functions over the per-episode counters Player accumulates
(jumps_fired, airborne_steps) plus steps and progress_x. Used by the watch
HUD so a trained genome can be measured, not just eyeballed."""

from __future__ import annotations


def behavior_metrics(jumps_fired: int, airborne_steps: int, steps: int,
                     progress_x: float) -> dict[str, float]:
    """Bounce metrics for one episode.

    Returns a dict with:
      - jumps_per_100px: jumps fired per 100px of forward progress (lower = less
        bounce-spam). 0.0 when progress_x <= 0.
      - airtime_pct: percentage of update ticks spent airborne, 0-100. 0.0 when
        steps <= 0.

    progress_x is expected non-negative (callers pass max_x - spawn_x, which only
    grows); non-positive values yield 0.0 rather than a negative/garbage rate."""
    jumps_per_100px = (100.0 * jumps_fired / progress_x) if progress_x > 0 else 0.0
    airtime_pct = (100.0 * airborne_steps / steps) if steps > 0 else 0.0
    return {"jumps_per_100px": jumps_per_100px, "airtime_pct": airtime_pct}
