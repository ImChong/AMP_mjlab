"""Restore ``wp.context`` removed from warp-lang 1.13+ public API (mjlab 1.2 still uses it)."""

import warp as wp

if not hasattr(wp, "context"):
  from warp import _src

  wp.context = _src.context  # type: ignore[attr-defined]
