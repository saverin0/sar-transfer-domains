"""One fixed rule, the same for every domain: two SAR channels -> encoder input.

Fixed 2026-09-26 before any new-domain number (no tuning per domain):

    mode "vv_vh_diff" (default): three channels (co-pol, cross-pol, co - cross),
        each standardised with mean/std from the TRAINING split. In dB the
        third channel is the log polarisation ratio, the usual third channel
        for dual-pol Sentinel-1.
    mode "vv_grey": co-pol only, standardised and copied three times -- the
        glacier study's input (one grey channel), kept for comparison.

No-data pixels (NaN) become 0 after standardisation, i.e. the training mean.
The decoder's image skip is the standardised co-pol channel mapped to [0, 1]
by clip((z + 3) / 6, 0, 1), which does not depend on the units.
"""

from __future__ import annotations

import numpy as np

MODES = ("vv_vh_diff", "vv_grey")


def channel_stats(images: np.ndarray, max_chips: int = 2000, seed: int = 0) -> dict:
    """Mean/std of co, cross and co - cross over finite pixels of up to `max_chips` chips."""
    rng = np.random.default_rng(seed)
    n = len(images)
    idx = np.sort(rng.choice(n, min(max_chips, n), replace=False))
    x = np.asarray(images[idx], np.float32)
    co, cross = x[:, 0], x[:, 1]
    out = {}
    for name, v in (("co", co), ("cross", cross), ("diff", co - cross)):
        f = v[np.isfinite(v)]
        if f.size == 0:
            out[name] = (0.0, 1.0)                       # channel absent everywhere
        else:
            out[name] = (float(f.mean()), float(max(f.std(), 1e-6)))
    return out


def _z(v: np.ndarray, ms: tuple[float, float]) -> np.ndarray:
    z = (v - ms[0]) / ms[1]
    return np.where(np.isfinite(z), z, 0.0).astype(np.float32)


def encoder_input(images: np.ndarray, stats: dict, mode: str = "vv_vh_diff") -> np.ndarray:
    """(N, 2, H, W) SAR -> (N, 3, H, W) float32 standardised encoder input."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} not in {MODES}")
    x = np.asarray(images, np.float32)
    co, cross = x[:, 0], x[:, 1]
    if mode == "vv_grey":
        z = _z(co, stats["co"])
        return np.stack([z, z, z], axis=1)
    return np.stack([_z(co, stats["co"]), _z(cross, stats["cross"]), _z(co - cross, stats["diff"])], axis=1)


def skip_image(images: np.ndarray, stats: dict) -> np.ndarray:
    """(N, 2, H, W) SAR -> (N, 1, H, W) float32 in [0, 1] for the decoder's image branch."""
    z = _z(np.asarray(images[:, 0], np.float32), stats["co"])
    return np.clip((z + 3.0) / 6.0, 0.0, 1.0)[:, None]
