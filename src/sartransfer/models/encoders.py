"""Frozen encoders behind one interface: tiles in, (N, gh, gw, dim) patch grid out.

Every backend receives the SAME input (see domains.s1input); each model's own
preprocessing (ImageNet / CLIP statistics) is bypassed on purpose, so encoders
are compared on identical inputs.

Backends, each checked against the model's own public code (2026-09-23):

- `dinov3` (Hugging Face transformers). ViT-L/16 at 512 px returns 1029 tokens:
  32x32 patches + 5 prefix (1 CLS + 4 registers). The prefix count is inferred
  from token count vs grid size, never hard-coded.
- `radio` (C-RADIOv4, trust_remote_code). `model(x)` returns (summary, spatial);
  spatial holds patch tokens only, NLC. It normally expects [0, 1] input and
  applies CLIP mean/std itself; `make_preprocessor_external()` switches that off
  so it takes already-standardised input. `preferred_resolution` is 512.

Both models are loaded from a pinned commit: DINOv3 at the snapshot every run in
this repository used, C-RADIO at a revision whose remote code is SHA-256-checked
against a reviewed copy before anything of it runs.
"""

from __future__ import annotations

from dataclasses import dataclass

from pathlib import Path

import numpy as np
import torch


@dataclass
class EncoderSpec:
    model_id: str
    name: str
    kind: str                       # dinov3 | radio
    trust_remote_code: bool = False
    batch: int = 32                 # tiles per forward pass at 512 px on an A100-40GB
    revision: str | None = None     # pinned commit; remote-code models are also hash-checked before loading


def verify_remote_code(spec: EncoderSpec, token: str | None = None) -> Path:
    """Refuse to run remote code that is not the reviewed copy.

    Downloads only the *.py files and config.json of the pinned revision into the
    HF cache (the same snapshot transformers loads the model code from) and
    compares the SHA-256 of every .py below the snapshot folder, subfolders
    included, and of config.json (its auto_map picks the module), keyed by
    relative path, with `remote_code_hashes.REMOTE_CODE_HASHES`. Any extra,
    missing or changed file raises before anything is executed.

    transformers runs its own copies of these files from HF_MODULES_CACHE and
    reuses a copy that exists, without comparing it to the snapshot. So the
    copies of this revision are deleted here, after the check; loading then
    copies them again from the checked snapshot.
    """
    import hashlib
    import shutil

    from huggingface_hub import snapshot_download

    from .remote_code_hashes import REMOTE_CODE_HASHES

    if spec.revision is None:
        raise ValueError(f"{spec.model_id}: remote code needs a pinned revision")
    expected = REMOTE_CODE_HASHES[spec.model_id]
    d = Path(snapshot_download(spec.model_id, revision=spec.revision, allow_patterns=["*.py", "config.json"],
                               token=token))
    got = {q.relative_to(d).as_posix(): hashlib.sha256(q.read_bytes()).hexdigest()
           for q in [*d.rglob("*.py"), d / "config.json"] if q.is_file()}
    bad = sorted(k for k in set(expected) | set(got) if expected.get(k) != got.get(k))
    if bad:
        raise RuntimeError(f"{spec.model_id}@{spec.revision[:12]}: remote code differs from the reviewed copy: {bad}")
    from transformers.utils import HF_MODULES_CACHE

    stale = [p for p in Path(HF_MODULES_CACHE).glob(f"transformers_modules/**/{spec.revision}") if p.is_dir()]
    for p in stale:
        shutil.rmtree(p)
    print(f"remote code verified: {len(got)} files == reviewed copy ({spec.model_id}@{spec.revision[:12]}); "
          f"{len(stale)} cached module copy removed, loading copies it again from the checked files")
    return d


ENCODERS = {
    # revision = the only snapshot in the Drive model cache that every run here used (read 2026-09-29)
    "dinov3-l-sat": EncoderSpec("facebook/dinov3-vitl16-pretrain-sat493m", "dinov3-l-sat", "dinov3",
                                revision="f692fa42da72c6797b67cd73494a168d1120d3ee"),
    "cradio-v4-h": EncoderSpec("nvidia/C-RADIOv4-H", "cradio-v4-h", "radio",
                               trust_remote_code=True, batch=16, revision="0057b339059c0b9e1b4ba996f975410ebbfdfcc8"),
}

# (import name, pip requirement) per encoder kind, installed only when the import is missing.
# C-RADIO's remote code is import-checked by transformers, so open_clip must exist even though
# its adaptor is never used here. Exact versions, pinned 2026-09-29 to the then current
# releases; the runs did not record the versions they installed.
EXTRA_PACKAGES = {
    "radio": [("timm", "timm==1.0.30"), ("einops", "einops==0.8.2"), ("open_clip", "open_clip_torch==3.3.0")],
}


def ensure_packages(kind: str) -> None:
    """Install the pinned extra packages of one encoder kind, if their import is missing."""
    import importlib
    import importlib.util
    import subprocess
    import sys

    missing = [pip for mod, pip in EXTRA_PACKAGES.get(kind, [])
               if importlib.util.find_spec(mod) is None]
    if missing:
        print("installing", missing)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
        importlib.invalidate_caches()


def package_versions() -> dict[str, str | None]:
    """Versions of the libraries that shape features and heads, stored with every run."""
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for p in ("torch", "transformers", "huggingface_hub", "timm", "einops", "open_clip_torch", "numpy"):
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    return out


class FrozenEncoder:
    """A frozen backbone that maps normalised tiles to a (gh, gw, dim) patch grid."""

    def __init__(self, spec: EncoderSpec, token: str | None = None,
                 dtype: torch.dtype = torch.bfloat16, device: str = "cuda"):
        self.spec = spec
        self.device = device
        self.dtype = dtype
        self.patch = 16
        self.dim = 0
        self._n_prefix: int | None = None

        if spec.kind == "dinov3":
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(spec.model_id, token=token, dtype=dtype,
                                                   revision=spec.revision)
            cfg = self.model.config
            self.patch = int(getattr(cfg, "patch_size", 16))
            self.dim = int(getattr(cfg, "hidden_size", 0))
        elif spec.kind == "radio":
            from transformers import AutoModel
            verify_remote_code(spec, token)          # pinned commit, files == reviewed copy
            self.model = AutoModel.from_pretrained(spec.model_id, token=token,
                                                   trust_remote_code=True, revision=spec.revision)
            self.model.make_preprocessor_external()   # we feed standardised input
            self.patch = int(self.model.patch_size)
        else:
            raise ValueError(f"unknown encoder kind {spec.kind!r}")

        self.model = self.model.to(device).eval()
        print(f"{spec.name}: {sum(q.numel() for q in self.model.parameters()) / 1e6:.1f} M parameters loaded")
        # Frozen. Nothing trainable sits in front during extraction, so
        # inference_mode below is safe.
        self.model.requires_grad_(False)

    def __repr__(self) -> str:
        return (f"FrozenEncoder({self.spec.name}, kind={self.spec.kind}, patch={self.patch}, "
                f"dim={self.dim}, n_prefix={self._n_prefix})")

    def _grid(self, tokens: torch.Tensor, tile: int) -> torch.Tensor:
        """(B, T, D) -> (B, gh, gw, D), dropping any prefix tokens."""
        gh = gw = tile // self.patch
        n_patch = gh * gw
        n_prefix = tokens.shape[1] - n_patch
        if n_prefix < 0 or n_prefix > 16:
            raise RuntimeError(
                f"{tokens.shape[1]} tokens for a {tile}px tile at patch {self.patch}: "
                f"expected {n_patch} patch tokens plus a small prefix, got prefix {n_prefix}")
        if self._n_prefix is None:
            self._n_prefix = n_prefix
        elif self._n_prefix != n_prefix:
            raise RuntimeError(f"prefix count changed: {self._n_prefix} -> {n_prefix}")
        self.dim = int(tokens.shape[-1])
        return tokens[:, n_prefix:, :].reshape(tokens.shape[0], gh, gw, -1)

    def _tokens(self, b: torch.Tensor) -> torch.Tensor:
        """One batch (B, 3, H, W) -> token sequence (B, T, D)."""
        kind = self.spec.kind
        if kind == "dinov3":
            return self.model(pixel_values=b.to(self.dtype)).last_hidden_state
        with torch.autocast("cuda", dtype=self.dtype):
            if kind == "radio":
                out = self.model(b)
                spatial = out[1] if isinstance(out, (tuple, list)) else out.features
                return spatial                               # NLC, patch tokens only
        raise ValueError(kind)

    @property
    def n_layers(self) -> int:
        """Number of transformer blocks."""
        kind = self.spec.kind
        if kind == "dinov3":
            return int(self.model.config.num_hidden_layers)
        if kind == "radio":
            inner = getattr(self.model, "radio_model", self.model)
            return len(inner.model.blocks)
        raise NotImplementedError(f"layer taps not supported for {kind}")

    def _layer_tokens(self, b: torch.Tensor, layers: list[int]) -> dict[int, torch.Tensor]:
        """Token sequences after block k (1-based), final norm applied, for each k.

        dinov3: HF `hidden_states[k]` is the output of block k (index 0 = the
                embeddings); the model's own backbone variant applies `norm` to
                intermediates, so the same is done here. k = n_layers equals the
                usual `last_hidden_state`.
        radio:  `forward_intermediates(indices=[k-1], norm=True)`; returns spatial
                tokens only (no prefix), NLC.
        """
        kind = self.spec.kind
        if kind == "dinov3":
            out = self.model(pixel_values=b.to(self.dtype), output_hidden_states=True)
            hs = out.hidden_states
            return {k: self.model.norm(hs[k]) for k in layers}
        if kind == "radio":
            inner = getattr(self.model, "radio_model", self.model)
            with torch.autocast("cuda", dtype=self.dtype):
                feats = inner.forward_intermediates(b, indices=[k - 1 for k in layers], norm=True,
                                                    output_fmt="NLC", intermediates_only=True)
            return {k: f for k, f in zip(layers, feats)}
        raise NotImplementedError(f"layer taps not supported for {kind}")

    @torch.inference_mode()
    def encode_layers(self, tiles: np.ndarray, layers: list[int],
                      batch: int | None = None) -> dict[int, np.ndarray]:
        """Like `encode`, but returns {layer: (N, gh, gw, dim) fp16} for several layers."""
        if tiles.ndim != 4 or tiles.shape[1] != 3:
            raise ValueError(f"expected (N, 3, H, W), got {tiles.shape}")
        batch = batch or self.spec.batch
        tile = tiles.shape[-1]
        out: dict[int, list] = {k: [] for k in layers}
        x = torch.from_numpy(tiles)
        for i in range(0, len(x), batch):
            b = x[i:i + batch].to(self.device, non_blocking=True)
            for k, tok in self._layer_tokens(b, layers).items():
                out[k].append(to_host_fp16(self._grid(tok, tile)))
        return {k: np.concatenate(v) for k, v in out.items()}

    @torch.inference_mode()
    def encode(self, tiles: np.ndarray, batch: int | None = None,
               layer: int | tuple[int, ...] | None = None) -> np.ndarray:
        """tiles (N, 3, tile, tile) float32 -> features (N, gh, gw, dim) float16.

        `layer` (1-based block index) taps an intermediate layer, final norm
        applied. None = the model's final output. A tuple of layers returns them
        concatenated on the channel axis (N, gh, gw, len(layers) * dim), in the
        given order.
        """
        if tiles.ndim != 4 or tiles.shape[1] != 3:
            raise ValueError(f"expected (N, 3, H, W), got {tiles.shape}")
        if isinstance(layer, (tuple, list)):
            d = self.encode_layers(tiles, list(layer), batch=batch)
            return np.concatenate([d[k] for k in layer], axis=-1)
        if layer is not None:
            return self.encode_layers(tiles, [layer], batch=batch)[layer]
        batch = batch or self.spec.batch
        tile = tiles.shape[-1]
        out = []
        x = torch.from_numpy(tiles)
        for i in range(0, len(x), batch):
            b = x[i:i + batch].to(self.device, non_blocking=True)
            tok = self._tokens(b)
            out.append(to_host_fp16(self._grid(tok, tile)))
        return np.concatenate(out) if out else np.empty((0,), dtype=np.float16)


def to_host_fp16(t: torch.Tensor) -> np.ndarray:
    """Features -> float16 numpy, cast on the device first (speed-up 2026-09-27).

    Copies half the bytes and skips numpy's float16 conversion on the CPU. Same
    values as the older `t.float().cpu().numpy().astype(np.float16)`: bf16 -> fp32
    is exact, and both paths round fp32 -> fp16 to nearest-even.
    """
    return t.to(torch.float16).cpu().numpy()
