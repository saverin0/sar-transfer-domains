"""Frozen features -> linear probe + small decoder -> metrics, for any prepared domain.

PRE-REGISTERED 2026-09-26, before any new-domain number (no tuning per domain):
- encoders and layers from the glacier study: dinov3-l-sat layer 21,
  cradio-v4-h layer 32 (chosen on the glacier validation split, not here);
- input: s1input mode "vv_vh_diff", channel statistics from the training split;
- head 1, linear probe: one linear layer on the 16 px patch features, patch
  label = majority vote of its valid pixels, balanced cross-entropy, AdamW
  lr 1e-3, batch 65,536 patches (the glacier probe's recipe), trained for at
  least 30 epochs AND at least ~1,000 optimiser steps -- the glacier probe's
  2.1 M patches gave ~1,000 steps in 30 epochs; small domains would otherwise
  get only a few dozen steps; pixel predictions = bilinear upsampling of the
  patch logits;
- head 2, small decoder: row 3 of the glacier study (GridDecoder with image
  skip = s1input.skip_image), 4,000 steps, batch 8, AdamW lr 1e-3 + OneCycle,
  balanced cross-entropy ignoring 255, horizontal flips, seeds 0, 1, 2, at most
  4,000 training chips (uniform sample, seed 0);
- nothing is selected on any split, so every non-train split (validation and
  tests) is scored in the same run; validation is reported as a check only;
- metrics per split: pixel confusion -> per-class IoU, mIoU, accuracy; for
  binary tasks also IoU, F1, precision and recall of class 1; per region
  (meta "region") pooled; a majority-class baseline from the training pixels.
- one no-data rule for every domain (added 2026-09-26, still before any
  number, because the converters differed): a pixel whose co- OR cross-pol
  value is missing (NaN) is ignored (label 255) in training and scoring;
- every result row carries the manifest's SAR units; units other than dB are
  reported, not converted (the third input channel is then not a log ratio);
- optional, off by default: `exclude_train_sharing` drops training chips that
  share a value of the given meta columns (e.g. "s2_scene") with any scored
  split, for scene-disjoint scoring.

Speed (2026-09-28, before any number; results unchanged): the pixel confusion is
counted per chip on the GPU right where the predictions are made
(`_chip_conf`); split and region totals are sums of those integer counts, so
they equal the earlier CPU `_confusion` exactly (measured on the PC: the CPU
count took about 16 min per encoder for snow's 3.5 billion test pixels).
`enc_cache` (a dict the caller keeps) loads each encoder once for all datasets
instead of once per dataset (1-2 min per load from the Drive model cache).

Outputs in res_dir, all prefixed dom_<dataset>__<encoder>__L<layer>__<mode>:
    __summary.csv                  one row per head / seed / split
    __<head>[__s<seed>]__regions.csv   per split and region
    __probe.pt, __decoder__s<seed>.pt  trained heads
    __info.json                    channel stats, chip counts, timings, the run's settings and
                                   package versions (a finished run is skipped only when the
                                   requested settings and the data's created_utc match)
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .prepared import IGNORE, load_manifest, load_split

PATCH = 16
PROBE_STEPS = 1000               # optimiser steps of the glacier probe (2.1 M patches, 30 epochs)
PRE_REGISTERED_LAYERS = {"dinov3-l-sat": 21, "cradio-v4-h": 32}
# run_domain's defaults = the pre-registered settings; runs finished before 2026-09-29 stored no
# settings in their info.json and are taken to have used exactly these.
PRE_REGISTERED_SETTINGS = {"max_train_chips": 4000, "probe_max_patches": 2_000_000, "steps": 4000,
                           "batch": 8, "lr": 1e-3}


def encoder_batch(spec_batch: int, size: int) -> int:
    """Chips per encoder forward pass for `size` px chips: the spec's batch is for 512 px chips.

    Scaled by the pixel count, (512 / size) ** 2, both ways, capped at 512. For the chip sizes that
    divide 512 (128, 256, 512 px) this equals the earlier rule; for larger chips it shrinks the
    batch (992 px -> 8 for DINOv3, 4 for C-RADIO), where the earlier rule kept the 512 px batch
    (the published alpine run used 32 and 16).
    """
    return int(max(1, min(512, spec_batch * (512 / size) ** 2)))


# ------------------------------------------------------------------ metrics

def seg_metrics(conf: np.ndarray, names: list[str]) -> dict:
    """Pixel confusion (rows truth, cols prediction) -> IoU per class, mIoU, accuracy (+ binary extras)."""
    tp = np.diag(conf).astype(np.float64)
    fp, fn = conf.sum(0) - tp, conf.sum(1) - tp
    denom = tp + fp + fn
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    out = {f"IoU_{n}": float(v) for n, v in zip(names, iou)}
    out["mIoU"] = float(np.nanmean(iou)) if np.isfinite(iou).any() else np.nan
    out["accuracy"] = float(tp.sum() / max(conf.sum(), 1))
    out["n_pixels"] = int(conf.sum())
    if len(names) == 2:
        p = tp[1] / max(tp[1] + fp[1], 1)
        r = tp[1] / max(tp[1] + fn[1], 1)
        out.update(IoU_pos=float(iou[1]), F1_pos=float(2 * p * r / max(p + r, 1e-12)),
                   precision_pos=float(p), recall_pos=float(r))
    return out


def _confusion(pred: np.ndarray, truth: np.ndarray, k: int) -> np.ndarray:
    """Reference CPU count (rows truth, cols prediction, 255 left out); scoring uses _chip_conf."""
    ok = truth != IGNORE
    return np.bincount(truth[ok].astype(np.int64) * k + pred[ok].astype(np.int64),
                       minlength=k * k).reshape(k, k)


def _chip_conf(pred, truth, k: int) -> np.ndarray:
    """(B, H, W) predicted and true class ids (torch tensors, one device) -> (B, k, k) int64 confusion per chip.

    Rows truth, cols prediction, label 255 left out; summed over chips it equals _confusion()."""
    import torch

    b = pred.shape[0]
    chip = torch.arange(b, device=pred.device).view(b, 1, 1).expand_as(truth)
    ok = truth != IGNORE
    code = (chip * (k * k) + truth.long() * k + pred.long())[ok]
    return torch.bincount(code, minlength=b * k * k).view(b, k, k).cpu().numpy()


def _labels_on(labels: np.ndarray, i: int, batch: int, device: str):
    import torch

    return torch.from_numpy(np.ascontiguousarray(labels[i:i + batch])).to(device)


def _majority_conf(labels: np.ndarray, k: int, maj: int, device: str, batch: int = 64) -> np.ndarray:
    """Per-chip confusion of the constant prediction `maj`."""
    import torch

    out = []
    for i in range(0, len(labels), batch):
        truth = _labels_on(labels, i, batch, device)
        out.append(_chip_conf(torch.full_like(truth, maj), truth, k))
    return np.concatenate(out)


# ------------------------------------------------------------------ features

def _encode_split(enc, images, stats: dict, mode: str, layer: int, batch: int) -> np.ndarray:
    """(N, 2, H, W) SAR -> (N, g, g, D) fp16 features at `layer`."""
    from .s1input import encoder_input

    out = []
    for i in range(0, len(images), batch):
        out.append(enc.encode(encoder_input(images[i:i + batch], stats, mode), batch=batch, layer=layer))
    return np.concatenate(out)


def _patch_labels(classes: np.ndarray, valid: np.ndarray, patch: int, n_classes: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """(H, W) pixel labels -> one label per patch by majority vote of its valid pixels.

    Returns (labels, patch_valid), both (H // patch, W // patch); a patch without a valid
    pixel is IGNORE and not valid.
    """
    h, w = classes.shape
    if h % patch or w % patch:
        raise ValueError(f"shape {classes.shape} is not a multiple of patch {patch}")
    gh, gw = h // patch, w // patch
    blocks = (classes.reshape(gh, patch, gw, patch).transpose(0, 2, 1, 3)
              .reshape(gh, gw, patch * patch))
    vblocks = (valid.reshape(gh, patch, gw, patch).transpose(0, 2, 1, 3)
               .reshape(gh, gw, patch * patch))
    counts = np.zeros((gh, gw, n_classes), dtype=np.int32)
    for cls in range(n_classes):
        counts[..., cls] = ((blocks == cls) & vblocks).sum(axis=-1)
    labels = counts.argmax(axis=-1).astype(np.uint8)
    keep = counts.sum(axis=-1) > 0
    labels[~keep] = IGNORE
    return labels, keep


def _patch_targets(labels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    labs, keeps = [], []
    for lab in labels:
        lab = np.asarray(lab)
        l, keep = _patch_labels(np.where(lab == IGNORE, 0, lab), lab != IGNORE, patch=PATCH, n_classes=k)
        labs.append(l); keeps.append(keep)
    return np.stack(labs), np.stack(keeps)


# ------------------------------------------------------------------ heads

class _TrainSet:
    """Training chips as tensors, built once per decoder seed (from part one).

    The chips live on the GPU when they fit (fp16 features, uint8 image and labels), else in
    host memory with only the fp16 batch copied per step. `batch(idx)` returns the same values
    as a per-step numpy path (fp16 -> fp32 is exact).
    """

    def __init__(self, feats: list, raws: list, labels: list, device: str, reserve_gb: float = 20.0):
        import torch

        n, (g1, g2, d), t = len(feats), feats[0].shape, raws[0].shape[-1]
        need = n * (g1 * g2 * d * 2 + 2 * t * t)
        fits = device == "cuda" and torch.cuda.mem_get_info()[0] > need + reserve_gb * 1e9
        self.device, self.store = device, (device if device != "cuda" or fits else "cpu")
        self.feats = torch.empty((n, g1, g2, d), dtype=torch.float16, device=self.store)
        self.raw = torch.empty((n, t, t), dtype=torch.uint8, device=self.store)
        self.lab = torch.empty((n, t, t), dtype=torch.uint8, device=self.store)
        for i in range(n):
            self.feats[i] = torch.from_numpy(np.ascontiguousarray(feats[i], dtype=np.float16))
            self.raw[i] = torch.from_numpy(np.ascontiguousarray(raws[i]))
            self.lab[i] = torch.from_numpy(np.ascontiguousarray(labels[i]))
        print(f"    training tiles held on {self.store} ({need / 1e9:.1f} GB)")

    def batch(self, idx):
        import torch

        ix = torch.as_tensor(np.asarray(idx), device=self.store)
        f = self.feats.index_select(0, ix).to(self.device, non_blocking=True).float().permute(0, 3, 1, 2)
        raw = self.raw.index_select(0, ix).to(self.device, non_blocking=True)[:, None]
        y = self.lab.index_select(0, ix).to(self.device, non_blocking=True).long()
        return f, raw, y


def _train_decoder(head, feats: np.ndarray, skip_u8: np.ndarray, labels: np.ndarray, k: int, device: str,
                   steps: int, batch: int, lr: float, seed: int, log_every: int = 1000):
    """Part one's row 3 training recipe with generic class ids."""
    import torch
    import torch.nn as nn

    rng = np.random.default_rng(seed)
    counts = np.zeros(k, np.float64)
    for lab in labels:
        counts += np.bincount(lab[lab != IGNORE].ravel(), minlength=k)[:k]
    w = torch.tensor(counts.sum() / (k * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w, ignore_index=IGNORE)
    head = head.to(device)
    ts = _TrainSet(list(feats), list(skip_u8), list(labels), device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    t0, running, n = time.perf_counter(), torch.zeros((), device=device), 0
    for it in range(1, steps + 1):
        head.train()
        idx = rng.choice(len(feats), batch, replace=len(feats) < batch)
        f, raw, y = ts.batch(idx)
        if rng.random() < 0.5:
            f, raw, y = f.flip(-1), raw.flip(-1), y.flip(-1)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            loss = loss_fn(head(f, raw.float() / 255.0).float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        running += loss.detach(); n += 1
        if it % log_every == 0 or it == steps:
            print(f"    step {it:>5}  loss {float(running) / n:.4f}  {(time.perf_counter() - t0) / 60:.1f} min")
            running, n = torch.zeros((), device=device), 0
    return head


def _decoder_conf(head, feats: np.ndarray, skip_u8: np.ndarray, labels: np.ndarray, k: int, device: str,
                  batch: int = 32) -> np.ndarray:
    """Per-chip pixel confusion (N, k, k) of the decoder's argmax, counted on `device`."""
    import torch

    head.eval()
    out = []
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        for i in range(0, len(feats), batch):
            f = torch.from_numpy(feats[i:i + batch]).to(device).float().permute(0, 3, 1, 2)
            img = torch.from_numpy(skip_u8[i:i + batch])[:, None].to(device).float() / 255.0
            pred = head(f, img).float().argmax(1)
            out.append(_chip_conf(pred, _labels_on(labels, i, batch, device), k))
    return np.concatenate(out)


def _probe_conf(probe, feats: np.ndarray, labels: np.ndarray, k: int, size: int, device: str,
                batch: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Patch argmax (N, g, g) and the per-chip pixel confusion (N, k, k) of the bilinearly upsampled logits."""
    import torch
    import torch.nn.functional as F

    W = probe.weight.detach().float().to(device)
    b = probe.bias.detach().float().to(device)
    pa, cc = [], []
    with torch.inference_mode():
        for i in range(0, len(feats), batch):
            f = torch.from_numpy(feats[i:i + batch]).to(device).float()           # (B, g, g, D)
            lg = (f @ W.T + b).permute(0, 3, 1, 2)                                 # (B, K, g, g)
            pa.append(lg.argmax(1).to(torch.uint8).cpu().numpy())
            up = F.interpolate(lg, size=(size, size), mode="bilinear", align_corners=False)
            cc.append(_chip_conf(up.argmax(1), _labels_on(labels, i, batch, device), k))
    return np.concatenate(pa), np.concatenate(cc)


# ------------------------------------------------------------------ driver

def run_domain(prepared_root: str | Path, dataset: str, encoder: str, res_dir: str | Path,
               token: str | None, layer: int | None = None, seeds=(0, 1, 2), mode: str = "vv_vh_diff",
               max_train_chips: int = 4000, probe_max_patches: int = 2_000_000, steps: int = 4000,
               batch: int = 8, lr: float = 1e-3, device: str = "cuda", force: bool = False,
               exclude_train_sharing: tuple[str, ...] = (), enc_cache: dict | None = None) -> pd.DataFrame:
    """Everything for one prepared dataset and one encoder; see the module docstring.

    enc_cache: a dict kept by the caller; the encoder is loaded into it on first use and reused by later
    calls (the caller frees it). None loads the encoder for this call only and frees it after the features.
    """
    import torch

    from ..models.decoder import GridDecoder
    from ..models.encoders import ENCODERS, FrozenEncoder, ensure_packages, package_versions
    from ..probe import train_linear_probe
    from .s1input import channel_stats, skip_image

    layer = PRE_REGISTERED_LAYERS[encoder] if layer is None else layer
    man = load_manifest(prepared_root, dataset)
    classes = {int(k): v for k, v in man["classes"].items()}
    k = len(classes)
    if sorted(classes) != list(range(k)):
        raise ValueError(f"class ids must be 0..{k - 1}, got {sorted(classes)}")
    names = [classes[i] for i in range(k)]
    splits = list(man["splits"])
    if "train" not in splits:
        raise ValueError(f"{dataset}: no train split in {splits}")
    evals = [s for s in splits if s != "train"]
    run = f"dom_{dataset}__{encoder}__L{layer}__{mode}" + ("__disjoint_" + "_".join(exclude_train_sharing)
                                                            if exclude_train_sharing else "")
    units = str(man.get("units", "dB"))
    if units != "dB":
        print(f"  NOTE: SAR units are {units!r}, not dB; results are reported with that label")
    res_dir = Path(res_dir); res_dir.mkdir(parents=True, exist_ok=True)
    spath = res_dir / f"{run}__summary.csv"
    settings = {"max_train_chips": max_train_chips, "probe_max_patches": probe_max_patches, "steps": steps,
                "batch": batch, "lr": lr, "data_created_utc": man.get("created_utc")}
    if spath.exists() and not force:
        s = pd.read_csv(spath)
        want = {("probe", -1)} | {("decoder", sd) for sd in seeds}
        if not (res_dir / f"{run}__info.json").exists():     # the summary is written last since 2026-09-29
            print(f"  {run}: summary without info.json (a run that stopped midway) -> run again")
        elif want <= set(zip(s["head"], s["seed"])):
            _check_same_settings(res_dir / f"{run}__info.json", settings)
            print(f"skip {run}: summary already has probe + decoder seeds {list(seeds)}")
            return s
    print(f"\n===== {run}: splits {man['splits']} | classes {names} | chip {man['chip_size']} px =====")
    t_all = time.perf_counter()
    data = {s: load_split(prepared_root, dataset, s) for s in splits}
    if exclude_train_sharing:
        shared = set()
        for s in evals:
            for c in exclude_train_sharing:
                shared |= {(c, v) for v in data[s]["meta"][c].astype(str) if v != ""}
        tm = data["train"]["meta"]
        keep = ~np.array([any((c, str(tm[c].iloc[i])) in shared for c in exclude_train_sharing)
                          for i in range(len(tm))])
        print(f"  scene-disjoint: {int((~keep).sum())} of {len(tm)} training chips share "
              f"{exclude_train_sharing} with a scored split -> dropped")
        idx = np.flatnonzero(keep)
        data["train"] = {"images": data["train"]["images"][idx], "labels": data["train"]["labels"][idx],
                         "meta": tm.iloc[idx].reset_index(drop=True)}
    stats = channel_stats(data["train"]["images"])
    size = int(man["chip_size"])

    # ---- frozen features, once per split
    spec = ENCODERS[encoder]
    ensure_packages(spec.kind)
    feats, times = {}, {}
    enc = None if enc_cache is None else enc_cache.get(encoder)
    if enc is None:
        t0 = time.perf_counter()
        enc = FrozenEncoder(spec, token=token)
        if enc_cache is not None:
            enc_cache[encoder] = enc
        times["encoder_load_min"] = (time.perf_counter() - t0) / 60
        print(f"  encoder loaded in {times['encoder_load_min']:.1f} min")
    else:
        print("  encoder reused (already loaded)")
    ebatch = encoder_batch(spec.batch, size)
    try:
        for s in splits:
            t0 = time.perf_counter()
            feats[s] = _encode_split(enc, data[s]["images"], stats, mode, layer, ebatch)
            times[f"extract_{s}_min"] = (time.perf_counter() - t0) / 60
            print(f"  features {s}: {feats[s].shape} in {times[f'extract_{s}_min']:.1f} min")
    finally:
        del enc                                         # with enc_cache the caller's dict keeps it
        if enc_cache is None:
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    D = feats["train"].shape[-1]
    labels = {}
    for s in splits:                      # one no-data rule: either channel missing -> ignore
        lab = np.array(data[s]["labels"])
        n_before = int((lab != IGNORE).sum())
        for i in range(0, len(lab), 256):
            miss = np.isnan(np.asarray(data[s]["images"][i:i + 256], np.float32)).any(axis=1)
            lab[i:i + 256][miss] = IGNORE
        labels[s] = lab
        print(f"  {s}: {n_before - int((lab != IGNORE).sum()):,} labelled pixels ignored for missing SAR")
    region = {s: data[s]["meta"]["region"].astype(str).to_numpy() for s in splits}

    rows, region_rows = [], []

    def score(head_name: str, seed: int, s: str, cc: np.ndarray) -> None:
        """cc: (N, k, k) pixel confusion per chip of split s; split and region totals are its sums."""
        rows.append({"run": run, "dataset": dataset, "encoder": encoder, "layer": layer, "mode": mode,
                     "units": units, "head": head_name, "seed": seed, "split": s, **seg_metrics(cc.sum(0), names)})
        for r in np.unique(region[s]):
            m = region[s] == r
            region_rows.append({"head": head_name, "seed": seed, "split": s, "region": r, "chips": int(m.sum()),
                                **seg_metrics(cc[m].sum(0), names)})

    # ---- majority baseline (most frequent training pixel class everywhere)
    tr_counts = np.bincount(labels["train"][labels["train"] != IGNORE].ravel(), minlength=k)[:k]
    maj = int(tr_counts.argmax())
    for s in evals:
        score("majority", -1, s, _majority_conf(labels[s], k, maj, device))

    # ---- head 1: linear probe on patch features
    t0 = time.perf_counter()
    plab, pkeep = _patch_targets(labels["train"], k)
    X, y = feats["train"][pkeep], plab[pkeep]
    if len(y) > probe_max_patches:
        sel = np.sort(np.random.default_rng(0).choice(len(y), probe_max_patches, replace=False))
        X, y = X[sel], y[sel]
    per_epoch = -(-len(y) // 65536)
    epochs = max(30, -(-PROBE_STEPS // per_epoch))
    probe = train_linear_probe(np.ascontiguousarray(X), y.astype(np.int64), balanced=True, n_classes=k,
                               epochs=epochs, lr=1e-3, batch=65536, device=device, verbose=False)
    torch.save(probe.state_dict(), res_dir / f"{run}__probe.pt")
    for s in evals:
        pa, cc = _probe_conf(probe, feats[s], labels[s], k, size, device)
        score("probe", -1, s, cc)
        el, ek = _patch_targets(labels[s], k)
        pconf = np.bincount(el[ek].astype(np.int64) * k + pa[ek].astype(np.int64), minlength=k * k).reshape(k, k)
        rows[-1]["patch_mIoU"] = seg_metrics(pconf, names)["mIoU"]
    times["probe_min"] = (time.perf_counter() - t0) / 60
    print(f"  probe: {len(y):,} patches, {epochs} epochs x {per_epoch} steps, {times['probe_min']:.1f} min")
    del X, y

    # ---- head 2: small decoder, several seeds, same capped uniform chip sample
    n_tr = len(feats["train"])
    pick = (np.sort(np.random.default_rng(0).choice(n_tr, max_train_chips, replace=False))
            if n_tr > max_train_chips else np.arange(n_tr))
    to_u8 = lambda im: np.round(skip_image(im, stats)[:, 0] * 255).astype(np.uint8)  # noqa: E731
    skip = {s: np.concatenate([to_u8(data[s]["images"][i:i + 256]) for i in range(0, len(data[s]["images"]), 256)])
            for s in splits}
    for sd in seeds:
        t0 = time.perf_counter()
        torch.manual_seed(sd)
        head = GridDecoder(D, n_classes=k)
        head = _train_decoder(head, feats["train"][pick], skip["train"][pick], labels["train"][pick], k,
                              device, steps, batch, lr, sd)
        torch.save(head.state_dict(), res_dir / f"{run}__decoder__s{sd}.pt")
        for s in evals:
            score("decoder", sd, s, _decoder_conf(head, feats[s], skip[s], labels[s], k, device))
        times[f"decoder_s{sd}_min"] = (time.perf_counter() - t0) / 60
        print(f"  decoder seed {sd}: {times[f'decoder_s{sd}_min']:.1f} min")
        del head
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    out = pd.DataFrame(rows)
    reg = pd.DataFrame(region_rows)
    for (h, sd), g in reg.groupby(["head", "seed"]):
        g.to_csv(res_dir / f"{run}__{h}{'' if sd < 0 else f'__s{sd}'}__regions.csv", index=False)
    times["total_min"] = (time.perf_counter() - t_all) / 60
    info = {"stats": stats, "splits": man["splits"], "train_chips_decoder": int(len(pick)),
            "probe_patches": int(min(probe_max_patches, int(pkeep.sum()))), "classes": names,
            "chip_size": size, "times": times, "encoder_batch": ebatch, "settings": settings,
            "versions": package_versions()}
    (res_dir / f"{run}__info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    tmp = spath.with_name(spath.name + ".partial")          # the summary last and whole: it marks the run done
    out.to_csv(tmp, index=False)
    tmp.replace(spath)
    cols = ["head", "seed", "split", "mIoU", "accuracy"] + (["IoU_pos", "F1_pos"] if k == 2 else [])
    print(out[cols].round(3).to_string(index=False))
    print(f"  total {times['total_min']:.1f} min")
    return out


def domain_table(res_dir: str | Path) -> pd.DataFrame:
    """All dom_*__summary.csv files -> one row per run / head / split, decoder as mean +- SD over seeds.

    Keyed on the run name (dataset, encoder, layer, input mode and any scene-disjoint variant), so a
    variant run is never averaged into the pre-registered one.
    """
    import glob

    files = sorted(glob.glob(str(Path(res_dir) / "dom_*__summary.csv")))
    if not files:
        raise FileNotFoundError(f"no dom_*__summary.csv in {res_dir}")
    s = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    keys = ["run", "dataset", "encoder", "head", "split"]
    agg = (s.groupby(keys)
             .agg(seeds=("seed", "nunique"), mIoU=("mIoU", "mean"), mIoU_sd=("mIoU", "std"),
                  accuracy=("accuracy", "mean"))
             .reset_index())
    if "IoU_pos" in s:
        pos = s.groupby(keys).agg(IoU_pos=("IoU_pos", "mean"), IoU_pos_sd=("IoU_pos", "std")).reset_index()
        agg = agg.merge(pos, on=keys, how="left")
    return agg


def _check_same_settings(info_path: Path, settings: dict) -> None:
    """Refuse to skip a finished run that was made with other settings or other data.

    Runs finished before 2026-09-29 stored no settings; they count as the pre-registered ones,
    and their data cannot be compared.
    """
    old = None
    if info_path.exists():
        old = json.loads(info_path.read_text(encoding="utf-8")).get("settings")
    if old is None:
        old = {**PRE_REGISTERED_SETTINGS, "data_created_utc": settings["data_created_utc"]}
    diff = {k: (old.get(k), v) for k, v in settings.items() if old.get(k) != v}
    if diff:
        raise ValueError(f"{info_path.name}: the finished run used other settings or data "
                         f"(finished, requested) {diff}; use force=True or another res_dir")
