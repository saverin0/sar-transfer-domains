"""What the frozen features look like on the new domains -- descriptive, trains nothing.

Embedding views (as part one's notebook 14), for every prepared dataset and both encoders at their
pre-registered layers, on the TEST split, with the run's own input (s1input "vv_vh_diff" and the
training-split channel statistics saved by run.py in <run>__info.json):

1. embedding maps: PCA 1-3 of a chip's patch features as RGB, and k-means (k = number of classes);
2. similarity search by example: cosine similarity of one query patch to every patch of its chip
   (map), and the class shares of the k most similar pure patches in the whole test split, with
   k = 100, or one less than the query class's pure patches when that is fewer (so a small class
   is not diluted by others);
3. class distances: per class c, margin = cos(patch, centre of c) - max over the other classes of
   cos(patch, their centre), for every "pure" test patch; histograms per class and a ranking AUC
   (margin of class-c patches against all others). Centres come from the test split's own labels,
   so the AUCs are optimistic; they describe the features, they are not scores of a model.
4. AnyUp versions of 1-3 (added 2026-09-29, run_all_anyup): the features after AnyUp
   upsampling to 1/4 of the chip size (4 px blocks), as part one's notebook 14 (option A) and head
   6. Guide = the decoder's skip image (standardised co-pol in [0, 1]) as grey, resized to the
   output size; bf16 autocast on CUDA as in the reported runs. Same chips A and B and the same
   query spot as the raw views (read from embedding_numbers.csv). The query vector is the mean of
   the unit vectors of the 16 blocks inside the query patch; those blocks are left out of the
   ranking; k = 16 x the raw k (capped as above). A block is pure by the same rule on its 4 x 4 px.
   Centres use every pure test block; the margins and AUCs use a fixed uniform random sample of
   at most 400,000 pure blocks (seed 0; uniform so the class mix stays as in the test split).

Fixed rules (set 2026-09-29 before looking at any view; nothing is chosen from results):
- a patch (16 x 16 px) is PURE for class c when at least half of its pixels are labelled and at
  least 90 % of those are c; only pure patches enter the similarity ranking and the distances;
- chips shown: A = the test chip with at least 90 % labelled pixels whose class mix is the most
  even (highest entropy of the class shares), B = the same rule among chips of another region
  (meta "region"; if there is only one region, the second best chip); ties -> the lower index.
  Corrected 2026-09-29 after the first run (display only; the AUCs use every pure test patch and
  did not change): the 90 % is counted within the chip's radar-covered area (both channels
  present), which must be at least half the chip. Alpine tiles sit padded inside 992 px chips,
  so on the whole chip only one test chip passed and it was shown twice, with almost no glacier.
- query: the target class (class 1 of a two-class set; otherwise the class with the most pure
  patches in chip A), the pure patch of that class nearest the chip centre, ties -> raster order;
  without a pure patch of that class in chip A, its purest patch there (reported as fallback).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .prepared import IGNORE, load_manifest, load_split

PATCH = 16
PURITY, MIN_LABELLED, CHIP_LABELLED = 0.9, 0.5, 0.9
TOP_K, SEED = 100, 0
BLOCK, SAMPLE_MAX, KMEANS_FIT = 4, 400_000, 20_000     # AnyUp views: 4 px blocks, AUC sample, k-means fit cap
LAYERS = {"dinov3-l-sat": 21, "cradio-v4-h": 32}
NAMES = {"dinov3-l-sat": "satellite DINOv3", "cradio-v4-h": "C-RADIO"}
MODE = "vv_vh_diff"


# ------------------------------------------------------------------ measures (from part one's notebook 14)

def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def pca3(X: np.ndarray) -> np.ndarray:
    """(N, D) -> (N, 3) first three principal components (exact SVD), sign fixed: largest loading positive."""
    import torch

    x = torch.from_numpy(np.asarray(X, np.float32))
    x = x - x.mean(0)
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    v = vh[:3]
    v = v * torch.sign(v.gather(1, v.abs().argmax(1, keepdim=True)))
    return (x @ v.T).numpy()


def pca_rgb(feats: np.ndarray, valid: np.ndarray, pct=(1, 99)) -> np.ndarray:
    """PCA 1-3 of one chip's valid patches as RGB in [0, 1] (1st-99th percentile), invalid patches black."""
    p = pca3(feats[valid])
    lo, hi = np.percentile(p, pct, axis=0)
    rgb = np.zeros(valid.shape + (3,), np.float32)
    rgb[valid] = np.clip((p - lo) / np.maximum(hi - lo, 1e-12), 0, 1)
    return rgb


def kmeans_map(feats: np.ndarray, valid: np.ndarray, k: int, seed: int = SEED, max_fit: int | None = None) -> np.ndarray:
    """k-means on one chip's valid cells; -1 on invalid cells. With more than `max_fit` cells (AnyUp
    blocks) it is fitted on `max_fit` of them drawn with the fixed seed, then every cell gets its
    nearest centre (as part one)."""
    from sklearn.cluster import KMeans

    out = np.full(valid.shape, -1, np.int16)
    X = np.asarray(feats[valid], np.float32)
    if len(X) < k:
        return out
    if max_fit is None or len(X) <= max_fit:
        out[valid] = KMeans(n_clusters=k, random_state=seed, n_init=3).fit(X).labels_
    else:
        pick = np.sort(np.random.default_rng(seed).choice(len(X), max_fit, replace=False))
        out[valid] = KMeans(n_clusters=k, random_state=seed, n_init=3).fit(X[pick]).predict(X)
    return out


def auc_rank(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(random pos > random neg), ties half (Mann-Whitney U from average ranks)."""
    from scipy.stats import rankdata

    pos, neg = np.asarray(pos, np.float64), np.asarray(neg, np.float64)
    if not len(pos) or not len(neg):
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ------------------------------------------------------------------ patches and chips

def patch_shares(labels: np.ndarray, k: int, patch: int = PATCH) -> tuple[np.ndarray, np.ndarray]:
    """(N, H, W) labels -> class shares among labelled pixels (N, g, g, k) and labelled fraction (N, g, g)."""
    n, h, w = labels.shape
    g1, g2 = h // patch, w // patch
    lab = np.asarray(labels)[:, :g1 * patch, :g2 * patch].reshape(n, g1, patch, g2, patch)
    counts = np.stack([(lab == c).sum(axis=(2, 4)) for c in range(k)], -1).astype(np.float32)
    tot = counts.sum(-1)
    return counts / np.maximum(tot, 1)[..., None], tot / (patch * patch)


def pure_classes(shares: np.ndarray, labelled: np.ndarray) -> np.ndarray:
    """Class id of every pure patch (>= 50 % labelled, >= 90 % one class), -1 elsewhere."""
    top = shares.argmax(-1)
    ok = (labelled >= MIN_LABELLED) & (shares.max(-1) >= PURITY)
    return np.where(ok, top, -1).astype(np.int16)


def pick_chips(labels: np.ndarray, covered: np.ndarray, region: np.ndarray, k: int) -> list[int]:
    """Chips A and B by the fixed rule of the module docstring (labels and radar coverage only, never results).

    covered: per chip, the number of pixels where both radar channels are present."""
    n = len(labels)
    ent = np.full(n, -1.0)
    for i in range(n):
        lab = np.asarray(labels[i])
        cnt = np.bincount(lab[lab != IGNORE].ravel(), minlength=k)[:k].astype(np.float64)
        if covered[i] >= 0.5 * lab.size and cnt.sum() >= CHIP_LABELLED * covered[i]:
            p = cnt[cnt > 0] / cnt.sum()
            ent[i] = float(-(p * np.log(p)).sum())
    order = np.argsort(-ent, kind="stable")                       # ties -> lower index
    order = order[ent[order] >= 0]
    if not len(order):
        raise ValueError("no test chip has at least 90 % of its radar-covered pixels labelled")
    a = int(order[0])
    other = [int(i) for i in order[1:] if region[i] != region[a]]
    b = other[0] if other else (int(order[1]) if len(order) > 1 else a)
    return [a, b]


def query_patch(pure_a: np.ndarray, shares_a: np.ndarray, cls: int) -> dict:
    """Pure patch of `cls` nearest the chip centre (ties raster order); else the purest one (fallback)."""
    g1, g2 = pure_a.shape
    cand = pure_a == cls
    fallback = not cand.any()
    if fallback:
        s = shares_a[..., cls]
        if s.max() <= 0:
            raise ValueError(f"chip A has no pixel of class {cls}")
        cand = s == s.max()
    rows, cols = np.nonzero(cand)
    d = (rows - (g1 - 1) / 2) ** 2 + (cols - (g2 - 1) / 2) ** 2
    i = int(np.argmin(d))
    return {"row": int(rows[i]), "col": int(cols[i]), "fallback": bool(fallback),
            "purity": float(shares_a[rows[i], cols[i], cls])}


# ------------------------------------------------------------------ features

def run_stats(res_dir: str | Path, dataset: str, encoder: str) -> dict:
    """Channel statistics of the training split, as used by run.py (its info.json)."""
    p = Path(res_dir) / f"dom_{dataset}__{encoder}__L{LAYERS[encoder]}__{MODE}__info.json"
    if not p.exists():
        raise FileNotFoundError(f"{p.name} not found: run notebook 02 for {dataset} x {encoder} first")
    return {k: tuple(v) for k, v in json.loads(p.read_text(encoding="utf-8"))["stats"].items()}


def test_features(enc, prepared_root, dataset: str, encoder: str, stats: dict) -> tuple[np.ndarray, dict, dict]:
    """(N, g, g, D) fp16 features of the test split at the pre-registered layer, the split, the manifest."""
    from .run import _encode_split, encoder_batch

    man = load_manifest(prepared_root, dataset)
    data = load_split(prepared_root, dataset, "test")
    size = int(man["chip_size"])
    ebatch = encoder_batch(enc.spec.batch, size)
    return _encode_split(enc, data["images"], stats, MODE, LAYERS[encoder], ebatch), data, man


def _labels_with_nodata(data: dict) -> np.ndarray:
    """run.py's no-data rule: a pixel with a missing co- or cross-pol value is ignored."""
    lab = np.array(data["labels"])
    for i in range(0, len(lab), 256):
        lab[i:i + 256][np.isnan(np.asarray(data["images"][i:i + 256], np.float32)).any(axis=1)] = IGNORE
    return lab


# ------------------------------------------------------------------ one dataset x encoder

def views(feats: np.ndarray, data: dict, man: dict, encoder: str) -> dict:
    """Everything shown and counted for one dataset and encoder (arrays + numbers)."""
    classes = [man["classes"][str(i)] for i in range(len(man["classes"]))]
    k = len(classes)
    labels = _labels_with_nodata(data)
    region = data["meta"]["region"].astype(str).to_numpy()
    shares, labelled = patch_shares(labels, k)
    pure = pure_classes(shares, labelled)
    covered = np.concatenate([np.isfinite(np.asarray(data["images"][i:i + 256], np.float32)).all(axis=1).sum(axis=(1, 2))
                              for i in range(0, len(labels), 256)])
    chips = pick_chips(labels, covered, region, k)
    a = chips[0]
    cls = 1 if k == 2 else int(np.bincount(pure[a][pure[a] >= 0], minlength=k).argmax())
    q = query_patch(pure[a], shares[a], cls)

    out = {"encoder": encoder, "classes": classes, "chips": chips, "chip_regions": [region[c] for c in chips],
           "query": q, "query_class": classes[cls], "maps": {}}
    for c in chips:                                               # 1. embedding maps
        valid = labelled[c] > 0
        f = np.asarray(feats[c], np.float32)
        out["maps"][c] = {"pca": pca_rgb(f, valid), "kmeans": kmeans_map(f, valid, k)}

    g1, g2 = pure.shape[1:]                                       # 2. similarity search by example
    fa = _unit(np.asarray(feats[a], np.float32).reshape(g1 * g2, -1))
    qv = fa[q["row"] * g2 + q["col"]]
    out["sim_map"] = (fa @ qv).reshape(g1, g2)
    idx = np.flatnonzero(pure.ravel() >= 0)
    X = _unit(np.asarray(feats.reshape(-1, feats.shape[-1])[idx], np.float32))
    y = pure.ravel()[idx].astype(np.int64)
    qflat = (a * g1 + q["row"]) * g2 + q["col"]
    sim = X @ qv
    order = np.argsort(-sim, kind="stable")
    kq = int(min(TOP_K, max(1, (y == cls).sum() - (pure.ravel()[qflat] == cls))))
    order = order[idx[order] != qflat][:kq]
    top = np.bincount(y[order], minlength=k)[:k] / max(len(order), 1)
    out["top_shares"] = {n: float(v) for n, v in zip(classes, top)}
    out["top_k"] = kq

    present = [c for c in range(k) if (y == c).any()]             # 3. class distances (classes with pure patches)
    out["margins"], out["auc"], out["n_pure"] = {}, {}, {}
    if len(present) >= 2:
        cent = _unit(np.stack([X[y == c].mean(0) for c in present]))
        cos = X @ cent.T
        for j, c in enumerate(present):
            m = cos[:, j] - np.delete(cos, j, axis=1).max(1)
            out["margins"][classes[c]] = (m[y == c], m[y != c])
            out["auc"][classes[c]] = auc_rank(m[y == c], m[y != c])
            out["n_pure"][classes[c]] = int((y == c).sum())
    out["mean_auc"] = float(np.nanmean(list(out["auc"].values()))) if out["auc"] else float("nan")
    return out


def number_rows(dataset: str, v: dict) -> list[dict]:
    base = {"dataset": dataset, "encoder": v["encoder"], "layer": LAYERS[v["encoder"]], "chip_A": v["chips"][0],
            "chip_B": v["chips"][1], "query_class": v["query_class"], "query_row": v["query"]["row"],
            "query_col": v["query"]["col"], "query_fallback": v["query"]["fallback"], "top_k": v["top_k"],
            "mean_auc": v["mean_auc"]}
    return [{**base, "class": c, "n_pure_patches": v["n_pure"].get(c, 0), "auc": v["auc"].get(c, np.nan),
             "top_k_share": v["top_shares"][c]} for c in v["classes"]]


# ------------------------------------------------------------------ figures (no colon in any title)

def _cat_colours(k: int):
    import matplotlib.pyplot as plt

    return plt.get_cmap("tab10" if k <= 10 else "tab20")(np.arange(k))[:, :3]


def _label_rgb(lab: np.ndarray, k: int) -> np.ndarray:
    rgb = np.zeros(lab.shape + (3,), np.float32)
    cols = _cat_colours(k)
    for c in range(k):
        rgb[lab == c] = cols[c]
    return rgb


def _radar(img: np.ndarray) -> np.ndarray:
    co = np.asarray(img[0], np.float32)
    ok = np.isfinite(co)
    if not ok.any():
        return np.zeros(co.shape, np.float32)
    lo, hi = np.percentile(co[ok], [2, 98])
    return np.where(ok, np.clip((co - lo) / max(hi - lo, 1e-6), 0, 1), 0)


def _shares_text(ts: dict, top_k: int) -> str:
    """Class shares of the most similar cells, largest first, two per line (classes under 0.5 % left out)."""
    items = [f"{c} {v:.2f}" for c, v in sorted(ts.items(), key=lambda kv: -kv[1]) if v >= 0.005]
    return f"top {top_k} in the test split\n" + "\n".join(", ".join(items[i:i + 2]) for i in range(0, len(items), 2))


def plot_dataset(dataset: str, data: dict, man: dict, per_enc: dict, out_dir: str | Path, tag: str = "",
                 what: str = "frozen features", unit: str = "pure test patches") -> list[Path]:
    """Three figures per dataset (maps, similarity, class distances), and every panel of them also as its own
    PNG in out_dir/panels/<dataset>/. Returns the three figure paths.

    tag / what / unit: file-name suffix and title words, e.g. "_anyup", "features after AnyUp (4 px blocks)",
    "sample of pure test blocks" for the AnyUp versions."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    out_dir = Path(out_dir)
    pdir = out_dir / "panels" / dataset
    pdir.mkdir(parents=True, exist_ok=True)
    encs = list(per_enc)
    v0 = per_enc[encs[0]]
    k, classes, chips = len(v0["classes"]), v0["classes"], v0["chips"]
    cols = _cat_colours(k)
    handles = [Patch(color=cols[c], label=classes[c]) for c in range(k)] + [Patch(color="black", label="no label")]
    lab = _labels_with_nodata(data)
    ext = (0, lab.shape[2], lab.shape[1], 0)
    au = " (AnyUp)" if tag else ""
    q = v0["query"]
    box = lambda ax: ax.add_patch(Rectangle((q["col"] * PATCH, q["row"] * PATCH), PATCH, PATCH,  # noqa: E731
                                            fill=False, ec="yellow", lw=2))

    def radar(ax, c):
        ax.imshow(_radar(data["images"][c]), cmap="gray")

    def labels(ax, c):
        ax.imshow(_label_rgb(lab[c], k))

    def pca(ax, e, c):
        ax.imshow(per_enc[e]["maps"][c]["pca"], extent=ext, interpolation="nearest")

    def kmeans(ax, e, c):
        ax.imshow(np.ma.masked_less(per_enc[e]["maps"][c]["kmeans"], 0), extent=ext, interpolation="nearest",
                  cmap="Set2", vmin=0, vmax=7)

    def query(ax):
        radar(ax, chips[0]); box(ax)

    def sim(ax, e):
        s = per_enc[e]["sim_map"]
        ax.imshow(s, extent=ext, interpolation="nearest", cmap="Blues", vmin=np.percentile(s, 2),
                  vmax=np.percentile(s, 98))
        box(ax)
        ax.set_xlabel(_shares_text(per_enc[e]["top_shares"], per_enc[e]["top_k"]), fontsize=7)

    def hist(ax, e, c):
        pos, neg = per_enc[e]["margins"][classes[c]]
        bins = np.linspace(min(pos.min(), neg.min()), max(pos.max(), neg.max()), 40)
        ax.hist(neg, bins=bins, density=True, color="0.6", alpha=0.7, label="other classes")
        ax.hist(pos, bins=bins, density=True, color=cols[c], alpha=0.8, label=classes[c])
        ax.tick_params(labelsize=6)

    n_single = [0]

    def single(name: str, draw, title: str, legend=None, ticks: bool = False) -> None:
        fig1, ax1 = plt.subplots(figsize=(4.4, 4.8), layout="constrained")
        draw(ax1)
        ax1.set_title(title, fontsize=9)
        if not ticks:
            ax1.set_xticks([]); ax1.set_yticks([])
        if legend:
            fig1.legend(handles=legend, loc="outside lower center", ncol=min(len(legend), 3), fontsize=7, frameon=False)
        fig1.savefig(pdir / name, dpi=110)
        plt.close(fig1)
        n_single[0] += 1

    paths = []
    ncol = 2 + 2 * len(encs)                                                  # 1. maps
    fig, axes = plt.subplots(len(chips), ncol, figsize=(2.6 * ncol, 3.0 * len(chips) + 0.6), squeeze=False,
                             layout="constrained")
    for r, c in enumerate(chips):
        reg = v0["chip_regions"][r]
        radar(axes[r, 0], c); axes[r, 0].set_title(f"chip {c} ({reg}) radar co-pol", fontsize=8)
        labels(axes[r, 1], c); axes[r, 1].set_title("labels", fontsize=8)
        single(f"chip{c}_radar.png", lambda ax: radar(ax, c), f"{dataset} chip {c} ({reg}) radar co-pol")
        single(f"chip{c}_labels.png", lambda ax: labels(ax, c), f"{dataset} chip {c} labels", legend=handles)
        for j, e in enumerate(encs):
            pca(axes[r, 2 + 2 * j], e, c)
            axes[r, 2 + 2 * j].set_title(f"{NAMES[e]} PCA 1-3{au}", fontsize=8)
            kmeans(axes[r, 3 + 2 * j], e, c)
            axes[r, 3 + 2 * j].set_title(f"{NAMES[e]} k-means, k = {k}{au}\n(cluster colours, not classes)", fontsize=8)
            single(f"chip{c}_{e}_pca{tag}.png", lambda ax: pca(ax, e, c),
                   f"{dataset} chip {c}, {NAMES[e]}{au}\nPCA 1-3 (same colour = similar features)")
            single(f"chip{c}_{e}_kmeans{tag}.png", lambda ax: kmeans(ax, e, c),
                   f"{dataset} chip {c}, {NAMES[e]}{au}\nk-means, k = {k} (cluster colours, not classes)")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.legend(handles=handles, loc="outside lower center", ncol=min(k + 1, 9), fontsize=8, frameon=False)
    fig.suptitle(f"{dataset} test chips, {what} (same colour = similar features)", fontsize=10)
    paths.append(out_dir / f"embed_maps{tag}_{dataset}.png"); fig.savefig(paths[-1], dpi=110); plt.close(fig)

    a = chips[0]                                                              # 2. similarity
    fig, axes = plt.subplots(1, 1 + len(encs), figsize=(3.8 * (1 + len(encs)), 4.8), squeeze=False,
                             layout="constrained")
    axes = axes[0]
    query(axes[0]); axes[0].set_title(f"chip {a}, query {v0['query_class']} patch (yellow)", fontsize=8)
    single(f"chip{a}_query.png", query, f"{dataset} chip {a}\nquery = one {v0['query_class']} patch (yellow)")
    for j, e in enumerate(encs):
        sim(axes[1 + j], e)
        axes[1 + j].set_title(f"{NAMES[e]}{au}\ndarker = more similar", fontsize=8)
        single(f"chip{a}_{e}_similarity{tag}.png", lambda ax: sim(ax, e),
               f"{dataset} chip {a}, {NAMES[e]}{au}\nsimilarity to the query, darker = more similar")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    paths.append(out_dir / f"embed_similarity{tag}_{dataset}.png"); fig.savefig(paths[-1], dpi=110); plt.close(fig)

    fig, axes = plt.subplots(len(encs), k, figsize=(max(2.4 * k, 6.0), 2.4 * len(encs) + 0.7), squeeze=False,
                             layout="constrained")                                   # 3. distances
    for i, e in enumerate(encs):
        for c in range(k):
            ax = axes[i, c]
            if classes[c] not in per_enc[e]["margins"]:
                ax.set_axis_off(); continue
            auc = per_enc[e]["auc"][classes[c]]
            hist(ax, e, c)
            ax.set_title(f"{NAMES[e]}, {classes[c]}\nAUC {auc:.3f}", fontsize=8)
            single(f"{e}_{classes[c]}_margins{tag}.png", lambda a_: hist(a_, e, c),
                   f"{dataset}, {NAMES[e]}{au}, {classes[c]}\nmargin to own centre, AUC {auc:.3f}", ticks=True,
                   legend=[Patch(color=cols[c], label=classes[c]), Patch(color="0.6", label="other classes")])
    fig.suptitle(f"{dataset} {unit}\nmargin = cos(own centre) - cos(nearest other centre)", fontsize=9)
    paths.append(out_dir / f"embed_margins{tag}_{dataset}.png"); fig.savefig(paths[-1], dpi=110); plt.close(fig)
    print(f"{dataset}: {n_single[0]} single-panel images in {pdir}")
    return paths


def _anyup_tensor(anyup, feats_chip: np.ndarray, image: np.ndarray, stats: dict, device: str = "cuda"):
    """One chip's (g, g, D) features -> AnyUp at 1/4 of the chip size, (size/4, size/4, D) float32 tensor on `device`."""
    import torch
    import torch.nn.functional as F

    from ..models.anyup_loader import guide_image, upsample_with_value
    from .s1input import skip_image

    size = int(image.shape[-1]) // BLOCK
    grey = np.round(skip_image(np.asarray(image, np.float32)[None], stats)[0, 0] * 255).astype(np.uint8)
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        ft = torch.from_numpy(np.asarray(feats_chip, np.float32)).permute(2, 0, 1)[None].to(device)
        guide = guide_image(torch.from_numpy(np.ascontiguousarray(grey))[None, None].to(device))
        guide = F.interpolate(guide, size=(size, size), mode="bilinear", align_corners=False)
        up = upsample_with_value(anyup, guide, ft, ft, out_size=(size, size))
    return up[0].permute(1, 2, 0).float()


def anyup_quarter(anyup, feats_chip: np.ndarray, image: np.ndarray, stats: dict, device: str = "cuda") -> np.ndarray:
    """As _anyup_tensor, as a float32 numpy array."""
    return _anyup_tensor(anyup, feats_chip, image, stats, device).cpu().numpy()


def _block_valid(image: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """4 px blocks with radar on at least half of their pixels (both channels present)."""
    ok = np.isfinite(np.asarray(image, np.float32)).all(0)
    h, w = ok.shape
    return ok[:h - h % block, :w - w % block].reshape(h // block, block, w // block, block).mean((1, 3)) >= 0.5


def _anyup_pass(enc, anyup, data: dict, man: dict, encoder: str, stats: dict, ref: dict, device: str) -> dict:
    """Views 1-3 on the AnyUp blocks of one dataset and encoder, in one streaming pass over the test split.

    ref: chips, query (row, col in patches), query class id, raw k, the block purity map and the sample of
    global block ids (from `_anyup_ref`)."""
    import torch

    with torch.inference_mode():
        return _anyup_pass_body(enc, anyup, data, man, encoder, stats, ref, device)


def _anyup_pass_body(enc, anyup, data: dict, man: dict, encoder: str, stats: dict, ref: dict, device: str) -> dict:
    import torch

    from .run import _encode_split, encoder_batch

    classes, k, pure = ref["classes"], ref["k"], ref["pure"]
    n, h4, w4 = pure.shape
    a, b = ref["chips"]
    cls, (qr, qc) = ref["cls"], ref["query"]
    size = int(man["chip_size"])
    ebatch = encoder_batch(enc.spec.batch, size)
    unit = lambda x: x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)  # noqa: E731
    step = PATCH // BLOCK

    fp = np.zeros((h4, w4), bool)                                   # the query patch's 16 blocks
    fp[qr * step:(qr + 1) * step, qc * step:(qc + 1) * step] = True
    img_a = np.asarray(data["images"][a], np.float32)
    fa = _encode_split(enc, img_a[None], stats, MODE, LAYERS[encoder], 1)[0]
    ua = unit(_anyup_tensor(anyup, fa, img_a, stats, device).reshape(h4 * w4, -1))
    qm = fp & _block_valid(img_a)
    qm = qm if qm.any() else fp
    qv = unit(ua[torch.from_numpy(qm.ravel()).to(device)].mean(0))
    sim_map = (ua @ qv).reshape(h4, w4).cpu().numpy()
    n_cls = int((pure == cls).sum()) - int((pure[a][fp] == cls).sum())
    kq = int(min(step * step * ref["raw_k"], max(1, n_cls)))

    D = ua.shape[-1]
    sums = torch.zeros(k, D, device=device)
    counts = torch.zeros(k, device=device)
    best_v = torch.empty(0, device=device)
    best_c = torch.empty(0, dtype=torch.long, device=device)
    samp_x, samp_y = [], []
    sample, per = ref["sample"], h4 * w4
    fp_t = torch.from_numpy(fp.ravel()).to(device)
    maps = {}
    for i0 in range(0, n, ebatch):
        imgs = np.asarray(data["images"][i0:i0 + ebatch], np.float32)
        feats = _encode_split(enc, imgs, stats, MODE, LAYERS[encoder], ebatch)
        for j in range(len(imgs)):
            i = i0 + j
            up = _anyup_tensor(anyup, feats[j], imgs[j], stats, device)
            u = unit(up.reshape(per, -1))
            pc = torch.from_numpy(pure[i].ravel().astype(np.int64)).to(device)
            ok = pc >= 0
            sums.index_add_(0, pc[ok], u[ok])                           # centres: every pure block
            counts += torch.bincount(pc[ok], minlength=k)[:k].float()
            rk = ok & ~fp_t if i == a else ok                           # ranking: query blocks left out
            vals = torch.cat([best_v, u[rk] @ qv])
            labs = torch.cat([best_c, pc[rk]])
            top = torch.topk(vals, min(kq, len(vals))).indices
            best_v, best_c = vals[top], labs[top]
            lo, hi = np.searchsorted(sample, [i * per, (i + 1) * per])
            if hi > lo:                                                 # the fixed AUC sample
                sel = torch.from_numpy(sample[lo:hi] - i * per).to(device)
                samp_x.append(u[sel].half().cpu().numpy())
                samp_y.append(pc[sel].cpu().numpy())
            if i in (a, b) and i not in maps:
                f = up.cpu().numpy()
                valid = _block_valid(imgs[j])
                maps[i] = {"pca": pca_rgb(f, valid), "kmeans": kmeans_map(f, valid, k, max_fit=KMEANS_FIT)}
    present = [c for c in range(k) if counts[c] > 0]
    X, y = np.concatenate(samp_x).astype(np.float32), np.concatenate(samp_y)
    out = {"encoder": encoder, "classes": classes, "chips": ref["chips"], "chip_regions": ref["regions"],
           "query": {"row": qr, "col": qc, "fallback": ref["fallback"]}, "query_class": classes[cls],
           "maps": maps, "sim_map": sim_map, "top_k": len(best_c),
           "top_shares": {nm: float(v) for nm, v in zip(classes, (torch.bincount(best_c, minlength=k)[:k].float()
                                                                   / max(len(best_c), 1)).cpu().numpy())},
           "margins": {}, "auc": {}, "n_pure": {classes[c]: int(counts[c]) for c in present}, "n_sample": len(y)}
    if len(present) >= 2:
        cent = unit(sums[present] / counts[present][:, None]).cpu().numpy()
        cos = X @ cent.T
        for j, c in enumerate(present):
            m = cos[:, j] - np.delete(cos, j, axis=1).max(1)
            out["margins"][classes[c]] = (m[y == c], m[y != c])
            out["auc"][classes[c]] = auc_rank(m[y == c], m[y != c])
    out["mean_auc"] = float(np.nanmean(list(out["auc"].values()))) if out["auc"] else float("nan")
    return out


def _anyup_ref(prepared_root, d: str, raw_row) -> dict:
    """Block purity map, the fixed AUC sample and the raw views' chips / query for dataset d."""
    data, man = load_split(prepared_root, d, "test"), load_manifest(prepared_root, d)
    classes = [man["classes"][str(i)] for i in range(len(man["classes"]))]
    k = len(classes)
    labels = _labels_with_nodata(data)
    shares, labelled = patch_shares(labels, k, patch=BLOCK)
    del labels
    pure = pure_classes(shares, labelled)
    del shares, labelled
    ids = np.flatnonzero(pure.ravel() >= 0)
    sample = np.sort(np.random.default_rng(SEED).choice(ids, min(len(ids), SAMPLE_MAX), replace=False))
    region = data["meta"]["region"].astype(str).to_numpy()
    chips = [int(raw_row.chip_A), int(raw_row.chip_B)]
    return {"data": data, "man": man, "classes": classes, "k": k, "pure": pure, "sample": sample, "chips": chips,
            "regions": [region[c] for c in chips], "query": (int(raw_row.query_row), int(raw_row.query_col)),
            "fallback": bool(raw_row.query_fallback), "cls": classes.index(raw_row.query_class),
            "raw_k": int(raw_row.top_k)}


def run_all_anyup(prepared_root, res_dir, out_dir, datasets, token: str | None, anyup_weights,
                  device: str = "cuda") -> pd.DataFrame:
    """View 4: views 1-3 on the AnyUp features (after run_all, whose chips and query spot it reuses).

    Writes embed_{maps,similarity,margins}_anyup_<dataset>.png and adds rows with features = "anyup_quarter"
    to embedding_numbers.csv (earlier AnyUp rows are replaced). Returns the whole table."""
    import gc

    import torch

    from ..models.anyup_loader import fetch_anyup_weights, load_anyup
    from ..models.encoders import ENCODERS, FrozenEncoder, ensure_packages

    out_dir = Path(out_dir)
    table = pd.read_csv(out_dir / "embedding_numbers.csv")
    if "features" not in table:
        table["features"] = "raw"
    raw = table[table.features == "raw"]
    anyup = load_anyup(fetch_anyup_weights(anyup_weights), device)
    refs = {d: _anyup_ref(prepared_root, d, raw[raw.dataset == d].iloc[0]) for d in datasets}
    per, rows = {d: {} for d in datasets}, []
    for e in LAYERS:
        spec = ENCODERS[e]
        ensure_packages(spec.kind)
        enc = FrozenEncoder(spec, token=token)
        for d in datasets:
            r = refs[d]
            v = _anyup_pass(enc, anyup, r["data"], r["man"], e, run_stats(res_dir, d, e), r, device)
            per[d][e] = v
            rows += [{**x, "features": "anyup_quarter"} for x in number_rows(d, v)]
            print(f"{d} x {NAMES[e]} AnyUp: top {v['top_k']} blocks, mean AUC {v['mean_auc']:.3f} "
                  f"(AUC sample {v['n_sample']:,} blocks)")
        del enc
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    for d in datasets:
        for p in plot_dataset(d, refs[d]["data"], refs[d]["man"], per[d], out_dir, tag="_anyup",
                              what="features after AnyUp (4 px blocks)",
                              unit="sample of pure test blocks (AnyUp, 4 px)"):
            print("saved", p)
    new = pd.DataFrame(rows)
    table = pd.concat([raw, new], ignore_index=True)
    table.to_csv(out_dir / "embedding_numbers.csv", index=False)
    return table


def run_all(prepared_root, res_dir, out_dir, datasets, token: str | None, device: str = "cuda") -> pd.DataFrame:
    """Every dataset x encoder: features of the test split, views, figures, one numbers CSV."""
    import gc

    import torch

    from ..models.encoders import ENCODERS, FrozenEncoder, ensure_packages

    per = {d: {} for d in datasets}
    rows = []
    for e in LAYERS:
        spec = ENCODERS[e]
        ensure_packages(spec.kind)
        enc = FrozenEncoder(spec, token=token)
        for d in datasets:
            feats, data, man = test_features(enc, prepared_root, d, e, run_stats(res_dir, d, e))
            v = views(feats, data, man, e)
            per[d][e] = v
            rows += [{**x, "features": "raw"} for x in number_rows(d, v)]
            print(f"{d} x {NAMES[e]}: chips {v['chips']}, query {v['query_class']} "
                  f"{'(fallback) ' if v['query']['fallback'] else ''}mean AUC {v['mean_auc']:.3f}")
            del feats
        del enc
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in datasets:
        man = load_manifest(prepared_root, d)
        for p in plot_dataset(d, load_split(prepared_root, d, "test"), man, per[d], out_dir):
            print("saved", p)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "embedding_numbers.csv", index=False)
    return table
