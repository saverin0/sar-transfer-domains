"""Offline test of domains.glacial_lakes on tiny synthetic files in the documented GLB/GLC format.

Layout, names, band order, dtypes and codes follow what was read from the Zenodo record
17917359, the two zip central directories and the ESSD preprint (see the module docstring).
Everything is written under the temp folder of tests/_setup.py (TMPDIR / TEMP / TMP).
Covers the review fixes: ambiguous 0-1 state stops, -32768 fill (squash / linear / dB),
missing split in the count check, hold-out twin drop + tile note, cross-CRS overlap, GLC
region 'unknown', stale part files, download disk / marker / Content-Range / ENOSPC.
"""
import errno, hashlib, http.server, io, os, shutil, sys, threading, time, types, zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from _setup import TMP_BASE  # noqa: E402  (this repo's src first, import guard, temp in one folder)
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.warp import transform as warp

from sartransfer.domains import prepared as P
from sartransfer.domains import glacial_lakes as G

TMP = TMP_BASE / "tmp_glacial_lakes"
shutil.rmtree(TMP, ignore_errors=True)
TMP.mkdir()
P.CHUNK = 3                                              # force several part files
rng = np.random.default_rng(0)
S = 256
T0 = time.time()


def expect_fail(fn, *words, exc=AssertionError):
    try:
        fn()
    except exc as e:
        msg = str(e)
        assert all(w in msg for w in words), f"message {msg!r} lacks {words}"
        return msg
    raise SystemExit(f"expected {exc.__name__} containing {words}")


def tif(arr, epsg, x0, y0, nodata=None, desc=None):
    """GeoTIFF bytes on a 10 m UTM grid, like the dataset (GDAL/rasterio GTiff, no compression)."""
    with MemoryFile() as mf:
        with mf.open(driver="GTiff", width=S, height=S, count=arr.shape[0], dtype=arr.dtype.name,
                     crs=f"EPSG:{epsg}", transform=from_origin(x0, y0, 10, 10), nodata=nodata) as ds:
            ds.write(arr)
            if desc:
                ds.descriptions = desc
        return mf.read()


def scene(mask, no_sar_rows=0, empty=False, fill=0.0):
    """Raw 11-band stack before normalisation: 9 optical/DEM bands + linear gamma0 VV, VH (lakes darker).

    Missing SAR rows get `fill`: 0.0 (GDAL default when the RTC has no nodata tag) or -32768
    (the PC RTC nodata, which rasterio's reproject copies when dst_nodata is not given).
    """
    opt = rng.random((9, S, S)).astype(np.float32)
    opt[8] *= 4000                                                            # elevation, m
    vv = rng.lognormal(np.log(0.08), 0.6, (S, S)).astype(np.float32)
    vh = rng.lognormal(np.log(0.015), 0.6, (S, S)).astype(np.float32)
    vv[mask == 1] *= 0.05
    vh[mask == 1] *= 0.05
    vv[-4:, -4:], vh[-4:, -4:] = 3.0, 0.6                                    # layover-bright slope: gamma0 > 1
    s1 = np.stack([vv, vh])
    s1[:, :no_sar_rows] = fill
    if empty:
        s1[:] = 0.0
    return np.concatenate([opt, s1])


def per_band_minmax(stack):
    """What [C] normalize_data._per_band_normalize does with nodata None (the preprint's 0-1 stretch)."""
    out = np.zeros_like(stack)
    for i in range(len(stack)):
        b = stack[i]
        lo, hi = b.min(), b.max()
        if hi > lo:
            out[i] = (b - lo) / (hi - lo)
    return out


def lake_mask():
    m = np.zeros((S, S), np.uint8)
    for _ in range(3):
        r, c, k = rng.integers(20, 230, 2).tolist() + [int(rng.integers(5, 20))]
        m[max(r - k, 0):r + k, max(c - k, 0):c + k] = 1
    return m


EPSG = {"5VPJ": 32605, "6VXN": 32606, "33XVH": 32633, "33XWG": 32633, "44RPU": 32644,
        "45RXM": 32645, "45UYQ": 32645, "24WWU": 32624}


def zone(name):
    return EPSG[[p for p in name.split("_") if p in EPSG][0]]


def stem(f):
    return Path(f).stem


# ---------------------------------------------------------------- unit checks: counts, polygon clip
recs = lambda counts: [{"official": k} for k, v in counts.items() for _ in range(v)]
G._check_counts(recs(G.EXPECTED))                                                   # all four exact
expect_fail(lambda: G._check_counts(recs({"train": 15300, "test": 1907, "challenge": 1105})),
            "val: 0 pairs, expected 1912")                                          # a whole split missing
expect_fail(lambda: G._check_counts(recs({"train": 15300, "val": 1912, "test": 1907})), "challenge: 0 pairs")
G._check_counts(recs({"train": 15300, "val": 1912, "test": 1907}), G.BENCH_SPLITS)  # challenge not asked for
sq, diamond = [(0, 0), (2, 0), (2, 2), (0, 2)], [(1, 0), (2, 1), (1, 2), (0, 1)]
assert G._clip_area(sq, (0, 0, 2, 2)) == 4 and G._clip_area(sq, (1, 0, 3, 2)) == 2 and G._clip_area(sq, (5, 5, 6, 6)) == 0
assert abs(G._clip_area(diamond, (0, 0, 2, 2)) - 2) < 1e-12 and abs(G._clip_area(diamond, (1, 0, 2, 2)) - 1) < 1e-12

# ---------------------------------------------------------------- variant A: the official zips, min-max state
GLB = {  # official folder -> file names (region prefix convention of the preprint 2.5)
    "train": ["AK_S2A_5VPJ_20200817_0_L2A_1", "AK_S2A_5VPJ_20200817_0_L2A_2", "SV_S2B_33XVH_20200801_0_L2A_5",
              "CA_S2B_44RPU_20200831_1_L2A_95", "SEE_S2B_44RPU_20200831_1_L2A_93", "SEE_S2B_44RPU_20200831_1_L2A_112",
              "SEE_S2B_44RPU_20200831_1_L2A_120", "GL_S2A_24WWU_20200805_2_L2A_7"],
    "val": ["NA_S2B_6VXN_20200828_1_L2A_231", "SV_S2A_33XWG_20200715_0_L2A_3"],
    "test": ["CA_S2B_44RPU_20200831_1_L2A_111", "SEE_S2A_45RXM_20201229_1_L2A_13"],
}
GLC = ["S2B_44RPU_20200831_1_L2A_7", "S2A_5VPJ_20200817_0_L2A_9", "S2B_33XVH_20201227_0_L2A_4",
       "S2A_45UYQ_20200901_0_L2A_2"]
TWINS = {"CA_S2B_44RPU_20200831_1_L2A_111": "SEE_S2B_44RPU_20200831_1_L2A_112",   # test = train (real archive)
         "SEE_S2B_44RPU_20200831_1_L2A_93": "CA_S2B_44RPU_20200831_1_L2A_95"}     # train = train, CA/SEE
SQUASHED = "NA_S2B_6VXN_20200828_1_L2A_231"          # -32768 fill before the per-band stretch
origin, raw, files, truth = {}, {}, {}, {}
k = 0
for split, names in GLB.items():
    for n in names:
        origin[n] = (300000 + 3000 * k, 7000000.0)
        k += 1
for n in GLC:
    origin[n] = (300000 + 3000 * k, 7000000.0)
    k += 1
origin["S2A_5VPJ_20200817_0_L2A_9"] = origin["AK_S2A_5VPJ_20200817_0_L2A_1"]    # GLC chip on a train footprint
# a test chip in UTM 45 on the footprint of a train chip in UTM 44, at the zone boundary (84 E, 28.3 N)
xs, ys = warp("EPSG:4326", "EPSG:32644", [83.99], [28.3])
origin["CA_S2B_44RPU_20200831_1_L2A_95"] = (xs[0] - 1280, ys[0] + 1280)
xs, ys = warp("EPSG:32644", "EPSG:32645", [xs[0]], [ys[0]])
origin["SEE_S2A_45RXM_20201229_1_L2A_13"] = (xs[0] - 1280, ys[0] + 1280)
for n in [x for v in GLB.values() for x in v] + GLC:
    if n in TWINS:
        continue
    m = lake_mask()
    if n == SQUASHED:
        st = scene(m, no_sar_rows=40, fill=G.PC_NODATA)
    else:
        st = scene(m, no_sar_rows=12 if n.startswith("AK_S2A_5VPJ_20200817_0_L2A_2") else 0,
                   empty=n.startswith("GL_"))
    norm = per_band_minmax(st)
    raw[n] = norm[9:11].copy()
    files[n] = (tif(norm, zone(n), *origin[n]), tif(m[None], zone(n), *origin[n]))
    truth[n] = m
for a, b in TWINS.items():                            # byte-identical images under two region prefixes
    files[a], raw[a], truth[a] = files[b], raw[b], truth[b]
assert raw[SQUASHED][:, 40:].min() > 0.999, "synthetic squash did not happen"
print(f"synthetic image file {len(files['AK_S2A_5VPJ_20200817_0_L2A_1'][0])} B "
      f"(archive: 2,885,555/2,885,556 B), mask {len(files['AK_S2A_5VPJ_20200817_0_L2A_1'][1])} B (archive: 65,943/65,944 B)")

RAW_A = TMP / "raw_a"
RAW_A.mkdir()
with zipfile.ZipFile(RAW_A / "Glacial_Lake_Bench.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
    z.writestr("Glacial_Lake_Bench/", b"")
    for kind in ("ann_dir", "img_dir"):                                         # archive order: ann_dir first
        z.writestr(f"Glacial_Lake_Bench/{kind}/", b"")
        for split in ("test", "train", "val"):
            z.writestr(f"Glacial_Lake_Bench/{kind}/{split}/", b"")
            if kind == "ann_dir" and split == "test":
                z.writestr("Glacial_Lake_Bench/ann_dir/test/Thumbs.db", b"\0" * 64)
            for n in sorted(GLB[split]):
                z.writestr(f"Glacial_Lake_Bench/{kind}/{split}/{n}.tif", files[n][kind == "ann_dir"])
    z.writestr("Glacial_Lake_Bench/Metadata.txt", b"Each image contains 11 channels: ... VV, and VH.")
with zipfile.ZipFile(RAW_A / "Glacial-Lake-Challenge.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
    z.writestr("Glacial-Lake-Challenge/", b"")
    for kind in ("image", "mask"):
        z.writestr(f"Glacial-Lake-Challenge/{kind}/", b"")
        for n in sorted(GLC):
            z.writestr(f"Glacial-Lake-Challenge/{kind}/{n}.tif", files[n][kind == "mask"])

# inventory: counts differ from the real archive -> the count check must fire; then without it
expect_fail(lambda: G.inventory(RAW_A), "archive directory", "15300")
inv = G.inventory(RAW_A, check_counts=False)
assert inv["state"]["bench"]["state"] == inv["state"]["challenge"]["state"] == "minmax", inv["state"]
assert inv["state"]["bench"]["evidence"]["squashed_chip_bands"] == 2, inv["state"]["bench"]["evidence"]
assert inv["state"]["challenge"]["evidence"]["squashed_chip_bands"] == 0
assert set(inv["extra_files"]) == {"Glacial_Lake_Bench/Metadata.txt", "Glacial_Lake_Bench/ann_dir/test/Thumbs.db"}
assert inv["challenge_region_source"] == {"none": 2, "GLB scene": 1, "GLB tile": 1}, inv["challenge_region_source"]
assert inv["splits"] == {"train": 8, "val": 2, "test": 2, "challenge": 4}, inv["splits"]
assert inv["leaks"]["scene_shared_with_train"] == {"challenge": "2/4", "test": "1/2", "val": "0/2"}, inv["leaks"]
assert inv["leaks"]["tiles_in_several_regions"] == {"CA/SEE": 1}, inv["leaks"]

OUT1, OUT2 = TMP / "out1", TMP / "out2"
man = G.convert(RAW_A, OUT1, check_counts=False, workers=1)
man2 = G.convert(RAW_A, OUT2, check_counts=False, workers=3)
assert man["splits"] == {"train": 8, "val": 2, "test": 2, "test_challenge": 4}, man["splits"]
assert man["units"] == "minmax01_per_chip" and man["channels"] == ["VV/HH", "VH/HV"] and man["chip_size"] == 256
assert man["classes"] == {"0": "background", "1": "lake"} or man["classes"] == {0: "background", 1: "lake"}
assert man["pixel_spacing_m"] == 10.0 and man["source"]["doi"] == "10.5281/zenodo.17917359"
assert "minmax" in man["notes"] and "without any SAR pixel" in man["notes"] and "-32768" in man["notes"]
assert "squashed by a fill (set to NaN) per split: {'train': 0, 'val': 1, 'test': 0, 'test_challenge': 0}" in man["notes"]
assert "not comparable with the preprint's GLBC baselines" in man["notes"]
assert "{'test': '1/2', 'test_challenge': '2/4', 'val': '0/2'}" in man["notes"]              # scene sharing
assert "keep_default_na=False" in man["notes"] and "overlap_train > 0 per split" in man["notes"]
assert man == P.load_manifest(OUT1, G.DATASET), "returned manifest differs from the file"
assert not list((OUT1 / G.DATASET).glob(".*part*")), "part files left behind"
v = P.validate(OUT1, G.DATASET)
assert set(v.split) == {"train", "val", "test", "test_challenge"}

official = {**{n: s for s, ns in GLB.items() for n in ns}, **{n: "test_challenge" for n in GLC}}
for split in man["splits"]:
    d1, d2 = P.load_split(OUT1, G.DATASET, split), P.load_split(OUT2, G.DATASET, split)
    assert d1["images"].dtype == np.float16 and d1["images"].shape[1:] == (2, S, S)
    assert d1["labels"].dtype == np.uint8 and d1["labels"].shape[1:] == (S, S)
    # determinism: 1 vs 3 worker threads -> identical arrays and meta
    assert np.array_equal(np.asarray(d1["images"]), np.asarray(d2["images"]), equal_nan=True), split
    assert np.array_equal(np.asarray(d1["labels"]), np.asarray(d2["labels"]))
    assert (OUT1 / G.DATASET / f"{split}_meta.csv").read_text() == (OUT2 / G.DATASET / f"{split}_meta.csv").read_text()
    meta = pd.read_csv(OUT1 / G.DATASET / f"{split}_meta.csv", keep_default_na=False, na_values=[""])
    names = [stem(f) for f in meta.source_file]
    assert names == sorted(n for n in official if official[n] == split), (split, names)   # sorted, complete
    for i, n in enumerate(names):
        exp = raw[n].copy()
        exp[exp == 0] = np.nan                                  # exact 0 = no data in the min-max state
        if n == SQUASHED:
            exp[:] = np.nan                                     # real pixels squashed to ~1 -> whole bands NaN
        assert np.array_equal(np.asarray(d1["images"][i]), exp.astype(np.float16), equal_nan=True), n
        lab = truth[n].copy()
        lab[np.isnan(exp).all(0)] = 255                         # no SAR at all -> ignore
        assert np.array_equal(np.asarray(d1["labels"][i]), lab), n
        assert set(np.unique(d1["labels"][i]).tolist()) <= {0, 1, 255}
        assert meta.s1_squashed[i] == (2 if n == SQUASHED else 0), (n, meta.s1_squashed[i])
        date = [p for p in n.split("_") if len(p) == 8 and p.isdigit()][0]
        assert meta.date[i] == f"{date[:4]}-{date[4:6]}-{date[6:]}", (n, meta.date[i])
        if split != "test_challenge":
            assert meta.region[i] == n.split("_")[0] and meta.region_source[i] == "file name"
        assert meta.crs[i] == f"EPSG:{zone(n)}" and np.isfinite(meta.lon[i]) and np.isfinite(meta.lat[i])
    if split != "train":
        assert "overlap_train" in meta.columns
assert "NA" in pd.read_csv(OUT1 / G.DATASET / "val_meta.csv", keep_default_na=False).region.tolist()

tr = P.load_split(OUT1, G.DATASET, "train")
m = tr["meta"].set_index(tr["meta"].source_file.map(stem))
assert (np.asarray(tr["labels"][list(m.index).index("GL_S2A_24WWU_20200805_2_L2A_7")]) == 255).all()
assert m.loc["GL_S2A_24WWU_20200805_2_L2A_7", "s1_valid"] == 0.0
assert 0.9 < m.loc["AK_S2A_5VPJ_20200817_0_L2A_2", "s1_valid"] < 0.96          # 12 no-data rows of 256
assert m.loc["AK_S2A_5VPJ_20200817_0_L2A_1", "overlap_test_challenge"] == 1.0
assert m.loc["AK_S2A_5VPJ_20200817_0_L2A_2", "overlap_test_challenge"] == 0.0
assert "Chips without any SAR pixel per split: {'train': 1, 'val': 1," in man["notes"]
va = pd.read_csv(OUT1 / G.DATASET / "val_meta.csv", keep_default_na=False, na_values=[""])
assert va.set_index(va.source_file.map(stem)).loc[SQUASHED, "s1_valid"] == 0.0
ch = P.load_split(OUT1, G.DATASET, "test_challenge")["meta"]
c = ch.set_index(ch.source_file.map(stem))
assert c.loc["S2A_5VPJ_20200817_0_L2A_9", ["region", "region_source"]].tolist() == ["AK", "GLB scene"]
assert c.loc["S2B_33XVH_20201227_0_L2A_4", ["region", "region_source"]].tolist() == ["SV", "GLB tile"]
for n in ("S2B_44RPU_20200831_1_L2A_7", "S2A_45UYQ_20200901_0_L2A_2"):        # scene in CA+SEE / tile unknown
    assert c.loc[n, ["region", "region_source"]].tolist() == ["unknown", "none"], c.loc[n]
assert c.loc["S2A_5VPJ_20200817_0_L2A_9", "overlap_train"] == 1.0
assert c.loc["S2A_45UYQ_20200901_0_L2A_2", "overlap_train"] == 0.0
te = P.load_split(OUT1, G.DATASET, "test")["meta"]
t = te.set_index(te.source_file.map(stem))
assert t.loc["CA_S2B_44RPU_20200831_1_L2A_111", "crc32"] == m.loc["SEE_S2B_44RPU_20200831_1_L2A_112", "crc32"]
assert t.loc["CA_S2B_44RPU_20200831_1_L2A_111", "overlap_train"] == 1.0         # duplicate across splits shows
assert str(t.loc["CA_S2B_44RPU_20200831_1_L2A_111", "crc32"]).startswith("0x")
# cross-CRS overlap: test chip (UTM 45) on a train footprint (UTM 44); check against a point sample
ov = t.loc["SEE_S2A_45RXM_20201229_1_L2A_13", "overlap_train"]
a0 = origin["SEE_S2A_45RXM_20201229_1_L2A_13"]
gx, gy = np.meshgrid(a0[0] + 12.8 * (np.arange(200) + 0.5), a0[1] - 12.8 * (np.arange(200) + 0.5))
px, py = warp("EPSG:32645", "EPSG:32644", gx.ravel().tolist(), gy.ravel().tolist())
b0 = origin["CA_S2B_44RPU_20200831_1_L2A_95"]
inside = ((np.array(px) >= b0[0]) & (np.array(px) <= b0[0] + 2560) & (np.array(py) <= b0[1]) & (np.array(py) >= b0[1] - 2560)).mean()
assert 0.8 < ov < 1.0 and abs(ov - inside) < 0.01, (ov, inside)
print(f"cross-CRS overlap {ov} (point sample {inside:.4f})")

# option: 10*log10 of the stretched values
OUT3 = TMP / "out3"
man3 = G.convert(RAW_A, OUT3, check_counts=False, minmax_to_log=True, include_challenge=False)
assert man3["units"] == "dB_rel_chip_max" and "test_challenge" not in man3["splits"]
d = P.load_split(OUT3, G.DATASET, "val")
for i, n in enumerate(stem(f) for f in d["meta"].source_file):
    x = raw[n].copy()
    x[x == 0] = np.nan
    if n == SQUASHED:
        x[:] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        exp = 10.0 * np.log10(x)
    assert np.array_equal(np.asarray(d["images"][i]), exp.astype(np.float16), equal_nan=True)
    if n != SQUASHED:
        assert np.nanmax(exp) == 0.0                                           # the chip maximum -> 0 dB

# option: region hold-out, then overwrite protection and clean replacement
OUT4 = TMP / "out4"
man4 = G.convert(RAW_A, OUT4, check_counts=False, holdout_regions=("SV",))
assert man4["splits"] == {"train": 7, "val": 1, "test": 2, "test_region_SV": 2, "test_challenge": 4}, man4["splits"]
h = P.load_split(OUT4, G.DATASET, "test_region_SV")["meta"]
assert set(h.region) == {"SV"} and set(h.official_split) == {"train", "val"}
assert "SV" not in set(P.load_split(OUT4, G.DATASET, "train")["meta"].region)
assert "0 train chips dropped" in man4["notes"]
expect_fail(lambda: G.convert(RAW_A, OUT4, check_counts=False, overwrite=False), "overwrite=True")
(OUT4 / G.DATASET / ".train_images.part00042.npy").write_bytes(b"stale")             # crashed run's leftovers
(OUT4 / G.DATASET / ".val_labels.part00007.npy").write_bytes(b"stale")
man4b = G.convert(RAW_A, OUT4, check_counts=False, limit_per_split=1)            # default: replace cleanly
assert man4b["splits"] == {"train": 1, "val": 1, "test": 1, "test_challenge": 1}
assert not (OUT4 / G.DATASET / "test_region_SV_images.npy").exists(), "stale held-out split left behind"
assert not list((OUT4 / G.DATASET).glob(".*part*")), "stale part files left behind"
expect_fail(lambda: G.convert(RAW_A, TMP / "x", check_counts=False, holdout_regions=("XX",)), "holdout region")
# a crashed run without manifest: leftovers go even with overwrite=False
OUT6 = TMP / "out6" / G.DATASET
OUT6.mkdir(parents=True)
for f in ("test_region_XX_images.npy", "test_region_XX_meta.csv", ".val_labels.part00001.npy"):
    (OUT6 / f).write_bytes(b"stale")
G.convert(RAW_A, TMP / "out6", check_counts=False, overwrite=False, limit_per_split=1, include_challenge=False)
assert sorted(p.name for p in OUT6.iterdir()) == sorted(["manifest.json"] + [f"{s}_{k}" for s in ("train", "val", "test")
                                                        for k in ("images.npy", "labels.npy", "meta.csv")])

# hold-out CA: byte-identical SEE twins leave train; a non-twin SEE chip on the shared tile stays and is noted
OUT5 = TMP / "out5"
man5 = G.convert(RAW_A, OUT5, check_counts=False, holdout_regions=("CA",), include_challenge=False, workers=2)
assert man5["splits"] == {"train": 5, "val": 2, "test": 1, "test_region_CA": 2}, man5["splits"]
tr5 = P.load_split(OUT5, G.DATASET, "train")["meta"]
ho5 = P.load_split(OUT5, G.DATASET, "test_region_CA")["meta"]
assert not set(tr5.crc32) & set(ho5.crc32), "a held-out chip's twin is still in train"
assert {stem(f) for f in tr5.source_file} == {"AK_S2A_5VPJ_20200817_0_L2A_1", "AK_S2A_5VPJ_20200817_0_L2A_2",
                                            "SV_S2B_33XVH_20200801_0_L2A_5", "SEE_S2B_44RPU_20200831_1_L2A_120",
                                            "GL_S2A_24WWU_20200805_2_L2A_7"}
assert ("2 train chips dropped as byte-identical (CRC-32) to a held-out chip ['SEE_S2B_44RPU_20200831_1_L2A_112.tif', "
        "'SEE_S2B_44RPU_20200831_1_L2A_93.tif']; 1 train chips kept on MGRS tiles that also hold held-out chips "
        "['44RPU']") in man5["notes"], man5["notes"]
assert ho5.overlap_train.tolist() == [0.0, 0.0]                                 # twins gone -> no overlap left
assert "'test_region_CA': '2/2'" in man5["notes"]                                # SEE_120 shares the S2 scene


# ---------------------------------------------------------------- variants B-E: extracted folders, other value states
def folder(root, chips, top="Glacial_Lake_Bench"):
    for (split, n), (img, msk) in chips.items():
        for kind, data in (("img_dir", img), ("ann_dir", msk)):
            p = root / top / kind / split / f"{n}.tif"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)


def chips_with(transform_s1, names, nodata=None, desc=None):
    out, s1raw, masks = {}, {}, {}
    for i, (split, n) in enumerate(names):
        msk = lake_mask()
        st = scene(msk, no_sar_rows=8 if i == 0 else 0)
        st[9:11] = transform_s1(st[9:11], i)
        out[(split, n)] = (tif(st, zone(n), 500000 + 3000 * i, 7100000.0, nodata=nodata, desc=desc),
                           tif(msk[None], zone(n), 500000 + 3000 * i, 7100000.0))
        s1raw[n], masks[n] = st[9:11].copy(), msk
    return out, s1raw, masks


NAMES = [("train", "AK_S2A_5VPJ_20200817_0_L2A_1"), ("train", "SV_S2B_33XVH_20200801_0_L2A_5"),
         ("train", "AK_S2A_5VPJ_20200817_0_L2A_4"), ("val", "AK_S2B_6VXN_20200828_1_L2A_231"),
         ("test", "SEE_S2A_45RXM_20200929_1_L2A_13")]


def lin_fill(s1, i):                                  # half of chip 1 outside the swath: PC RTC nodata -32768
    s1 = s1.copy()
    if i == 1:
        s1[:, :, :128] = G.PC_NODATA
    return s1


# B: linear gamma0 (the generation code's default output) -> dB; 10 % of the pixels are -32768
RAW_B = TMP / "raw_b"
cb, sb, mb = chips_with(lin_fill, NAMES, desc=tuple(["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "NDWI", "Slope",
                                                     "Elevation", "VV", "VH"]))
folder(RAW_B, cb)
inv_b = G.inventory(RAW_B, check_counts=False)
assert inv_b["state"]["bench"]["state"] == "linear" and "challenge" not in inv_b["state"]
assert inv_b["state"]["bench"]["evidence"]["share_pc_nodata_-32768"] > 0.05
expect_fail(lambda: G.convert(RAW_B, TMP / "out_b", check_counts=False), "include_challenge=False")
man_b = G.convert(RAW_B, TMP / "out_b", check_counts=False, include_challenge=False)
assert man_b["units"] == "dB" and man_b["splits"] == {"train": 3, "val": 1, "test": 1}
d = P.load_split(TMP / "out_b", G.DATASET, "train")
for i, n in enumerate(stem(f) for f in d["meta"].source_file):
    x = sb[n].copy()
    x[~(x > 0)] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        exp = 10.0 * np.log10(x)
    assert np.array_equal(np.asarray(d["images"][i]), exp.astype(np.float16), equal_nan=True), n
    assert d["meta"].pol[i] == "VV/VH"                                        # from band descriptions
    lab = mb[n].copy()
    lab[np.isnan(exp).all(0)] = 255
    assert np.array_equal(np.asarray(d["labels"][i]), lab)
assert (np.asarray(d["labels"][list(map(stem, d["meta"].source_file)).index("SV_S2B_33XVH_20200801_0_L2A_5")])[:, :128] == 255).all()
v = P.validate(TMP / "out_b", G.DATASET)
assert -40 < v.loc[0, "VV/HH_p1"] < v.loc[0, "VV/HH_p99"] < 10                   # plausible dB levels

# C: dB with a nodata value in the GeoTIFF tag, plus some -32768
RAW_C = TMP / "raw_c"


def to_db(s1, i):
    with np.errstate(divide="ignore"):
        x = (10 * np.log10(s1)).astype(np.float32)
    x[~np.isfinite(x)] = -9999.0
    x[:, -5:, :7] = np.nan
    if i == 2:
        x[:, :3] = G.PC_NODATA
    return x


cc, sc, _ = chips_with(to_db, NAMES[:3], nodata=-9999.0)
folder(RAW_C, cc)
man_c = G.convert(RAW_C, TMP / "out_c", check_counts=False, include_challenge=False)
assert man_c["units"] == "dB" and "'db'" in man_c["notes"]
d = P.load_split(TMP / "out_c", G.DATASET, "train")
for i, n in enumerate(stem(f) for f in d["meta"].source_file):
    exp = sc[n].copy()
    exp[(exp == -9999.0) | (exp == G.PC_NODATA)] = np.nan
    assert np.array_equal(np.asarray(d["images"][i]), exp.astype(np.float16), equal_nan=True), n
assert np.isnan(np.asarray(d["images"][list(map(stem, d["meta"].source_file)).index("AK_S2A_5VPJ_20200817_0_L2A_4")])[:, :3]).all()

# D: per-chip 2-98 % stretch of dB ([C] per_modality)
RAW_D = TMP / "raw_d"


def pct(s1, i):
    out = np.empty_like(s1)
    for b in range(2):
        x = 10 * np.log10(np.clip(s1[b], 1e-10, None))
        lo, hi = np.percentile(x, 2), np.percentile(x, 98)
        out[b] = np.clip((x - lo) / (hi - lo), 0, 1)
    return out


cd, sd, _ = chips_with(pct, NAMES[:3])
folder(RAW_D, cd)
man_d = G.convert(RAW_D, TMP / "out_d", check_counts=False, include_challenge=False)
assert man_d["units"] == "pct2_98_dB_per_chip"
d = P.load_split(TMP / "out_d", G.DATASET, "train")
n0 = stem(d["meta"].source_file[0])
assert np.array_equal(np.asarray(d["images"][0]), sd[n0].astype(np.float16), equal_nan=True)   # kept as is

# D2: min-max with an epsilon, (x - lo) / (hi - lo + 1e-6): still "minmax"
RAW_D2 = TMP / "raw_d2"
cd2, _, _ = chips_with(lambda s1, i: np.stack([(b - b.min()) / (b.max() - b.min() + 1e-6) for b in s1]), NAMES[:3])
folder(RAW_D2, cd2)
assert G.inventory(RAW_D2, check_counts=False)["state"]["bench"]["state"] == "minmax"

# B2: every value in 0-1 but chip maxima below 1 -> ambiguous (scene/dataset-wide stretch or dark gamma0): STOP
AMBIG = ("ambiguous SAR value state", "whole scene or the whole dataset", "s1_scale='linear'")
RAW_B2 = TMP / "raw_b2"
cb2, sb2, _ = chips_with(lambda s1, i: np.clip(s1, 0, 0.9), NAMES[:3])         # dark linear gamma0
folder(RAW_B2, cb2)
expect_fail(lambda: G.inventory(RAW_B2, check_counts=False), *AMBIG)
expect_fail(lambda: G.convert(RAW_B2, TMP / "out_b2", check_counts=False, include_challenge=False), *AMBIG)
man_b2 = G.convert(RAW_B2, TMP / "out_b2", s1_scale="linear", check_counts=False, include_challenge=False)
assert man_b2["units"] == "dB" and "forced" in man_b2["notes"]                 # the user's explicit decision
d = P.load_split(TMP / "out_b2", G.DATASET, "train")
n0 = stem(d["meta"].source_file[0])
x = sb2[n0].copy()
x[~(x > 0)] = np.nan
with np.errstate(divide="ignore", invalid="ignore"):
    assert np.array_equal(np.asarray(d["images"][0]), (10 * np.log10(x)).astype(np.float16), equal_nan=True)
RAW_B3 = TMP / "raw_b3"                                                         # one stretch over the dataset
cb3, _, _ = chips_with(lambda s1, i: s1 / 5.0, NAMES[:3])
folder(RAW_B3, cb3)
expect_fail(lambda: G.inventory(RAW_B3, check_counts=False), *AMBIG)

# E: failures that must stop the run with a message naming the assumption
RAW_E = TMP / "raw_e"                                              # one stretch over all 11 bands
ce = {}
for i, (split, n) in enumerate(NAMES[:3]):
    msk = lake_mask()
    st = scene(msk)
    st = (st - st.min()) / (st.max() - st.min())
    ce[(split, n)] = (tif(st, zone(n), 0.0 + 3000 * i, 7e6), tif(msk[None], zone(n), 0.0 + 3000 * i, 7e6))
folder(RAW_E, ce)
expect_fail(lambda: G.convert(RAW_E, TMP / "out_e", check_counts=False, include_challenge=False),
            "all 11 bands look stretched TOGETHER")
RAW_F = TMP / "raw_f"                                              # 10 bands, mask value 2, bad name, no mask
msk = lake_mask()
folder(RAW_F, {("train", "AK_S2A_5VPJ_20200817_0_L2A_1"): (tif(np.zeros((10, S, S), np.float32), 32605, 0, 7e6),
                                                          tif(msk[None], 32605, 0, 7e6))})
expect_fail(lambda: G.convert(RAW_F, TMP / "out_f", s1_scale="minmax", check_counts=False,
                              include_challenge=False), "10 bands", "documented 11")
shutil.rmtree(RAW_F)
bad = msk.copy()
bad[0, 0] = 2
st = per_band_minmax(scene(msk))
folder(RAW_F, {("train", "AK_S2A_5VPJ_20200817_0_L2A_1"): (tif(st, 32605, 0, 7e6), tif(bad[None], 32605, 0, 7e6))})
expect_fail(lambda: G.convert(RAW_F, TMP / "out_f", check_counts=False, include_challenge=False),
            "mask values [0, 1, 2]", "documented 0 background, 1 lake")
shutil.rmtree(RAW_F)
folder(RAW_F, {("train", "AK_S2A_5VPJ_20200817_0_L2A_1"): (tif(st, 32605, 0, 7e6), tif(msk[None], 32605, 50, 7e6))})
expect_fail(lambda: G.convert(RAW_F, TMP / "out_f", check_counts=False, include_challenge=False),
            "differs from its image")
shutil.rmtree(RAW_F)
folder(RAW_F, {("train", "XX_S2A_5VPJ_20200817_0_L2A_1"): (tif(st, 32605, 0, 7e6), tif(msk[None], 32605, 0, 7e6))})
expect_fail(lambda: G.inventory(RAW_F, check_counts=False), "region code XX")
(RAW_F / "Glacial_Lake_Bench/ann_dir/train/XX_S2A_5VPJ_20200817_0_L2A_1.tif").unlink()
(RAW_F / "Glacial_Lake_Bench/img_dir/train/XX_S2A_5VPJ_20200817_0_L2A_1.tif").rename(
    RAW_F / "Glacial_Lake_Bench/img_dir/train/AK_S2A_5VPJ_20200817_0_L2A_1.tif")
expect_fail(lambda: G.inventory(RAW_F, check_counts=False), "without a mask of the same name")
# an extracted folder with a whole split missing (partial extraction) stops the count check
expect_fail(lambda: G.inventory(RAW_B), "train: 3 pairs, expected 15300", "val: 1 pairs, expected 1912")
shutil.rmtree(RAW_B / "Glacial_Lake_Bench" / "img_dir" / "val")
expect_fail(lambda: G.inventory(RAW_B), "val: 0 pairs, expected 1912")


# ---------------------------------------------------------------- download: resume, md5, skip, restarts, disk (local server only)
class Handler(http.server.BaseHTTPRequestHandler):
    data, bad_range, gets = b"", False, 0

    def do_GET(self):
        Handler.gets += 1
        d, r = Handler.data, self.headers.get("Range")
        if r:
            start = 0 if Handler.bad_range else int(r.split("=")[1].split("-")[0])
            body = d[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(d) - 1}/{len(d)}")
        else:
            body = d
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


os.environ["NO_PROXY"] = "127.0.0.1,localhost"
Handler.data = rng.integers(0, 256, 3_000_000, dtype=np.uint8).tobytes()
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
saved, saved_md5 = G.FILES, G._md5
try:
    MD5 = hashlib.md5(Handler.data).hexdigest()
    G.FILES = {"bench": {**saved["bench"], "url": f"http://127.0.0.1:{srv.server_port}/b",
                         "size": len(Handler.data), "md5": MD5}}
    dl = TMP / "dl"
    dl.mkdir()
    zp, mk, pt = dl / "Glacial_Lake_Bench.zip", dl / "Glacial_Lake_Bench.zip.md5ok", dl / "Glacial_Lake_Bench.zip.part"
    pt.write_bytes(Handler.data[:1_234_567])                                          # interrupted earlier
    got = G.download(dl, which=("bench",), chunk_mb=1)
    assert got == [zp] and zp.read_bytes() == Handler.data and mk.read_text() == MD5
    t = zp.stat().st_mtime_ns
    G.download(dl, which=("bench",))                                                   # complete -> skipped
    assert zp.stat().st_mtime_ns == t
    # a marker from another md5 does not count: the file is verified again
    calls = []
    G._md5 = lambda p: (calls.append(p), saved_md5(p))[1]
    mk.write_text("0" * 32)
    G.download(dl, which=("bench",))
    assert len(calls) == 1 and mk.read_text() == MD5
    G._md5 = saved_md5
    # a 206 whose Content-Range does not start at the resume offset: restart from 0, result still correct
    zp.unlink(); mk.unlink()
    pt.write_bytes(Handler.data[:777_777])
    Handler.bad_range, Handler.gets = True, 0
    G.download(dl, which=("bench",), chunk_mb=1)
    assert zp.read_bytes() == Handler.data and Handler.gets == 2, Handler.gets
    Handler.bad_range = False
    # not enough free disk for the rest: stop before fetching
    zp.unlink(); mk.unlink()
    Handler.gets = 0
    G.shutil = types.SimpleNamespace(disk_usage=lambda p: types.SimpleNamespace(free=1_000_000))
    expect_fail(lambda: G.download(dl, which=("bench",)), "still to download", "GB free")
    G.shutil = shutil
    assert Handler.gets == 0
    # disk full while writing: raised at once, not retried 8 times
    opened = []

    class Full(io.RawIOBase):
        def write(self, b):
            raise OSError(errno.ENOSPC, "No space left on device")

    G.open = lambda *a, **k: (opened.append(a), Full())[1]
    t1 = time.time()
    e = expect_fail(lambda: G.download(dl, which=("bench",)), "not retried", exc=OSError)
    del G.open
    assert len(opened) == 1 and time.time() - t1 < 5, (opened, time.time() - t1)
    # md5 mismatch -> .bad
    G.FILES["bench"]["md5"] = "0" * 32
    G.download(dl, which=("bench",), verify=False)                                     # fetch without md5
    assert zp.read_bytes() == Handler.data
    expect_fail(lambda: G.download(dl, which=("bench",)), "md5")
    assert (dl / "Glacial_Lake_Bench.zip.bad").exists() and not zp.exists()
finally:
    G.FILES, G._md5, G.shutil = saved, saved_md5, shutil
    if "open" in vars(G):
        del G.open
    srv.shutdown()

assert G.FILES["bench"]["size"] == 44268636160 and G.FILES["challenge"]["md5"] == "f2a9a1e83eb4ca0a5d37544f282ae51f"
import gc
for name in ("d", "d1", "d2", "tr", "tr5", "ho5", "h", "ch", "te"):          # memory maps hold Windows file locks
    globals().pop(name, None)
gc.collect()
shutil.rmtree(TMP)
assert not TMP.exists()
print(f"glacial_lakes verified in {time.time() - T0:.0f} s: zip + folder sources, inventory, value states "
      f"minmax (+eps, +-32768 squash) / minmax->log / linear->dB (+-32768) / dB+nodata (+-32768) / pct stretch, "
      f"ambiguous 0-1 stop + forced linear, joint-stretch + 10-band + mask-code + footprint + name + pairing + "
      f"missing-split failures, official splits + test_challenge + region hold-out (twin drop, tile note), GLC region "
      f"'unknown', ignore on no-SAR, overlap (same + cross CRS) + crc meta, determinism across threads, overwrite + "
      f"stale part files, download resume/md5/skip/marker/Content-Range/disk-free/ENOSPC")
