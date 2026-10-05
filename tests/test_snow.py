"""Offline test of domains.snow on tiny synthetic files in the documented SnowSAR layout.

Layout, names, band order, dtypes and codes follow the README and the zip's
central directory (read 2026-09-26): snowsar_dataset/<split>/values/<orbit>_<YYYYMMDD>.tif
(2-band float32 dB, VV then VH) paired by name with
snowsar_dataset/<split>/labels/{raw,cni,ks}/<same>.tif (1-band uint8, 0 no snow,
1 snow, -1 no data -> stored as 255 in uint8). File names below are real names
from the listing. The scenes are small, so RAM stays low.

Section 4 also covers the fixes after the independent review: calibration-free
value checks (uncalibrated positive dB accepted, linear power and amplitude refused,
check_values override), exact VV == VH == 0 pairs recorded instead of stopping
(also when only a middle scene has them, and with > 50 % zero fill), statistics
printed before any assertion, band descriptions, a label nodata tag of 0, the
float16 range, the water_year column and README periods, and the Table 1 shape
print. Section 6 runs inventory() on the REAL zip listing (central directory only,
no member data) and checks TABLE1_SHAPES against the real member sizes.
"""
import contextlib
import hashlib
import io
import pickle
import shutil
import struct
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from _setup import REF, TMP_BASE  # noqa: E402  (this repo's src first, import guard, temp in one folder)
from sartransfer.domains import prepared as P, snow as S  # noqa: E402

import rasterio  # noqa: E402,F401
from rasterio.errors import NotGeoreferencedWarning  # noqa: E402
from rasterio.io import MemoryFile  # noqa: E402

warnings.simplefilter("ignore", NotGeoreferencedWarning)
TMP = TMP_BASE / "tmp_snow"                # under the tests' temp folder
shutil.rmtree(TMP, ignore_errors=True)
TMP.mkdir()
CH = 512
P.CHUNK = 5                                # force several part files per split


def tif(arr, dtype, nodata=None, desc=None, compress=None) -> bytes:
    arr = np.asarray(arr)
    arr = arr[None] if arr.ndim == 2 else arr
    prof = dict(driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0], dtype=dtype)
    if nodata is not None:
        prof["nodata"] = nodata
    if compress:
        prof["compress"] = compress
    with MemoryFile() as mf:
        with mf.open(**prof) as dst:
            dst.write(arr.astype(dtype))
            for i, d in enumerate(desc or [], 1):
                dst.set_band_description(i, d)
        return mf.read()


def sar(h, w, seed):
    r = np.random.default_rng(seed)
    x = np.stack([r.normal(-10, 3, (h, w)), r.normal(-17, 3, (h, w))]).astype(np.float32)
    x[:, :20, :30] = np.nan                # no-data corner
    x[0, 40, 40] = -np.inf                 # 10*log10(0)
    x[1, 45, 45] = np.inf
    return x


def label(h, w, seed, cloudy=False, unlabelled_below=None):
    r = np.random.default_rng(seed + 1000)
    lab = np.zeros((h, w), np.uint8)
    lab[:, w // 2:] = 1
    lab[r.random((h, w)) < 0.2] = 255      # cloud gaps: README -1, stored in uint8 as 255
    if unlabelled_below is not None:
        lab[unlabelled_below:] = 255
    if cloudy:
        lab[:] = 255
    return lab


# folder, file, h, w, options
SCENES = [
    ("train", "139_20180918.tif", 600, 1100, dict(unlabelled_below=512)),   # bottom chips skipped
    ("train", "66_20190417.tif", 520, 1030, dict(cloudy=True)),             # raw all no-data -> 0 chips
    ("train", "88_20181219.tif", 512, 1024, {}),
    ("validation", "139_20180924.tif", 600, 700, {}),
    ("validation", "88_20180908.tif", 530, 600, {}),
    ("test/no_transfer", "66_20181206.tif", 520, 1030, {}),
    ("test/no_transfer", "88_20181002.tif", 512, 512, {}),
    ("test/spatial_transfer", "139_20180918.tif", 600, 800, {}),            # same name as a train scene
    ("test/spatial_transfer", "161_20190312.tif", 700, 600, {}),
    ("test/temporal_transfer", "139_20190901.tif", 600, 700, {}),
    ("test/temporal_transfer", "88_20200530.tif", 512, 600, {}),
]
EXP_SCENES = {"train": 3, "validation": 2, "test/no_transfer": 2, "test/spatial_transfer": 2,
              "test/temporal_transfer": 2}
SPLIT_OF = {k: v[0] for k, v in S.SPLITS.items()}
BASIN = {k: v[1] for k, v in S.SPLITS.items()}


def dirs_for(names):
    out = set()
    for n in names:
        parts = n.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            out.add("/".join(parts[:i]) + "/")
    return sorted(out)


def build_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for d in dirs_for(members):
            zf.writestr(zipfile.ZipInfo(d), b"")                     # directory entries, as in the real zip
        for n, b in members.items():
            zf.writestr(n, b)


def make_dataset(root: Path):
    """The full synthetic dataset; returns (arrays by member name) for exact checks."""
    root.mkdir(parents=True, exist_ok=True)
    members, truth = {}, {}
    for k, (folder, f, h, w, opt) in enumerate(SCENES):
        x = sar(h, w, k)
        lab = label(h, w, k, **opt)
        v = f"snowsar_dataset/{folder}/values/{f}"
        members[v] = tif(x, "float32")
        truth[v] = x
        variants = {"raw": lab}
        if folder == "train":
            filled = label(h, w, k + 50)                             # interpolated: fewer gaps
            filled[filled == 255] = 0
            variants.update(cni=filled, ks=1 - filled)
        for var, arr in variants.items():
            ln = f"snowsar_dataset/{folder}/labels/{var}/{f}"
            members[ln] = tif(arr, "uint8", compress="lzw")
            truth[ln] = arr
    bio = io.BytesIO()
    np.savez_compressed(bio, D139=np.ones((4, 4), bool))
    members["snowsar_dataset/projection/masks.npz"] = bio.getvalue()
    members["snowsar_dataset/stats/Guil_2018.pkl"] = pickle.dumps({"mean": [-10.0, -17.0], "std": [3.0, 3.0]})
    members["snowsar_dataset/reference_images/ref_Guil_2018_139.tif"] = tif(sar(64, 64, 99), "float32")
    build_zip(root / S.ZIP_NAME, members)
    (root / "README.md").write_text("synthetic stand-in for the official README\n", encoding="utf-8")
    return truth


def expected_chips(x, lab, chip=CH):
    """Independent re-statement of the chip rule: pad NaN/255, ignore where SAR NaN, keep if any label."""
    x = np.where(np.isinf(x), np.nan, x)
    h, w = lab.shape
    out = []
    for y0 in range(0, h, chip):
        for x0 in range(0, w, chip):
            img = np.full((2, chip, chip), np.nan, np.float32)
            lc = np.full((chip, chip), 255, np.uint8)
            sub = x[:, y0:y0 + chip, x0:x0 + chip]
            img[:, :sub.shape[1], :sub.shape[2]] = sub
            lc[:sub.shape[1], :sub.shape[2]] = lab[y0:y0 + chip, x0:x0 + chip]
            lc[np.isnan(img).any(0)] = 255
            if (lc != 255).any():
                out.append((y0, x0, img, lc))
    return out


def check_output(out_root, truth, label_var_train="raw", splits=None):
    man = P.load_manifest(out_root, "snow")
    exp_counts = {}
    for folder, f, h, w, opt in SCENES:
        sp = SPLIT_OF[folder]
        if splits and sp not in splits:
            continue
        var = label_var_train if folder == "train" else "raw"
        v = f"snowsar_dataset/{folder}/values/{f}"
        n = len(expected_chips(truth[v], truth[f"snowsar_dataset/{folder}/labels/{var}/{f}"]))
        exp_counts[sp] = exp_counts.get(sp, 0) + n
    exp_counts = {k: v for k, v in exp_counts.items() if v}
    assert man["splits"] == dict(sorted(exp_counts.items())), (man["splits"], exp_counts)
    assert man["classes"] == {"0": "no_snow", "1": "snow"} and man["ignore_index"] == 255
    assert man["channels"] == ["VV", "VH"] and man["units"] == "dB" and man["chip_size"] == CH
    assert man["pixel_spacing_m"] is None and man["source"]["doi"] == "10.57745/IMTSFL"
    assert man["source"]["name"].startswith("Replication Data for : Weakly")      # exact API title
    assert "radiometric calibration is not stated" in man["notes"] and "masks.npz" in man["notes"]
    assert "kept as 0.0 and counted per chip" in man["notes"]
    for sp in man["splits"]:
        d = P.load_split(out_root, "snow", sp, mmap=False)
        img, lab, meta = d["images"], d["labels"], d["meta"]
        assert img.dtype == np.float16 and img.shape[1:] == (2, CH, CH) and lab.dtype == np.uint8
        assert set(np.unique(lab)) <= {0, 1, 255}
        assert (lab[np.isnan(img).any(1)] == 255).all(), "label must be ignore where SAR is NaN"
        assert ((lab != 255).reshape(len(lab), -1).any(1)).all(), "chips without labels must be skipped"
        assert np.isfinite(img[~np.isnan(img)]).all()
        for col in ("chip_id", "region", "date", "water_year", "orbit", "orbit_pass", "source", "label_source",
                    "y0", "x0", "zero_pairs"):
            assert col in meta.columns, col
        folder = {v: k for k, v in SPLIT_OF.items()}[sp]
        assert set(meta.region) == {BASIN[folder]}, (sp, set(meta.region))
        assert set(meta.water_year) == {S.WATER_YEAR[folder]}, (sp, set(meta.water_year))
        assert (meta.zero_pairs == 0).all()
        var = label_var_train if sp == "train" else "raw"
        assert meta.label_source.str.contains(f"/labels/{var}/", regex=False).all()
        # every chip equals the independently cut chip of its scene
        for i, row in meta.iterrows():
            f = row.source.split("/")[-1]
            orbit, ymd = f[:-4].split("_")
            assert row.date == f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}" and int(row.orbit) == int(orbit)
            assert row.orbit_pass == {"139": "descending", "66": "descending", "88": "ascending",
                                      "161": "ascending"}[orbit]
            ref = {(a, b): (c, e) for a, b, c, e in expected_chips(truth[row.source], truth[row.label_source])}
            ei, el = ref[(int(row.y0), int(row.x0))]
            assert np.array_equal(img[i], ei.astype(np.float16), equal_nan=True), (sp, i)
            assert np.array_equal(lab[i], el), (sp, i)
    return man


# ---------------------------------------------------------------- 1. inventory + convert + validate
raw = TMP / "raw"
truth = make_dataset(raw)
rep = S.inventory(raw, sample=2, expected_scenes=EXP_SCENES)
assert rep["scenes"] == {"train": 3, "val": 2, "test": 2, "test_spatial": 2, "test_temporal": 2}, rep["scenes"]
assert set(rep["label_codes"]) <= {0, 1, 255} and rep["zero_pairs"] == 0
for r in rep["sampled"]:
    assert r["shape_vs_table1"] == "differs" and r["VV_tail_ratio"] < S.LINEAR_TAIL and r["VV_negative_share"] > 0.9
try:
    S.inventory(raw, sample=1)                                 # the real counts (88, 24, ...) must not match
    raise SystemExit("inventory accepted wrong scene counts")
except AssertionError as e:
    assert "published zip lists" in str(e)

man1 = S.convert(raw, TMP / "out1")
assert set(man1["splits"]) == {"train", "val", "test", "test_spatial", "test_temporal"}, man1["splits"]
check_output(TMP / "out1", truth)
assert "train: " in man1["notes"] and "scenes without a negative value: VV 0/3, VH 0/3" in man1["notes"]
v = P.validate(TMP / "out1", "snow")
assert set(v.split) == set(man1["splits"]) and (v.nan_share > 0).all()
m = P.load_split(TMP / "out1", "snow", "train")["meta"]
assert "66_20190417" not in "".join(m.source), "all-cloud raw scene must give no chips"
assert not list((TMP / "out1" / "snow").glob(".*part*"))

# ---------------------------------------------------------------- 2. determinism
S.convert(raw, TMP / "out2")
for sp in man1["splits"]:
    a, b = P.load_split(TMP / "out1", "snow", sp, mmap=False), P.load_split(TMP / "out2", "snow", sp, mmap=False)
    assert np.array_equal(a["images"], b["images"], equal_nan=True) and np.array_equal(a["labels"], b["labels"])
    pd.testing.assert_frame_equal(a["meta"], b["meta"])

# ---------------------------------------------------------------- 3. label variant, subset, dry run
man3 = S.convert(raw, TMP / "out3", labels="cni", splits=("train", "val"))
assert set(man3["splits"]) == {"train", "val"}
check_output(TMP / "out3", truth, label_var_train="cni", splits=("train", "val"))
assert "66_20190417" in "".join(P.load_split(TMP / "out3", "snow", "train")["meta"].source)
man4 = S.convert(raw, TMP / "out4", splits=("test_spatial",), max_scenes_per_split=1)
m4 = P.load_split(TMP / "out4", "snow", "test_spatial")["meta"]
assert list(man4["splits"]) == ["test_spatial"] and set(m4.date) == {"2018-09-18"} and set(m4.region) == {"Gyronde"}
assert "DRY RUN" in man4["notes"]
man5 = S.convert(raw, TMP / "out5", labels="cni", splits=("train",), max_scenes_per_split=2)
m5 = P.load_split(TMP / "out5", "snow", "train")["meta"]
assert {s.split("/")[-1] for s in m5.source} == {"139_20180918.tif", "66_20190417.tif"}, "subset = first + last date"
for bad in (dict(labels="interpolated"), dict(splits=("validation",))):
    try:
        S.convert(raw, TMP / "outx", **bad)
        raise SystemExit(f"accepted {bad}")
    except ValueError:
        pass


# ---------------------------------------------------------------- 4. run-time checks of unverified facts
def mini(name, x=None, lab=None, xdtype="float32", ldtype="uint8", lab_file=True, lab_nodata=None,
         files=("88_20181219.tif",), **kw):
    """A zip with train scenes of 512 x 600; x and lab may be one array or a list with one entry per file."""
    r = TMP / name
    r.mkdir()
    mem = {}
    for k, f in enumerate(files):
        xk = x[k] if isinstance(x, list) else x
        xk = sar(512, 600, 7 + k) if xk is None else xk
        lk = lab[k] if isinstance(lab, list) else lab
        lk = label(512, 600, 7) if lk is None else lk
        mem[f"snowsar_dataset/train/values/{f}"] = tif(xk, xdtype, **kw)
        if lab_file:
            for var in ("raw", "cni", "ks"):                       # train has all three variants
                mem[f"snowsar_dataset/train/labels/{var}/{f}"] = tif(lk, ldtype, nodata=lab_nodata, compress="lzw")
    build_zip(r / S.ZIP_NAME, mem)
    return r


def refuses(root, text, **kw):
    try:
        S.convert(root, root / "out", splits=("train",), **kw)
    except AssertionError as e:
        assert text in str(e), str(e)
        return str(e)
    raise SystemExit(f"{root.name}: accepted")


def ok(root, out="ok", **kw):
    man = S.convert(root, root / out, splits=("train",), **kw)
    return man, P.load_split(root / out, "snow", "train", mmap=False)


def inv_refuses(root, text):
    """inventory() must print the scene's statistics row BEFORE the assertion stops it."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            S.inventory(root, sample=1, expected_scenes=None)
            raise SystemExit(f"{root.name}: inventory accepted")
        except AssertionError as e:
            assert text in str(e), str(e)
    return buf.getvalue()


rng = np.random.default_rng(5)

# 4a. exact VV == VH == 0 pairs along the bottom: default keeps + counts them, no stop (review risk 2)
xz = sar(512, 600, 7)
xz[:, 500:, :] = 0.0
rz = mini("zeros", x=xz)
man, d = ok(rz, "k")
assert d["meta"].zero_pairs.sum() == 12 * 600 and sorted(d["meta"].zero_pairs) == [12 * 88, 12 * 512]
zc = ((d["images"][:, 0] == 0) & (d["images"][:, 1] == 0)).reshape(len(d["meta"]), -1).sum(1)
assert (zc == d["meta"].zero_pairs.to_numpy()).all(), "exact zeros survive float16 and match meta zero_pairs"
assert "7200 VV == VH == 0.0 pixels in 1 scenes (7200 in kept chips, kept as 0.0)" in man["notes"]
S.convert(rz, rz / "t", splits=("train",), zero_is_nodata=True)
d = P.load_split(rz / "t", "snow", "train", mmap=False)
top = (d["meta"].y0 == 0).to_numpy()
assert np.isnan(d["images"][top][:, :, 500:512]).all() and (d["labels"][top][:, 500:512] == 255).all()
assert d["meta"].zero_pairs.sum() == 7200 and "set to NaN / label 255" in P.load_manifest(rz / "t", "snow")["notes"]
S.convert(rz, rz / "f", splits=("train",), zero_is_nodata=False)
d = P.load_split(rz / "f", "snow", "train", mmap=False)
assert (d["images"][((d["meta"].y0 == 0) & (d["meta"].x0 == 0)).to_numpy()][:, :, 500:512] == 0).all()
rep = S.inventory(rz, sample=1, expected_scenes=None)
assert rep["zero_pairs"] == 12 * 600 and rep["sampled"][0]["zero_rows_cols"] == (500, 511, 0, 599)

# 4b. > 50 % exact-zero fill: inventory reports it (no "linear" misdiagnosis), convert runs (review nit 4)
xh = sar(512, 600, 7)
xh[:, :, :330] = 0.0
rh = mini("zerohalf", x=xh)
rep = S.inventory(rh, sample=1, expected_scenes=None)
assert rep["zero_pairs"] == 512 * 330 and rep["sampled"][0]["VV_negative_share"] > 0.9
man, d = ok(rh)
assert d["meta"].zero_pairs.sum() == 512 * 330

# 4c. zero pairs only in a MIDDLE scene (inventory samples first + last): convert no longer stops on scene 2
files3 = ("139_20180906.tif", "139_20180918.tif", "139_20180930.tif")
xm = sar(512, 600, 21)
xm[:, 500:, :] = 0.0
rm = mini("latezero", x=[None, xm, None], files=files3)
rep = S.inventory(rm, sample=2, expected_scenes=None)
assert rep["zero_pairs"] == 0 and [r["file"] for r in rep["sampled"]] == [files3[0], files3[2]]
man, d = ok(rm)
assert man["splits"] == {"train": 6}
per_scene = d["meta"].groupby("source").zero_pairs.sum().to_dict()
assert per_scene == {f"snowsar_dataset/train/values/{f}": (7200 if f == files3[1] else 0) for f in files3}

# 4d. uncalibrated amplitude in dB (all values positive) is accepted (review risk 1)
xu = np.stack([rng.normal(38, 3, (512, 600)), rng.normal(31, 3, (512, 600))]).astype(np.float32)
assert xu.min() > 0
man, d = ok(mini("uncal_sym", x=xu))
assert man["splits"] == {"train": 2} and "scenes without a negative value: VV 1/1, VH 1/1" in man["notes"]
assert abs(float(np.nanmedian(d["images"][:, 0].astype(np.float32))) - 38) < 0.2
inten = rng.exponential(1.0, (2, 512, 600))                     # single-look speckle, clipped at p0.5/p99.5
inten = np.clip(inten, np.percentile(inten, 0.5), np.percentile(inten, 99.5))
xs = np.stack([40 + 10 * np.log10(inten[0]), 33 + 10 * np.log10(inten[1])]).astype(np.float32)
assert xs.min() > 0
rs = mini("uncal_speckle", x=xs)
rep = S.inventory(rs, sample=1, expected_scenes=None)
assert rep["sampled"][0]["VV_negative_share"] == 0 and rep["sampled"][0]["VV_tail_ratio"] < 0.6
ok(rs)

# 4e. linear power and linear amplitude are refused; check_values=False overrides and stores unchanged
xl = sar(512, 600, 7)
rl = mini("linear", x=np.where(np.isfinite(xl), 10 ** (xl / 10), xl))
refuses(rl, "look linear")
assert "VV_p1_p50_p99" in inv_refuses(rl, "look linear")
amp = 100 * np.sqrt(rng.exponential(1.0, (2, 512, 600)))       # Rayleigh amplitude, tail ratio ~1.8
amp[1] *= 0.4
amp = amp.astype(np.float32)
ra = mini("linear_amp", x=amp)
refuses(ra, "look linear")
man, d = ok(ra, check_values=False)
assert "Value checks (dB look, band order): OFF" in man["notes"]
assert np.array_equal(d["images"][(d["meta"].x0 == 0).to_numpy()][0], amp[:, :, :512].astype(np.float16))

# 4f. per-band standardised values fail the band-order check with a message that says so
xzs = np.stack([rng.normal(-0.05, 1, (512, 600)), rng.normal(0.05, 1, (512, 600))]).astype(np.float32)
rzs = mini("zscored", x=xzs)
refuses(rzs, "standardised")
ok(rzs, check_values=False)

# 4g. band descriptions: only a wrong polarisation order stops the run (review nit 6)
refuses(mini("desc", desc=["VH", "VV"]), "wrong order")
refuses(mini("desc_vv2", desc=["VV", "VV"]), "wrong order")
for nm, ds in (("desc_ok", ["VV", "VH"]), ("desc_other", ["Band 1", "Band 2"]),
               ("desc_sigma", ["Sigma0_VV_db", "Sigma0_VH_db"])):
    ok(mini(nm, desc=ds))

# 4h. a label nodata tag of 0 collides with class 0: the README wins, the tag is ignored and reported
rt = mini("labtag0", lab_nodata=0)
rep = S.inventory(rt, sample=1, expected_scenes=None)
assert rep["sampled"][0]["label_nodata_tag"] == 0.0 and rep["sampled"][0]["label_tag_ignored"] == 0.0
man, d = ok(rt)
ref = P.load_split(rz / "f", "snow", "train", mmap=False)       # same scene seed, same labels, no tag
assert np.array_equal(ref["labels"], d["labels"]) and "1 label files with an ignored nodata tag 0/1" in man["notes"]

# 4i. values beyond float16 would become inf in the prepared file: NaN + label 255 instead, counted
xb = sar(512, 600, 7)
xb[0, 100, 100:105] = 1e6
xb[1, 101, 100:103] = -1e5
man, d = ok(mini("f16", x=xb))
sel = (d["meta"].x0 == 0).to_numpy()
c, lc = d["images"][sel][0], d["labels"][sel][0]
assert np.isnan(c[0, 100, 100:105]).all() and np.isnan(c[1, 101, 100:103]).all()
assert (lc[100, 100:105] == 255).all() and (lc[101, 100:103] == 255).all() and not np.isinf(d["images"]).any()
assert "8 values beyond float16 -> NaN" in man["notes"]

# 4j. water_year from the date; a date outside the split's README period stops at the listing
for nm, f in (("wy_late", "139_20190905.tif"), ("wy_summer", "139_20180715.tif")):
    r = mini(nm, files=(f,))
    refuses(r, "README period")
    try:
        S.inventory(r, sample=0, expected_scenes=None)
        raise SystemExit(f"{nm}: inventory accepted")
    except AssertionError as e:
        assert "README period" in str(e)
assert [S._water_year(t) for t in ("2018-09-01", "2019-06-30", "2019-09-01", "2020-05-30", "2019-07-15")] == \
    ["2018-19", "2018-19", "2019-20", "2019-20", None]

# 4k. read shape printed next to paper Table 1 (Guil A88 = 1996 x 7833 after the row swap)
rt1 = mini("t1")
rep = S.inventory(rt1, sample=1, expected_scenes=None)
assert rep["sampled"][0]["table1_shape"] == (1996, 7833) and rep["sampled"][0]["shape_vs_table1"] == "differs"
saved = S.TABLE1_SHAPES[("Guil", 88)]
S.TABLE1_SHAPES[("Guil", 88)] = (512, 600)
try:
    assert S.inventory(rt1, sample=1, expected_scenes=None)["sampled"][0]["shape_vs_table1"] == "same"
finally:
    S.TABLE1_SHAPES[("Guil", 88)] = saved

# 4l. the other documented-format checks still stop, with the statistics printed first by inventory()
lab2 = label(512, 600, 7)
lab2[0, 0] = 2
r2 = mini("code2", lab=lab2)
refuses(r2, "outside the README's codes")
assert "label_dtype" in inv_refuses(r2, "outside the README's codes")
lab_i8 = label(512, 600, 7).astype(np.int16)
lab_i8[lab_i8 == 255] = -1                                     # -1 in a signed file is accepted too
r8 = mini("int8", lab=lab_i8, ldtype="int8")
S.convert(r8, r8 / "out", splits=("train",))
got = P.load_split(r8 / "out", "snow", "train", mmap=False)
assert np.array_equal(ref["labels"][:, :500], got["labels"][:, :500])
rsw = mini("swapped", x=sar(512, 600, 7)[::-1].copy())
refuses(rsw, "band order")
assert "VH_p1_p50_p99" in inv_refuses(rsw, "band order")
refuses(mini("shape", lab=label(500, 600, 7)), "one SAR grid")
refuses(mini("unpaired", lab_file=False), "pairing by name")
xn = sar(512, 600, 7)
xn[:, 100:110, :] = -9999.0
rn = mini("ndtag", x=xn, nodata=-9999.0)
S.convert(rn, rn / "out", splits=("train",))
d = P.load_split(rn / "out", "snow", "train", mmap=False)
assert np.isnan(d["images"][(d["meta"].y0 == 0).to_numpy()][:, :, 100:110]).all()

# 4m. the zip's REAL label format (looked at 2026-09-28): float32, band 1 = README codes -1/0/1, band 2 = 1.0
#     (undocumented), ks with 3 bands. Band 1 is the label; NaN -> 255; where band 2 is not 1 -> 255, counted.
base = label(512, 600, 7).astype(np.float32)
base[base == 255] = -1
base[300, 300:310] = np.nan
b2 = np.ones((512, 600), np.float32)
b2[200:210, 0:50] = 0.0                                            # 500 px flagged by band 2
rf = mini("float2band", lab=np.stack([base, b2]), ldtype="float32")
row = S.inventory(rf, sample=1, expected_scenes=None)["sampled"][0]
assert (row["label_dtype"], row["label_bands"], row["label_band2_not_1"], row["label_nan"]) == ("float32", 2, 500, 10)
assert set(row["label_codes"]) <= {-1, 0, 1}, row["label_codes"]
man, d = ok(rf)
exp = ref["labels"].copy()                                         # same scene seed, 1-band uint8 labels
first = int(np.flatnonzero(((ref["meta"].y0 == 0) & (ref["meta"].x0 == 0)).to_numpy())[0])
exp[first, 200:210, 0:50] = 255
exp[first, 300, 300:310] = 255
assert np.array_equal(d["labels"], exp)
assert "label files with 2 bands; 500 label pixels in 1 scenes had band 2 != 1 (set to 255)" in man["notes"]
assert "band 1 = the README codes (used)" in man["notes"]
lab3, st3 = S._label(np.stack([base, b2, np.full_like(b2, 7.5)]),
                     {"count": 3, "dtypes": ("float32",) * 3, "nodata": (None,) * 3}, "ks")
lab2_, st2 = S._label(np.stack([base, b2]), {"count": 2, "dtypes": ("float32",) * 2, "nodata": (None,) * 2}, "raw")
assert np.array_equal(lab3, lab2_) and st3["bands"] == 3 and st3["band2_not_1"] == 500   # band 3 not used
half = base.copy()
half[5, 5] = 0.5
refuses(mini("float_half", lab=np.stack([half, b2]), ldtype="float32"), "not whole numbers")
two = base.copy()
two[5, 5] = 2.0
refuses(mini("float_two", lab=np.stack([two, b2]), ldtype="float32"), "outside the README's codes")

# ---------------------------------------------------------------- 5. download: resume, md5, skip (fake HTTP)
import requests  # noqa: E402

blob = np.random.default_rng(3).bytes(3_000_000)
readme = b"# fake readme\n"
fake_files = {"README.md": {"id": 1, "size": len(readme), "md5": hashlib.md5(readme).hexdigest()},
              S.ZIP_NAME: {"id": 2, "size": len(blob), "md5": hashlib.md5(blob).hexdigest()}}
calls = []


class FakeResp:
    def __init__(self, data, start):
        self.data, self.start = data, start
        self.status_code = 206 if start else 200
        self.headers = {"Content-Range": f"bytes {start}-{len(data) - 1}/{len(data)}"} if start else {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk):
        body = self.data[self.start:]
        first_zip = sum(c[0].endswith("/2") for c in calls) == 1 and len(self.data) == len(blob)
        cut = len(body) // 3 if first_zip else len(body)
        for i in range(0, cut, chunk):
            yield body[i:min(i + chunk, cut)]
        if cut < len(body):
            raise requests.exceptions.ChunkedEncodingError("simulated drop")


def fake_get(url, headers=None, **kw):
    calls.append((url, dict(headers or {})))
    data = blob if url.endswith("/2") else readme
    start = int(headers["Range"][6:-1]) if headers and "Range" in headers else 0
    return FakeResp(data, start)


orig_get, orig_files = requests.get, S.FILES
requests.get, S.FILES = fake_get, fake_files
try:
    dl = TMP / "dl"
    paths = S.download(dl, chunk_mb=1, backoff_s=0)
    assert [p.name for p in paths] == ["README.md", S.ZIP_NAME]
    assert (dl / S.ZIP_NAME).read_bytes() == blob and (dl / "README.md").read_bytes() == readme
    zip_calls = [c for c in calls if c[0].endswith("/2")]
    assert len(zip_calls) == 2 and "Range" in zip_calls[1][1], zip_calls   # dropped once, resumed by range
    assert zip_calls[0][0] == f"{S.API}/access/datafile/2"
    n = len(calls)
    S.download(dl, backoff_s=0)
    assert len(calls) == n, "complete + verified files must be skipped"
    (dl / f"{S.ZIP_NAME}.md5ok").unlink()
    (dl / S.ZIP_NAME).write_bytes(b"x" * len(blob))            # right size, wrong content
    try:
        S.download(dl, files=(S.ZIP_NAME,), backoff_s=0)
        raise SystemExit("corrupt file accepted")
    except RuntimeError as e:
        assert "md5" in str(e)
    try:
        S.download(dl, files=("other.zip",))
        raise SystemExit("unknown file accepted")
    except ValueError:
        pass
finally:
    requests.get, S.FILES = orig_get, orig_files
assert S.FILES[S.ZIP_NAME]["size"] == 43036387541

# ---------------------------------------------------------------- 6. the REAL zip listing (no member data)
# The published central directory (134,392 B, range-read earlier; metadata only) plus ZIP64 end records built
# here with the directory at offset 0. zipfile lists the 981 real entries; no member is ever read.
cd_path = next((p for r in REF for p in (r / "snow_src" / "central_directory.bin",
                                         r / "review_snow_src" / "central_directory.bin") if p.exists()), None)
if cd_path is None:
    print("section 6 skipped: the real central directory was not found (set SARTRANSFER_TEST_REF)")
    real_note = "real listing: skipped"
else:
    cd = cd_path.read_bytes()
    n_entries, pos = 0, 0
    while pos < len(cd):
        assert cd[pos:pos + 4] == b"PK\x01\x02", pos
        ln, le, lc = struct.unpack_from("<HHH", cd, pos + 28)
        pos += 46 + ln + le + lc
        n_entries += 1
    assert n_entries == 981
    tail = (struct.pack("<4sQHHIIQQQQ", b"PK\x06\x06", 44, 45, 45, 0, 0, n_entries, n_entries, len(cd), 0)
            + struct.pack("<4sIQI", b"PK\x06\x07", 0, len(cd), 1)
            + struct.pack("<4sHHHHIIH", b"PK\x05\x06", 0, 0, 0xFFFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0))
    rr = TMP / "real_listing"
    rr.mkdir()
    (rr / S.ZIP_NAME).write_bytes(cd + tail)
    rep = S.inventory(rr, sample=0)            # default EXPECTED_SCENES, pairing, orbits, README periods
    assert rep["scenes"] == {"train": 88, "val": 24, "test": 22, "test_spatial": 111, "test_temporal": 136}
    n_chips, groups = 0, set()
    with zipfile.ZipFile(rr / S.ZIP_NAME) as zf:
        names = zf.namelist()
        for folder, (split, basin) in S.SPLITS.items():
            for s in S._scenes(names, folder, "raw"):
                h, w = S.TABLE1_SHAPES[(basin, s["orbit"])]
                tiles = -(-h // 1024) * -(-w // 1024)
                assert zf.getinfo(s["values"]).file_size == tiles * 1024 * 1024 * 8 + 170 + 8 * tiles, s["values"]
                assert s["water_year"] == S.WATER_YEAR[folder]
                n_chips += -(-h // 512) * -(-w // 512)
                groups.add((basin, s["orbit"]))
    assert groups == set(S.TABLE1_SHAPES) and n_chips == 24508, (groups, n_chips)
    real_note = f"real listing: 981 entries, 381 scenes, member sizes = 1024-tiled Table 1 sizes, {n_chips} chips max"

shutil.rmtree(TMP, ignore_errors=True)
print("snow verified: layout + pairing, inventory, 5 official splits, chips equal the source crops, "
      "NaN/inf/-1 handling, determinism, cni variant, dry run, run-time format checks, resumable md5 download; "
      "review fixes: uncalibrated dB accepted, linear refused, check_values override, zero pairs recorded "
      "(middle scene, >50 % fill), stats printed before asserts, band descriptions, label tag 0, float16 range, "
      f"water_year + README periods, Table 1 shape; {real_note}")
