"""Offline test of domains.alpine on tiny synthetic files in the documented GlaViTU layout.

Layout mirrored from the HDF5 metadata read on 2026-09-26: official split file names,
one group per tile named "<prefix>-<row>-<col>", the same group attributes, datasets
(H, W, C) float64 contiguous with fill 0, padding == (stored - original) // 2 with
stored = next multiple of the padding unit (384 in the real files, 32 here),
outlines one-hot (channel 1 = glacier), SAR as (log10(sigma0 + 1e-6) - min)/(max - min)
with the repo's min/max (assumption A), band 0 = ascending (assumption B).
The val file stores its string attributes as fixed-length bytes, the others as str
(the real storage type is not known).
No network: the HTTP session, the clock and (in most parts) the remote reader are
replaced by local fakes that serve the synthetic files.
"""
import atexit
import hashlib
import json
import shutil
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

from _setup import TMP_BASE  # noqa: E402  (this repo's src first, import guard, temp in one folder; h5py via SARTRANSFER_TEST_PYLIB if not installed)
import h5py  # noqa: E402
import requests  # noqa: E402
from sartransfer.domains import alpine as A, prepared as P  # noqa: E402

PS = 32                                       # padding unit (384 in the real files)
rng = np.random.default_rng(0)
tmp = Path(tempfile.mkdtemp(prefix="alpine_", dir=TMP_BASE))  # under the tests' temp folder
atexit.register(shutil.rmtree, tmp, ignore_errors=True)            # also when an assertion fails
server, raw = tmp / "server", tmp / "raw"
server.mkdir()
TRUTH = {}


def encode(db, feature, band):
    """compile_features.py forward model: dB -> (log10(sigma0 + 1e-6) - min) / (max - min)."""
    lo, hi = A.LOG10_RANGE[feature][band]
    return (np.log10(10.0 ** (db / 10.0) + 1e-6) - lo) / (hi - lo)


def decode(x, feature, band):
    """Its inverse, written out here independently of alpine._to_db."""
    lo, hi = A.LOG10_RANGE[feature][band]
    return 10.0 * np.log10(10.0 ** (x * (hi - lo) + lo) - 1e-6)


def make_tile(f, name, sub, region, oh, ow, asc=True, desc=True, asc_cols=None, extremes=False, both=False,
              bytes_attrs=False):
    H, W = PS * (oh // PS + 1), PS * (ow // PS + 1)
    ph, pw = (H - oh) // 2, (W - ow) // 2
    yy, xx = np.mgrid[0:oh, 0:ow].astype(float)
    z = 1500 + 400 * np.sin(xx / 5.0 + rng.uniform(0, 6)) + 300 * np.cos(yy / 7.0)
    t = np.tanh(np.gradient(z, axis=1) / 40.0)          # eastward slope: bright in asc, dark in desc
    db_co = {0: -11 + 5 * t + rng.normal(0, 1, (oh, ow)), 1: -11 - 5 * t + rng.normal(0, 1, (oh, ow))}
    db_cr = {b: db_co[b] - 7 + rng.normal(0, 0.5, (oh, ow)) for b in (0, 1)}
    ins = (slice(ph, ph + oh), slice(pw, pw + ow))
    co, cr = np.zeros((H, W, 2)), np.zeros((H, W, 2))
    for b, on in ((0, asc), (1, desc)):
        if on:
            co[ins + (b,)] = encode(db_co[b], "co_pol_sar", b)
            cr[ins + (b,)] = encode(db_cr[b], "cross_pol_sar", b)
    if asc_cols is not None:                            # asc covers only the first asc_cols columns
        co[ph:ph + oh, pw + asc_cols:pw + ow, 0] = 0
        cr[ph:ph + oh, pw + asc_cols:pw + ow, 0] = 0
    co[ph + 1, pw + 1:pw + 4, :] = 0                    # no-data pixels inside the tile
    if extremes:                                        # desc maximum of the repo stats, and a value just above 0
        for (y, x), v in (((3, 3), 1.0), ((4, 4), 1e-7)):
            co[ph + y, pw + x, 1] = v
            db_co[1][y, x] = decode(v, "co_pol_sar", 1)
    glacier = (yy - oh / 2) ** 2 + (xx - ow / 3) ** 2 < (min(oh, ow) / 4) ** 2
    out = np.zeros((H, W, 2))
    out[ins + (1,)] = glacier
    out[ins + (0,)] = ~glacier
    out[ph + 2, pw + 5, :] = 0                          # an unlabelled (0,0) pixel inside
    if both:
        out[ph + 3, pw + 6, :] = 1                      # a (1,1) pixel inside
    dem = np.zeros((H, W, 2))
    dem[ins + (0,)] = (z + 103.29675627387357) / (8349.427068840781 + 103.29675627387357)
    dem[ins + (1,)] = rng.random((oh, ow))
    g = f.create_group(name)
    lat0, lon0 = 45.0 + rng.uniform(0, 1), 7.0 + rng.uniform(0, 1)
    for k, v in dict(epsg=4326, height=H, width=W, original_height=oh, original_width=ow, padding_height=ph,
                     padding_width=pw, region=region, subregion=sub, tile_name=name, xmin=lon0,
                     xmax=lon0 + ow * 1.0e-4, ymin=lat0, ymax=lat0 + oh * 1.0e-4).items():
        g.attrs[k] = np.bytes_(v) if bytes_attrs and isinstance(v, str) else v
    for k, v in dict(optical=rng.random((H, W, 6)), dem=dem, co_pol_sar=co, cross_pol_sar=cr,
                     in_sar=np.zeros((H, W, 2)), outlines=out).items():
        g.create_dataset(k, data=v)
    g.create_dataset("bright_dark_outlines", data=np.zeros((H, W, 3), np.uint8))
    TRUTH[name] = dict(H=H, W=W, ph=ph, pw=pw, oh=oh, ow=ow, db_co=db_co, db_cr=db_cr, glacier=glacier, both=both,
                       n_co=[int((co[ins + (b,)] != 0).sum()) for b in (0, 1)],
                       n_cr=[int((cr[ins + (b,)] != 0).sum()) for b in (0, 1)])


TILES = {"train": [("ALP-1-1", "ALP", 44, 47, {}), ("ALP-1-2", "ALP", 40, 45, dict(asc=False)),
                   ("ALP-2-1", "ALP", 46, 42, {}), ("ALP-2-2", "ALP", 41, 50, {}), ("NZ1-1-1", "NZ", 45, 44, {})],
         "val": [("ALP-3-1", "ALP", 45, 45, {}), ("ALP-3-2", "ALP", 43, 44, dict(asc=False, desc=False)),
                 ("ALP-3-3", "ALP", 42, 46, dict(asc_cols=1))],
         "test": [("ALP-4-1", "ALP", 47, 46, dict(both=True)),
                  ("ALP-4-2", "ALP", 40, 41, dict(asc=False, extremes=True))]}
for split, tiles in TILES.items():
    with h5py.File(server / A.SPLIT_FILES[split], "w") as f:
        for name, sub, oh, ow, kw in tiles:
            make_tile(f, name, sub, "NZ" if sub == "NZ" else "ALP", oh, ow, bytes_attrs=split == "val", **kw)
    p = server / A.SPLIT_FILES[split]
    A.FILES[p.name] = (p.stat().st_size, hashlib.md5(p.read_bytes()).hexdigest())
A.SUBREGIONS["ALP"] = dict(A.SUBREGIONS["ALP"], tiles=(4, 3, 2))
A.SUBREGIONS["NZ"] = dict(A.SUBREGIONS["NZ"], tiles=(1, 0, 0))
N_TILES = {s: sum(1 for t in TILES[s] if t[1] == "ALP") for s in A.SPLITS}

# ---------------------------------------------------------------- HTTP layer (fake session)
blob = rng.integers(0, 256, 10_000, dtype=np.uint8).tobytes()
A.FILES["fake.bin"] = (len(blob), hashlib.md5(blob).hexdigest())


class FakeResp:
    def __init__(self, status, headers=None, chunks=(), drop_after=None):
        self.status_code, self.headers, self.chunks, self.drop_after = status, headers or {}, chunks, drop_after

    def __enter__(self): return self
    def __exit__(self, *a): return False

    def iter_content(self, n):
        for i, c in enumerate(self.chunks):
            if self.drop_after is not None and i >= self.drop_after:
                raise requests.ConnectionError("simulated drop")
            yield c


class FakeSession:
    events, redirects = [], 0

    def get(self, url, allow_redirects=True, headers=None, stream=False, timeout=None):
        if url.startswith(A.BASE):
            FakeSession.redirects += 1
            return FakeResp(302, {"Location": "https://signed.invalid/x"})
        a, b = (int(v) for v in headers["Range"].split("=")[1].split("-"))
        ev = FakeSession.events.pop(0) if FakeSession.events else "ok"
        if ev == "403":
            return FakeResp(403)
        data = blob[a:b + 1]
        return FakeResp(206, {"Content-Range": f"bytes {a}-{b}/{len(blob)}"},
                        [data[i:i + 1000] for i in range(0, len(data), 1000)], 2 if ev == "drop" else None)


real_session, real_sleep = A.requests.Session, A.time.sleep
A.requests.Session, A.time.sleep = FakeSession, (lambda s: None)
try:
    FakeSession.events = ["403", "drop"]
    assert A._Nird("fake.bin", len(blob)).read(100, 9_000) == blob[100:9_000]
    assert FakeSession.redirects == 2, FakeSession.redirects          # re-signed after the 403
    dl = tmp / "dl"
    dl.mkdir()
    (dl / "fake.bin.part").write_bytes(blob[:3_000])                   # resume from a partial file
    FakeSession.events = ["drop"]
    assert A.download(dl, mode="files", files=["fake.bin"]) == [dl / "fake.bin"]
    assert (dl / "fake.bin").read_bytes() == blob and (dl / "fake.bin.md5ok").exists()
    (dl / "fake.bin").write_bytes(blob[:-1] + b"\x00")                 # corrupted, same size
    (dl / "fake.bin.md5ok").unlink()
    try:
        A.download(dl, mode="files", files=["fake.bin"]); raise SystemExit("md5 mismatch accepted")
    except IOError:
        assert (dl / "fake.bin.bad").exists()
finally:
    A.requests.Session, A.time.sleep = real_session, real_sleep

# ---------------------------------------------------------------- subset download (fake remote)
calls = {"open": 0, "get": 0}
real_open_remote = A._open_remote


def fake_remote(name):
    calls["open"] += 1
    path = server / name

    def rng_fn(a, b):
        with open(path, "rb") as fh:
            fh.seek(a)
            return fh.read(b - a)
    r = A._RangeReader(path.stat().st_size, rng_fn, block=4096)
    get = r.get

    def counted(start, n):
        calls["get"] += 1
        return get(start, n)
    r.get = counted
    return r


def same_subsets(x_root, y_root):
    for split in A.SPLITS:
        name = A._subset_name(split, "ALP")
        with h5py.File(x_root / name, "r") as x, h5py.File(y_root / name, "r") as y:
            assert sorted(x) == sorted(y) and dict(x.attrs) == dict(y.attrs), split
            for k in x:
                assert sorted(x[k]) == sorted(y[k]) and A._attrs(x[k]) == A._attrs(y[k]), (split, k)
                for ft in x[k]:
                    assert np.array_equal(x[k][ft][()], y[k][ft][()]), (split, k, ft)


A._open_remote = fake_remote
paths = A.download(raw, subregions=("ALP",), check_tiles=2, workers=2)
assert [p.name for p in paths] == [A._subset_name(s, "ALP") for s in A.SPLITS], paths
n_get = 3 * sum(N_TILES.values()) + 2                                         # dem only for 2 train tiles
assert calls == {"open": 3, "get": n_get}, calls
for split in A.SPLITS:
    with h5py.File(server / A.SPLIT_FILES[split], "r") as s, h5py.File(raw / A._subset_name(split, "ALP"), "r") as d:
        assert d.attrs["complete"] and d.attrs["subregion"] == "ALP"
        assert sorted(d.keys()) == sorted(k for k in s.keys() if k.startswith("ALP-"))
        for k in d:
            want = {"co_pol_sar", "cross_pol_sar", "outlines"} | ({"dem"} if split == "train" and k in ("ALP-1-1", "ALP-1-2") else set())
            assert set(d[k].keys()) == want, (k, set(d[k].keys()))
            for ft in want:
                assert d[k][ft].dtype == s[k][ft].dtype and np.array_equal(d[k][ft][()], s[k][ft][()]), (k, ft)
            assert {a: A._attr(v) for a, v in s[k].attrs.items()}.items() <= A._attrs(d[k]).items()
            assert isinstance(d[k].attrs["tile_name"], bytes if split == "val" else str), (k, type(d[k].attrs["tile_name"]))
A.download(raw, subregions=("ALP",), check_tiles=2)
assert calls == {"open": 3, "get": n_get}, calls                               # complete files skipped
with h5py.File(raw / A._subset_name("train", "ALP"), "r+") as f:
    del f["ALP-2-2"]
    f.attrs["complete"] = False
A.download(raw, subregions=("ALP",), check_tiles=2)
assert calls == {"open": 4, "get": n_get + 3}, calls                           # only the missing tile


# ---------------------------------------------------------------- real _open_remote over a fake NIRD session
class Clock:
    """Stands in for the time module inside alpine: every time() call advances 5 s."""
    now, lock = 0.0, threading.Lock()

    @classmethod
    def time(cls):
        with cls.lock:
            cls.now += 5.0
            return cls.now

    @staticmethod
    def sleep(s):
        pass


class FileSession:
    """Serves the synthetic server files like NIRD: 302 to a signed URL, then 206 range replies."""
    sessions = redirects = ranges = 0
    events, lock = [], threading.Lock()

    def __init__(self):
        with FileSession.lock:
            FileSession.sessions += 1

    def get(self, url, allow_redirects=True, headers=None, stream=False, timeout=None):
        if url.startswith(A.BASE):
            assert not allow_redirects
            with FileSession.lock:
                FileSession.redirects += 1
                n = FileSession.redirects
            return FakeResp(302, {"Location": f"https://signed.invalid/{url[len(A.BASE):]}?sig={n}"})
        path = server / url.split("?")[0].rsplit("/", 1)[1]
        a, b = (int(v) for v in headers["Range"].split("=")[1].split("-"))
        with FileSession.lock:
            FileSession.ranges += 1
            ev = FileSession.events.pop(0) if FileSession.events else "ok"
        if ev == "403":
            return FakeResp(403)
        with open(path, "rb") as fh:
            fh.seek(a)
            data = fh.read(b + 1 - a)
        return FakeResp(206, {"Content-Range": f"bytes {a}-{b}/{path.stat().st_size}"},
                        [data[i:i + 65536] for i in range(0, len(data), 65536)])


raw_http = tmp / "raw_http"
A._open_remote, A.requests.Session, A.time = real_open_remote, FileSession, Clock
try:
    FileSession.events = ["403"]                                               # first range read: link expired
    A.download(raw_http, subregions=("ALP",), check_tiles=2, workers=2)
finally:
    A._open_remote, A.requests.Session, A.time = fake_remote, real_session, time
print(f"remote reader: {FileSession.sessions} sessions, {FileSession.redirects} link resolutions, "
      f"{FileSession.ranges} range requests")
assert FileSession.sessions >= 3, FileSession.sessions                         # one per file and thread
assert FileSession.redirects > FileSession.sessions + 1, (FileSession.redirects, FileSession.sessions)  # 40 s timer
same_subsets(raw, raw_http)                                                    # byte-identical to the local path

# ---------------------------------------------------------------- inventory
inv = A.inventory(raw)
assert inv["tiles"] == {f"{s}/ALP": N_TILES[s] for s in A.SPLITS}, inv["tiles"]
assert inv["chip_size_auto"] == 64 and inv["checks"]["scale_ok"] and inv["checks"]["orbit_status"] == "confirmed"
assert inv["checks"]["plausible_scales"] == ["minmax_log10"], inv["checks"]["plausible_scales"]
n_in_test = sum(TRUTH[k]["oh"] * TRUTH[k]["ow"] for k in ("ALP-4-1", "ALP-4-2"))
assert inv["outlines_00_share_sample"]["test/ALP"] == round(2 / n_in_test, 6), inv["outlines_00_share_sample"]
assert set(inv["outlines_00_share_sample"]) == {"train/ALP", "val/ALP", "test/ALP"}
t = TRUTH["ALP-3-3"]
s33 = [d for d in inv["sample"] if d["tile"] == "ALP-3-3"][0]
assert s33["co_pol_sar[0]"]["valid_frac_inside"] == round(t["n_co"][0] / (t["oh"] * t["ow"]), 4), s33["co_pol_sar[0]"]
assert s33["co_pol_sar[1]"]["valid_frac_inside"] == round(t["n_co"][1] / (t["oh"] * t["ow"]), 4)


# ---------------------------------------------------------------- convert + validate
def stats(man):
    return json.loads(man["notes"].split("Stats: ", 1)[1])


def check_chips(root, dataset, expect, pref="asc"):
    """Every chip against the synthetic truth; expect = {tile: orbit} where it differs from pref."""
    for split in A.SPLITS:
        d = P.load_split(root, dataset, split, mmap=False)
        img, lab, meta = d["images"], d["labels"], d["meta"]
        assert img.dtype == np.float16 and lab.dtype == np.uint8 and img.shape[1:] == (2, 64, 64)
        assert set(np.unique(lab)) <= {0, 1, 255} and (meta.region == "ALP").all() and (meta.date == "").all()
        assert (meta.ref_year == 2015).all() and (meta.source_file == A.SPLIT_FILES[split]).all()
        for i, row in meta.iterrows():
            t = TRUTH[row.tile]
            assert row.orbit == expect.get(row.tile, pref), (row.tile, row.orbit)
            b = 0 if row.orbit == "asc" else 1
            y, x = t["ph"] - row.y0, t["pw"] - row.x0             # original extent inside the chip
            ins = (slice(y, y + t["oh"]), slice(x, x + t["ow"]))
            co, cr = img[i, 0].astype(np.float32), img[i, 1].astype(np.float32)
            ok, okc = np.isfinite(co[ins]), np.isfinite(cr[ins])
            assert ok.sum() == t["n_co"][b] and okc.sum() == t["n_cr"][b], (row.tile, ok.sum(), okc.sum())
            tol = 0.02 + np.abs(t["db_co"][b][ok]) * 2.0 ** -10               # float16 storage
            assert (np.abs(co[ins][ok] - t["db_co"][b][ok]) <= tol).all(), row.tile   # units: dB
            assert np.abs(cr[ins][okc] - t["db_cr"][b][okc]).max() < 0.02
            keep = np.zeros((64, 64), bool)
            keep[ins] = True
            assert np.isnan(img[i][:, ~keep].astype(np.float32)).all() and (lab[i][~keep] == 255).all()
            want = np.where(t["glacier"], 1, 0).astype(np.uint8)
            want[2, 5] = 255                                                  # (0,0) -> ignore
            if t["both"]:
                want[3, 6] = 255                                              # (1,1) -> ignore
            assert np.array_equal(lab[i][ins], want), row.tile
            n_in = t["oh"] * t["ow"]
            assert 0 < row.glacier_frac < 0.5
            assert abs(row.sar_valid_frac - t["n_co"][b] / n_in) < 1e-4, (row.tile, row.sar_valid_frac)
            assert abs(row.other_orbit_valid_frac - t["n_co"][1 - b] / n_in) < 1e-4, (row.tile, row.other_orbit_valid_frac)


out1, out2 = tmp / "out1", tmp / "out2"
man = A.convert(raw, out1)
assert man["dataset"] == A.DATASET and man["chip_size"] == 64 and man["units"] == "dB"
assert man["splits"] == {"test": 2, "train": 4, "val": 2}, man["splits"]      # ALP-3-2 has no SAR: dropped
assert man["classes"] == {0: "non-glacier", 1: "glacier"} and man["channels"] == ["co-pol", "cross-pol"]
v = P.validate(out1, A.DATASET)
assert set(v.split) == {"train", "val", "test"} and (v.nan_share > 0).all()
COVERAGE = {"ALP-1-2": "desc", "ALP-3-3": "desc", "ALP-4-2": "desc"}           # asc absent or 1 column only
check_chips(out1, A.DATASET, COVERAGE)
st = stats(man)
assert st["orbit_rule"] == "coverage" and st["tiles_without_sar"] == ["val/ALP-3-2"]
assert st["chips_per_split_orbit"] == {"test/asc": 1, "test/desc": 1, "train/asc": 3, "train/desc": 1,
                                       "val/asc": 1, "val/desc": 1}, st["chips_per_split_orbit"]
assert st["chips_where_other_orbit_has_more_valid_px"] == {"train": 0, "val": 0, "test": 0}
u = st["outline_px_in_chips"]
assert {s: (u[s]["outlines_00"], u[s]["outlines_11"]) for s in A.SPLITS} == {"train": (4, 0), "val": (2, 0),
                                                                              "test": (2, 1)}, u
assert u["test"]["inside_px"] == n_in_test
d = st["db_per_split_channel_before_masking"]
hi_db, lo_db = decode(1.0, "co_pol_sar", 1), decode(1e-7, "co_pol_sar", 1)
print(f"extremes in the synthetic test tile: {hi_db:.3f} dB and {lo_db:.3f} dB; recorded {d['test']['co-pol']}")
assert hi_db > 150 and lo_db < -100 and st["db_range"] is None and st["outside_range"] == "keep"
assert abs(d["test"]["co-pol"]["max"] - hi_db) < 1e-3 and abs(d["test"]["co-pol"]["min"] - lo_db) < 1e-3
assert all(-40 < d[s][ch]["min"] and d[s][ch]["max"] < 10 for s in ("train", "val") for ch in A.CHANNELS)
assert all(x["below"] == x["above"] == 0 for s in d.values() for x in s.values())       # no db_range given
man2 = A.convert(raw, out2)
for split in A.SPLITS:                                                 # determinism
    a, b = P.load_split(out1, A.DATASET, split, mmap=False), P.load_split(out2, A.DATASET, split, mmap=False)
    assert a["images"].tobytes() == b["images"].tobytes() and a["labels"].tobytes() == b["labels"].tobytes()
    assert a["meta"].equals(b["meta"])
assert stats(man2) == st
out3 = tmp / "out3"                                                     # official files, no subset
A.convert(server, out3)
for split in A.SPLITS:
    a, b = P.load_split(out1, A.DATASET, split, mmap=False), P.load_split(out3, A.DATASET, split, mmap=False)
    assert a["images"].tobytes() == b["images"].tobytes() and a["labels"].tobytes() == b["labels"].tobytes()

# ---------------------------------------------------------------- orbit rules
many = A.convert(raw, tmp / "oany", orbit_rule="any", dataset="tany")
check_chips(tmp / "oany", "tany", {"ALP-1-2": "desc", "ALP-4-2": "desc"})      # 'any': the 1-column asc is kept
assert stats(many)["chips_where_other_orbit_has_more_valid_px"] == {"train": 0, "val": 1, "test": 0}
r = P.load_split(tmp / "oany", "tany", "val")["meta"].set_index("tile").loc["ALP-3-3"]
print(f"orbit_rule='any' keeps ALP-3-3 asc: sar_valid_frac {r.sar_valid_frac}, other orbit {r.other_orbit_valid_frac}")
assert r.sar_valid_frac < 0.05 and r.other_orbit_valid_frac > 0.99
A.convert(raw, tmp / "odesc", orbit="desc", dataset="tdesc")
check_chips(tmp / "odesc", "tdesc", {}, pref="desc")                           # ties go to the preferred orbit

# ---------------------------------------------------------------- dB range: count, then mask
mr = A.convert(raw, tmp / "orng", db_range=(-60, 30), dataset="trng")
d = stats(mr)["db_per_split_channel_before_masking"]
assert (d["test"]["co-pol"]["below"], d["test"]["co-pol"]["above"]) == (1, 1), d["test"]
assert sum(x["below"] + x["above"] for s in d.values() for x in s.values()) == 2
assert stats(mr)["db_range"] == [-60.0, 30.0] and "kept as they are" in mr["notes"]
mn = A.convert(raw, tmp / "onan", db_range=(-60, 30), outside_range="nan", dataset="tnan")
assert "set to NaN below -60 dB and above 30 dB" in mn["notes"]
assert stats(mn)["db_per_split_channel_before_masking"] == d                  # counted before masking
for split in A.SPLITS:
    a = P.load_split(out1, A.DATASET, split, mmap=False)
    kept, masked = P.load_split(tmp / "orng", "trng", split, mmap=False), P.load_split(tmp / "onan", "tnan", split, mmap=False)
    assert kept["images"].tobytes() == a["images"].tobytes()                  # "keep" changes nothing
    x, y = a["images"].astype(np.float32), masked["images"].astype(np.float32)
    newnan = np.isnan(y) & ~np.isnan(x)
    assert newnan.sum() == (2 if split == "test" else 0) and np.array_equal(x[~newnan], y[~newnan], equal_nan=True)
    if split == "test":
        i = int(np.flatnonzero(masked["meta"].tile == "ALP-4-2")[0])
        assert newnan[i, 0].sum() == 2 and newnan[i, 1].sum() == 0
        vals = x[i, 0][newnan[i, 0]]
        assert (vals < -100).sum() == 1 and (vals > 150).sum() == 1, vals
        t = TRUTH["ALP-4-2"]
        assert abs(masked["meta"].sar_valid_frac[i] - (t["n_co"][1] - 2) / (t["oh"] * t["ow"])) < 1e-4

# ---------------------------------------------------------------- chip layouts, flushing (RAM), more subregions
held, parts = {"max": 0}, Counter()
real_add, real_flush = P.PreparedWriter.add, P.PreparedWriter._flush


def spy_add(self, split, image, label, **meta):
    real_add(self, split, image, label, **meta)
    held["max"] = max(held["max"], sum(a.nbytes for b in self._buf.values() for a in b["img"] + b["lab"]))


def spy_flush(self, split):
    if self._buf.get(split) and self._buf[split]["img"]:
        parts[split] += 1
    real_flush(self, split)


P.PreparedWriter.add, P.PreparedWriter._flush = spy_add, spy_flush
try:
    m32 = A.convert(raw, tmp / "o32", chip_size=32, dataset="t32")            # default flush_mb=256
    default_run = (held["max"], dict(parts))
    held["max"] = 0
    parts.clear()
    A.convert(raw, tmp / "o32f", chip_size=32, dataset="t32", flush_mb=0.02)  # 20 kB = ~4 chips of 5,120 B
    small_run = (held["max"], dict(parts))
finally:
    P.PreparedWriter.add, P.PreparedWriter._flush = real_add, real_flush
chip = 5 * 32 * 32
assert m32["splits"] == {"test": 8, "train": 16, "val": 8}, m32["splits"]      # 2 x 2 windows per tile
print(f"writer buffer peak: {default_run} with flush_mb=256, {small_run} with flush_mb=0.02")
assert default_run == (16 * chip, {"train": 1, "val": 1, "test": 1}), default_run   # one split in RAM at a time
assert small_run[0] <= 20_000 + chip and small_run[1] == {"train": 4, "val": 2, "test": 2}, small_run
for split in A.SPLITS:                                                 # flushing does not change the output
    a, b = P.load_split(tmp / "o32", "t32", split, mmap=False), P.load_split(tmp / "o32f", "t32", split, mmap=False)
    assert a["images"].tobytes() == b["images"].tobytes() and a["labels"].tobytes() == b["labels"].tobytes()
    assert a["meta"].equals(b["meta"])
m80 = A.convert(raw, tmp / "o80", chip_size=80, dataset="t80")                 # windows beyond the array
a64, a80 = P.load_split(out1, A.DATASET, "train", mmap=False), P.load_split(tmp / "o80", "t80", "train", mmap=False)
for i in range(4):
    y, x = -int(a80["meta"].y0[i]), -int(a80["meta"].x0[i])
    assert np.array_equal(a80["images"][i][:, y:y + 64, x:x + 64], a64["images"][i], equal_nan=True)
    assert np.array_equal(a80["labels"][i][y:y + 64, x:x + 64], a64["labels"][i])
    assert (a80["labels"][i][:y] == 255).all() and np.isnan(a80["images"][i][:, :y].astype(np.float32)).all()
mnz = A.convert(server, tmp / "onz", subregions=("ALP", "NZ"))
assert mnz["dataset"] == "glavitu_alp_nz" and mnz["splits"] == {"test": 2, "train": 5, "val": 2}
assert set(P.load_split(tmp / "onz", "glavitu_alp_nz", "train")["meta"].region) == {"ALP", "NZ"}


# ---------------------------------------------------------------- failures are loud
def must_fail(fn, what, expect=""):
    try:
        fn()
    except AssertionError as e:
        assert expect in str(e), (what, str(e)[:200])
        print(f"refused as expected ({what}): {str(e)[:110]}")
        return
    raise SystemExit(f"accepted: {what}")


must_fail(lambda: A.convert(raw, tmp / "bad1", sar_scale="linear"), "linear scale on min-max data")
must_fail(lambda: A.convert(raw, tmp / "bad2", sar_scale="log10"), "log10 scale on min-max data")
must_fail(lambda: A.convert(raw, tmp / "bad3", orbit_bands=("desc", "asc")), "swapped orbit bands", "contradicted")
A.SUBREGIONS["ALP"] = dict(A.SUBREGIONS["ALP"], tiles=(5, 3, 2))
must_fail(lambda: A.convert(raw, tmp / "bad4"), "tile count differs from Table 2", "Table 2")
A.SUBREGIONS["ALP"] = dict(A.SUBREGIONS["ALP"], tiles=(4, 3, 2))
bad = tmp / "raw_bad"
shutil.copytree(raw, bad)
with h5py.File(bad / A._subset_name("test", "ALP"), "r+") as f:
    f["ALP-4-1/outlines"][20, 20, 1] = 0.5
must_fail(lambda: A.convert(bad, tmp / "bad5"), "soft outline values", "one-hot")
must_fail(lambda: A.inventory(bad), "soft outline values (inventory)", "one-hot")
nocross = tmp / "raw_nocross"
shutil.copytree(raw, nocross)
with h5py.File(nocross / A._subset_name("val", "ALP"), "r+") as f:
    del f["ALP-3-1/cross_pol_sar"]
must_fail(lambda: A.convert(nocross, tmp / "bad6"), "cross-pol missing where Table 2 lists it", "cross_pol_sar")
for kw in (dict(outside_range="nan"), dict(outside_range="clip", db_range=(-60, 30)), dict(db_range=(30, -60)),
           dict(db_range=(float("nan"), 0)), dict(orbit_rule="most"), dict(flush_mb=0)):
    try:
        A.convert(raw, tmp / "badarg", **kw)
    except ValueError as e:
        print(f"refused as expected ({kw}): {e}")
    else:
        raise SystemExit(f"accepted: {kw}")
assert not (tmp / "badarg").exists()                                   # refused before anything is written

shutil.rmtree(tmp)
print("alpine converter verified: HTTP retry/resume + md5, subset copy (byte-exact, resumable, dem only for "
      "check tiles), real remote reader over a fake NIRD session (per-thread sessions, 40 s link refresh, 403), "
      "bytes and str attributes, inventory ((0,0) share, valid share per band), Table 2 counts, dB units, "
      "orbit rules 'coverage' / 'any' and the preferred orbit, other_orbit_valid_frac, terrain check, dB min/max "
      "per split, db_range counting and NaN masking, one-hot labels with (0,0)/(1,1) counts, ignore/NaN outside "
      "tiles, region/date meta, byte-bounded flushing, chip layouts, determinism, missing cross-pol, loud failures")
