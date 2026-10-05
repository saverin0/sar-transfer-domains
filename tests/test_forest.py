"""Offline test of domains.forest (ForTy v1) on synthetic shards in the documented format.

Format written here, as documented (features.json, dataset_info.json, TFDS
example_serializer, TFRecord record_writer): one tf.train.Example per record,
all 12 features at their real shapes, uint8/int64 -> packed int64_list,
float32/float64 -> packed float_list, C order; plain TFRecord framing with
masked CRC32C; file names forty_v1-{split}.tfrecord-NNNNN-of-01024.
Masked S1 pixels carry mask 0 and value 0 (geeflow). Temp files stay in the tests' temp folder.
"""
import base64, gc, hashlib, http.server, json, shutil, struct, sys, threading, urllib.parse
from pathlib import Path

import numpy as np

from _setup import TMP_BASE  # noqa: E402  (this repo's src first, import guard, temp in one folder)
from sartransfer.domains import forest as F, prepared as P  # noqa: E402

TMP = TMP_BASE / "tmp_forest"
shutil.rmtree(TMP, ignore_errors=True)
TMP.mkdir(parents=True)
P.CHUNK = 4                                                   # several part files per split


def expect_error(fn, text):
    try:
        fn()
    except Exception as e:
        assert text in str(e), (text, str(e))
        return
    raise SystemExit(f"no error containing {text!r}")


# ---------------------------------------------------------------- 1. CRC32C and varints
assert F.crc32c(b"123456789") == 0xE3069283                  # CRC-32C check value
rng = np.random.default_rng(0)
for n in (0, 1, 1023, 1024, 1025, 5000):
    d = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
    assert F.crc32c(d) == F._crc_scalar(0xFFFFFFFF, d) ^ 0xFFFFFFFF, n


def varint(v):
    v &= (1 << 64) - 1
    out = bytearray()
    while True:
        b, v = v & 0x7F, v >> 7
        out.append(b | 0x80 if v else b)
        if not v:
            return bytes(out)


vals = [0, 1, 127, 128, 300, 2 ** 40, -1, -5]
got = F._varints(np.frombuffer(b"".join(varint(x) for x in vals), np.uint8))
assert got.tolist() == vals, got


# ---------------------------------------------------------------- 2. writers for the documented format
def ld(field, payload):
    return varint(field << 3 | 2) + varint(len(payload)) + payload


def packed_ints(a):
    a = np.asarray(a).ravel()
    if a.size and a.min() >= 0 and a.max() < 128:
        return a.astype(np.uint8).tobytes()
    return b"".join(varint(int(x)) for x in a)


def f_float(a):
    return ld(2, ld(1, np.asarray(a, "<f4").ravel().tobytes()))     # Feature.float_list, packed


def f_int(a):
    return ld(3, ld(1, packed_ints(a)))                              # Feature.int64_list, packed


def example(feats):                                                   # Example{features{feature map}}
    return ld(1, b"".join(ld(1, ld(1, k.encode()) + ld(2, v)) for k, v in feats.items()))


def mask_crc(data):                                                   # tsl crc32c::Mask, written out again
    c = F.crc32c(data)
    return (((c >> 15) | (c << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def frame(data):
    h = struct.pack("<Q", len(data))
    return h + struct.pack("<I", mask_crc(h)) + data + struct.pack("<I", mask_crc(data))


BASE_VV = np.array([-12, -8, -9, -10, -12, -5, -20, -14, -16], float)     # dB per source code
ROWS = [(0, 16, 0), (16, 32, 1), (32, 40, 2), (40, 48, 3), (48, 88, 4), (88, 96, 5),
        (96, 104, 6), (104, 106, 7), (106, 128, 8)]                        # Fig. 3a-like shares
LAB = np.zeros((128, 128), np.uint8)
for a, b, c in ROWS:
    LAB[a:b] = c
OFFS = np.array([0.0, 2.0, 4.0, 6.0])                                      # dB per seasonal step
EXPECT = {}
ID0 = 170_000                         # real ids are 0..199,999 (stats/*_id.json); 3 varint bytes here
WANT_IDS = np.array([255, 0, 1, 2, 3, 4, 5, 6, 7] + [254] * 247, np.uint8)  # source code -> prepared id


def record(key, idx, lat, lon, empty=False, labels=LAB, mask_value=1, scale="dB", sid=None):
    """The scene (backscatter) always follows LAB; `labels` are the codes stored in the file."""
    col = 0.01 * np.arange(128)
    vv = BASE_VV[LAB][None] + OFFS[:, None, None] + col + 0.1 * (idx % 3)
    s1 = np.stack([vv, vv - 7, np.broadcast_to(30 + 15 * np.arange(128) / 127, vv.shape)], -1)
    if scale == "linear":
        s1[..., :2] = 10 ** (s1[..., :2] / 10)
    m = np.ones((4, 128, 128, 3), np.uint8)
    m[1, 50, 50, 0] = mask_value                       # 1, or an undocumented mask value
    m[3, :10, :10] = 0                                 # one season missing
    m[:, 124:, 120:] = 0                               # no season at all (bare-ground rows) -> both NaN -> 255
    m[:, 60, 10, 0] = 0                                # VV missing in every season, VH present -> label kept
    if empty:
        m[:] = 0
    asc = np.where(m == 0, 0.0, s1).astype(np.float32)             # geeflow: masked -> 0
    asc[0, 5, 60, 0] = np.nan                                       # non-finite under mask 1
    desc = np.where(m == 0, 0.0, asc + np.array([1.0, 1.0, 0.0])).astype(np.float32)
    ok = (m[..., :2] == 1) & np.isfinite(asc[..., :2])
    x = np.where(ok, asc[..., :2].astype(np.float64), 0.0)
    mean = np.where(ok.sum(0) > 0, x.sum(0) / np.maximum(ok.sum(0), 1), np.nan)
    EXPECT[key] = {"mean": mean.transpose(2, 0, 1), "desc2": np.where(
        (m[2, ..., :2] == 1) & np.isfinite(desc[2, ..., :2]), desc[2, ..., :2], np.nan).transpose(2, 0, 1)}
    none = (m[..., :2] == 0).all(0).all(-1)
    EXPECT[key]["labels"] = np.where(none, 255, WANT_IDS[labels]).astype(np.uint8)
    feats = {"climate": f_float(np.zeros((4, 1, 1, 14))), "elevation": f_float(np.zeros((128, 128, 3))),
             "id": f_int([ID0 + idx if sid is None else sid]), "lat": f_float([lat]), "lon": f_float([lon]),
             "s1_asc": f_float(asc), "s1_asc_mask": f_int(m), "s1_desc": f_float(desc), "s1_desc_mask": f_int(m),
             "s2": f_float(np.zeros((4, 128, 128, 10))), "s2_mask": f_int(np.ones((4, 128, 128, 10))),
             "segmentation_labels": f_int(labels)}
    return frame(example(feats))


def tensor(dtype, shape, cls="tensor_feature.Tensor"):
    t = {"dtype": dtype, "encoding": "none", "shape": {"dimensions": [str(d) for d in shape]} if shape else {}}
    return {"pythonClassName": f"tensorflow_datasets.core.features.{cls}", "tensor": t}


# the real bucket features.json (fetched 2026-09-26), same classes, dtypes, shapes and nesting
FEATURES = {"featuresDict": {"features": {
    "climate": tensor("float32", [4, 1, 1, 14]), "elevation": tensor("float32", [128, 128, 3]),
    "id": tensor("int64", [], "scalar.Scalar"), "lat": tensor("float64", [], "scalar.Scalar"),
    "lon": tensor("float64", [], "scalar.Scalar"),
    "s1_asc": tensor("float32", [4, 128, 128, 3]), "s1_asc_mask": tensor("uint8", [4, 128, 128, 3]),
    "s1_desc": tensor("float32", [4, 128, 128, 3]), "s1_desc_mask": tensor("uint8", [4, 128, 128, 3]),
    "s2": tensor("float32", [4, 128, 128, 10]), "s2_mask": tensor("uint8", [4, 128, 128, 10]),
    "segmentation_labels": tensor("uint8", [128, 128])}},
    "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict"}
POS = [(47.3, 8.5), (-3.2, -60.1), (61.0, 25.4), (5.5, 101.2), (-33.9, 151.2)]


def build(raw, plan):
    """plan: {split: [[record kwargs, ...] per shard]} -> shards + features.json + dataset_info.json."""
    raw.mkdir(parents=True)
    (raw / "features.json").write_text(json.dumps(FEATURES, indent=4), encoding="utf-8")
    info = {"name": "forty_v1", "version": "1.0.0", "fileFormat": "tfrecord", "splits": []}
    k = 0
    for split in ("train", "validation", "test"):
        lengths = [20] * 1024
        for si, rs in enumerate(plan.get(split, [])):
            blob = b""
            for ri, kw in enumerate(rs):
                lat, lon = POS[k % len(POS)]
                blob += record((raw.name, split, si, ri), k, lat, lon, **kw)
                k += 1
            (raw / F.shard_name(split, si)).write_bytes(blob)
            lengths[si] = len(rs)
        info["splits"].append({"filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}", "name": split,
                               "numBytes": "0", "shardLengths": [str(x) for x in lengths]})
    (raw / "dataset_info.json").write_text(json.dumps(info), encoding="utf-8")


def recs(n, **kw):
    return [dict(kw) for _ in range(n)]


SRC = TMP / "src"
build(SRC, {"train": [recs(5), recs(5)], "validation": [recs(2) + [{"empty": True}]], "test": [recs(3)]})
# size of one synthetic record = the record size that reproduces the real numBytes of every split to the
# byte (features.json + per-id counts in stats/*_id.json): 5,456,392 bytes for a 3-byte id, + 16 framing
assert len(record(("size",), 0, 47.3, 8.5)) == 5_456_392 + 16

# ---------------------------------------------------------------- 3. download() against a local server
LISTING, RANGES = [], []
for split in ("test", "train", "validation"):
    for i in range(1024):
        p = SRC / F.shard_name(split, i)
        LISTING.append({"name": F.LIST_PREFIX + p.name, "size": str(p.stat().st_size) if p.exists() else "1",
                        "md5Hash": base64.b64encode(hashlib.md5(p.read_bytes()).digest()).decode() if p.exists() else "x"})
for n in ("features.json", "dataset_info.json"):
    LISTING.append({"name": F.LIST_PREFIX + n, "size": str((SRC / n).stat().st_size),
                    "md5Hash": base64.b64encode(hashlib.md5((SRC / n).read_bytes()).digest()).decode()})


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path == "/storage/v1/b/forest_typology/o":
            assert q["prefix"] == [F.LIST_PREFIX]
            s = int(q.get("pageToken", ["0"])[0])
            body = {"items": LISTING[s:s + 1000]}
            if s + 1000 < len(LISTING):
                body["nextPageToken"] = str(s + 1000)
            data, code = json.dumps(body).encode(), 200
        else:
            p = SRC / u.path.rsplit("/", 1)[-1]
            if not p.exists():
                self.send_error(404)
                return
            data, code = p.read_bytes(), 200
            if r := self.headers.get("Range"):
                s = int(r.split("=")[1].split("-")[0])
                RANGES.append((p.name, s))
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {s}-{len(data) - 1}/{len(data)}")
                data, code = data[s:], None
        if code:
            self.send_response(code)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
F.BASE_URL = f"http://127.0.0.1:{srv.server_address[1]}/forest_typology/forty_v1/1.0.0/"
F.LIST_URL = f"http://127.0.0.1:{srv.server_address[1]}/storage/v1/b/forest_typology/o"
RAW = TMP / "raw"
SUB = {"train": 2, "validation": 1, "test": 1}
paths = F.download(RAW, shards=SUB)
assert [p.name for p in paths] == ["features.json", "dataset_info.json", F.shard_name("train", 0),
                                   F.shard_name("train", 1), F.shard_name("validation", 0), F.shard_name("test", 0)]
assert all(p.read_bytes() == (SRC / p.name).read_bytes() for p in paths)
victim = RAW / F.shard_name("train", 1)                               # interrupted download
half = victim.read_bytes()[: victim.stat().st_size // 2]
victim.unlink()
(RAW / (victim.name + ".part")).write_bytes(half)
F.download(RAW, shards=SUB)
assert RANGES == [(victim.name, len(half))], RANGES                   # resumed, others skipped by size
assert victim.read_bytes() == (SRC / victim.name).read_bytes() and not list(RAW.glob("*.part"))
for it in LISTING:                                                      # published md5 disagrees -> refused
    if it["name"].endswith(F.shard_name("test", 0)):
        it["md5Hash"] = "AAAAAAAAAAAAAAAAAAAAAA=="
(RAW / F.shard_name("test", 0)).unlink()
expect_error(lambda: F.download(RAW, shards=SUB, retries=2), "failed after 2 attempts")
assert not (RAW / F.shard_name("test", 0)).exists()
srv.shutdown()
shutil.copy(SRC / F.shard_name("test", 0), RAW)

# ---------------------------------------------------------------- 4. inventory + convert + validate
inv = F.inventory(RAW)
assert inv["sampled"] == 4 + 3 + 3 and inv["label_check"] == "pass", inv
assert inv["splits"]["train"]["shards"] == 2 and inv["splits"]["train"]["records_expected"] == 10

OUT1, OUT2 = TMP / "out1", TMP / "out2"
man = F.convert(RAW, OUT1, shards=SUB, min_chips_label_check=5)
assert man["splits"] == {"test": 3, "train": 10, "val": 3}, man["splits"]
assert man["classes"] == F.CLASSES and man["ignore_index"] == 255
assert man["channels"] == ["VV", "VH"] and man["units"] == "dB" and man["chip_size"] == 128
assert man["pixel_spacing_m"] == 10.0 and "Label-code check passed" in man["notes"], man["notes"]
for text in ("passed early (after 5 train chips) and at the end (10 train chips)",   # early + final check
             "Label 255 where VV and VH are both NaN",                                 # no-SAR rule
             "'train': 0.002, 'val': 0.3346, 'test': 0.002",                           # no-SAR pixel shares
             "exactly one channel NaN (label kept): {'train': 10, 'val': 2, 'test': 3}",
             "NOT the held-out unit", "run options, not pre-registered", "UNVERIFIED",
             "planted forest and tree crops cannot be detected"):
    assert text in man["notes"], (text, man["notes"])
v = P.validate(OUT1, F.DATASET)
assert set(v.split) == {"train", "val", "test"}
m2 = json.loads((OUT1 / F.DATASET / "manifest.json").read_text(encoding="utf-8"))
assert m2["classes"] == {str(k): n for k, n in F.CLASSES.items()} and set(m2["classes"].values()) == {
    "natural_forest", "planted_forest", "tree_crops", "other_vegetation", "built", "water", "ice", "bare_ground"}

want_ids = {0: 255, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}           # source code -> prepared id
order = [("train", 0, i) for i in range(5)] + [("train", 1, i) for i in range(5)]
for split, keys in (("train", order), ("val", [("validation", 0, i) for i in range(3)]),
                    ("test", [("test", 0, i) for i in range(3)])):
    d = P.load_split(OUT1, F.DATASET, split)
    assert d["images"].shape == (len(keys), 2, 128, 128) and d["images"].dtype == np.float16
    assert d["labels"].shape == (len(keys), 128, 128) and d["labels"].dtype == np.uint8
    for a, b, c in ROWS:                                                  # codes -> ids outside the no-SAR corner
        assert (d["labels"][[j for j, k in enumerate(keys) if k != ('validation', 0, 2)], a:b, :120]
                == want_ids[c]).all(), (split, c)
    for j, (sp, si, ri) in enumerate(keys):                               # exact, incl. 255 where no SAR
        lab = np.array(d["labels"][j])                                    # a copy, not a view of the map
        assert np.array_equal(lab, EXPECT[("src", sp, si, ri)]["labels"]), (split, j)
        img = np.asarray(d["images"][j], np.float32)
        assert np.array_equal(lab == 255, np.isnan(img).all(0) | (LAB == 0)), (split, j)   # ignore = unknown | no SAR
    meta = d["meta"]
    assert meta["sample_id"].between(0, 199_999).all()
    assert list(meta.columns[:3]) == ["chip_id", "region", "date"] and meta["date"].fillna("").eq("").all()
    assert {"source_file", "record", "orbit", "valid_share", "sample_id", "lat", "lon"} <= set(meta.columns)
    for j, (sp, si, ri) in enumerate(keys):
        img = np.asarray(d["images"][j], np.float32)
        exp = EXPECT[("src", sp, si, ri)]["mean"]
        assert np.array_equal(np.isnan(img), np.isnan(exp)), (split, j)
        diff = np.abs(img - exp)[np.isfinite(exp)]
        assert diff.size == 0 or diff.max() < 0.02, (split, j, diff.max())
        assert meta.source_file[j] == F.shard_name(sp, si) and meta.record[j] == ri and meta.orbit[j] == "asc"
        assert meta.region[j] == F.region_of(meta.lat[j], meta.lon[j])
im = np.asarray(P.load_split(OUT1, F.DATASET, "train")["images"][0], np.float32)
assert abs(im[0, 20, 64] - (-8 + 3 + 0.64)) < 0.01                  # natural forest: mean of 4 steps in dB
assert abs(im[0, 2, 2] - (-12 + 2 + 0.02)) < 0.01                   # step 3 masked: mean of steps 0-2
assert abs(im[0, 5, 60] - (-12 + 4 + 0.60)) < 0.01                  # NaN in step 0 ignored: mean of 2, 4, 6
assert abs(im[1, 20, 64] - (-15 + 3 + 0.64)) < 0.01                 # VH channel, dB kept (no conversion)
lin_mean_db = 10 * np.log10(np.mean(10 ** ((-5 + 0.64 + OFFS - 3) / 10)))
assert abs(im[0, 20, 64] - lin_mean_db) > 0.3                        # really a mean of dB, not of linear
HOLE = np.zeros((128, 128), bool)
HOLE[124:, 120:] = True
assert np.isnan(im[:, HOLE]).all() and np.isfinite(im[1][~HOLE]).all()
assert np.isnan(im[0, 60, 10]) and np.isfinite(im[1, 60, 10])       # only VV missing at (60, 10)
vv_ok = ~HOLE
vv_ok[60, 10] = False
assert np.isfinite(im[0][vv_ok]).all()                              # VV finite everywhere else
tl = np.asarray(P.load_split(OUT1, F.DATASET, "train")["labels"][0])
assert LAB[HOLE].min() == 8 and (tl[HOLE] == 255).all()              # labelled bare ground, no SAR -> 255
assert (tl[124:, :120] == want_ids[8]).all() and tl[60, 10] == want_ids[LAB[60, 10]] == 3   # one NaN -> kept
tr = P.load_split(OUT1, F.DATASET, "train")["meta"]
assert tr.region[:5].tolist() == ["lat+40_lon+000", "lat-10_lon-070", "lat+60_lon+020", "lat+00_lon+100", "lat-40_lon+150"]
assert tr.sample_id[0] == ID0 and abs(tr.lat[0] - 47.3) < 1e-5
va = P.load_split(OUT1, F.DATASET, "val")
assert va["meta"].valid_share.tolist()[2] == 0.0 and np.isnan(np.asarray(va["images"][2], np.float32)).all()
assert (va["labels"][2] == 255).all()                                 # no SAR at all -> nothing scored
del tl

# ignore_no_sar=False keeps the old behaviour (labels scored without input), and says so in the notes
man4 = F.convert(RAW, TMP / "out4", shards={"train": 0, "validation": 1, "test": 0}, ignore_no_sar=False)
assert man4["splits"] == {"val": 3} and "Labels KEPT where VV and VH are both NaN" in man4["notes"]
assert "skipped (0 train chips < 500)" in man4["notes"]
v4 = P.load_split(TMP / "out4", F.DATASET, "val")["labels"]
assert np.array_equal(v4[2], v4[0]) and (v4[2] != 255).sum() == (LAB != 0).sum()
del v4

# determinism: a second run gives identical arrays and meta
man2 = F.convert(RAW, OUT2, shards=SUB, min_chips_label_check=5)
for split in ("train", "val", "test"):
    a, b = P.load_split(OUT1, F.DATASET, split, mmap=False), P.load_split(OUT2, F.DATASET, split, mmap=False)
    assert np.array_equal(a["images"], b["images"], equal_nan=True) and np.array_equal(a["labels"], b["labels"])
    assert (OUT1 / F.DATASET / f"{split}_meta.csv").read_bytes() == (OUT2 / F.DATASET / f"{split}_meta.csv").read_bytes()
assert {k: v for k, v in man.items() if k != "created_utc"} == {k: v for k, v in man2.items() if k != "created_utc"}

# other orbit, single seasonal step
man3 = F.convert(RAW, TMP / "out3", orbit="desc", reduce=2, shards={"train": 1, "validation": 0, "test": 0},
                 min_chips_label_check=5)
assert man3["splits"] == {"train": 5} and "seasonal step 2" in man3["notes"]
d3 = P.load_split(TMP / "out3", F.DATASET, "train")
for j in range(5):
    e = EXPECT[("src", "train", 0, j)]["desc2"]
    img = np.asarray(d3["images"][j], np.float32)
    assert np.array_equal(np.isnan(img), np.isnan(e)) and np.abs(img - e)[np.isfinite(e)].max() < 0.02
assert (d3["meta"].orbit == "desc").all()

# ---------------------------------------------------------------- 5. every run-time assumption check fires
# what check_label_codes can and cannot tell apart. Shares: paper Fig. 3a train bars (read off the chart);
# VV means: ILLUSTRATIVE dB values per class for this unit test only, not dataset facts.
names = list(F.SOURCE_CODES.values())
SHARE = dict(zip(names, [.141, .198, .082, .097, .271, .062, .058, .004, .087]))
VVC = dict(zip(names, [-10.0, -7.5, -8.0, -8.0, -11.0, -6.0, -19.0, -13.0, -15.0]))


def verdict(order):                                                    # order[c] = class stored under code c
    return F.check_label_codes(np.array([SHARE[n] for n in order]) * 1e6, np.array([VVC[n] for n in order]))


readme = ["unknown", "natural_forest", "planted_forest", "tree_crops", "other_vegetation", "water", "ice",
          "bare_ground", "built"]
assert verdict(names) == []
for order in (readme, readme[1:] + ["unknown"], names[1:] + ["unknown"],           # README 1-/0-based, bar 0-based
              names[:5] + ["bare_ground", "water", "ice", "built"]):              # built <-> bare
    assert verdict(order), order                                                  # caught
assert any("built <-> bare" in s for s in verdict(names[:5] + ["bare_ground", "water", "ice", "built"]))
for order in ([names[0], names[2], names[1]] + names[3:], [names[0], names[1], names[3], names[2]] + names[4:],
              [names[0], names[3], names[2], names[1]] + names[4:]):
    assert verdict(order) == [], order                 # documented LIMIT: swaps among the forest types pass

readme_order = np.array([0, 1, 2, 3, 4, 8, 5, 6, 7], np.uint8)[LAB]       # codes as in the README list order
swap58 = np.array([0, 1, 2, 3, 4, 8, 6, 7, 5], np.uint8)[LAB]             # built and bare codes swapped
for name, plan, text, kw in [
    ("permuted", {"train": [recs(5, labels=readme_order)]}, "label-code assumption", {}),
    ("swap58", {"train": [recs(5, labels=swap58)]}, "built <-> bare swapped?", {}),
    ("code9", {"train": [recs(1, labels=np.full((128, 128), 9, np.uint8))]}, "outside 0..8", {}),
    ("linear", {"train": [recs(2, scale="linear")]}, "failed", {}),
    ("mask2", {"train": [recs(2, mask_value=2)]}, "s1_asc_mask has values", {}),
]:
    build(TMP / name, plan)
    expect_error(lambda: F.convert(TMP / name, TMP / f"o_{name}", shards={"train": 1, "validation": 0, "test": 0},
                                   min_chips_label_check=5, **kw), text)
assert not (TMP / "o_permuted" / F.DATASET / "manifest.json").exists()

# the early check stops the run before later shards are read: shard 1 is corrupt, but the error is the
# label-code one after 5 chips; with the threshold above 5 chips the corrupt shard is reached first
build(TMP / "early", {"train": [recs(5, labels=readme_order), recs(5, labels=readme_order)]})
sh = TMP / "early" / F.shard_name("train", 1)
b = bytearray(sh.read_bytes())
b[5000] ^= 0xFF
sh.write_bytes(bytes(b))
EARLY = {"train": 2, "validation": 0, "test": 0}
expect_error(lambda: F.convert(TMP / "early", TMP / "o_early", shards=EARLY, min_chips_label_check=5),
             "failed at the early check after 5 train chips")
expect_error(lambda: F.convert(TMP / "early", TMP / "o_early2", shards=EARLY, min_chips_label_check=10),
             "data CRC mismatch")
assert not (TMP / "o_early" / F.DATASET / "manifest.json").exists()

# a sample id seen twice (here: across train and validation) stops the run
build(TMP / "dupid", {"train": [recs(2)], "validation": [[{"sid": ID0}]]})
expect_error(lambda: F.convert(TMP / "dupid", TMP / "o_dupid", shards={"train": 1, "validation": 1, "test": 0}),
             f"sample id {ID0} in {F.shard_name('validation', 0)}#0 was already read from")
bad = TMP / "corrupt"
shutil.copytree(TMP / "code9", bad)
sh = bad / F.shard_name("train", 0)
b = bytearray(sh.read_bytes())
b[5000] ^= 0xFF
sh.write_bytes(bytes(b))
expect_error(lambda: list(F.read_records(sh)), "data CRC mismatch")
sh.write_bytes(bytes(b[:-100]))
expect_error(lambda: list(F.read_records(sh, verify_crc=False)), "truncated record")
build(TMP / "count_ok", {"train": [recs(2)]})
info = json.loads((TMP / "count_ok" / "dataset_info.json").read_text())
next(s for s in info["splits"] if s["name"] == "train")["shardLengths"][0] = "3"
(TMP / "count_ok" / "dataset_info.json").write_text(json.dumps(info))
expect_error(lambda: F.convert(TMP / "count_ok", TMP / "o_count", shards={"train": 1, "validation": 0, "test": 0},
                               min_chips_label_check=5), "dataset_info.json says 3")
expect_error(lambda: F.convert(RAW, TMP / "o_missing", shards={"train": 3}), "train shards missing")

del d, va, d3                                                          # release memory maps (Windows)
gc.collect()
shutil.rmtree(TMP)
print("forest verified: CRC32C + varints, record size, download (listing pages, md5, resume, refusal), inventory, "
      "convert (splits, ids, ignore incl. no-SAR -> 255 and one-channel kept, ignore_no_sar=False, dB mean of "
      "valid steps, NaN no-data, region/date meta, notes), validate, determinism, desc/step option, label-code "
      "check scope (caught vs documented limits), early label-code check, built/bare swap, duplicate ids, and "
      "every run-time assumption check")
