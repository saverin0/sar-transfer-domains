"""Forest domain: ForTy v1 (Google DeepMind forest typology) -> prepared format.

Every fact below was read from a primary source on 2026-09-26. Sources:
  [R]  forty_v1/README.md, github.com/google-deepmind/forest_typology
  [P]  Jiang & Neumann, "Not every tree is a forest: benchmarking forest types
       from satellite remote sensing", IGARSS 2025, arXiv:2505.01805v1
  [F]  features.json in the bucket (the TFDS feature spec)
  [D]  dataset_info.json in the bucket (split names, numBytes, 1024 shard lengths each)
  [S]  stats/ in the bucket (the dataset's own statistics over each full split):
       <split>_s1_{asc,desc}_band_{0,1,2}.json, <split>_s1_{asc,desc}_mask_band_{0,1,2}.json,
       <split>_id.json (count per sample id). It has no label statistics.
  [G]  github.com/google-deepmind/geeflow (the export library the jeo pipeline
       uses): ee_data.Sentinel1, pipelines._add_mask_and_rename
  [E]  Earth Engine catalogue COPERNICUS/S1_GRD + guides/sentinel1
  [T]  TFRecord framing: xla/tsl/lib/io/record_writer.h, tsl/lib/hash/crc32c.h;
       tf.train.Example: tensorflow/core/example/{example,feature}.proto;
       TFDS serializer: tensorflow_datasets/core/example_serializer.py

Files (verified: [R], [D], the GCS listing): gs://forest_typology/forty_v1/1.0.0/
    forty_v1-{train,validation,test}.tfrecord-NNNNN-of-01024, 1024 shards per split,
    plain uncompressed TFRecords: in all three splits the listed file bytes equal
    dataset_info numBytes + 16 bytes of framing per record, exactly. The record
    layout below reproduces numBytes of EVERY split to the byte (train
    874,064,877,546; validation 107,278,121,550; test 109,935,384,392), computed
    from features.json and the per-id counts in [S] <split>_id.json (a sample id
    takes 1-3 varint bytes). So there are no other features, lat/lon are stored
    as float32, and every mask and label value is < 128. Every object has an md5
    in the listing.
    train 160,191 / validation 19,661 / test 20,148 samples ([R], [D]). Sample ids
    occur once per split, no two splits share one, and together they are exactly
    0..199,999 ([S] *_id.json). convert() re-checks uniqueness at run time.

Record format (verified: [F], [T]): one tf.train.Example per record. TFDS stores a
    Tensor(encoding=none) as int64_list (integer dtypes, incl. uint8) or float_list
    (float dtypes; float64 lat/lon are downcast to float32), flattened in C order,
    keyed by the feature name. Used here:
        s1_asc, s1_desc            float32 (4, 128, 128, 3)
        s1_asc_mask, s1_desc_mask  uint8   (4, 128, 128, 3)
        segmentation_labels        uint8   (128, 128)
        lat, lon                   float64 (stored float32); id int64 (Scalar)
    We decode this with our own TFRecord reader (length + masked CRC32C framing,
    CRC32C implemented here) and our own protobuf decoder. No TensorFlow.

Channels and units (verified):
    s1 channel 0 = VV, 1 = VH, 2 = incidence angle. geeflow's Sentinel1 source has
    BANDS = ["VV", "VH", "angle"] from COPERNICUS/S1_GRD [G]; the Earth Engine
    catalogue gives VV/VH in dB (sigma0, 10*log10, thermal noise removed,
    orthorectified, no radiometric terrain flattening) and "angle" in degrees [E].
    The dataset's own train stats agree [S]: band 0 mean -10.8 dB, band 1 mean
    -17.9 dB, band 2 range 29.4..46.1 (Sentinel-1 IW incidence range). So we keep
    channels 0 and 1 as they are: dB, no conversion. The angle is dropped.
    Checked again at run time on the first records (_check_channels).

No data (verified [G] + [S]): geeflow writes <source>_mask = the Earth Engine
    mask as uint8 and fills masked pixels with 0. In [S] the mask has only values
    0/1 and the band statistics count exactly the mask==1 pixels, so 1 = valid.
    Valid train pixel-steps: ascending 69.1 %, descending 84.1 %. The VV and VH
    masks have almost the same counts (train mask==1 of 10,498,277,376 pixel-steps:
    asc 7,258,742,457 VV vs 7,258,801,078 VH; desc 8,823,375,352 vs 8,823,375,357;
    validation equal in both orbits) [S]. A step counts as valid where mask == 1
    and the value is finite; a channel with no valid step at a pixel is NaN there.
    Masks other than 0/1 stop the conversion.
    Labels without SAR: with ignore_no_sar=True (default) the label is 255 wherever
    BOTH VV and VH are NaN, because a SAR-only model cannot see those pixels and
    run.py scores every label that is not 255. This is the glacial_lakes rule;
    snow.py uses "either channel NaN". By the mask counts above the two rules
    differ on very few pixels here; convert() records how many pixels have exactly
    one channel NaN and how many were set to 255 for lack of SAR.

Seasonal steps: [R] "Seasonal mosaics for 2020"; Sentinel-1 has the "same temporal
    extent as Sentinel-2". The ORDER of the 4 steps is not documented (the paper's
    Fig. 2 shows a "VV winter-spring-summer composition"; geeflow builds date
    ranges in increasing time order, and it also has a per-hemisphere option).
    The compositing reducer is not documented either ([P] says "mosaics").
    reduce="mean": per pixel, the mean of the dB values over the valid steps,
    which does not depend on the step order. reduce=<0..3> takes one step; that
    is only meaningful once the order is known (UNVERIFIED).

Converter defaults, fixed before any number. They are NOT part of the
    pre-registered protocol, which only says "ForTy v1 (shard subset)":
    orbit="asc", reduce="mean", shards 32 / 8 / 8, ignore_no_sar=True.
    Notebook 01 sets orbit="desc" and 32 / 32 / 32 shards (decided 2026-09-27).
    Descending has more valid SAR (84.1 % vs 69.1 % of train pixel-steps [S]).
    The orbit is a convert() option only: every record holds both orbits, so the
    download is the same either way.

Classes. [R] lists 8 classes plus Unknown without integer codes, and no public
    source gives them (searched: [R], [P], the forest_typology repo tree incl. its
    demo notebook, the geeflow and jeo trees, and [S], which has no label
    statistics). The source CODES are taken from the order of the label colour
    bar (paper Fig. 2, repo image forty_v1_examples.png) and the x-axis of the
    class histogram (paper Fig. 3a), which agree and differ from the README list
    order:
        0 unknown, 1 natural forest, 2 planted forest, 3 tree crops,
        4 other vegetation, 5 built, 6 water, 7 ice, 8 bare ground
    This is an inference (UNVERIFIED). Run-time checks: every label value must be
    in 0..8 (else the run stops), and check_label_codes() on the train chips, run
    EARLY (as soon as min_chips_label_check train chips are read: about 4 shards,
    ~1 min, at the default 500) and again at the end; a failure stops the run
    before any manifest is written:
      - code 0 share in 3-35 % (Fig. 3a: unknown ~14 %);
      - code 4 the most frequent of codes 1-8 (other vegetation ~27 %);
      - code 7 share < 2 % (ice ~0.4 %);
      - mean VV of code 6 (water) at least 3 dB below code 1 (natural forest);
      - mean VV of code 5 (built) at least 3 dB above code 8 (bare ground). This
        is a physical expectation (buildings scatter C-band VV strongly back;
        bare soil, sand and rock much less), not a dataset fact.
    These catch the README list order (1- or 0-based), the colour-bar order
    0-based, unknown stored as 9, and a built <-> bare swap. LIMIT: neither the
    SAR nor the shares can tell natural forest, planted forest and tree crops
    apart (codes 1-3; swaps among them pass every check), so their per-class
    results rest on the inferred order, and any write-up must say so.
    Prepared ids: source code c in 1..8 -> c - 1 (0..7); code 0 (unknown) -> 255.

Splits (verified [R], [P], [D]): official train / validation / test, split by
    100 x 100 km blocks assigned 8:1:1 at random. Mapped to "train" / "val" /
    "test". Subset: the first N shards of each split in sorted order, default
    32 / 8 / 8 = 5,052 / 161 / 146 chips, 29.24 GB to download (numbers from [D]
    and the listing). Val/test at 8 shards are small (rare classes such as ice
    may get almost no pixels); 32 shards give 618 / 636 chips, 64 give 1,244 /
    1,272 ([D]). Shard lengths look like random hash assignment (variance/mean
    0.99-1.06, lag-1 correlation -0.11 to -0.01 in [D]), so the first shards are
    very likely a random sample: evidence, not proof. convert() prints how many
    10-degree regions each split covers.

Chip: 128 x 128 px at 10 m ([R]: 1280 x 1280 m plots, 10 m S1), not resampled.
Meta: region = 10-degree lat/lon cell (lower-left corner) of the sample's lat/lon.
    It is NOT the held-out unit: the official split is by 100 km blocks and the
    block is not stored in the records ([F]), so train and val/test share
    10-degree cells and per-region scores are per cell, not held-out-region
    results. date = "" (seasonal composites of 2020, no acquisition date). Also
    shard file, record index, sample id, lat, lon, orbit, valid_share (finite VV
    share of the chip).
Licence: CC BY-SA 4.0 ([R], [P] footnote 1), so derived data are share-alike.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import struct
import time
from pathlib import Path

import numpy as np

from .prepared import IGNORE, PreparedWriter

DATASET = "forty_v1"

BASE_URL = "https://storage.googleapis.com/forest_typology/forty_v1/1.0.0/"
LIST_URL = "https://storage.googleapis.com/storage/v1/b/forest_typology/o"
LIST_PREFIX = "forty_v1/1.0.0/"

SOURCE = {
    "name": "ForTy v1 (FORest TYpes benchmark), version 1.0.0",
    "provider": "Google DeepMind",
    "bucket": "gs://forest_typology/forty_v1/1.0.0",
    "https": BASE_URL,
    "readme": "https://github.com/google-deepmind/forest_typology/blob/main/forty_v1/README.md",
    "paper": "Y. Jiang, M. Neumann, 'Not every tree is a forest: benchmarking forest types from "
             "satellite remote sensing', IEEE IGARSS 2025, https://arxiv.org/abs/2505.01805",
    "licence": "CC BY-SA 4.0 (Creative Commons Attribution ShareAlike 4.0 International)",
    "citation": "@inproceedings{jiang2025:forty-igarss, title={Not every tree is a forest: "
                "benchmarking forest types from satellite remote sensing}, author={Yuchang Jiang "
                "and Maxim Neumann}, booktitle={Presented at IEEE IGARSS 2025}, "
                "url={https://arxiv.org/abs/2505.01805}, pages={1-6}, year={2025}}",
}

REQUIRES: list[tuple[str, str]] = []           # numpy, pandas, requests only

SPLITS = {"train": "train", "validation": "val", "test": "test"}     # official -> ours
DEFAULT_SHARDS = {"train": 32, "validation": 8, "test": 8}
N_SHARDS = 1024
SIZE = 128
SOURCE_CODES = {0: "unknown", 1: "natural_forest", 2: "planted_forest", 3: "tree_crops",
                4: "other_vegetation", 5: "built", 6: "water", 7: "ice", 8: "bare_ground"}
CLASSES = {c - 1: n for c, n in SOURCE_CODES.items() if c}          # 0..7
_LUT = np.full(256, IGNORE, np.uint8)
_LUT[1:9] = np.arange(8, dtype=np.uint8)

# the part of features.json the converter relies on: name -> (dtype, shape)
SPEC = {"s1_asc": ("float32", [4, SIZE, SIZE, 3]), "s1_asc_mask": ("uint8", [4, SIZE, SIZE, 3]),
        "s1_desc": ("float32", [4, SIZE, SIZE, 3]), "s1_desc_mask": ("uint8", [4, SIZE, SIZE, 3]),
        "segmentation_labels": ("uint8", [SIZE, SIZE]), "lat": ("float64", []),
        "lon": ("float64", []), "id": ("int64", [])}
_KIND = {"float32": "float", "float64": "float", "uint8": "int64", "int64": "int64"}


def shard_name(split: str, i: int) -> str:
    return f"forty_v1-{split}.tfrecord-{i:05d}-of-{N_SHARDS:05d}"


# ----------------------------------------------------------------- CRC32C (Castagnoli)
def _crc_table(poly: int = 0x82F63B78) -> np.ndarray:
    t = np.zeros(256, np.uint32)
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ poly if c & 1 else c >> 1
        t[i] = c
    return t


_T = _crc_table()
_TL = [int(x) for x in _T]
_BLOCK = 1024


def _crc_scalar(r: int, data: bytes) -> int:
    for b in data:
        r = _TL[(r ^ b) & 0xFF] ^ (r >> 8)
    return r


def _zero_tables(n: int) -> list[list[int]]:
    """Tables for the linear map 'feed n zero bytes' on the CRC register, one per register byte."""
    cols = []
    for bit in range(32):
        r = 1 << bit
        for _ in range(n):
            r = _TL[r & 0xFF] ^ (r >> 8)
        cols.append(r)
    out = []
    for p in range(4):
        z = [0] * 256
        for v in range(256):
            acc = 0
            for b in range(8):
                if v >> b & 1:
                    acc ^= cols[8 * p + b]
            z[v] = acc
        out.append(z)
    return out


_Z = _zero_tables(_BLOCK)


def crc32c(data: bytes) -> int:
    """CRC-32C of `data`. Blocks of 1024 bytes run in parallel with numpy, then get chained:
    the register update is linear, so crc(A + B) = shift(crc(A), len B) xor crc0(B)."""
    buf = np.frombuffer(data, np.uint8)
    k = buf.size // _BLOCK
    r = 0xFFFFFFFF
    if k:
        blocks = np.ascontiguousarray(buf[:k * _BLOCK].reshape(k, _BLOCK).T)
        c = np.zeros(k, np.uint32)
        for j in range(_BLOCK):
            c = _T[(c ^ blocks[j]) & 0xFF] ^ (c >> 8)
        z0, z1, z2, z3 = _Z
        for ci in c.tolist():
            r = z0[r & 0xFF] ^ z1[(r >> 8) & 0xFF] ^ z2[(r >> 16) & 0xFF] ^ z3[r >> 24] ^ ci
    return _crc_scalar(r, buf[k * _BLOCK:].tobytes()) ^ 0xFFFFFFFF


def masked_crc(data: bytes) -> int:
    c = crc32c(data)
    return ((((c >> 15) | (c << 17)) & 0xFFFFFFFF) + 0xA282EAD8) & 0xFFFFFFFF


def read_records(path: str | Path, verify_crc: bool = True):
    """Yield the payload of each record of an uncompressed TFRecord file."""
    path = Path(path)
    with open(path, "rb") as f:
        if f.read(2) == b"\x1f\x8b":
            raise ValueError(f"{path.name} is gzip-compressed; expected plain TFRecord")
        f.seek(0)
        i = 0
        while True:
            head = f.read(12)
            if not head:
                return
            if len(head) < 12:
                raise ValueError(f"{path.name}: truncated header at record {i}")
            n, lcrc = struct.unpack("<QI", head)
            if masked_crc(head[:8]) != lcrc:
                raise ValueError(f"{path.name}: length CRC mismatch at record {i} (not a TFRecord, or corrupt)")
            data, foot = f.read(n), f.read(4)
            if len(data) < n or len(foot) < 4:
                raise ValueError(f"{path.name}: truncated record {i} (partial download?)")
            if verify_crc and masked_crc(data) != struct.unpack("<I", foot)[0]:
                raise ValueError(f"{path.name}: data CRC mismatch at record {i} (corrupt file)")
            yield data
            i += 1


# ----------------------------------------------------------------- tf.train.Example decoder
def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if b < 0x80:
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("malformed varint")


def _fields(buf: bytes, start: int, end: int):
    """(field number, wire type, int value or (start, end) byte range) of one message."""
    pos = start
    while pos < end:
        key, pos = _varint(buf, pos)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, pos = _varint(buf, pos)
            yield fn, wt, v
            continue
        if wt == 1:
            n = 8
        elif wt == 5:
            n = 4
        elif wt == 2:
            n, pos = _varint(buf, pos)
        else:
            raise ValueError(f"unsupported protobuf wire type {wt}")
        yield fn, wt, (pos, pos + n)
        pos += n
    if pos != end:
        raise ValueError("protobuf field runs past its message")


def _varints(b: np.ndarray) -> np.ndarray:
    """Decode a packed run of varints (uint8 array) to int64, vectorised."""
    if b.size == 0:
        return np.zeros(0, np.int64)
    if int(b.max()) < 0x80:                         # every value < 128: one byte each
        return b.astype(np.int64)
    if b[-1] >= 0x80:
        raise ValueError("truncated packed varint")
    ends = np.flatnonzero(b < 0x80)
    starts = np.concatenate(([0], ends[:-1] + 1))
    pos = np.arange(b.size) - np.repeat(starts, ends - starts + 1)
    if pos.max() > 9:
        raise ValueError("varint longer than 10 bytes")
    vals = (b & 0x7F).astype(np.uint64) << (7 * pos).astype(np.uint64)
    return np.add.reduceat(vals, starts).view(np.int64)


def _feature(buf: bytes, start: int, end: int) -> tuple[str, object]:
    """tensorflow.Feature -> ("bytes" | "float" | "int64" | "none", values)."""
    for fn, wt, rng in _fields(buf, start, end):
        if wt != 2:
            continue
        s, e = rng
        if fn == 2:                                                   # FloatList
            parts = []
            for f2, w2, v in _fields(buf, s, e):
                if f2 == 1 and w2 == 2:
                    parts.append(np.frombuffer(buf, "<f4", (v[1] - v[0]) // 4, v[0]))
                elif f2 == 1 and w2 == 5:
                    parts.append(np.frombuffer(buf, "<f4", 1, v[0]))
            return "float", (np.concatenate(parts) if parts else np.zeros(0, np.float32))
        if fn == 3:                                                   # Int64List
            parts = []
            for f2, w2, v in _fields(buf, s, e):
                if f2 == 1 and w2 == 2:
                    parts.append(_varints(np.frombuffer(buf, np.uint8, v[1] - v[0], v[0])))
                elif f2 == 1 and w2 == 0:
                    parts.append(np.array([v], np.uint64).view(np.int64))
            return "int64", (np.concatenate(parts) if parts else np.zeros(0, np.int64))
        if fn == 1:                                                   # BytesList
            return "bytes", [bytes(buf[v[0]:v[1]]) for f2, w2, v in _fields(buf, s, e) if f2 == 1 and w2 == 2]
    return "none", None


def parse_example(buf: bytes, keys=None) -> dict[str, tuple[str, object]]:
    """tf.train.Example bytes -> {feature name: (kind, values)}; only `keys` are decoded."""
    out = {}
    for fn, wt, rng in _fields(buf, 0, len(buf)):
        if fn != 1 or wt != 2:                      # Example.features
            continue
        for f2, w2, ent in _fields(buf, *rng):
            if f2 != 1 or w2 != 2:                  # Features.feature (map entry)
                continue
            key, val = None, None
            for f3, w3, v in _fields(buf, *ent):
                if f3 == 1 and w3 == 2:
                    key = bytes(buf[v[0]:v[1]]).decode("utf-8")
                elif f3 == 2 and w3 == 2:
                    val = v
            if key is None:
                raise ValueError("feature map entry without a key")
            if keys is None or key in keys:
                out[key] = _feature(buf, *val) if val else ("none", None)
    return out


def _get(ex: dict, key: str) -> np.ndarray:
    """Checked array for one feature of SPEC: kind and length as TFDS writes them."""
    dtype, shape = SPEC[key]
    if key not in ex:
        raise ValueError(f"feature {key!r} missing; present: {sorted(ex)}")
    kind, v = ex[key]
    n = int(np.prod(shape)) if shape else 1
    if kind != _KIND[dtype] or v is None or len(v) != n:
        raise ValueError(f"feature {key!r}: {kind} x {None if v is None else len(v)}, expected "
                         f"{_KIND[dtype]} x {n} (TFDS Tensor {dtype} {shape}, features.json)")
    return v.reshape(shape) if shape else v


# ----------------------------------------------------------------- one record -> one chip
def _chip(ex: dict, orbit: str, reduce) -> tuple[np.ndarray, np.ndarray, dict]:
    """(image (2,H,W) float32 dB with NaN, raw label codes (H,W) uint8, meta)."""
    s1 = _get(ex, f"s1_{orbit}")[..., :2]                          # (4,H,W,2) VV, VH
    m = _get(ex, f"s1_{orbit}_mask")
    if m.min() < 0 or m.max() > 1:
        raise ValueError(f"s1_{orbit}_mask has values {np.unique(m)[:10]}; assumption from geeflow "
                         f"and stats/*_mask_band_*.json: 0 = no data, 1 = valid")
    ok = (m[..., :2] == 1) & np.isfinite(s1)
    if reduce == "mean":
        n = ok.sum(0)
        tot = np.where(ok, s1, 0.0).sum(0, dtype=np.float64)
        img = np.where(n > 0, tot / np.maximum(n, 1), np.nan)
    else:
        img = np.where(ok[reduce], s1[reduce], np.nan)
    img = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)
    lab = _get(ex, "segmentation_labels")
    if lab.min() < 0 or lab.max() > 8:
        raise ValueError(f"label values {np.unique(lab)[:20]} outside 0..8 (source codes assumed from "
                         f"paper Fig. 2/3a: 0 unknown .. 8 bare)")
    lat, lon = float(_get(ex, "lat")[0]), float(_get(ex, "lon")[0])
    meta = {"sample_id": int(_get(ex, "id")[0]), "lat": round(lat, 6), "lon": round(lon, 6)}
    return img, lab.astype(np.uint8), meta


def region_of(lat: float, lon: float, deg: int = 10) -> str:
    """Lower-left corner of the `deg`-degree cell, e.g. lat+40_lon-010."""
    la, lo = int(math.floor(lat / deg) * deg), int(math.floor(lon / deg) * deg)
    return f"lat{la:+03d}_lon{lo:+04d}"


# ----------------------------------------------------------------- run-time checks
def _check_channels(vals: list[np.ndarray]) -> dict:
    """vals: valid values of channels 0, 1, 2. Units/order as in the dataset's own stats [S]."""
    if min(v.size for v in vals) < 1000:
        raise ValueError(f"only {[v.size for v in vals]} valid S1 pixels in the sample; cannot check units")
    med = [float(np.median(v)) for v in vals]
    ok = (-25 <= med[0] <= -2 and -32 <= med[1] <= -8 and med[1] < med[0] - 2 and 25 <= med[2] <= 50)
    msg = (f"S1 channel medians {np.round(med, 2).tolist()}; expected VV dB ~-10.8, VH dB ~-17.9, "
           f"angle deg 29-46 (stats/train_s1_asc_band_*.json)")
    if not ok:
        raise ValueError("assumption 's1 channels = (VV dB, VH dB, incidence angle deg)' failed: " + msg)
    return {"median_vv": med[0], "median_vh": med[1], "median_angle": med[2]}


def _sample_channels(path: Path, orbit: str, n_records: int = 16) -> dict:
    """Pool the valid pixels of the first `n_records` records (several plots, not one lake)."""
    vals = [[], [], []]
    for i, rec in enumerate(read_records(path)):
        if i >= n_records:
            break
        ex = parse_example(rec, {f"s1_{orbit}", f"s1_{orbit}_mask"})
        x, m = _get(ex, f"s1_{orbit}"), _get(ex, f"s1_{orbit}_mask")
        for c in range(3):
            v = x[..., c][m[..., c] == 1]
            vals[c].append(v[np.isfinite(v)])
    return _check_channels([np.concatenate(v) for v in vals])


def check_label_codes(counts: np.ndarray, vv_mean: np.ndarray) -> list[str]:
    """Problems with the assumed source codes; counts / vv_mean indexed by source code 0..8."""
    share = counts / max(counts.sum(), 1)
    bad = []
    if not 0.03 <= share[0] <= 0.35:
        bad.append(f"code 0 (unknown) share {share[0]:.3f} not in 0.03-0.35 (Fig. 3a ~0.14)")
    if int(np.argmax(share[1:])) + 1 != 4:
        bad.append(f"most frequent of codes 1-8 is {int(np.argmax(share[1:])) + 1}, expected 4 other veg (~0.27)")
    if share[7] >= 0.02:
        bad.append(f"code 7 (ice) share {share[7]:.3f} >= 0.02 (Fig. 3a ~0.004)")
    if not (np.isfinite(vv_mean[6]) and np.isfinite(vv_mean[1]) and vv_mean[6] <= vv_mean[1] - 3):
        bad.append(f"mean VV water (code 6) {vv_mean[6]:.1f} dB not >= 3 dB below natural forest "
                   f"(code 1) {vv_mean[1]:.1f} dB")
    if not (np.isfinite(vv_mean[5]) and np.isfinite(vv_mean[8]) and vv_mean[5] >= vv_mean[8] + 3):
        bad.append(f"mean VV built (code 5) {vv_mean[5]:.1f} dB not >= 3 dB above bare ground "
                   f"(code 8) {vv_mean[8]:.1f} dB (built <-> bare swapped?)")
    return bad


def _code_table(counts: np.ndarray, vv_mean: np.ndarray) -> str:
    share = counts / max(counts.sum(), 1)
    return "\n".join(f"    code {c} {SOURCE_CODES[c]:17s} share {share[c]:.4f}  mean VV {vv_mean[c]:7.2f} dB"
                     for c in range(9))


def _vv_mean(vv_sum: np.ndarray, vv_n: np.ndarray) -> np.ndarray:
    return np.where(vv_n > 0, vv_sum / np.maximum(vv_n, 1), np.nan)


def _require_label_codes(counts: np.ndarray, vv_sum: np.ndarray, vv_n: np.ndarray, n_chips: int,
                         when: str) -> None:
    """Print the code table of the train chips read so far; stop the run if check_label_codes fails."""
    vv_mean = _vv_mean(vv_sum, vv_n)
    table = _code_table(counts, vv_mean)
    print(f"  train label codes, {when} check ({n_chips} chips):\n{table}")
    bad = check_label_codes(counts, vv_mean)
    if bad:
        raise ValueError(f"label-code assumption (0 unknown, 1 natural, 2 planted, 3 tree crops, 4 other veg, "
                         f"5 built, 6 water, 7 ice, 8 bare; paper Fig. 2/3a order) failed at the {when} check "
                         f"after {n_chips} train chips: " + "; ".join(bad)
                         + ". Nothing was finalised (no new manifest written). Code table:\n" + table)


def _read_spec(raw: Path) -> dict:
    fj = raw / "features.json"
    if not fj.exists():
        raise FileNotFoundError(f"{fj} missing; download() fetches it")
    feats = json.loads(fj.read_text(encoding="utf-8"))["featuresDict"]["features"]
    for k, (dtype, shape) in SPEC.items():
        t = feats[k]["tensor"]
        got = (t["dtype"], [int(d) for d in t["shape"].get("dimensions", [])], t.get("encoding"))
        if got != (dtype, shape, "none"):
            raise ValueError(f"features.json {k}: {got}, expected {(dtype, shape, 'none')}")
    return feats


def _shard_lengths(raw: Path) -> dict[str, list[int]]:
    info = json.loads((raw / "dataset_info.json").read_text(encoding="utf-8"))
    out = {s["name"]: [int(x) for x in s["shardLengths"]] for s in info["splits"]}
    if set(out) != set(SPLITS) or any(len(v) != N_SHARDS for v in out.values()):
        raise ValueError(f"dataset_info.json splits {[(k, len(v)) for k, v in out.items()]}, expected "
                         f"{sorted(SPLITS)} x {N_SHARDS}")
    return out


def _shard_files(raw: Path, shards: dict[str, int]) -> dict[str, list[Path]]:
    out = {}
    for split, k in shards.items():
        if split not in SPLITS or not 0 <= k <= N_SHARDS:
            raise ValueError(f"shards {shards}: splits must be {sorted(SPLITS)}, counts 0..{N_SHARDS}")
        files = [raw / shard_name(split, i) for i in range(k)]
        missing = [p.name for p in files if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} {split} shards missing in {raw}, e.g. {missing[:3]}")
        out[split] = files
    return out


# ----------------------------------------------------------------- download
def _listing(timeout: float = 60) -> dict[str, dict]:
    """Official object listing (GCS JSON API): file name -> {size, md5 (base64)}."""
    import requests
    out, token = {}, None
    while True:
        params = {"prefix": LIST_PREFIX, "fields": "items(name,size,md5Hash),nextPageToken", "maxResults": 1000}
        if token:
            params["pageToken"] = token
        r = requests.get(LIST_URL, params=params, timeout=timeout)
        r.raise_for_status()
        d = r.json()
        for it in d.get("items", []):
            out[it["name"][len(LIST_PREFIX):]] = {"size": int(it["size"]), "md5": it.get("md5Hash")}
        token = d.get("nextPageToken")
        if not token:
            return out


def _fetch(url: str, path: Path, size: int, md5: str | None, retries: int, chunk: int, timeout: float) -> str:
    import requests
    if path.exists() and path.stat().st_size == size:
        return "skip"
    part = path.with_name(path.name + ".part")
    for attempt in range(retries):
        try:
            have = part.stat().st_size if part.exists() else 0
            if have > size:
                part.unlink()
                have = 0
            h = hashlib.md5()
            if have:
                with open(part, "rb") as f:
                    while b := f.read(chunk):
                        h.update(b)
            if have < size:
                hdr = {"Range": f"bytes={have}-"} if have else {}
                with requests.get(url, headers=hdr, stream=True, timeout=timeout) as r:
                    r.raise_for_status()
                    if have and r.status_code != 206:           # range ignored: start again
                        have, h = 0, hashlib.md5()
                    with open(part, "ab" if have else "wb") as f:
                        for b in r.iter_content(chunk):
                            f.write(b)
                            h.update(b)
            got = part.stat().st_size
            if got != size:
                raise OSError(f"{path.name}: {got} bytes, expected {size}")
            if md5 and base64.b64encode(h.digest()).decode() != md5:
                part.unlink()
                raise OSError(f"{path.name}: md5 mismatch, removed")
            part.replace(path)
            return "ok"
        except (OSError, requests.RequestException) as e:
            wait = min(60, 2 ** attempt)
            print(f"    retry {attempt + 1}/{retries} in {wait}s: {e}")
            time.sleep(wait)
    raise RuntimeError(f"{path.name}: failed after {retries} attempts")


def download(dest, shards: dict | None = None, retries: int = 6, chunk_mb: int = 8,
             timeout: float = 120) -> list[Path]:
    """Fetch features.json, dataset_info.json and the first N shards per split into `dest`.

    Resumable: complete files (size as listed) are skipped, partial ones continue
    from their .part file. Each new file is checked against the listing's md5.
    Default 32 train / 8 validation / 8 test shards = 29.24 GB.
    """
    shards = {**DEFAULT_SHARDS, **(shards or {})}
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    listing = _listing(timeout)
    names = ["features.json", "dataset_info.json"]
    for split, k in shards.items():
        avail = sorted(n for n in listing if n.startswith(f"forty_v1-{split}.tfrecord-"))
        if split not in SPLITS or len(avail) != N_SHARDS or not 0 <= k <= N_SHARDS:
            raise ValueError(f"{split}: {len(avail)} shards listed, asked for {k}")
        names += avail[:k]
    total = sum(listing[n]["size"] for n in names)
    print(f"ForTy v1: {len(names)} files, {total / 1e9:.2f} GB -> {dest}")
    done, t0, out = 0, time.time(), []
    for i, n in enumerate(names, 1):
        st = _fetch(BASE_URL + n, dest / n, listing[n]["size"], listing[n]["md5"], retries,
                    chunk_mb << 20, timeout)
        done += listing[n]["size"]
        el = time.time() - t0
        print(f"  [{i}/{len(names)}] {st:4s} {n}  {done / 1e9:.2f}/{total / 1e9:.2f} GB  {el / 60:.1f} min")
        out.append(dest / n)
    return out


# ----------------------------------------------------------------- inventory
def inventory(raw_root, orbit: str = "asc", records_per_split: int = 4) -> dict:
    """Print what is on disk and a small decoded sample; assert the documented format."""
    raw = Path(raw_root)
    _read_spec(raw)
    lengths = _shard_lengths(raw)
    out = {"splits": {}}
    print(f"ForTy v1 inventory of {raw}: features.json matches the spec for {sorted(SPEC)}")
    for split in SPLITS:
        files = sorted(raw.glob(f"forty_v1-{split}.tfrecord-*-of-{N_SHARDS:05d}"))
        idx = [int(re.search(r"-(\d{5})-of-", p.name).group(1)) for p in files]
        gb = sum(p.stat().st_size for p in files) / 1e9
        exp = sum(lengths[split][i] for i in idx)
        contiguous = idx == list(range(len(idx)))
        print(f"  {split:10s} {len(files):4d} shards ({'first ' + str(len(idx)) if contiguous else 'NOT the first N'}),"
              f" {gb:.2f} GB, {exp} records expected (dataset_info.json)")
        out["splits"][split] = {"shards": len(files), "gb": round(gb, 3), "records_expected": exp}
    counts, vv_sum, vv_n, vals, n = np.zeros(9, np.int64), np.zeros(9), np.zeros(9), [[], [], []], 0
    for split in SPLITS:
        files = sorted(raw.glob(f"forty_v1-{split}.tfrecord-*-of-{N_SHARDS:05d}"))
        if not files:
            continue
        for i, rec in enumerate(read_records(files[0])):
            if i >= records_per_split:
                break
            ex = parse_example(rec)
            if n == 0:
                print("  record keys: " + ", ".join(
                    f"{k}={kind}x{0 if v is None else len(v)}" for k, (kind, v) in sorted(ex.items())))
            x, m = _get(ex, f"s1_{orbit}"), _get(ex, f"s1_{orbit}_mask")
            for c in range(3):
                v = x[..., c][m[..., c] == 1]
                vals[c].append(v[np.isfinite(v)])
            filled = np.unique(x[m == 0])
            img, lab, meta = _chip(ex, orbit, "mean")
            counts += np.bincount(lab.ravel(), minlength=9)[:9]
            fin = np.isfinite(img[0])
            vv_sum += np.bincount(lab[fin], img[0][fin], 9)[:9]
            vv_n += np.bincount(lab[fin], None, 9)[:9]
            print(f"  {split}/{files[0].name}#{i}: id {meta['sample_id']} lat {meta['lat']:.3f} lon {meta['lon']:.3f}"
                  f" region {region_of(meta['lat'], meta['lon'])}; mask values {np.unique(m).tolist()},"
                  f" valid steps {m[..., 0].mean():.2f}; values under mask 0: {filled[:5].tolist()};"
                  f" labels {np.unique(lab).tolist()}")
            n += 1
    if n == 0:
        raise FileNotFoundError(f"no shards in {raw}")
    ch = _check_channels([np.concatenate(v) for v in vals])
    vv_mean = np.where(vv_n > 0, vv_sum / np.maximum(vv_n, 1), np.nan)
    print(f"  channels OK: {ch}")
    print(f"  label codes over {n} sampled chips (codes assumed from paper Fig. 2/3a):\n{_code_table(counts, vv_mean)}")
    bad = check_label_codes(counts, vv_mean)
    print("  label-code check: " + ("PASS" if not bad else "not conclusive on a small sample: " + "; ".join(bad)))
    out.update(channels=ch, label_counts=counts.tolist(), label_check=bad or "pass", sampled=n)
    return out


# ----------------------------------------------------------------- convert
def convert(raw_root, out_root, orbit: str = "asc", reduce="mean", shards: dict | None = None,
            region_deg: int = 10, verify_crc: bool = True, min_chips_label_check: int = 500,
            ignore_no_sar: bool = True) -> dict:
    """Write the prepared format from the first N shards per split (default 32/8/8).

    orbit "asc" or "desc"; reduce "mean" over the valid seasonal steps or a step
    index 0..3 (order unverified). Converter defaults, fixed before any number; the
    user confirms them before the first real run (see the module docstring).
    ignore_no_sar: label 255 where VV and VH are both NaN (no SAR input).
    Train shards are read first; the label-code check runs as soon as
    min_chips_label_check train chips are in, and again at the end. Any failed
    check stops the run before the manifest is written.
    """
    if orbit not in ("asc", "desc"):
        raise ValueError("orbit must be 'asc' or 'desc'")
    if reduce != "mean" and reduce not in (0, 1, 2, 3):
        raise ValueError("reduce must be 'mean' or a step index 0..3")
    raw = Path(raw_root)
    shards = {**DEFAULT_SHARDS, **(shards or {})}
    _read_spec(raw)
    lengths = _shard_lengths(raw)
    files = _shard_files(raw, shards)
    first = next((f[0] for f in files.values() if f), None)
    if first is None:
        raise ValueError("no shards selected")
    ch = _sample_channels(first, orbit)
    print(f"ForTy v1 -> {Path(out_root) / DATASET}: orbit {orbit}, reduce {reduce}, shards {shards}; channels {ch}")
    notes = (f"S1 {'ascending' if orbit == 'asc' else 'descending'} (s1_{orbit}), 2020 seasonal composites; "
             f"{'per-pixel mean of dB over the valid of 4 seasonal steps' if reduce == 'mean' else f'seasonal step {reduce} (step order unverified)'}"
             f" (valid = mask 1 and finite, none valid -> NaN). Channels VV, VH = s1[..., 0:2], sigma0 dB as stored; "
             f"channel 2 (incidence angle) dropped. Labels: source codes 1..8 -> ids 0..7, 0 unknown -> 255 "
             f"(codes INFERRED from paper Fig. 2/3a order, UNVERIFIED; checked below, but swaps among natural "
             f"forest, planted forest and tree crops cannot be detected). "
             + ("Label 255 where VV and VH are both NaN (no SAR input, ignore_no_sar=True). " if ignore_no_sar else
                "Labels KEPT where VV and VH are both NaN (ignore_no_sar=False): these pixels are scored without "
                "input. ")
             + f"Official splits train/validation/test -> train/val/test, first shards {shards} (orbit, "
             f"reducer and shard counts are run options, not pre-registered). region = {region_deg}-degree lat/lon cell, NOT the held-out unit "
             f"(official split by 100 km blocks, not stored in the records): train and val/test share cells, so "
             f"per-region scores are not held-out-region results. date empty.")
    w = PreparedWriter(out_root, DATASET, SIZE, CLASSES, SOURCE, channels=("VV", "VH"), units="dB",
                       pixel_spacing_m=10.0, notes=notes)
    counts, vv_sum, vv_n = np.zeros(9, np.int64), np.zeros(9), np.zeros(9)
    n_train, early = 0, None
    seen: dict[int, str] = {}
    chips, empty, regions, no_sar, one_nan, t0 = {}, {}, {}, {}, {}, time.time()
    for split in SPLITS:                            # train first, so the early label-code check can fire
        ours = SPLITS[split]
        for p in files.get(split, []):
            si = int(re.search(r"-(\d{5})-of-", p.name).group(1))
            n = 0
            for rec in read_records(p, verify_crc):
                ex = parse_example(rec, {f"s1_{orbit}", f"s1_{orbit}_mask", "segmentation_labels", "lat", "lon", "id"})
                img, lab, meta = _chip(ex, orbit, reduce)
                sid = meta["sample_id"]
                if sid in seen:
                    raise ValueError(f"sample id {sid} in {p.name}#{n} was already read from {seen[sid]}; "
                                     f"stats/*_id.json: ids are unique and the official splits are disjoint")
                seen[sid] = f"{p.name}#{n}"
                fin = np.isfinite(img[0])
                nan = np.isnan(img)
                none = nan.all(0)
                if split == "train":
                    counts += np.bincount(lab.ravel(), minlength=9)[:9]
                    vv_sum += np.bincount(lab[fin], img[0][fin], 9)[:9]
                    vv_n += np.bincount(lab[fin], None, 9)[:9]
                    n_train += 1
                    if early is None and n_train >= min_chips_label_check:
                        _require_label_codes(counts, vv_sum, vv_n, n_train, "early")
                        early = n_train
                vs = float(fin.mean())
                chips[ours] = chips.get(ours, 0) + 1
                empty[ours] = empty.get(ours, 0) + int(vs == 0)
                no_sar[ours] = no_sar.get(ours, 0) + int(none.sum())
                one_nan[ours] = one_nan.get(ours, 0) + int((nan.any(0) & ~none).sum())
                reg = region_of(meta["lat"], meta["lon"], region_deg)
                regions.setdefault(ours, set()).add(reg)
                ids = _LUT[lab]
                if ignore_no_sar:
                    ids[none] = IGNORE
                w.add(ours, img, ids, region=reg, date="",
                      source_file=p.name, record=n, orbit=orbit, valid_share=round(vs, 4), **meta)
                n += 1
            if n != lengths[split][si]:
                raise ValueError(f"{p.name}: {n} records, dataset_info.json says {lengths[split][si]}")
            print(f"  {p.name}: {n} chips  ({(time.time() - t0) / 60:.1f} min)")
    if n_train >= min_chips_label_check:
        _require_label_codes(counts, vv_sum, vv_n, n_train, "final")
        check = f"passed early (after {early} train chips) and at the end ({n_train} train chips)"
    else:
        print(f"  train label codes ({n_train} chips):\n{_code_table(counts, _vv_mean(vv_sum, vv_n))}")
        check = f"skipped ({n_train} train chips < {min_chips_label_check})"
    n_reg = {k: len(v) for k, v in regions.items()}
    no_sar_share = {k: round(v / (SIZE * SIZE * chips[k]), 4) for k, v in no_sar.items()}
    print(f"  label-code check {check}; chips without any valid VV pixel: {empty}; pixel share with no SAR "
          f"(both channels NaN): {no_sar_share}; pixels with one channel NaN: {one_nan}; regions per split: {n_reg}")
    w.notes += (f" Label-code check {check}. Chips with no valid VV pixel: {empty}. Pixel share with no SAR (both "
                f"channels NaN{', label set to 255' if ignore_no_sar else ', label kept'}): {no_sar_share}. Pixels "
                f"with exactly one channel NaN (label kept): {one_nan}. Regions per split: {n_reg}.")
    return w.close()
