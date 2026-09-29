"""Snow cover: Sentinel-1 VV/VH + MODIS snow masks, French Alps (SnowSAR replication data, Briand et al.).

Dataset: Briand, Weissgerber, Lobry, Idier (2025), "Replication Data for : Weakly
supervised learning for snow cover segmentation in mountainous areas from
Sentinel-1 SAR images using interpolated NDSI time series" (title exactly as the
API gives it), Recherche Data Gouv, doi:10.57745/IMTSFL, version 1.0 released
2025-12-15, licence Etalab Open Licence 2.0. First two authors: DTIS, ONERA,
Universite Paris-Saclay [API]. "SnowSAR" is OUR short label, taken from the file
name snowsar_dataset.zip; it is not an official name. Paper: Briand et al., ISPRS J.
Photogramm. Remote Sens. 239, 670-688 (2026), doi:10.1016/j.isprsjprs.2026.05.013
(preprint hal-05186268).

Sources opened 2026-09-26 (nothing below is guessed; see "unverified" for the rest):
  [API]  Dataverse API .../api/datasets/:persistentId/?persistentId=doi:10.57745/IMTSFL
         two files: README.md (id 708620, 5,617 B, md5 a30cfe33...) and
         snowsar_dataset.zip (id 708968, 43,036,387,541 B, md5 94a41d1a...)
  [README] README.md from the API (md5 checked)
  [ZIP]  the zip's central directory only (HTTP range requests on the last
         ~200 kB of the file: ZIP64 end records + 134,392 B directory, 981 entries),
         with each member's name and compressed/uncompressed size
  [PAPER] preprint hal.science/hal-05186268v1/file/draft_preprint.pdf, Sections 2-4
         (incl. 2.3, 4.1, 4.2, 4.4.1), Tables 1 and 5; Crossref metadata of the journal DOI

Layout inside the zip [ZIP] (the README draws the same tree, see the label note):
    snowsar_dataset/<split>/values/<orbit>_<YYYYMMDD>.tif      SAR, 2 bands
    snowsar_dataset/<split>/labels/<variant>/<same name>.tif   snow mask (see "Classes")
    <split> = train | validation | test/no_transfer | test/spatial_transfer
              | test/temporal_transfer; plus projection/*.npz, reference_images/,
              stats/*.pkl (not used here)
  Scenes per split [ZIP]: train 88, validation 24, no_transfer 22,
  spatial_transfer 111, temporal_transfer 136. Every values file has a label of
  the same name; train also has cni/ and ks/ labels for all 88 [ZIP]. 88/24/22
  and "134 acquisitions" for Guil 2018 match the paper (Sec. 4.1).
  Image and label are paired by identical file name.

Label variants. The README calls them no_interpolation, cni, ks, and puts only
no_interpolation under validation/test. The zip stores them as labels/raw,
labels/cni and labels/ks, with only labels/raw under validation/test [ZIP]. So
"raw" is the README's no_interpolation, i.e. observed MODIS labels with cloud
gaps. This is inferred from the matching structure; it is not spelled out anywhere.

Classes [README "File Format"]: label 0 = no snow, 1 = snow, -1 = no data, "1 band,
uint8". A uint8 cannot hold -1. The zip itself (version 1.0, looked at on
2026-09-28) stores float32 labels with 2 bands (raw, cni) or 3 bands (ks): band 1
holds the README codes -1 / 0 / 1; band 2 was 1.0 in every file looked at and is
not documented. The label is band 1. Pixels where band 2 is not exactly 1 become no
data (255) and are counted (inventory row, convert printout, manifest notes), so a
pixel the dataset may flag is never used; band 3 is not used. The converter
accepts exactly these encodings of band 1 and raises on any other value:
    uint8 file: 255 (the uint8 wrap-around of -1) = no data
    signed integer or float file: -1 = no data (float: whole numbers only, NaN = no data)
  A GDAL nodata tag, if present, also marks no data, except a tag of 0 or 1: that
  collides with the README's classes, so the README wins, and the tag is ignored
  and reported (inventory row, convert printout, manifest notes).
  Prepared classes: {0: "no_snow", 1: "snow"}, no data -> 255 (ignore). The
  labels are MODIS MOD10A1 C6.1 NDSI > 0.4, same-day as the SAR (paper Sec. 3.3.1),
  i.e. 500 m blocks projected onto the SAR grid, not hand-drawn outlines.

Splits [README "Data Splits" table + ZIP]:
    train                    -> "train"          Guil,    2018-09-01..2019-06-30
    validation               -> "val"            Guil,    same period
    test/no_transfer         -> "test"           Guil,    same period
    test/spatial_transfer    -> "test_spatial"   Gyronde, 2018-2019
    test/temporal_transfer   -> "test_temporal"  Guil,    2019-09-01..2020-06-30
  Only "train" may be used for training. Default labels everywhere: the observed
  no_interpolation labels (pre-registered 2026-09-26: train AND score on them);
  cni/ks exist for train only and can be chosen with `labels=`. Val and tests
  always use no_interpolation, because that is all they have.
  Every file's date must lie in its split's README period (Sep 1 - Jun 30 of water
  year 2018-19, or 2019-20 for temporal_transfer). This is checked on the zip listing
  before any scene is read; the real listing passes.

Meta per chip: region = basin (Guil / Gyronde, taken from the split folder per the
README table); date = the file name's YYYYMMDD; water_year = "2018-19" or "2019-20"
(the held-out unit of test_temporal, where region stays "Guil"). The file names follow
the 6-day Sentinel-1 repeat for every orbit (12-day for 161) [ZIP], which supports
reading them as acquisition dates. orbit = relative orbit from the file name;
orbit_pass from the README list (descending 139, 66; ascending 88, 161). Also: the
source and label zip members, the chip corner (y0, x0), the scene size, the labelled
share, and zero_pairs (see "No data").

Units: dB, per the README ("2 bands (VV, VH), float32, dB scale") and the Dataverse
description ("VV and VH backscatter in dB"). The values are stored unchanged.
Radiometric calibration is NOT stated anywhere: the paper writes the values as
sigma0 but calls them "backscatter amplitude" from IW SLC with the phase discarded,
estimates references with a Rayleigh law, and describes no calibration step (Sec. 2,
2.3, 4.2). Calibrated sigma0 in dB is mostly negative; uncalibrated amplitude in dB
is typically positive. So no check depends on the sign alone. Checked per scene on
every 8th pixel where both bands are finite and not an exact VV == VH == 0 pair
(skipped below 1000 such pixels):
  (a) linear look: a band is refused only if NONE of its values is negative AND its
      upper tail is longer than 1.5x its lower tail, (p99 - p50) / (p50 - p1) > 1.5.
      For homogeneous single-look speckle this ratio is 0.45 in dB (any offset, 10 or
      20 log10), 1.0 for a symmetric dB histogram, but 1.79 for linear amplitude and
      5.7 for linear intensity (computed from the exponential distribution). Linear
      data are also never negative. A multi-looked linear amplitude (4 looks: 1.24)
      would pass, but the paper describes SLC amplitude with no multilooking.
  (b) band order: median band 1 > median band 2 (co-pol brighter than cross-pol);
      this does not depend on calibration.
  If the file has band descriptions, they must not name the polarisations in the
  wrong order (VH in band 1 or VV in band 2); other descriptions are only printed.
  check_values=False skips (a) and (b); inventory() prints the percentiles first.
  The per-channel standardisation in s1input removes a constant offset and a 10 vs
  20 log10 scale, so calibration matters for the model only if it varies within a
  scene (e.g. with range). The manifest notes record, per split, the range of scene
  medians and how many scenes have no negative value.
Channels: 0 = VV, 1 = VH.

Geometry: radar (range/azimuth) geometry, not map-projected [README, PAPER]. The
README's "5x20m" is the range x azimuth RESOLUTION (paper Sec. 3.2). Pixel spacing
is not published, so pixel_spacing_m = None. The SAR comes from Sentinel-1 IW SLC,
debursted and coregistered (LabSAR), with phase discarded [PAPER].

Chips: non-overlapping 512 x 512 (pre-registered; the paper's validation patches
are also non-overlapping 512 px, Sec. 4.1), on a grid starting at the scene's top-
left corner. Right and bottom edges are padded with NaN (image) and 255 (label).
Chips with no labelled pixel are skipped.

No data:
  image: non-finite -> NaN (-inf is what 10*log10(0) gives); values equal to a
         band's GDAL nodata tag -> NaN; |value| >= 65504 (beyond float16, the prepared
         format; impossible for dB) -> NaN, counted.
  label: the no-data code (above) -> 255; also 255 wherever either SAR channel is
         NaN, so metrics only count pixels that have an input.
  VV == VH == 0.0 exactly: no source says whether this is fill. It does NOT stop the
         run. By default the pixels are kept as 0.0 and counted: per chip in the meta
         column zero_pairs, per split in the printout and the manifest notes. Both
         channels exactly 0.0 survive the float16 cast, so they can still be masked
         later on the prepared arrays ((VV == 0) & (VH == 0) -> NaN, label 255)
         without re-reading the zip. zero_is_nodata=True masks them here instead.
  Member sizes [ZIP, PAPER]: every values member is exactly as large as an
         uncompressed, 1024 x 1024-tiled, pixel-interleaved 2-band float32 GeoTIFF
         of the paper's Table 1 size: 8,388,608 B per tile + 170 B + 8 B per tile
         (Guil D139: 1872 x 8889 -> 2 x 9 tiles -> 150,995,258 B, as listed; all six
         basin/orbit groups match, reproduced by writing such files with rasterio).
         The extra bytes are TIFF edge-tile padding, which GDAL does not return, so
         scenes are read at their true size. A raster padded to multiples of 1024 px
         would give the same byte count, so this is a strong inference, not proof;
         inventory() prints each sampled scene's read shape next to Table 1. The sizes
         fit only if Table 1's Guil D66 and A88 rows are swapped (D66 1934 x 10745,
         A88 1996 x 7833); TABLE1_SHAPES holds them that way.

Not used, and a comparability note: the zip's projection/ folder also holds
masks.npz and altitudes.npz, which the README does not describe (it lists lat/lon
grids only). The paper's metric section (4.4.1) cumulates the confusion matrix over
all test dates and mentions no basin mask. If its scores were restricted to a mask,
our whole-scene scores are not directly comparable to its F1 values.

Unverified until the data are opened (checked or reported at run time):
  - label no-data really stored as 255 (uint8) or -1 (signed), no other codes (asserted);
  - values float32, 2 bands (asserted), VV then VH, and dB rather than linear
    (value checks (a), (b) above, switchable with check_values);
  - radiometric calibration (not stated; not checked; the notes record the medians);
  - label and image of the same name have the same height and width (asserted);
  - fill value: exact VV == VH == 0 pairs are counted, not decided (see above);
  - read size == Table 1 size, i.e. no raster padding (printed by inventory());
  - labels/raw == the README's "no_interpolation" (structural inference, not checked);
  - file-name date == SAR acquisition date (6-day spacing supports it; not stated);
  - which Kalman-smoother lambda the published ks labels use (not stated; the paper
    picks 0.3 in Sec. 5.2.2). Irrelevant to the default.
"""

from __future__ import annotations

import hashlib
import re
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import requests

from .prepared import IGNORE, PreparedWriter

DATASET = "snow"
API = "https://entrepot.recherche.data.gouv.fr/api"
ZIP_NAME = "snowsar_dataset.zip"
TOP = "snowsar_dataset"

# Dataverse API, dataset version 1.0 (read 2026-09-26): file id, bytes, published md5.
FILES = {
    "README.md": {"id": 708620, "size": 5617, "md5": "a30cfe33e9cdfa822017b2af049f1010"},
    ZIP_NAME: {"id": 708968, "size": 43036387541, "md5": "94a41d1ab882a1618e015b2ddbfb0317"},
}

TITLE = ("Replication Data for : Weakly supervised learning for snow cover segmentation in mountainous areas "
         "from Sentinel-1 SAR images using interpolated NDSI time series")          # exactly as the API gives it

SOURCE = {
    "name": TITLE,
    "short_name": "SnowSAR (our own label, taken from the file name snowsar_dataset.zip; not an official name)",
    "doi": "10.57745/IMTSFL",
    "url": "https://doi.org/10.57745/IMTSFL",
    "landing_page": "https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/IMTSFL",
    "repository": "Recherche Data Gouv (Dataverse), dataset version 1.0, released 2025-12-15",
    "licence": "Etalab Open Licence 2.0 (etalab-2.0), https://spdx.org/licenses/etalab-2.0.html",
    "citation": f"Briand, S., Weissgerber, F., Lobry, S., Idier, J. (2025). {TITLE}. Recherche Data Gouv, V1. "
                "https://doi.org/10.57745/IMTSFL",
    "paper": "Briand, S., Weissgerber, F., Lobry, S., Idier, J. (2026). Weakly supervised learning for snow "
             "cover segmentation in mountainous areas from Sentinel-1 SAR images using interpolated NDSI "
             "time series. ISPRS Journal of Photogrammetry and Remote Sensing 239, 670-688. "
             "https://doi.org/10.1016/j.isprsjprs.2026.05.013 (preprint: https://hal.science/hal-05186268)",
    "sar": "Sentinel-1 IW SLC, VV/VH backscatter amplitude in dB (README: dB scale; radiometric calibration not "
           "stated), radar geometry (debursted, coregistered with LabSAR)",
    "labels": "MODIS MOD10A1 C6.1 (https://doi.org/10.5067/MODIS/MOD10A1.061), NDSI > 0.4, same day as SAR, "
              "projected on SAR geometry",
    "files": FILES,
}

REQUIRES = [("rasterio", "rasterio==1.5.1")]      # exact version, pinned 2026-09-29

CLASSES = {0: "no_snow", 1: "snow"}
CHIP = 512

# zip split folder -> (prepared split, basin); README "Data Splits" table.
SPLITS = {
    "train": ("train", "Guil"),
    "validation": ("val", "Guil"),
    "test/no_transfer": ("test", "Guil"),
    "test/spatial_transfer": ("test_spatial", "Gyronde"),
    "test/temporal_transfer": ("test_temporal", "Guil"),
}
# README "Temporal coverage": 20180901-20190630 (training/validation; its table puts no_transfer and
# spatial_transfer in 2018-2019 too) and 20190901-20200630 (temporal transfer test).
WATER_YEAR = {"train": "2018-19", "validation": "2018-19", "test/no_transfer": "2018-19",
              "test/spatial_transfer": "2018-19", "test/temporal_transfer": "2019-20"}
EXPECTED_SCENES = {"train": 88, "validation": 24, "test/no_transfer": 22,
                   "test/spatial_transfer": 111, "test/temporal_transfer": 136}      # [ZIP]
ORBIT_PASS = {139: "descending", 66: "descending", 88: "ascending", 161: "ascending"}  # [README]
# Paper Table 1 (height x width, px) per (basin, orbit), with the Guil D66 and A88 rows SWAPPED: only that way
# do they match the zip member sizes (module docstring, "Member sizes"). Printed by inventory(), never asserted.
TABLE1_SHAPES = {("Guil", 139): (1872, 8889), ("Guil", 66): (1934, 10745), ("Guil", 88): (1996, 7833),
                 ("Gyronde", 139): (2038, 4214), ("Gyronde", 88): (1964, 5107), ("Gyronde", 161): (2147, 6019)}
LABEL_DIRS = {"no_interpolation": "raw", "raw": "raw", "cni": "cni", "ks": "ks"}      # README name -> zip folder
NAME_RE = re.compile(r"^(\d+)_(\d{8})\.tif$")
LINEAR_TAIL = 1.5            # (p99 - p50) / (p50 - p1) above this AND no negative value -> "looks linear"
MIN_CHECK_PX = 1000          # the value checks need at least this many sampled pixels
F16_MAX = 65504.0            # largest finite float16; the prepared images are float16

# ------------------------------------------------------------------ download

def _md5(path: Path, every_s: float = 60.0) -> str:
    h, done, size, t0, last = hashlib.md5(), 0, path.stat().st_size, time.time(), time.time()
    with open(path, "rb") as f:
        while True:
            b = f.read(16 << 20)
            if not b:
                break
            h.update(b)
            done += len(b)
            if time.time() - last > every_s:
                last = time.time()
                print(f"  md5 {path.name}: {done / 1e9:.1f}/{size / 1e9:.1f} GB")
    return h.hexdigest()


def _fetch(url: str, path: Path, size: int, retries: int, chunk: int, timeout: float,
           backoff_s: float, every_s: float = 30.0) -> None:
    """Stream `url` to `path`, resuming with HTTP Range; every retry asks the API for a fresh signed URL."""
    for attempt in range(retries + 1):
        have = path.stat().st_size if path.exists() else 0
        if have == size:
            return
        if have > size:
            raise RuntimeError(f"{path} is {have} bytes, larger than the published {size}; delete it and rerun")
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=timeout, allow_redirects=True) as r:
                r.raise_for_status()
                if have and r.status_code != 206:
                    print(f"  {path.name}: server ignored the range request, restarting from 0")
                    have = 0
                elif have and not r.headers.get("Content-Range", "").startswith(f"bytes {have}-"):
                    raise requests.ConnectionError(f"unexpected Content-Range {r.headers.get('Content-Range')!r}")
                t0, last, start = time.time(), time.time(), have
                with open(path, "ab" if have else "wb") as f:
                    for block in r.iter_content(chunk):
                        f.write(block)
                        have += len(block)
                        if time.time() - last > every_s:
                            last = time.time()
                            rate = (have - start) / max(last - t0, 1e-6) / 1e6
                            print(f"  {path.name}: {have / 1e9:.2f}/{size / 1e9:.2f} GB, {rate:.0f} MB/s")
            if have == size:
                return
            print(f"  {path.name}: stream ended at {have} of {size} bytes")
        except requests.RequestException as e:
            print(f"  {path.name}: attempt {attempt + 1}/{retries + 1} failed: {type(e).__name__}: {e}")
        if attempt < retries:
            time.sleep(min(backoff_s * 2 ** attempt, 120.0))
    raise RuntimeError(f"{path.name}: not complete after {retries + 1} attempts; rerun download() to resume")


def download(dest: str | Path, files: tuple[str, ...] = ("README.md", ZIP_NAME), retries: int = 30,
             chunk_mb: int = 8, timeout: float = 60.0, verify: bool = True, backoff_s: float = 5.0) -> list[Path]:
    """Download the official files from Recherche Data Gouv into `dest` (Colab: /content/...).

    README.md (5.6 kB) and snowsar_dataset.zip (43.0 GB, one file; the repository offers
    no smaller parts, and all five splits are needed). Resumable: a file of the
    published size is skipped, a shorter one is continued with an HTTP Range request.
    Each file is checked against the md5 the Dataverse API publishes. A
    `<file>.md5ok` marker saves re-hashing the 43 GB on later calls.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for name in files:
        if name not in FILES:
            raise ValueError(f"{name!r} is not an official file of this dataset: {sorted(FILES)}")
        info = FILES[name]
        path, marker = dest / name, dest / f"{name}.md5ok"
        url = f"{API}/access/datafile/{info['id']}"
        ok = f"{info['md5']} {info['size']}"
        if path.exists() and path.stat().st_size == info["size"] and marker.exists() \
                and marker.read_text().strip() == ok:
            print(f"{name}: complete and md5-verified earlier, skipped")
            out.append(path)
            continue
        print(f"{name}: {info['size'] / 1e9:.3f} GB from {url}")
        _fetch(url, path, info["size"], retries, chunk_mb << 20, timeout, backoff_s)
        if verify:
            got = _md5(path)
            if got != info["md5"]:
                raise RuntimeError(f"{name}: md5 {got} != published {info['md5']}; the file is corrupt, "
                                   f"delete {path} and rerun download()")
            marker.write_text(ok)
            print(f"{name}: md5 ok")
        out.append(path)
    return out


# ------------------------------------------------------------------ reading

def _zip_path(raw_root: str | Path, zip_name: str) -> Path:
    p = Path(raw_root) / zip_name
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run download() first")
    return p


def _water_year(date: str) -> str | None:
    """"2018-19" for 2018-09-01..2019-06-30 (the README periods run Sep 1 - Jun 30); None for July and August."""
    y, m = int(date[:4]), int(date[5:7])
    if m in (7, 8):
        return None
    start = y if m >= 9 else y - 1
    return f"{start}-{(start + 1) % 100:02d}"


def _scenes(names: list[str], folder: str, label_dir: str) -> list[dict]:
    """Scenes of one split folder, paired values <-> labels/<label_dir> by file name, sorted (date, orbit)."""
    vpre, lpre = f"{TOP}/{folder}/values/", f"{TOP}/{folder}/labels/{label_dir}/"
    vals = {n[len(vpre):]: n for n in names if n.startswith(vpre) and n != vpre and "/" not in n[len(vpre):]}
    labs = {n[len(lpre):]: n for n in names if n.startswith(lpre) and n != lpre and "/" not in n[len(lpre):]}
    bad = [f for f in list(vals) + list(labs) if not NAME_RE.match(f)]
    assert not bad, f"{folder}: file names not <orbit>_<YYYYMMDD>.tif as in the zip listing: {bad[:5]}"
    missing = sorted(set(vals) ^ set(labs))
    assert not missing, (f"{folder}: values/ and labels/{label_dir}/ must hold the same file names "
                         f"(pairing by name); unmatched: {missing[:5]}")
    out = []
    for f in vals:
        orbit, ymd = NAME_RE.match(f).groups()
        orbit = int(orbit)
        assert orbit in ORBIT_PASS, f"{folder}/{f}: orbit {orbit} not in the README's list {sorted(ORBIT_PASS)}"
        date = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
        wy = _water_year(date)
        if folder in WATER_YEAR:
            assert wy == WATER_YEAR[folder], (f"{folder}/{f}: date {date} lies outside this split's README period "
                                              f"(water year {WATER_YEAR[folder]}, Sep 1 - Jun 30)")
        out.append({"file": f, "orbit": orbit, "date": date, "water_year": wy or "",
                    "values": vals[f], "label": labs[f]})
    return sorted(out, key=lambda s: (s["date"], s["orbit"]))


def _read_tif(zf: zipfile.ZipFile, member: str) -> tuple[np.ndarray, dict]:
    """Read one GeoTIFF straight from the zip (member bytes -> GDAL memory file); CRC checked by zipfile."""
    from rasterio.errors import NotGeoreferencedWarning
    from rasterio.io import MemoryFile

    data = zf.read(member)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with MemoryFile(data) as mf, mf.open() as src:
            arr = src.read()
            info = {"count": src.count, "dtypes": tuple(src.dtypes), "height": src.height, "width": src.width,
                    "nodata": tuple(src.nodatavals), "descriptions": tuple(src.descriptions),
                    "compress": str(src.compression.value) if src.compression else None,
                    "crs": str(src.crs) if src.crs else None}
    del data
    return arr, info


def _check_header(info: dict, member: str) -> None:
    """README: 2 bands (VV, VH), float32. Descriptions, if any, must not name the polarisations the wrong way round."""
    assert info["count"] == 2, f"{member}: {info['count']} bands; README: 2 bands (VV, VH)"
    assert all(d == "float32" for d in info["dtypes"]), f"{member}: dtypes {info['dtypes']}; README: float32"
    d1, d2 = ((d or "").upper() for d in info["descriptions"])
    swapped = ("VH" in d1 and "VV" not in d1) or ("VV" in d2 and "VH" not in d2)
    assert not swapped, (f"{member}: band descriptions {info['descriptions']} name the polarisations in the wrong "
                         f"order; README: band 1 VV, band 2 VH")


def _sar_stats(x: np.ndarray, zero: np.ndarray) -> dict:
    """Per band on every 8th pixel (both bands finite, not an exact zero pair): p1/p50/p99, negative share, tail ratio."""
    s = x[:, ::8, ::8]
    ok = np.isfinite(s).all(0) & ~zero[::8, ::8]
    st = {"n": int(ok.sum())}
    for b, name in enumerate(("VV", "VH")):
        v = s[b][ok]
        if not v.size:
            st[name] = None
            continue
        p1, p50, p99 = (float(q) for q in np.percentile(v, [1, 50, 99]))
        st[name] = {"neg_share": float((v < 0).mean()), "p1": p1, "p50": p50, "p99": p99,
                    "tail_ratio": (p99 - p50) / (p50 - p1) if p50 > p1 else float("nan")}
    return st


def _check_values(st: dict, member: str) -> None:
    """Value checks (a) linear look and (b) band order of the module docstring ("Units"); both calibration-free."""
    if st["n"] < MIN_CHECK_PX:
        return
    for name in ("VV", "VH"):
        b = st[name]
        assert not (b["neg_share"] == 0 and b["tail_ratio"] > LINEAR_TAIL), (
            f"{member}: no {name} value is negative and its upper tail is {b['tail_ratio']:.2f}x its lower tail "
            f"(p1/p50/p99 = {b['p1']:.4g}/{b['p50']:.4g}/{b['p99']:.4g}). The README says dB. dB of speckled SAR, "
            f"calibrated or not, has the longer tail at the LOW end; linear power or amplitude is never negative "
            f"and has it at the HIGH end, so these look linear, not dB. If you have checked that they are dB, "
            f"rerun with check_values=False.")
    mvv, mvh = st["VV"]["p50"], st["VH"]["p50"]
    assert mvv > mvh, (
        f"{member}: median band 1 {mvv:.4g} <= median band 2 {mvh:.4g}. The README band order is (VV, VH) and "
        f"co-pol backscatter is normally higher than cross-pol, so the bands look swapped. Values already "
        f"standardised per band would also fail here, but the README ships the normalisation statistics "
        f"separately (stats/), so they should not be. If you have checked, rerun with check_values=False.")


def _sar(arr: np.ndarray, info: dict, member: str, zero_is_nodata: bool | None = False,
         check_values: bool = True) -> tuple[np.ndarray, dict]:
    """(2, H, W) float32 with NaN for no data, and statistics (key "zero" = H x W mask of exact VV == VH == 0).

    Header checks always run; the value checks only with check_values. zero_is_nodata=True turns exact zero
    pairs into NaN; False or None keeps them. Nothing here stops on zero pairs.
    """
    _check_header(info, member)
    x = arr.astype(np.float32, copy=False)
    n_inf = n_big = 0
    for b, nd in enumerate(info["nodata"]):
        xb = x[b]
        if nd is not None and not np.isnan(nd):
            xb[xb == np.float32(nd)] = np.nan
        inf = np.isinf(xb)
        big = np.abs(xb) >= F16_MAX                     # includes +-inf; NaN compares False
        n_inf += int(inf.sum())
        n_big += int(big.sum()) - int(inf.sum())
        xb[big] = np.nan
        del inf, big
    zero = (x[0] == 0) & (x[1] == 0)
    n_zero = int(zero.sum())
    box = None
    if n_zero:
        r, c = np.flatnonzero(zero.any(1)), np.flatnonzero(zero.any(0))
        box = (int(r[0]), int(r[-1]), int(c[0]), int(c[-1]))
        if zero_is_nodata:
            x[:, zero] = np.nan
    st = {"inf": n_inf, "out_of_f16": n_big, "zero_pairs": n_zero, "zero_rows_cols": box, "zero": zero,
          **_sar_stats(x, zero)}
    if check_values:
        _check_values(st, member)
    return x, st


def _label(arr: np.ndarray, info: dict, member: str) -> tuple[np.ndarray, dict]:
    """(H, W) uint8 from band 1: 0 no snow, 1 snow, 255 no data; raises on any code outside the README's {0, 1, -1}.

    The zip stores float32 with 2 or 3 bands (module docstring, "Classes"): band 1 is the label; where band 2
    is not exactly 1 the pixel becomes 255 and is counted ("band2_not_1"); further bands are not used.
    """
    assert info["count"] >= 1, f"{member}: no band"
    dt = np.dtype(info["dtypes"][0])
    assert dt.kind in "iuf", f"{member}: dtype {dt}; README: uint8"
    a = arr[0]
    nodata = {255} if dt == np.uint8 else {-1}
    tag, ignored = info["nodata"][0], None
    if tag is not None and not np.isnan(tag):
        if float(tag) in (0.0, 1.0):
            ignored = float(tag)                        # collides with a README class: the README wins
        else:
            nodata.add(int(tag))
    n_nan = 0
    if dt == np.uint8:
        cnt = np.bincount(a.ravel(), minlength=256)
        found = {int(c): int(cnt[c]) for c in np.flatnonzero(cnt)}
    else:
        v = a
        if dt.kind == "f":
            fin = ~np.isnan(a)
            n_nan = int(a.size - fin.sum())
            v = a[fin]
        codes, counts = np.unique(v, return_counts=True)
        frac = codes[codes != np.round(codes)]
        assert not frac.size, (f"{member}: label values {frac[:5].tolist()} are not whole numbers; README codes "
                               f"0 = no snow, 1 = snow, -1 = no data")
        found = {int(c): int(n) for c, n in zip(codes, counts)}
    bad = sorted(set(found) - {0, 1} - nodata)
    assert not bad, (f"{member}: label values {bad} are outside the README's codes 0 = no snow, 1 = snow, "
                     f"-1 = no data (stored as {sorted(nodata)} in this {dt} file); all codes found: {found}")
    out = np.full(a.shape, IGNORE, np.uint8)
    out[a == 0] = 0
    out[a == 1] = 1
    n_b2 = 0
    if info["count"] >= 2:                              # undocumented band 2 (1.0 in every file looked at)
        off = arr[1] != 1                               # NaN counts as "not 1"
        n_b2 = int(off.sum())
        out[off] = IGNORE
    return out, {"codes": found, "nodata_codes": sorted(nodata), "ignored_tag": ignored, "nan": n_nan,
                 "bands": int(info["count"]), "band2_not_1": n_b2}


def _chips(x: np.ndarray, lab: np.ndarray, chip: int):
    """Non-overlapping chips from the top-left corner; edge chips padded with NaN / 255; skip unlabelled ones."""
    h, w = lab.shape
    for y0 in range(0, h, chip):
        for x0 in range(0, w, chip):
            yh, xw = min(chip, h - y0), min(chip, w - x0)
            img = np.full((2, chip, chip), np.nan, np.float32)
            img[:, :yh, :xw] = x[:, y0:y0 + yh, x0:x0 + xw]
            lc = np.full((chip, chip), IGNORE, np.uint8)
            lc[:yh, :xw] = lab[y0:y0 + yh, x0:x0 + xw]
            lc[np.isnan(img).any(0)] = IGNORE
            n = int((lc != IGNORE).sum())
            if n:
                yield y0, x0, img, lc, n / chip / chip


def _spread(scenes: list[dict], n: int) -> list[dict]:
    """n scenes evenly spread over the date-sorted list (first and last included), deterministic."""
    if n >= len(scenes):
        return list(scenes)
    return [scenes[i] for i in sorted({int(round(k)) for k in np.linspace(0, len(scenes) - 1, n)})]


def _label_dir(labels: str) -> str:
    if labels not in LABEL_DIRS:
        raise ValueError(f"labels {labels!r} not in {sorted(LABEL_DIRS)}")
    return LABEL_DIRS[labels]


def _span(v: list[float]) -> str:
    return f"{min(v):.1f}..{max(v):.1f}" if v else "-"


# ------------------------------------------------------------------ inventory

def inventory(raw_root: str | Path, sample: int = 2, zip_name: str = ZIP_NAME,
              expected_scenes: dict | None = EXPECTED_SCENES, check_values: bool = True) -> dict:
    """Print what the zip holds and check the documented format on `sample` scenes per split.

    Reads the zip in place. Checks on the listing: split folders, scene counts (the zip
    listing of version 1.0), values <-> labels pairing for raw (all splits) and cni/ks
    (train), file-name pattern, orbits and README periods. On the sampled scenes (first
    and last date, then evenly spread) it first PRINTS the header, the read shape next to
    paper Table 1, value percentiles, negative share, tail ratio, no-data and zero-pair
    counts and label codes, and only then asserts band count, dtype, label codes,
    matching shapes and (with check_values) the value checks. A sample cannot rule out a
    problem in another scene: convert() re-checks every scene (exact VV == VH == 0 pixels
    never stop it; they are counted).
    """
    zpath = _zip_path(raw_root, zip_name)
    readme = Path(raw_root) / "README.md"
    print(f"{zpath}: {zpath.stat().st_size:,} bytes (published {FILES[ZIP_NAME]['size']:,}); "
          f"README.md {'present' if readme.exists() else 'missing'}; md5 "
          f"{'verified by download()' if (Path(raw_root) / f'{zip_name}.md5ok').exists() else 'NOT verified here'}")
    report = {"scenes": {}, "sampled": [], "label_codes": {}, "zero_pairs": 0}
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        tops = sorted({n[len(TOP) + 1:].split("/")[0] for n in names if n.startswith(TOP + "/") and n != TOP + "/"})
        print(f"{len(names)} zip entries; top-level folders: {tops}")
        for folder, (split, basin) in SPLITS.items():
            variants = ("raw", "cni", "ks") if folder == "train" else ("raw",)
            scenes = {v: _scenes(names, folder, v) for v in variants}
            sc = scenes["raw"]
            orbits = {o: sum(s["orbit"] == o for s in sc) for o in sorted({s["orbit"] for s in sc})}
            print(f"\n{folder} -> {split} ({basin}): {len(sc)} scenes, orbits {orbits}, "
                  f"dates {sc[0]['date'] if sc else '-'}..{sc[-1]['date'] if sc else '-'}, labels {list(variants)}")
            if expected_scenes is not None:
                assert len(sc) == expected_scenes[folder], \
                    f"{folder}: {len(sc)} scenes, the published zip lists {expected_scenes[folder]}"
            report["scenes"][split] = len(sc)
            for s in _spread(sc, sample):
                row, vst = {"split": split, "file": s["file"]}, None
                try:
                    arr, vi = _read_tif(zf, s["values"])
                    la, li = _read_tif(zf, s["label"])
                    shape, t1 = (vi["height"], vi["width"]), TABLE1_SHAPES.get((basin, s["orbit"]))
                    row.update(shape=shape, label_shape=(li["height"], li["width"]), table1_shape=t1,
                               shape_vs_table1="-" if t1 is None else ("same" if shape == t1 else "differs"),
                               sar_bands=vi["count"], sar_dtypes=vi["dtypes"], sar_nodata_tag=vi["nodata"],
                               descriptions=vi["descriptions"], crs=vi["crs"], sar_compress=vi["compress"],
                               label_dtype=li["dtypes"][0], label_nodata_tag=li["nodata"][0],
                               label_compress=li["compress"])
                    x, vst = _sar(arr, vi, s["values"], zero_is_nodata=False, check_values=False)
                    del arr
                    vst.pop("zero")
                    fin = np.isfinite(x)
                    row.update(nan_share=round(float(np.isnan(x).mean()), 4), inf_pixels=vst["inf"],
                               out_of_f16=vst["out_of_f16"], zero_pairs=vst["zero_pairs"],
                               zero_rows_cols=vst["zero_rows_cols"], checked_px=vst["n"],
                               min_max={b: [round(float(np.nanmin(x[k])), 2), round(float(np.nanmax(x[k])), 2)]
                                        if fin[k].any() else None for k, b in enumerate(("VV", "VH"))})
                    for b in ("VV", "VH"):
                        v = vst[b]
                        row[f"{b}_p1_p50_p99"] = None if v is None else [round(v[k], 2) for k in ("p1", "p50", "p99")]
                        row[f"{b}_negative_share"] = None if v is None else round(v["neg_share"], 4)
                        row[f"{b}_tail_ratio"] = None if v is None else round(v["tail_ratio"], 3)
                    del x, fin
                    lab, lst = _label(la, li, s["label"])
                    del la
                    row.update(label_codes=lst["codes"], label_tag_ignored=lst["ignored_tag"],
                               label_bands=lst["bands"], label_band2_not_1=lst["band2_not_1"],
                               label_nan=lst["nan"])
                    assert lab.shape == shape, (f"{s['file']}: label {lab.shape} != image {shape}; "
                                                f"assumed both on one SAR grid")
                    del lab
                finally:
                    print("  ", row)
                if check_values:
                    _check_values(vst, s["values"])
                report["sampled"].append(row)
                report["zero_pairs"] += row["zero_pairs"]
                for c, n in row["label_codes"].items():
                    report["label_codes"][c] = report["label_codes"].get(c, 0) + n
    print(f"\nlabel codes in the sample (raw values): {report['label_codes']}")
    if report["zero_pairs"]:
        print(f"{report['zero_pairs']} sampled pixels have VV == VH == 0.0 exactly ('zero_rows_cols' gives their "
              f"row/column extent per scene). convert() keeps them as 0.0 and counts them per chip (meta "
              f"'zero_pairs'); zero_is_nodata=True masks them instead")
    else:
        print("no VV == VH == 0.0 pixels in the sample (convert() counts them in every scene anyway)")
    if report["sampled"]:
        same = sum(r["shape_vs_table1"] == "same" for r in report["sampled"])
        print(f"read shape == paper Table 1 (Guil D66/A88 rows swapped) in {same}/{len(report['sampled'])} sampled "
              f"scenes; 'differs' would mean the rasters themselves are padded or cropped")
    return report


# ------------------------------------------------------------------ convert

def convert(raw_root: str | Path, out_root: str | Path, chip_size: int = CHIP, labels: str = "no_interpolation",
            splits: tuple[str, ...] | None = None, max_scenes_per_split: int | None = None,
            zero_is_nodata: bool | None = False, zip_name: str = ZIP_NAME, check_values: bool = True) -> dict:
    """Write <out_root>/snow/ in the prepared format, reading the zip in place, one scene at a time.

    labels: training labels, "no_interpolation" (default, pre-registered; zip folder
        labels/raw), "cni" or "ks". Val and test splits always use no_interpolation.
    splits: subset of ("train", "val", "test", "test_spatial", "test_temporal").
    max_scenes_per_split: a dry run (or a deterministic subset) on N scenes per split, evenly
        spread over the date-sorted list, so that a subset does not cover only the early season.
    zero_is_nodata: False (default; None means the same) keeps exact VV == VH == 0.0 pixels as
        0.0 and counts them per chip (meta zero_pairs) and per split (printout, manifest notes);
        True turns them into NaN / label 255. The run never stops on them.
    check_values: False skips the dB-look and band-order checks (module docstring, "Units").
        The other format checks (band count, dtype, label codes, shapes) always run and stop
        the run, naming the scene.
    Scenes go in (date, orbit) order and chips row-major, so reruns give identical arrays.

    Size: at most 24,508 chips (the zip's scene counts x ceil(H/512) x ceil(W/512) with
    TABLE1_SHAPES, before unlabelled chips are skipped) x 1.31 MB (float16 image + uint8
    label) = about 32 GB. Peak RAM about 1.6 GB: up to ~0.55 GB while one Guil D66 member is
    inflated and read, plus the writer's 512-chip buffer (0.67 GB) and its np.stack copy
    when it writes a part file (0.67 GB).
    """
    zpath = _zip_path(raw_root, zip_name)
    train_dir = _label_dir(labels)
    known = {v[0] for v in SPLITS.values()}
    if splits is not None and set(splits) - known:
        raise ValueError(f"unknown splits {sorted(set(splits) - known)}; choose from {sorted(known)}")
    notes = (
        f"SnowSAR (doi:10.57745/IMTSFL; 'SnowSAR' is our label) in radar geometry, resolution ~5 x 20 m (range x "
        f"azimuth), pixel spacing not published. Units: dB as the README states; radiometric calibration is not "
        f"stated (the paper writes sigma0 but calls the values backscatter amplitude and describes no calibration "
        f"step), so values may be offset from calibrated sigma0 in dB; s1input's per-channel standardisation "
        f"removes a constant offset or scale. Training labels: {labels} (zip labels/{train_dir}); val/test labels: "
        f"no_interpolation (zip labels/raw), MODIS MOD10A1 NDSI > 0.4, cloud gaps -> 255. Label files: the README "
        f"says 1 band uint8; the zip has float32 with 2 bands (3 for ks): band 1 = the README codes (used), band 2 "
        f"undocumented (1.0 in every file looked at; pixels where it is not 1 -> 255, counted below), band 3 not "
        f"used. Splits: train, "
        f"val=validation, test=no_transfer (Guil 2018-19), test_spatial=spatial_transfer (Gyronde 2018-19), "
        f"test_temporal=temporal_transfer (Guil 2019-20; meta water_year). Non-overlapping {chip_size} px chips "
        f"from the top-left, edges padded NaN/255, chips without labelled pixels skipped; label 255 wherever a SAR "
        f"channel is NaN. Exact VV == VH == 0.0 pixels (whether they are fill is not documented): "
        + ("set to NaN / label 255 (zero_is_nodata=True)." if zero_is_nodata else
           "kept as 0.0 and counted per chip in meta zero_pairs; to treat them as fill later, set both channels "
           "to NaN and the label to 255 where VV == 0 and VH == 0.")
        + f" Value checks (dB look, band order): {'on' if check_values else 'OFF (check_values=False)'}."
        + " Not used: projection/masks.npz and altitudes.npz (not described in the README). The paper does not "
          "say whether its MODIS-label scores were restricted to a basin mask, so whole-scene scores here are "
          "not directly comparable to its F1."
        + (f" DRY RUN / SUBSET: {max_scenes_per_split} scenes per split, evenly spread over the dates."
           if max_scenes_per_split else ""))
    w = PreparedWriter(out_root, DATASET, chip_size, CLASSES, SOURCE, channels=("VV", "VH"), units="dB",
                       pixel_spacing_m=None, notes=notes)
    flush = getattr(w, "flush", None) or w._flush       # public flush(split) once prepared.py has one (requested)
    run = []
    t0 = time.time()
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        for folder, (split, basin) in SPLITS.items():
            if splits is not None and split not in splits:
                continue
            ldir = train_dir if folder == "train" else "raw"
            scenes = _scenes(names, folder, ldir)
            if max_scenes_per_split:
                scenes = _spread(scenes, max_scenes_per_split)
            tot = dict(chips=0, zero=0, zero_scenes=0, zero_in_chips=0, f16=0, tags=0, no_neg_vv=0, no_neg_vh=0,
                       b2=0, b2_scenes=0, bands=set())
            med = {"VV": [], "VH": []}
            for i, s in enumerate(scenes):
                t1 = time.time()
                arr, vi = _read_tif(zf, s["values"])
                x, st = _sar(arr, vi, s["values"], zero_is_nodata, check_values)
                del arr
                zero = st.pop("zero")
                la, li = _read_tif(zf, s["label"])
                lab, lst = _label(la, li, s["label"])
                del la
                assert lab.shape == x.shape[1:], (f"{s['file']}: label {lab.shape} != image {x.shape[1:]}; "
                                                  f"assumed both on one SAR grid")
                h, wd = lab.shape
                n_all = -(-h // chip_size) * -(-wd // chip_size)
                kept = zin = 0
                for y0, x0, img, lc, share in _chips(x, lab, chip_size):
                    nz = int(zero[y0:y0 + chip_size, x0:x0 + chip_size].sum()) if st["zero_pairs"] else 0
                    w.add(split, img, lc, region=basin, date=s["date"], water_year=s["water_year"],
                          orbit=s["orbit"], orbit_pass=ORBIT_PASS[s["orbit"]], source=s["values"],
                          label_source=s["label"], y0=y0, x0=x0, scene_h=h, scene_w=wd,
                          labelled_share=round(share, 4), zero_pairs=nz)
                    kept += 1
                    zin += nz
                del x, lab, zero
                tot["chips"] += kept
                tot["zero"] += st["zero_pairs"]
                tot["zero_scenes"] += bool(st["zero_pairs"])
                tot["zero_in_chips"] += zin
                tot["f16"] += st["out_of_f16"]
                tot["tags"] += lst["ignored_tag"] is not None
                tot["b2"] += lst["band2_not_1"]
                tot["b2_scenes"] += bool(lst["band2_not_1"])
                tot["bands"].add(lst["bands"])
                extra = ""
                if st["VV"] is not None:
                    med["VV"].append(st["VV"]["p50"])
                    med["VH"].append(st["VH"]["p50"])
                    tot["no_neg_vv"] += st["VV"]["neg_share"] == 0
                    tot["no_neg_vh"] += st["VH"]["neg_share"] == 0
                    extra += f", medians VV {st['VV']['p50']:.1f} VH {st['VH']['p50']:.1f}"
                if st["zero_pairs"]:
                    extra += f", {st['zero_pairs']} px VV == VH == 0"
                if st["out_of_f16"]:
                    extra += f", {st['out_of_f16']} values beyond float16 -> NaN"
                if lst["ignored_tag"] is not None:
                    extra += f", label nodata tag {lst['ignored_tag']:g} ignored (README codes win)"
                if lst["band2_not_1"]:
                    extra += f", {lst['band2_not_1']} label px with band 2 != 1 -> 255"
                print(f"[{split} {i + 1}/{len(scenes)}] {s['file']} {h}x{wd}: {kept}/{n_all} chips kept{extra}, "
                      f"{time.time() - t1:.1f} s (total {(time.time() - t0) / 60:.1f} min)")
            flush(split)                        # write this split's buffered chips before the next split
            n = len(scenes)
            line = (f"{split}: {tot['chips']} chips from {n} scenes; scene medians VV {_span(med['VV'])}, "
                    f"VH {_span(med['VH'])}; scenes without a negative value: VV {tot['no_neg_vv']}/{n}, "
                    f"VH {tot['no_neg_vh']}/{n}; {tot['zero']} VV == VH == 0.0 pixels in {tot['zero_scenes']} "
                    f"scenes ({tot['zero_in_chips']} in kept chips, "
                    f"{'set to NaN' if zero_is_nodata else 'kept as 0.0'}); {tot['f16']} values beyond float16 "
                    f"-> NaN; {tot['tags']} label files with an ignored nodata tag 0/1; label files with "
                    f"{'/'.join(str(b) for b in sorted(tot['bands'])) or '-'} bands; {tot['b2']} label pixels in "
                    f"{tot['b2_scenes']} scenes had band 2 != 1 (set to 255).")
            print(line)
            run.append(line)
    w.notes = notes + " Run: " + " ".join(run)
    man = w.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min: {man['splits']}")
    return man
