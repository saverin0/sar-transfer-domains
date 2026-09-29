"""Alpine glaciers: the GlaViTU glacier-mapping benchmark -> prepared format.

Dataset
    Maslov, Persello, Schellenberger, Stein, "Globally scalable glacier mapping by
    deep learning matches expert delineation accuracy", Nat. Commun. 16, 43 (2025),
    doi:10.1038/s41467-024-54956-x. Data on the NIRD Research Data Archive,
    doi:10.11582/2024.00168, CC-BY-4.0, version 1 released 2024-10-30, 14 files,
    487,045,529,767 bytes.

Sources, all opened 2026-09-26. Every statement below is tagged with one of them.
    [TOC]   NIRD table_of_contents_10.11582_2024.00168.csv and landing page: file
            names, bytes, md5, licence.
    [PAPER] the Nature Communications article, Methods and Table 2.
    [CODE]  github.com/konstantin-a-maslov/scalable_glacier_mapping, read for the
            FORMAT only (nothing imported or vendored): dataloaders (samplers,
            plugins, filters), predict.py, evaluate.py, compile_features.py,
            utils/dataset_stats.pickle, dataloaders/global_stats.pickle (the two
            pickles were loaded with an unpickler that only allows numpy arrays).
    [H5]    HDF5 metadata of the three main files, read with HTTP range requests:
            tile names, group attributes, dataset dtype/shape/layout. No pixel
            values were read.

File layout [TOC, H5]
    20230905_{train,val,test}_global_ps384.hdf5 (243.1 / 81.4 / 86.0 GB). There
    is one file per split and no file per region. The root holds one group per
    tile, named "<prefix>-<row>-<col>" (e.g. "ALP-10-5"). Group attributes are
    region, subregion, tile_name, epsg (4326), height and width (stored, padded),
    original_height, original_width, padding_height, padding_width and
    xmin/xmax/ymin/ymax (lon/lat bounds). No acquisition date is stored.
    Every dataset is (height, width, C) float64, contiguous and uncompressed,
    with fill value 0.0: co_pol_sar C=2, cross_pol_sar C=2, outlines C=2, dem
    C=2, in_sar C=2, optical C=6, thermal C=1 where present, and a uint8
    bright_dark_outlines C=3. The original tile sits at [padding_height :
    padding_height + original_height, padding_width : ...], and padding ==
    (stored - original) // 2 for all 556 tiles whose attributes were read. This
    is the padding of compile_features.py [CODE], with stored sizes as multiples
    of 384.
    ALP: 177 / 59 / 60 tiles in train / val / test [H5, equal to PAPER Table 2].
    All are stored 1152 x 1152, originals are 780-906 x 846-985 px, and every ALP
    tile has co_pol_sar and cross_pol_sar.

SAR: what is known, what is assumed, and how it is CHECKED at run time
    [PAPER] "sigma0-calibrated amplitude images acquired by ENVISAT and
    Sentinel-1 from both ascending and descending orbital paths", and the co-pol
    input has "two channels of stacked backscattering images from ascending and
    descending orbital paths". Per Table 2, both co- and cross-pol exist only in
    ALP 2015, NZL 2019, SCA2 2018 and SVAL1-3 2020. SAN1 and SAN2 (2016) have
    co-pol only. The paper names only those two SAR sensors, and ENVISAT ended in
    2012, so every subregion here is Sentinel-1 (an inference; the paper gives no
    sensor per subregion). Polarisation names (VV/VH or HH/HV) are not stated,
    hence the channel names "co-pol" / "cross-pol".
    [CODE] The training loaders feed the HDF5 arrays to the network unchanged.
    compile_features.py prepares NEW scenes as
    (log10(sigma0 + 1e-6) - min) / (max - min), with min/max from
    utils/dataset_stats.pickle (co_pol_sar min (-6, -6), max (1.7576, 15.6351);
    cross_pol_sar min (-6, -6), max (1.3886, 15.9050)). The README's "sigma0 in
    linear scale" describes those input GeoTIFFs.
    ASSUMPTION A (sar_scale="minmax_log10", the default): the HDF5 stores the
        same normalised log10 values, so dB = 10*log10(10**(x*(max-min)+min) - 1e-6).
        "linear" (dB = 10*log10(x)) and "log10" (dB = 10*log10(10**x - 1e-6)) are
        the alternatives.
    ASSUMPTION B (orbit_bands=("asc", "desc")): band 0 is ascending and band 1
        is descending. dataloaders/global_stats.pickle gives the same maxima under
        the names co_pol_intensity_asc (1.7576) and _desc (15.6351), which match
        dataset_stats bands 0 and 1. Neither the paper nor the README states the
        order.
    Neither assumption can be read from metadata, so both are checked on the first
    `check_tiles` training tiles before anything is written:
      * Scale: for "minmax_log10", >= 99.9 % of the values must lie in [0, 1]; for
        "linear", >= 99.9 % must be >= 0. The implied dB must also look like C-band
        backscatter: median co-pol in [-25, 0] dB and median co minus cross in
        [+2, +15] dB. These are our own plausibility bounds, not the source's. A
        wrong guess misses them by a wide margin: min-max values read as linear
        give co ~ cross, and linear values read as min-max give co ~ -50 dB.
      * Orbit order: Sentinel-1 looks right, so an ascending pass looks east and
        slopes facing west (elevation rising eastward) come out brighter;
        descending is the opposite. The Spearman correlation of co-pol dB with the
        eastward elevation gradient (dem channel 0 = elevation, per
        compile_features.py and "stacked elevation and slope" [PAPER]) must be
        > +0.05 for "asc" and < -0.05 for "desc". The wrong sign raises; |r| <
        0.05 only warns. This needs "dem", which download() fetches for the first
        `check_tiles` training tiles only. Row 0 = north and column 0 = west
        follows the coordinate code in dataloaders/plugins/plugins.py [CODE].

Labels [CODE, H5, PAPER]
    "outlines" is (H, W, 2) float64 one-hot. predict.py keeps channel -1 as the
    ground truth, and evaluate.py scores class 1 = glacier (the IoU of pred == 1).
    Mapping: channel 1 == 1 -> 1 "glacier", channel 0 == 1 -> 0 "non-glacier".
    Anything else inside the tile ((0,0) or (1,1)), and everything outside the
    original extent (padding or beyond the array), -> 255 ignore. Debris-covered
    ice is part of "glacier" [PAPER]. Checked at run time: outline values are
    only 0/1, and the glacier share of labelled pixels is < 0.5. The converted
    glacier area is printed next to Table 2 (ALP 1317.40 km2), using pixel areas
    from each tile's lon/lat box. Alps labels come from Paul et al. 2020, ESSD
    12:1805, a Sentinel-2 inventory [PAPER ref. 40]: optical, so independent of
    the SAR.
    Difference to the official metric [CODE], stated in the manifest notes with
    the counts per split: training uses FocalLoss only (configs/training.py), and
    losses.FocalLoss sums y_true * log(y_pred) over the two channels, so a (0,0)
    pixel gives zero loss, i.e. training ignores it, as here. evaluate.py,
    however, scores true = outlines[..., -1], so there a (0,0) pixel counts as
    non-glacier (and a (1,1) pixel, if any, as glacier). predict.py also crops
    [pad:-pad] on both sides, which keeps one padding row / column where stored -
    original is odd (102 / 100 of the 296 ALP tiles in H / W [H5]); here those
    pixels are outside the original extent -> 255. An IoU computed here is
    therefore not on exactly the paper's pixel set. inventory() prints the (0,0)
    share of its sample tiles.

Splits [PAPER, TOC, H5]
    This is the official random ~60/20/20 tile split, one file per split: train ->
    "train", val -> "val", test -> "test". The tile counts per subregion must
    equal Table 2 (checked). The split is random by tile, not a region hold-out,
    so neighbouring tiles of the same subregion and year appear in both train and
    test. The "independent acquisition test dataset" (*_features.pickle +
    *_reference.tif) is not used: the paper lists only optical data and DEMs for
    it (Landsat 5 / Sentinel-2 with SRTM / AW3D30 / Cop30DEM), so it has no SAR
    to test on.

Chips
    Original tiles are not square, their size varies, and it is not a multiple of
    16 (ALP: 780-906 x 846-985 px). chip_size=None picks the smallest multiple of
    16 that holds the largest original side among the selected tiles (ALP: 985 ->
    992, one chip per tile). Each tile is covered by ceil(h/c) x ceil(w/c)
    windows, centred on its original extent. Pixels outside that extent are NaN
    in the image and 255 in the label.

No data, orbit rule, units
    0.0 is the fill and padding value. Under assumption A it maps to
    log10(0 + 1e-6) = min, i.e. sigma0 = 0, so a raw 0 -> NaN, and sigma0 <= 0
    after the inverse -> NaN. A valid pixel is one with a finite dB value.
    Orbit rule, orbit_rule="coverage" (the default; set 2026-09-26 after review,
    still before any real data was opened): per tile, use the orbit whose co-pol
    band has more valid pixels inside the original extent, and `orbit` ("asc")
    on a tie. It has no threshold. orbit_rule="any" is the first rule: `orbit` if
    its co-pol band has any valid pixel in the tile, else the other orbit; it can
    keep an almost empty ascending image over a complete descending one. Co- and
    cross-pol always come from the same orbit. The meta column
    other_orbit_valid_frac is the other orbit's valid co-pol share of each chip,
    and the chips where it beats the chosen orbit are counted in the notes. A
    tile with no valid co-pol pixel in either orbit is dropped, and so is a window
    with no valid co-pol pixel (keep_empty=False); both are counted.
    Extreme values: the repo stats allow log10 sigma0 up to 15.6 (co-pol) and
    15.9 (cross-pol) in the descending bands, i.e. about +156 / +159 dB, and
    stored values just above 0 give sigma0 near 0, i.e. below -100 dB. They are
    kept by default. convert() records the finite pixel count and dB min/max per
    split and channel in the manifest notes. db_range=(lo, hi) also counts the
    pixels below lo and above hi, and outside_range="nan" sets them to NaN. The
    range is the user's choice; no bound is built in.
    Output units: dB. Pixel spacing: 10 m per the paper ("resampled to
    10 m"). The tiles are epsg 4326, though, and box / original size gives
    0.99-1.16e-4 degree per pixel over all 296 ALP tiles [H5] (about 11-13 m N-S
    and 7.6-8.9 m E-W). So "10 m" is nominal. The meta records degrees per pixel
    and an approximate chip-centre lat/lon, linear within the tile's box.

Memory
    PreparedWriter keeps a split's chips in RAM until it holds 512 of them, which
    at 992 px is ~2.5 GB. convert() therefore writes a split's buffered chips to
    disk whenever they pass flush_mb (default 256 MB) and after each split,
    through the writer's flush. Peak RAM is then about 2 x flush_mb (the buffer plus the
    np.stack inside the flush) plus one tile's working set, whatever the
    subregions or chip size.

Unverified until real data is opened (all checked at run time except the last three):
    assumption A (SAR scale), assumption B (band order), dem channel 0 = elevation
    (used only by the orbit check), outlines channel 1 = glacier, VV/VH naming,
    Sentinel-1 per subregion (inferred from the year), and whether the date-free
    tiles really have one SAR date each (the paper: SAR within up to a month of
    the optical date).
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import threading
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

from .prepared import IGNORE, PreparedWriter

DATASET = "glavitu_alp"
NIRD_ID = "243c5285-2bf0-438f-bf2c-74b7bdba182e"
BASE = f"https://data.archive.sigma2.no/dataset/{NIRD_ID}/download/"

SOURCE = {
    "name": "Globally Scalable Glacier Mapping by Deep Learning Matches Expert Delineation Accuracy "
            "(GlaViTU tile-based dataset)",
    "doi": "10.11582/2024.00168",
    "landing_page": "https://archive.sigma2.no/dataset/800DB4CE-FE69-4925-9B90-AC78C6408945",
    "download_base": BASE,
    "table_of_contents": BASE + "table_of_contents_10.11582_2024.00168.csv",
    "licence": "CC-BY-4.0",
    "version": "1 (released 2024-10-30)",
    "citation": "University of Twente, Maslov, K. (2024). Globally Scalable Glacier Mapping by Deep Learning "
                "Matches Expert Delineation Accuracy [Data set]. NIRD RDA. https://doi.org/10.11582/2024.00168",
    "paper": "Maslov, K.A., Persello, C., Schellenberger, T., Stein, A.: Globally scalable glacier mapping by deep "
             "learning matches expert delineation accuracy. Nat Commun 16, 43 (2025). "
             "https://doi.org/10.1038/s41467-024-54956-x",
    "format_reference": "https://github.com/konstantin-a-maslov/scalable_glacier_mapping (read only, not used)",
    "labels_alps": "Paul et al. 2020, Earth Syst. Sci. Data 12, 1805-1821 (Sentinel-2 inventory), paper ref. 40",
}
REQUIRES = [("h5py", "h5py==3.16.0")]      # exact version, pinned 2026-09-29

# name: (bytes, md5) from the table of contents [TOC]
FILES = {
    "20230905_train_global_ps384.hdf5": (243137679016, "eef2724f0ce5451cf821a63328ab13b3"),
    "20230905_val_global_ps384.hdf5": (81435045168, "bd84d0bf932cae7eab92c9e9a4d53645"),
    "20230905_test_global_ps384.hdf5": (86000989312, "fe6c7e9b63b1e8a85f500f2798f6f79c"),
    "gsgm_train_DEMO.hdf5": (24873991224, "40f65c47d378be9880f9f1cfa2938535"),
    "gsgm_val_DEMO.hdf5": (8195045688, "338b36cd501c50cc3fb3728cb5d57164"),
    "gsgm_test_DEMO.hdf5": (8275266000, "03aea14fb8d9543540a45893ba7ccc13"),
    "swiss_alps_features-004.pickle": (2312111277, "cdf7635d0abf0f3b257673863ade1f76"),
    "swiss_alps_reference.tif": (28318524, "cf87dcf4e9872ab33c28d5b16d5ac2e2"),
    "sc1_features-001.pickle": (13860865201, "34a3e6ff6050ea0245b16019a153081c"),
    "sc1_reference.tif": (170109342, "727a6c1a53aec37f7051eed340b4e616"),
    "alaska_features-003.pickle": (10121381043, "e36b5b60655496a71b050ab74fe8c7aa"),
    "alaska_reference.tif": (123961790, "166ef8a5018ced2dbd3b53ab2c36c036"),
    "southern_canada_features-002.pickle": (8410891443, "426e779de8fa8f92dc5804bdf5b4026c"),
    "southern_canada_reference.tif": (99874739, "1a4b48d3929fd8a3da7d187d13228ea2"),
}
SPLITS = ("train", "val", "test")
SPLIT_FILES = {s: f"20230905_{s}_global_ps384.hdf5" for s in SPLITS}

# HDF5 "subregion" attribute -> Table 2 row [PAPER]; prefix = tile-name prefix [H5].
# Only subregions whose SAR year rules out ENVISAT (i.e. Sentinel-1).
SUBREGIONS = {
    "ALP": dict(paper="ALP", prefix="ALP", year=2015, tiles=(177, 59, 60), cross=True, area_km2=1317.40),
    "NZ": dict(paper="NZL", prefix="NZ1", year=2019, tiles=(92, 31, 31), cross=True, area_km2=652.44),
    "SC2": dict(paper="SCA2", prefix="SC2", year=2018, tiles=(13, 5, 5), cross=True, area_km2=54.75),
    "SVAL1": dict(paper="SVAL1", prefix="SVAL", year=2020, tiles=(120, 32, 33), cross=True, area_km2=2782.39),
    "SVAL2": dict(paper="SVAL2", prefix="SVAL", year=2020, tiles=(68, 31, 30), cross=True, area_km2=599.32),
    "SVAL3": dict(paper="SVAL3", prefix="SVAL", year=2020, tiles=(38, 13, 13), cross=True, area_km2=850.49),
    "SA1": dict(paper="SAN1", prefix="SA", year=2016, tiles=(58, 19, 20), cross=False, area_km2=541.49),
    "SA2": dict(paper="SAN2", prefix="SA", year=2016, tiles=(207, 69, 70), cross=False, area_km2=7375.86),
}
FEATURES = ("co_pol_sar", "cross_pol_sar", "outlines")   # copied for every tile
CHECK_FEATURES = ("dem",)                                # copied for the first check_tiles train tiles

# log10(sigma0 + 1e-6) min/max per band, from utils/dataset_stats.pickle [CODE]
LOG10_EPS = 1e-6
LOG10_RANGE = {"co_pol_sar": ((-6.0, 1.757612490680689), (-6.0, 15.635085643187638)),
               "cross_pol_sar": ((-6.0, 1.3885907807268798), (-6.0, 15.905003697873322))}
SAR_SCALES = ("minmax_log10", "log10", "linear")
ORBIT_BANDS = ("asc", "desc")                            # assumption B
ORBIT_RULES = ("coverage", "any")                        # see "No data, orbit rule, units"
OUTSIDE_RANGE = ("keep", "nan")
PLAUSIBLE_CO_DB = (-25.0, 0.0)                           # our physical bounds, not the source's
PLAUSIBLE_CO_MINUS_CROSS_DB = (2.0, 15.0)
ORBIT_R_MIN = 0.05
CLASSES = {0: "non-glacier", 1: "glacier"}
CHANNELS = ("co-pol", "cross-pol")
PIXEL_SPACING_M = 10.0
EARTH_RADIUS_KM = 6371.0088
REQUIRED_ATTRS = ("epsg", "height", "width", "original_height", "original_width", "padding_height",
                  "padding_width", "region", "subregion", "tile_name", "xmin", "xmax", "ymin", "ymax")


# ------------------------------------------------------------------ remote reading

class _Nird:
    """HTTP range reads of one NIRD file.

    The download URL answers with a 302 to a presigned S3 URL that expires after
    60 s and refuses HEAD (seen 2026-09-26), so the link is re-resolved every 40 s
    and after a 403. Ranges resume where they broke off.
    """

    def __init__(self, name: str, size: int, timeout: int = 60, retries: int = 8):
        self.url, self.size, self.timeout, self.retries = BASE + name, size, timeout, retries
        self.s = requests.Session()
        self._signed, self._t = None, 0.0

    def _link(self) -> str:
        if self._signed is None or time.time() - self._t > 40:
            r = self.s.get(self.url, allow_redirects=False, timeout=self.timeout)
            if r.status_code not in (301, 302, 303, 307, 308) or "Location" not in r.headers:
                raise IOError(f"{self.url}: expected a redirect to the file, got HTTP {r.status_code}")
            self._signed, self._t = r.headers["Location"], time.time()
        return self._signed

    def stream(self, start: int, stop: int, chunk: int = 1 << 20):
        """Yield the bytes [start, stop), retrying and resuming after errors."""
        pos, fails = start, 0
        while pos < stop:
            try:
                with self.s.get(self._link(), headers={"Range": f"bytes={pos}-{stop - 1}"},
                                stream=True, timeout=self.timeout) as r:
                    if r.status_code == 403:
                        self._signed = None
                        raise IOError("HTTP 403 (presigned link expired)")
                    if r.status_code != 206:
                        raise IOError(f"HTTP {r.status_code} for a range request")
                    cr = r.headers.get("Content-Range", "")
                    if not cr.startswith(f"bytes {pos}-") or not cr.endswith(f"/{self.size}"):
                        raise IOError(f"Content-Range {cr!r}: expected start {pos} of a {self.size}-byte file")
                    for part in r.iter_content(chunk):
                        if part:
                            pos += len(part)
                            fails = 0
                            yield part
                if pos < stop:
                    raise IOError(f"stream ended at byte {pos} of {stop}")
            except (requests.RequestException, IOError) as e:
                fails += 1
                if fails > self.retries:
                    raise IOError(f"{self.url}: giving up at byte {pos}: {e}") from e
                wait = min(60, 2 ** fails)
                print(f"    retry {fails}/{self.retries} at byte {pos:,} in {wait} s ({e})")
                time.sleep(wait)

    def read(self, start: int, stop: int) -> bytes:
        b = b"".join(self.stream(start, stop))
        if len(b) != stop - start:
            raise IOError(f"{self.url}: got {len(b)} bytes for [{start}, {stop})")
        return b


class _RangeReader(io.RawIOBase):
    """Seekable read-only file over range_fn(start, stop) -> bytes, for h5py metadata.

    Small reads go through a block cache; get() fetches pixel data directly.
    """

    def __init__(self, size: int, range_fn, block: int = 1 << 15, max_blocks: int = 2048):
        super().__init__()
        self.size, self._range, self.block, self.max_blocks = size, range_fn, block, max_blocks
        self.pos, self._cache, self.requests, self.bytes = 0, {}, 0, 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _blk(self, k):
        b = self._cache.get(k)
        if b is None:
            if len(self._cache) >= self.max_blocks:
                self._cache.clear()
            a = k * self.block
            b = self._cache[k] = self._range(a, min(a + self.block, self.size))
            self.requests += 1
            self.bytes += len(b)
        return b

    def readinto(self, buf):
        n = max(0, min(len(buf), self.size - self.pos))
        out, p = bytearray(), self.pos
        while len(out) < n:
            k, o = divmod(p, self.block)
            take = self._blk(k)[o:o + n - len(out)]
            out += take
            p += len(take)
        buf[:n] = out
        self.pos += n
        return n

    def get(self, start: int, nbytes: int) -> bytes:
        b = self._range(start, start + nbytes)
        if len(b) != nbytes:
            raise IOError(f"range read returned {len(b)} of {nbytes} bytes")
        return b


def _open_remote(name: str) -> _RangeReader:
    """A _RangeReader over the official file; one HTTP session per thread."""
    size, local = FILES[name][0], threading.local()

    def rng(a, b):
        if not hasattr(local, "c"):
            local.c = _Nird(name, size)
        return local.c.read(a, b)
    return _RangeReader(size, rng)


# ------------------------------------------------------------------------ download

def download(dest, mode: str = "subset", subregions=("ALP",), splits=SPLITS, files=None,
             check_tiles: int = 8, workers: int = 4, verify_md5: bool = True) -> list[Path]:
    """Get the data into `dest` (on Colab: /content/...). Resumable.

    mode="subset" (default): read the three official split files IN PLACE over
        HTTP range requests and copy only the selected subregions' tiles into
        <file stem>.<SUBREGION>.subset.hdf5, with the same group names,
        attributes, dtypes and shapes. Only co_pol_sar, cross_pol_sar and
        outlines are copied, plus dem for the first `check_tiles` training tiles
        (orbit check). ALP: 296 x 3 x 21.2 MB = 18.9 GB + 0.17 GB dem, instead of
        410 GB. Every copied dataset is verified by its byte count. The official
        md5 covers whole files only, so it cannot be checked for a subset.
        Finished tiles are skipped when the call is repeated.
    mode="files": download whole official files (`files`, default the three split
        files), resuming by byte range, then check the md5 from the table of
        contents. The train file alone is 243 GB.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if mode == "files":
        names = list(files) if files else [SPLIT_FILES[s] for s in splits]
        bad = [n for n in names if n not in FILES]
        if bad:
            raise ValueError(f"not in the table of contents: {bad}")
        return [_download_file(dest, n, verify_md5) for n in names]
    if mode != "subset":
        raise ValueError(f"mode {mode!r} not in ('subset', 'files')")
    subs = _subregions(subregions)
    return [_download_subset(dest, split, sub, check_tiles, workers) for split in splits for sub in subs]


def _md5(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _download_file(dest: Path, name: str, verify_md5: bool) -> Path:
    size, md5 = FILES[name]
    out, ok, part = dest / name, dest / (name + ".md5ok"), dest / (name + ".part")
    if not (out.exists() and out.stat().st_size == size):
        have = part.stat().st_size if part.exists() else 0
        if have > size:
            part.unlink()
            have = 0
        print(f"{name}: {size / 1e9:.1f} GB, starting at {have / 1e9:.2f} GB")
        t0 = last = time.time()
        start = have
        with open(part, "ab") as fh:
            for chunk in _Nird(name, size).stream(have, size):
                fh.write(chunk)
                have += len(chunk)
                if time.time() - last > 30:
                    last = time.time()
                    print(f"  {have / 1e9:.2f} / {size / 1e9:.2f} GB, {(have - start) / 1e6 / (last - t0):.1f} MB/s")
        part.replace(out)
    if verify_md5 and not (ok.exists() and ok.read_text().strip() == md5):
        print(f"{name}: checking md5 ...")
        got = _md5(out)
        if got != md5:
            out.replace(out.with_name(name + ".bad"))
            raise IOError(f"{name}: md5 {got} != {md5} from the table of contents; renamed to .bad")
        ok.write_text(md5)
    print(f"{name}: complete ({size:,} bytes{', md5 ok' if verify_md5 else ''})")
    return out


def _subset_name(split: str, sub: str) -> str:
    return f"{Path(SPLIT_FILES[split]).stem}.{sub}.subset.hdf5"


def _layout(ds, where: str):
    """(offset, nbytes, dtype, shape) of a contiguous, uncompressed dataset."""
    if ds.chunks is not None or ds.compression is not None:
        raise AssertionError(f"{where}: chunks={ds.chunks}, compression={ds.compression}; the byte-range "
                             "copy expects the contiguous, uncompressed layout seen in the metadata")
    off, nb = ds.id.get_offset(), ds.id.get_storage_size()
    if off is None or nb != ds.size * ds.dtype.itemsize:
        raise AssertionError(f"{where}: offset {off}, {nb} bytes stored for shape {ds.shape} {ds.dtype}")
    return off, nb, ds.dtype.str, ds.shape


def _bounded(ex, fn, items, window):
    """ex.map with at most `window` results in flight, in input order."""
    q = deque()
    for it in items:
        q.append(ex.submit(fn, it))
        if len(q) >= window:
            yield q.popleft().result()
    while q:
        yield q.popleft().result()


def _download_subset(dest: Path, split: str, sub: str, check_tiles: int, workers: int) -> Path:
    import h5py
    name = SPLIT_FILES[split]
    n_exp = SUBREGIONS[sub]["tiles"][SPLITS.index(split)]
    out = dest / _subset_name(split, sub)
    done = set()
    if out.exists():
        try:
            with h5py.File(out, "r") as f:
                if f.attrs.get("complete") and int(f.attrs.get("n_tiles", -1)) == n_exp:
                    print(f"{out.name}: complete, {n_exp} tiles, skipped")
                    return out
                done = {k for k in f if f[k].attrs.get("_complete")}
        except OSError as e:
            out.replace(out.with_name(out.name + ".corrupt"))
            print(f"{out.name}: unreadable ({e}); moved aside, starting again")
    rd = _open_remote(name)
    fobj = io.BufferedReader(rd, buffer_size=rd.block)
    with h5py.File(fobj, "r") as src:
        pre = SUBREGIONS[sub]["prefix"]
        tiles = [k for k in sorted(src.keys()) if k.split("-")[0] == pre
                 and _attr(src[k].attrs["subregion"]) == sub]
        if len(tiles) != n_exp:
            raise AssertionError(f"{name}: {len(tiles)} {sub} tiles, Table 2 of the paper says {n_exp}")
        feats = [f for f in FEATURES if f != "cross_pol_sar" or SUBREGIONS[sub]["cross"]]
        plan = []
        for i, k in enumerate(tiles):
            if k in done:
                continue
            g, items = src[k], []
            extra = list(CHECK_FEATURES) if split == "train" and i < check_tiles else []
            for ft in feats + extra:
                if ft not in g:
                    raise AssertionError(f"{name}/{k} has no {ft!r}; members {sorted(g)}")
                items.append((ft,) + _layout(g[ft], f"{name}/{k}/{ft}"))
            plan.append((k, dict(g.attrs), items))
    total = sum(it[2] for _, _, its in plan for it in its)
    print(f"{name}: {n_exp} {sub} tiles, {len(done)} already local, fetching {len(plan)} "
          f"({total / 1e9:.2f} GB); metadata took {rd.requests} requests, {rd.bytes / 1e6:.1f} MB")

    def fetch(item):
        k, attrs, items = item
        return k, attrs, {ft: np.frombuffer(rd.get(off, nb), dtype=dt).reshape(shape)
                          for ft, off, nb, dt, shape in items}

    t0, got = time.time(), 0
    with h5py.File(out, "a") as dst, ThreadPoolExecutor(max(1, workers)) as ex:
        dst.attrs.update({"source_file": name, "source_url": BASE + name, "source_bytes": FILES[name][0],
                          "source_md5_whole_file": FILES[name][1], "subregion": sub, "complete": False,
                          "n_tiles": n_exp, "note": "subset copy of whole datasets; byte counts verified"})
        for i, (k, attrs, arrs) in enumerate(_bounded(ex, fetch, plan, max(1, workers) + 1)):
            if k in dst:
                del dst[k]
            g = dst.create_group(k)
            for a, v in attrs.items():
                g.attrs[a] = v
            for ft, arr in arrs.items():
                g.create_dataset(ft, data=arr)
            g.attrs["_complete"] = 1
            dst.flush()
            got += sum(a.nbytes for a in arrs.values())
            dt = max(time.time() - t0, 1e-9)
            print(f"  {i + 1}/{len(plan)} {k}: {got / 1e9:.2f} / {total / 1e9:.2f} GB, {got / 1e6 / dt:.1f} MB/s")
        n = sum(1 for k in dst if dst[k].attrs.get("_complete"))
        if n != n_exp:
            raise AssertionError(f"{out.name}: {n} complete tiles, expected {n_exp}")
        dst.attrs["complete"] = True
    print(f"{out.name}: complete, {n_exp} tiles")
    return out


# ----------------------------------------------------------------- local reading

def _attr(v):
    if isinstance(v, bytes):
        return v.decode()
    return v.item() if isinstance(v, np.generic) else v


def _attrs(g) -> dict:
    return {k: _attr(v) for k, v in g.attrs.items()}


def _subregions(subregions) -> tuple:
    subs = tuple(dict.fromkeys(subregions))
    bad = [s for s in subs if s not in SUBREGIONS]
    if not subs or bad:
        raise ValueError(f"unknown subregions {bad}; choose from {sorted(SUBREGIONS)} (HDF5 'subregion' codes)")
    for s in subs:
        if not SUBREGIONS[s]["cross"]:
            print(f"note: {s} ({SUBREGIONS[s]['paper']}) has co-pol only (Table 2); cross-pol will be NaN")
    return subs


def _open_split(raw_root: Path, split: str, sub: str):
    """(h5py.File, sorted tile names, file kind); asserts the Table 2 tile count."""
    import h5py
    n_exp = SUBREGIONS[sub]["tiles"][SPLITS.index(split)]
    sp, full = raw_root / _subset_name(split, sub), raw_root / SPLIT_FILES[split]
    if sp.exists():
        f, kind = h5py.File(sp, "r"), "subset"
        tiles = sorted(k for k in f if f[k].attrs.get("_complete"))
    elif full.exists():
        f, kind = h5py.File(full, "r"), "official"
        pre = SUBREGIONS[sub]["prefix"]
        tiles = sorted(k for k in f if k.split("-")[0] == pre and _attr(f[k].attrs["subregion"]) == sub)
    else:
        raise FileNotFoundError(f"neither {sp.name} nor {full.name} in {raw_root}; run download() first")
    if len(tiles) != n_exp:
        fname = Path(f.filename).name
        f.close()
        raise AssertionError(f"{fname}: {len(tiles)} complete {sub} tiles in {split}; Table 2 of "
                             f"the paper says {n_exp} (incomplete download or a different release)")
    return f, tiles, kind


def _open_all(raw_root, subs) -> dict:
    handles = {}
    try:
        for split in SPLITS:
            for sub in subs:
                handles[(split, sub)] = _open_split(Path(raw_root), split, sub)
    except Exception:
        _close(handles)
        raise
    return handles


def _close(handles):
    for f, _, _ in handles.values():
        try:
            f.close()
        except Exception:
            pass


def _check_attrs(g, name: str) -> dict:
    a = _attrs(g)
    miss = [k for k in REQUIRED_ATTRS if k not in a]
    if miss:
        raise AssertionError(f"{name}: attributes {miss} missing (seen in the metadata: {REQUIRED_ATTRS})")
    H, W, oh, ow = (int(a[k]) for k in ("height", "width", "original_height", "original_width"))
    if not (0 < oh <= H and 0 < ow <= W and a["padding_height"] == (H - oh) // 2
            and a["padding_width"] == (W - ow) // 2 and a["tile_name"] == name.split("/")[-1]):
        raise AssertionError(f"{name}: attributes break the padding rule padding == (stored - original) // 2: {a}")
    return a


def _inside(a) -> tuple[slice, slice]:
    ph, pw = int(a["padding_height"]), int(a["padding_width"])
    return slice(ph, ph + int(a["original_height"])), slice(pw, pw + int(a["original_width"]))


def _to_db(x, feature: str, band: int, sar_scale: str) -> np.ndarray:
    """Stored SAR value -> sigma0 dB (float32); 0, non-finite and sigma0 <= 0 -> NaN."""
    x = np.asarray(x, np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if sar_scale == "minmax_log10":
            lo, hi = LOG10_RANGE[feature][band]
            s = 10.0 ** (x * (hi - lo) + lo) - LOG10_EPS
        elif sar_scale == "log10":
            s = 10.0 ** x - LOG10_EPS
        elif sar_scale == "linear":
            s = x.copy()
        else:
            raise ValueError(f"sar_scale {sar_scale!r} not in {SAR_SCALES}")
        s[(x == 0) | ~np.isfinite(x)] = np.nan
        db = 10.0 * np.log10(np.where(s > 0, s, np.nan))
    db[~np.isfinite(db)] = np.nan
    return db.astype(np.float32)


def _labels(o: np.ndarray, inside, where: str) -> np.ndarray:
    if o.ndim != 3 or o.shape[-1] != 2:
        raise AssertionError(f"{where}: outlines shape {o.shape}, expected (H, W, 2) one-hot")
    vals = np.unique(o)
    if not np.isin(vals, (0.0, 1.0)).all():
        raise AssertionError(f"{where}: outlines values {vals[:10]} are not only 0/1 (one-hot assumption)")
    c0, c1 = o[..., 0] == 1, o[..., 1] == 1
    lab = np.full(o.shape[:2], IGNORE, np.uint8)
    lab[c0 & ~c1] = 0
    lab[c1 & ~c0] = 1
    keep = np.zeros(o.shape[:2], bool)
    keep[inside] = True
    lab[~keep] = IGNORE
    return lab, keep & ~c0 & ~c1, keep & c0 & c1          # label, (0,0) and (1,1) inside the tile


def _auto_chip(dims) -> int:
    return 16 * math.ceil(max(max(h, w) for h, w in dims) / 16)


def _windows(a, c: int):
    """(row, col, y0, x0) of ceil(h/c) x ceil(w/c) windows centred on the original extent."""
    oh, ow = int(a["original_height"]), int(a["original_width"])
    ny, nx = -(-oh // c), -(-ow // c)
    y0 = int(a["padding_height"]) - (ny * c - oh) // 2
    x0 = int(a["padding_width"]) - (nx * c - ow) // 2
    return [(iy, ix, y0 + iy * c, x0 + ix * c) for iy in range(ny) for ix in range(nx)]


def _cut(arr: np.ndarray, y0: int, x0: int, c: int, fill) -> np.ndarray:
    H, W = arr.shape[-2:]
    out = np.full(arr.shape[:-2] + (c, c), fill, arr.dtype)
    ys, xs, ye, xe = max(y0, 0), max(x0, 0), min(y0 + c, H), min(x0 + c, W)
    if ys < ye and xs < xe:
        out[..., ys - y0:ye - y0, xs - x0:xe - x0] = arr[..., ys:ye, xs:xe]
    return out


def _px_area_km2(a) -> float:
    """Pixel area from the tile's lon/lat box over its original extent (sphere)."""
    lon = math.radians(abs(float(a["xmax"]) - float(a["xmin"])))
    lat = abs(math.sin(math.radians(float(a["ymax"]))) - math.sin(math.radians(float(a["ymin"]))))
    return EARTH_RADIUS_KM ** 2 * lon * lat / (int(a["original_height"]) * int(a["original_width"]))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(np.float64)
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


# ------------------------------------------------------------------------ checks

def _checks(handles, subs, sar_scale: str, orbit: str, orbit_bands, check_tiles: int,
            orbit_check: str = "error", strict: bool = True) -> dict:
    """Check assumptions A (SAR scale) and B (band order) on the first training tiles."""
    acc = {b: defaultdict(list) for b in (0, 1)}
    used = []
    for sub in subs:
        f, tiles, _ = handles[("train", sub)]
        for k in tiles[:check_tiles]:
            g = f[k]
            ins = _inside(_check_attrs(g, k))
            co = g["co_pol_sar"][ins]
            cr = g["cross_pol_sar"][ins] if "cross_pol_sar" in g else None
            gx = np.gradient(g["dem"][ins][..., 0], axis=1) if "dem" in g else None
            used.append(k)
            for b in (0, 1):
                m = (co[..., b] != 0) & np.isfinite(co[..., b])
                if cr is not None:
                    m &= (cr[..., b] != 0) & np.isfinite(cr[..., b])
                idx = np.flatnonzero(m)
                if idx.size == 0:
                    continue
                idx = idx[::max(1, idx.size // 50000)]
                acc[b]["co"].append(co[..., b].ravel()[idx])
                acc[b]["cr"].append(cr[..., b].ravel()[idx] if cr is not None else np.full(idx.size, np.nan))
                if gx is not None:                      # orbit check: only tiles that carry dem
                    acc[b]["co_g"].append(co[..., b].ravel()[idx])
                    acc[b]["gx"].append(gx.ravel()[idx])
    rows = []
    for b in (0, 1):
        if not acc[b]["co"]:
            continue
        co, cr = np.concatenate(acc[b]["co"]), np.concatenate(acc[b]["cr"])
        fin_cr = cr[np.isfinite(cr)]
        for h in SAR_SCALES:
            cdb = _to_db(co, "co_pol_sar", b, h)
            med_co = float(np.nanmedian(cdb)) if np.isfinite(cdb).any() else float("nan")
            d = cdb - _to_db(cr, "cross_pol_sar", b, h)
            med_d = float(np.nanmedian(d)) if np.isfinite(d).any() else float("nan")
            raw = np.concatenate([co, fin_cr])
            in01, nonneg = float(np.mean((raw >= 0) & (raw <= 1))), float(np.mean(raw >= 0))
            rng_ok = {"minmax_log10": in01 >= 0.999, "linear": nonneg >= 0.999, "log10": True}[h]
            ok = (rng_ok and PLAUSIBLE_CO_DB[0] <= med_co <= PLAUSIBLE_CO_DB[1]
                  and (math.isnan(med_d) or PLAUSIBLE_CO_MINUS_CROSS_DB[0] <= med_d <= PLAUSIBLE_CO_MINUS_CROSS_DB[1]))
            rows.append({"sar_scale": h, "band": b, "orbit": orbit_bands[b], "pixels": int(co.size),
                         "median_co_db": round(med_co, 2), "median_co_minus_cross_db": round(med_d, 2),
                         "frac_in_0_1": round(in01, 5), "frac_nonneg": round(nonneg, 5), "plausible": bool(ok)})
    pref = orbit_bands.index(orbit)
    band = pref if acc[pref]["co"] else (1 - pref if acc[1 - pref]["co"] else None)
    print(f"SAR scale check on {len(used)} training tiles (bounds: co {PLAUSIBLE_CO_DB} dB, "
          f"co - cross {PLAUSIBLE_CO_MINUS_CROSS_DB} dB):")
    for r in rows:
        print("  " + ", ".join(f"{k}={v}" for k, v in r.items()))
    chosen = [r for r in rows if r["sar_scale"] == sar_scale and r["band"] == band]
    scale_ok = chosen[0]["plausible"] if chosen else None
    alts = sorted({r["sar_scale"] for r in rows if r["band"] == band and r["plausible"]})

    rs = {}
    for b in (0, 1):
        if acc[b]["gx"]:
            db = _to_db(np.concatenate(acc[b]["co_g"]), "co_pol_sar", b, sar_scale)
            gx = np.concatenate(acc[b]["gx"])
            m = np.isfinite(db) & np.isfinite(gx)
            if m.sum() >= 1000:
                rs[orbit_bands[b]] = round(_spearman(db[m], gx[m]), 4)
    sign = {"asc": 1.0, "desc": -1.0}
    if not rs:
        status = "not run (no dem in the check tiles)"
    elif any(r * sign[o] <= -ORBIT_R_MIN for o, r in rs.items()):
        status = "contradicted"
    elif all(r * sign[o] >= ORBIT_R_MIN for o, r in rs.items()):
        status = "confirmed"
    else:
        status = "inconclusive"
    print(f"orbit order check (Spearman r of co-pol dB with eastward elevation gradient; "
          f"asc should be > +{ORBIT_R_MIN}, desc < -{ORBIT_R_MIN}): {rs} -> {status}")
    out = {"tiles": used, "scale_table": rows, "scale_band": band, "scale_ok": scale_ok,
           "plausible_scales": alts, "orbit_r": rs, "orbit_status": status}
    if strict:
        if scale_ok is None:
            raise AssertionError(f"no valid SAR pixel in the first {check_tiles} training tiles; cannot check the "
                                 "SAR scale (raise check_tiles)")
        if not scale_ok:
            raise AssertionError(
                f"SAR scale assumption {sar_scale!r} failed on band {band}: see the table above. Assumption A "
                "(HDF5 holds (log10(sigma0 + 1e-6) - min) / (max - min), inferred from compile_features.py) or its "
                f"alternative does not hold. Scales that pass the same test: {alts or 'none'}.")
        if status == "contradicted" and orbit_check == "error":
            raise AssertionError(
                f"orbit order assumption B (band 0 = {orbit_bands[0]}, band 1 = {orbit_bands[1]}) contradicted by "
                f"terrain geometry: r = {rs}. If the bands are swapped, pass orbit_bands={tuple(orbit_bands[::-1])}; "
                "orbit_check='warn' keeps going without a decision.")
        if status != "confirmed":
            print(f"WARNING: orbit order not confirmed ({status}); orbit labels rest on assumption B")
    return out


# ------------------------------------------------------------------------ inventory

def inventory(raw_root, subregions=("ALP",), sample: int = 3, sar_scale: str = "minmax_log10",
              orbit: str = "asc", orbit_bands=ORBIT_BANDS) -> dict:
    """Print what is on disk and assert the documented layout on the selected tiles.

    Asserts: Table 2 tile counts, group attributes and the padding rule for every
    tile, and (H, W, 2) float64 SAR and outlines with only 0/1 outline values on
    `sample` tiles per split. Prints value ranges and valid shares per SAR band
    (the orbit rule's input), the (0,0) outline share per split over the sampled
    tiles, and the two assumption checks. It does not raise on those checks;
    convert() does.
    """
    raw_root = Path(raw_root)
    subs = _subregions(subregions)
    files = sorted(p for p in raw_root.iterdir() if p.is_file())
    print(f"{raw_root}: {len(files)} files")
    for p in files:
        size, tag = p.stat().st_size, ""
        if p.name in FILES:
            full = FILES[p.name][0]
            tag = "official, size ok" if size == full else f"official, INCOMPLETE {size / full:.1%}"
        elif p.name.endswith(".subset.hdf5"):
            tag = "subset copy"
        print(f"  {p.name:52s} {size / 1e9:9.3f} GB  {tag}")
    out = {"files": {p.name: p.stat().st_size for p in files}, "tiles": {}, "sample": []}
    handles = _open_all(raw_root, subs)
    try:
        dims = []
        for (split, sub), (f, tiles, kind) in handles.items():
            at = [_check_attrs(f[k], k) for k in tiles]
            dims += [(a["original_height"], a["original_width"]) for a in at]
            mem = Counter(tuple(sorted(f[k].keys())) for k in tiles)
            hw = Counter((a["height"], a["width"]) for a in at)
            oh = [a["original_height"] for a in at] or [0]
            ow = [a["original_width"] for a in at] or [0]
            print(f"{split}/{sub}: {len(tiles)} tiles ({kind} file {Path(f.filename).name}); stored {dict(hw)}; "
                  f"original H {min(oh)}-{max(oh)}, W {min(ow)}-{max(ow)}; regions {sorted({a['region'] for a in at})}")
            for m, n in mem.items():
                print(f"    members {list(m)}: {n} tiles")
            out["tiles"][f"{split}/{sub}"] = len(tiles)
        out["chip_size_auto"] = _auto_chip(dims) if dims else None
        print(f"auto chip size: {out['chip_size_auto']} px (smallest multiple of 16 >= largest original side)")
        out["outlines_00_share_sample"] = {}
        for (split, sub), (f, tiles, _) in handles.items():
            ds_ = [_describe(f[k], split, sub, sar_scale) for k in tiles[:sample]]
            out["sample"] += ds_
            n_in = sum(d.get("inside_px", 0) for d in ds_)
            if n_in:
                sh = sum(d["outlines_00_px_inside"] for d in ds_) / n_in
                out["outlines_00_share_sample"][f"{split}/{sub}"] = round(sh, 6)
                print(f"{split}/{sub}: outlines (0,0) share inside {len(ds_)} sampled tiles {sh:.4%} (-> 255 here; "
                      "the official evaluate.py scores them as non-glacier)")
        out["checks"] = _checks(handles, subs, sar_scale, orbit, orbit_bands, max(sample, 1), strict=False)
    finally:
        _close(handles)
    return out


def _describe(g, split: str, sub: str, sar_scale: str) -> dict:
    a = _check_attrs(g, g.name.strip("/"))
    H, W = a["height"], a["width"]
    ins = _inside(a)
    d = {"split": split, "subregion": sub, "tile": a["tile_name"], "stored": (H, W),
         "original": (a["original_height"], a["original_width"])}
    for ft in ("co_pol_sar", "cross_pol_sar", "outlines", "dem"):
        if ft not in g:
            continue
        ds = g[ft]
        if ft != "dem" and (ds.shape != (H, W, 2) or ds.dtype != np.float64):
            raise AssertionError(f"{split}/{a['tile_name']}/{ft}: {ds.shape} {ds.dtype}, "
                                 f"expected ({H}, {W}, 2) float64")
        d[f"{ft}_shape"], d[f"{ft}_dtype"] = ds.shape, str(ds.dtype)
        if ft in ("co_pol_sar", "cross_pol_sar"):
            v = ds[ins]
            for b in range(v.shape[-1]):
                x = v[..., b]
                ok = x[(x != 0) & np.isfinite(x)]
                q = np.percentile(ok, [0, 0.1, 1, 50, 99, 100]).round(5).tolist() if ok.size else []
                db = _to_db(ok, ft, b, sar_scale)
                db = db[np.isfinite(db)]
                qd = np.percentile(db, [0.1, 50, 99.9]).round(2).tolist() if db.size else []
                d[f"{ft}[{b}]"] = {"zero_frac": round(float(np.mean(x == 0)), 4),
                                   "nan_frac": round(float(np.mean(np.isnan(x))), 4),
                                   "valid_frac_inside": round(db.size / max(x.size, 1), 4),
                                   "min_p0.1_p1_p50_p99_max": q, f"dB_p0.1_p50_p99.9 ({sar_scale})": qd}
        if ft == "outlines":
            o = ds[()]
            keep = np.zeros((H, W), bool)
            keep[ins] = True
            s = o.sum(-1)
            d["outlines_values"] = np.unique(o).tolist()
            if not set(d["outlines_values"]) <= {0.0, 1.0}:
                raise AssertionError(f"{split}/{a['tile_name']}/outlines: values {d['outlines_values'][:10]} are not "
                                     "only 0/1 (one-hot assumption)")
            d["outlines_onehot_frac_inside"] = round(float(np.mean(s[keep] == 1)), 5)
            d["inside_px"] = int(keep.sum())
            d["outlines_00_px_inside"] = int(((o[..., 0] == 0) & (o[..., 1] == 0) & keep).sum())
            d["outlines_11_px_inside"] = int(((o[..., 0] == 1) & (o[..., 1] == 1) & keep).sum())
            d["outlines_padding_values"] = np.unique(o[~keep]).tolist() if (~keep).any() else []
            d["glacier_frac_inside"] = round(float(np.mean(o[..., 1][keep] == 1)), 4)
    print(f"  {split}/{a['tile_name']}: " + "; ".join(f"{k}={v}" for k, v in d.items()
                                                       if k not in ("split", "subregion", "tile")))
    return d


# ------------------------------------------------------------------------- convert

def _pick_band(n_valid, pref: int, orbit_rule: str):
    """Band (0/1) from the valid co-pol pixel counts per band inside the tile; None if both are 0.

    "coverage": the band with more valid pixels, `pref` on a tie.
    "any": `pref` if it has any valid pixel, else the other band.
    """
    alt = 1 - pref
    if orbit_rule == "coverage":
        return None if not max(n_valid) else (alt if n_valid[alt] > n_valid[pref] else pref)
    if orbit_rule == "any":
        return pref if n_valid[pref] else (alt if n_valid[alt] else None)
    raise ValueError(f"orbit_rule {orbit_rule!r} not in {ORBIT_RULES}")


def _read_tile(g, a, sub: str, sar_scale: str, pref: int, orbit_rule: str, where: str) -> dict:
    """One tile -> {"img" (2, H, W) float32 dB, NaN = no data or outside the original extent;
    "lab" (H, W) uint8; "inside" (H, W) bool original extent; "band" used (None: no valid co-pol
    pixel in either orbit); "valid" [2 x (H, W) bool] valid co-pol per band inside the tile;
    "m00" / "m11" (H, W) bool outline pixels inside the tile that are (0,0) / (1,1)}."""
    H, W = int(a["height"]), int(a["width"])
    ins = _inside(a)
    inside = np.zeros((H, W), bool)
    inside[ins] = True
    if SUBREGIONS[sub]["cross"] and "cross_pol_sar" not in g:
        raise AssertionError(f"{where}: no 'cross_pol_sar', but Table 2 of the paper lists cross-pol for "
                             f"{SUBREGIONS[sub]['paper']}; members {sorted(g)}")
    co = g["co_pol_sar"][()]
    if co.shape != (H, W, 2):
        raise AssertionError(f"{where}/co_pol_sar: {co.shape}, expected ({H}, {W}, 2)")
    co_db = [_to_db(co[..., b], "co_pol_sar", b, sar_scale) for b in (0, 1)]
    del co
    valid = [np.isfinite(d) & inside for d in co_db]
    band = _pick_band([int(v.sum()) for v in valid], pref, orbit_rule)
    img = np.full((2, H, W), np.nan, np.float32)
    if band is not None:
        img[0] = co_db[band]
        if "cross_pol_sar" in g:
            cr = g["cross_pol_sar"][()]
            if cr.shape != (H, W, 2):
                raise AssertionError(f"{where}/cross_pol_sar: {cr.shape}, expected ({H}, {W}, 2)")
            img[1] = _to_db(cr[..., band], "cross_pol_sar", band, sar_scale)
        img[:, ~inside] = np.nan
    lab, m00, m11 = _labels(g["outlines"][()], ins, where)
    return {"img": img, "lab": lab, "inside": inside, "band": band, "valid": valid, "m00": m00, "m11": m11}


def convert(raw_root, out_root, subregions=("ALP",), orbit: str = "asc", sar_scale: str = "minmax_log10",
            chip_size: int | None = None, check_tiles: int = 8, orbit_bands=ORBIT_BANDS,
            orbit_check: str = "error", keep_empty: bool = False, dataset: str | None = None,
            orbit_rule: str = "coverage", db_range=None, outside_range: str = "keep",
            flush_mb: float = 256.0) -> dict:
    """Write the prepared format from the local files that download() made (or the official files).

    Reads each tile in place with h5py, one at a time. Runs the assumption checks
    first and raises on failure. Splits are the official files; region = HDF5
    subregion code; date = "" (not stored). Returns the manifest.

    orbit_rule     "coverage" (default): per tile, the orbit with more valid co-pol
                   pixels, `orbit` on a tie. "any": `orbit` if it has any valid pixel.
    db_range       None (default) or (lo, hi) in dB, set by the user: the pixels below
                   lo / above hi are counted per split and channel (manifest notes).
    outside_range  "keep" (default) or "nan": with db_range, set those pixels to NaN.
    flush_mb       write a split's buffered chips to disk whenever they pass this many
                   MB, and after each split; peak RAM ~2 x flush_mb + one tile.
    """
    subs = _subregions(subregions)
    if orbit not in ORBIT_BANDS or sorted(orbit_bands) != sorted(ORBIT_BANDS):
        raise ValueError(f"orbit {orbit!r} / orbit_bands {orbit_bands!r}: use 'asc'/'desc'")
    if sar_scale not in SAR_SCALES:
        raise ValueError(f"sar_scale {sar_scale!r} not in {SAR_SCALES}")
    if orbit_check not in ("error", "warn"):
        raise ValueError(f"orbit_check {orbit_check!r} not in ('error', 'warn')")
    if orbit_rule not in ORBIT_RULES:
        raise ValueError(f"orbit_rule {orbit_rule!r} not in {ORBIT_RULES}")
    lo = hi = None
    if db_range is not None:
        lo, hi = (float(v) for v in db_range)
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError(f"db_range {db_range!r}: need finite (lo, hi) in dB with lo < hi")
    if outside_range not in OUTSIDE_RANGE or (outside_range == "nan" and db_range is None):
        raise ValueError(f"outside_range {outside_range!r}: 'keep', or 'nan' together with db_range=(lo, hi)")
    if not flush_mb > 0:
        raise ValueError(f"flush_mb {flush_mb!r} must be > 0")
    orbit_bands = tuple(orbit_bands)
    pref = orbit_bands.index(orbit)
    name = dataset or (DATASET if subs == ("ALP",) else "glavitu_" + "_".join(s.lower() for s in subs))
    handles = _open_all(raw_root, subs)
    try:
        attrs = {(sp, sb, k): _check_attrs(f[k], k) for (sp, sb), (f, tiles, _) in handles.items() for k in tiles}
        c = chip_size or _auto_chip([(a["original_height"], a["original_width"]) for a in attrs.values()])
        print(f"chip size {c} px ({'auto' if chip_size is None else 'given'}); dataset folder {name}")
        chk = _checks(handles, subs, sar_scale, orbit, orbit_bands, check_tiles, orbit_check, strict=True)
        used = {f"{sp}/{sb}": f"{Path(f.filename).name} ({kind})" for (sp, sb), (f, _, kind) in handles.items()}
        w = PreparedWriter(out_root, name, c, CLASSES, source={**SOURCE, "subregions": list(subs), "files_used": used},
                           channels=CHANNELS, units="dB", pixel_spacing_m=PIXEL_SPACING_M)
        flush = w.flush
        budget, per_chip = flush_mb * 1e6, 5 * c * c         # float16 image (2 x 2 B) + uint8 label per pixel
        print(f"RAM: a split's buffered chips go to disk every {flush_mb:g} MB and after the split; "
              f"peak ~2 x {flush_mb:g} MB + one tile")
        per_orbit, dropped_tiles, dropped_win = Counter(), [], 0
        area, n_lab, n_gla = defaultdict(float), 0, 0
        dbs = {s: {ch: {"finite_px": 0, "min": None, "max": None, "below": 0, "above": 0} for ch in CHANNELS}
               for s in SPLITS}
        unl = {s: {"inside_px": 0, "outlines_00": 0, "outlines_11": 0} for s in SPLITS}
        other_more = Counter()
        for split in SPLITS:
            buffered = 0
            for sub in subs:
                f, tiles, _ = handles[(split, sub)]
                info = SUBREGIONS[sub]
                for i, k in enumerate(tiles):
                    a, where = attrs[(split, sub, k)], f"{SPLIT_FILES[split]}/{k}"
                    t = _read_tile(f[k], a, sub, sar_scale, pref, orbit_rule, where)
                    img, lab, inside, band = t["img"], t["lab"], t["inside"], t["band"]
                    g1 = int((lab == 1).sum())
                    area[sub] += g1 * _px_area_km2(a)
                    n_lab += int((lab != IGNORE).sum())
                    n_gla += g1
                    if band is None:
                        dropped_tiles.append(f"{split}/{k}")
                        continue
                    for ch, cname in enumerate(CHANNELS):          # source dB range, before any masking
                        v = img[ch][np.isfinite(img[ch])]
                        if v.size:
                            st = dbs[split][cname]
                            vmin, vmax = float(v.min()), float(v.max())
                            st["finite_px"] += int(v.size)
                            st["min"] = vmin if st["min"] is None else min(st["min"], vmin)
                            st["max"] = vmax if st["max"] is None else max(st["max"], vmax)
                            if db_range is not None:
                                st["below"] += int((v < lo).sum())
                                st["above"] += int((v > hi).sum())
                    if outside_range == "nan":
                        with np.errstate(invalid="ignore"):
                            img[(img < lo) | (img > hi)] = np.nan
                    own, other = t["valid"][band], t["valid"][1 - band]
                    oh, ow = int(a["original_height"]), int(a["original_width"])
                    dx = (float(a["xmax"]) - float(a["xmin"])) / ow
                    dy = (float(a["ymax"]) - float(a["ymin"])) / oh
                    for iy, ix, y0, x0 in _windows(a, c):
                        im, lb = _cut(img, y0, x0, c, np.nan), _cut(lab, y0, x0, c, IGNORE)
                        ins = _cut(inside, y0, x0, c, False)
                        fin = np.isfinite(im[0])
                        if not fin.any() and not keep_empty:
                            dropped_win += 1
                            continue
                        n_in, n_l = max(int(ins.sum()), 1), int((lb != IGNORE).sum())
                        ov = float(_cut(other, y0, x0, c, False).sum()) / n_in
                        other_more[split] += int(ov > float(_cut(own, y0, x0, c, False).sum()) / n_in)
                        u = unl[split]
                        u["inside_px"] += int(ins.sum())
                        u["outlines_00"] += int(_cut(t["m00"], y0, x0, c, False).sum())
                        u["outlines_11"] += int(_cut(t["m11"], y0, x0, c, False).sum())
                        w.add(split, im, lb, region=sub, date="", tile=k, subregion_paper=info["paper"],
                              ref_year=info["year"], source_file=SPLIT_FILES[split], orbit=orbit_bands[band],
                              win_row=iy, win_col=ix, y0=int(y0), x0=int(x0),
                              lat=round(float(a["ymax"]) - (y0 + c / 2 - int(a["padding_height"])) * dy, 6),
                              lon=round(float(a["xmin"]) + (x0 + c / 2 - int(a["padding_width"])) * dx, 6),
                              deg_per_px_x=round(dx, 9), deg_per_px_y=round(dy, 9),
                              sar_valid_frac=round(float((fin & ins).sum()) / n_in, 4),
                              other_orbit_valid_frac=round(ov, 4),
                              label_valid_frac=round(n_l / n_in, 4),
                              glacier_frac=round(float((lb == 1).sum()) / max(n_l, 1), 4))
                        per_orbit[(split, orbit_bands[band])] += 1
                        buffered += per_chip
                        if buffered >= budget:
                            flush(split)
                            buffered = 0
                    if (i + 1) % 25 == 0 or i + 1 == len(tiles):
                        print(f"  {split}/{sub}: {i + 1}/{len(tiles)} tiles")
            flush(split)
        share = n_gla / max(n_lab, 1)
        if share >= 0.5:
            raise AssertionError(f"glacier share of labelled pixels is {share:.2f}; outlines channel 1 = glacier "
                                 "(from predict.py/evaluate.py) looks wrong")
        area_txt = "; ".join(f"{s} {area[s]:.1f} km2 vs Table 2 {SUBREGIONS[s]['area_km2']:.2f} km2 "
                             f"(ratio {area[s] / SUBREGIONS[s]['area_km2']:.3f})" for s in subs)
        print(f"glacier area in the converted tiles (pixel area from the lon/lat box): {area_txt}")
        for s in subs:
            r = area[s] / SUBREGIONS[s]["area_km2"]
            if not 0.8 <= r <= 1.25:
                print(f"WARNING: {s} glacier area ratio {r:.3f} is far from 1; check labels and pixel size")
        for st in (x for d in dbs.values() for x in d.values()):
            for key in ("min", "max"):
                st[key] = None if st[key] is None else round(st[key], 3)
        print("dB per split and channel, before any masking (finite px, min, max"
              + (f", below {lo:g} / above {hi:g}" if db_range is not None else "") + "):")
        for s in SPLITS:
            for ch in CHANNELS:
                st = dbs[s][ch]
                if st["finite_px"]:
                    print(f"  {s} {ch}: {st['finite_px']} px, {st['min']} .. {st['max']} dB"
                          + (f", {st['below']} / {st['above']}" if db_range is not None else ""))
        if db_range is None:
            print("  (db_range=(lo, hi) counts the values outside a range you choose; outside_range='nan' masks them)")
        print(f"chips where the other orbit has more valid co-pol pixels: {dict(other_more)}")
        print(f"outline pixels (0,0) / (1,1) inside the written chips (-> 255): {unl}")
        best = [r for r in chk["scale_table"] if r["sar_scale"] == sar_scale and r["band"] == chk["scale_band"]][0]
        rule_txt = {"coverage": f"per tile the orbit with more valid co-pol pixels, {orbit} on a tie",
                    "any": f"{orbit} if its co-pol band has any valid pixel, else the other"}[orbit_rule]
        mask_txt = ("kept as they are" if outside_range == "keep" else
                    f"set to NaN below {lo:g} dB and above {hi:g} dB")
        u_txt = "; ".join(f"{s} {u['outlines_00']} (0,0) + {u['outlines_11']} (1,1) of {u['inside_px']} px"
                          for s, u in unl.items() if u["inside_px"])
        stats = {"db_per_split_channel_before_masking": dbs, "db_range": [lo, hi] if db_range is not None else None,
                 "outside_range": outside_range, "orbit_rule": orbit_rule,
                 "chips_per_split_orbit": {f"{k[0]}/{k[1]}": v for k, v in sorted(per_orbit.items())},
                 "chips_where_other_orbit_has_more_valid_px": {s: other_more[s] for s in SPLITS},
                 "outline_px_in_chips": unl, "tiles_without_sar": dropped_tiles, "windows_without_sar": dropped_win}
        w.notes = (
            f"Official GlaViTU split (one HDF5 file per split), subregions {list(subs)}. region = HDF5 subregion "
            f"code; date empty (not stored in the files); ref_year = Table 2 year (SAR within ~1 month, per paper). "
            f"Chips {c} px, centred windows over each tile's original extent; outside -> NaN / 255. "
            f"Units dB from sar_scale={sar_scale} (assumption A); check on band {chk['scale_band']}: median co "
            f"{best['median_co_db']} dB, co-cross {best['median_co_minus_cross_db']} dB -> plausible. "
            f"Orbit rule {orbit_rule!r}: {rule_txt}; bands {list(orbit_bands)} (assumption B), terrain check "
            f"{chk['orbit_status']} {chk['orbit_r']}. Chips where the other orbit has more valid co-pol pixels: "
            f"{dict(other_more)} (meta other_orbit_valid_frac). Tiles without SAR dropped: {len(dropped_tiles)}; "
            f"windows without SAR dropped: {dropped_win}. dB values {mask_txt}; the repo stats allow log10 sigma0 "
            f"up to 15.6 / 15.9 in the descending bands (~ +156 dB), so see the dB min/max per split in the stats. "
            f"Glacier share of labelled pixels {share:.4f}; area {area_txt}. Labels: outlines (0,0) or (1,1) inside "
            f"a tile -> 255 ({u_txt}). The authors' FocalLoss gives (0,0) zero loss (ignored in training, as here), "
            f"but their evaluate.py scores (0,0) as non-glacier ((1,1) as glacier), and predict.py's [pad:-pad] "
            f"crop keeps one padding row/column where stored - original is odd (255 here), so an IoU here is not "
            f"on exactly the paper's "
            f"pixel set. Channels named co-pol/cross-pol: the source does not name the polarisations. Pixel spacing "
            f"10 m is the paper's nominal value; the grid is EPSG:4326 (see deg_per_px_x/y). "
            f"Stats: " + json.dumps(stats, sort_keys=True))
        man = w.close()
    finally:
        _close(handles)
    print(f"{name}: {man['splits']} chips; tiles without SAR dropped {len(dropped_tiles)}, "
          f"windows dropped {dropped_win}; per orbit {dict(per_orbit)}")
    return man
