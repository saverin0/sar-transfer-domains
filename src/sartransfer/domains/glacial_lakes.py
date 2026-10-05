"""Glacial lakes: Glacial-Lake-Bench (GLB) + Glacial-Lake-Challenge (GLC) -> prepared format.

Sources opened 2026-09-26 (nothing below is guessed; open points are checked at run time):
  [Z] Zenodo record 17917359, API JSON (https://zenodo.org/api/records/17917359): file
      names, byte sizes, md5, licence CC BY 4.0, description (11 channels "Blue, Green,
      Red, NIR, SWIR1, SWIR2, NDWI, Slope, Elevation, VV, and VH"; mask "0 for background
      and 1 for lake pixels"; the 18 region codes; leave-one-region-out recommended).
  [D] The central directory of both zips, read with HTTP range requests (directory bytes
      only): exact member names, sizes and CRC-32. (The first build's 1 KB range read also
      covered the 538-byte Metadata.txt member; it repeats [Z], and no fact here rests on it.)
      Re-read 2026-09-27 with zenodo_directory(): GLB 38,240 files in 2 requests (5.6 MB);
      the img_dir + ann_dir .tif files sum to 56,431,403,680 B (the extracted size).
  [P] Kaushik, Tellman, Howat, Haritashya: Glacial-Lake-Bench ..., ESSD Discussions
      preprint, https://doi.org/10.5194/essd-2026-474 (sections 2.1, 2.2, 2.5, 3.2, Table 1).
  [C] The dl4eo code, github.com/Sk-2103/dl4eo (read only to learn the format; not
      imported). NOTE: pushed 2026-06, after the Dec 2025 release, and it builds 10 bands
      (one DEM band, no Slope), so it is NOT the release's generator -- only a hint.
  [R] Planetary Computer STAC collection sentinel-1-rtc: vv/vh/hh/hv "terrain-corrected
      gamma naught", float32, nodata -32768, 10 m, IW only.
  [G] rasterio 1.5.0, docstring of rasterio.warp.reproject: src_nodata defaults to the
      source band's nodata; dst_nodata "Defaults to the nodata value of the destination
      image (if set), the value of src_nodata, or 0 (GDAL default)".
  [H] Hugging Face API, dataset repo Sk-21/Cryo-Bench, and a 4 MB read of the start of its
      data/GLB.tar.gz (2026-09-27): 44,287,419,921 B, LFS sha256 ef47a250...7ddfc085. First
      members: GLB/train/, GLB/train/global_stats.json (767 B, added by Cryo-Bench),
      GLB/train/images/, GLB/train/images/GL_S2B_21WXV_20200717_1_L2A_91.tif (2,885,556 B),
      i.e. Zenodo file names. The mask folder and the val/test folder names come later in the
      stream and are NOT known. Cryo-Bench has no challenge set.
  [L] Read locally 2026-09-27: huggingface_hub 1.31.0 (_hf_hub_download_to_local_dir,
      _download_to_tmp_and_move): with local_dir the file goes to a process-unique temp file
      in <local_dir>/.cache/huggingface/download/ and is then moved to <local_dir>/<filename>;
      a failed call deletes its temp file (no resume across calls); the HF_HOME cache is only
      read (a copy already there is copied). hf_xet 1.6.0 (strings of the binary) reads
      HF_XET_CACHE, HF_HOME, XDG_CACHE_HOME and HF_XET_CHUNK_CACHE_SIZE_BYTES.

Layout [D]
  Glacial_Lake_Bench.zip (44,268,636,160 B):
    Glacial_Lake_Bench/img_dir/{train,val,test}/<REG>_<S2A|S2B>_<MGRS>_<YYYYMMDD>_<n>_L2A_<k>.tif
    Glacial_Lake_Bench/ann_dir/{train,val,test}/<same file name>      (+ Metadata.txt, Thumbs.db)
    pairs: train 15,300, val 1,912, test 1,907 = 19,119. [P] says 19,115 (abstract) and
    its Table 1 sums to 19,150; the archive is what we count.
  Glacial-Lake-Challenge.zip (2,588,538,597 B):
    Glacial-Lake-Challenge/{image,mask}/<S2A|S2B>_<MGRS>_<YYYYMMDD>_<n>_L2A_<k>.tif
    1,105 pairs ([P]: 1,105), NO region prefix. [P] calls it GLBC, [Z] calls it GLC.
  File sizes [D]: images 2,885,555/2,885,556 B, masks 65,943/65,944 B (+ 274 GLB and 38
  GLC masks 6,144 B larger, cause unknown). A plain GDAL GeoTIFF of 11 x 256 x 256 float32,
  uncompressed, in a UTM CRS, without nodata tag or band descriptions has exactly these
  sizes (the odd byte follows the zone's digit count in every file); a nodata tag would
  add 12-19 B. So missing SAR is most likely a pixel value, not a tag -- an inference from
  sizes, which is why every chip is still checked.
  Either the zips sit in raw_root (read in place, never extracted) or their extracted
  top folders do (raw_root/Glacial_Lake_Bench/..., raw_root/Glacial-Lake-Challenge/...).
  Hugging Face copy of GLB [H] (download(bench_source="hf")): GLB.tar.gz is extracted to
    raw_root/GLB/<train|val|validation|test>/images/<file name> + one mask folder per split,
    found at run time as the one sibling of images/ whose .tif names equal the images'.
    Its files are read under their Zenodo names (meta 'source_file' as from the zip) and
    only after check_identity() found every Zenodo image and mask present and byte-identical
    (size + CRC-32) to the zip's central directory [D]. Chips the Hugging Face copy has and
    the Zenodo release does not (6 test pairs seen on 2026-09-28) are listed in the check's
    record and never read, so the splits are Zenodo's. The challenge set still comes from
    Zenodo. The zip is used when both are in raw_root.

Channels: bands 10 and 11 of 11 [Z][P] -> channel 0 (co-pol), channel 1 (cross-pol). All
  other bands are dropped. [P] 2.1: "For Arctic regions, where VV and VH data may be
  unavailable, HH and HV polarization images are used instead"; [C] takes VV+VH when the
  RTC item has both, else HH+HV. File names do not say which, so the manifest names the
  channels "VV/HH" and "VH/HV"; meta 'pol' is filled only when the GeoTIFF band
  descriptions name the polarisation (and they are checked against the band order).

Classes: 0 background, 1 lake [Z][P Fig. 2]. Labels are rasterised from the Zhang et al.
  (2024) inventory made from 2020 Sentinel-2 images [P], i.e. optical, not drawn on the SAR.
  255 (ignore) where both SAR channels are no data (ignore_no_sar=True), because a
  SAR-only model cannot see those pixels. Any mask value other than 0/1 stops the run.

Splits: official folders train -> "train", val -> "val", test -> "test". [P] 3.2: a random
  80/10/10 split, and "LORO [is] the primary evaluation because it ... is robust to the
  spatial autocorrelation that inflates random-split estimates". In the archive [D] 1,774
  of 1,907 test chips and 1,786 of 1,912 val chips come from a Sentinel-2 scene that also
  gives train chips (1,630 of 3,695 scene keys span more than one official folder), so the
  official "test" measures in-distribution performance; the manifest notes repeat these
  counts for the actual output splits.
  GLC -> "test_challenge" (hard cases: cloud, shadow, frozen and small lakes). [P] 3.2
  Phase 2 trained its GLBC models on the FULL GLB (all three folders) "excluding tiles with
  filenames matching those in the GLBC set"; here only "train" is trained on (contract),
  so test_challenge scores are NOT comparable with [P]'s GLBC baselines. In the archive no
  GLB name (minus region prefix) equals a GLC name and no image is byte-identical (CRC-32),
  yet 1,037 of 1,105 GLC chips come from Sentinel-2 scenes that also give GLB train chips.
  Region hold-out ([Z][P] LORO): holdout_regions=("SV", ...) moves all GLB chips of those
  regions (any official folder) to "test_region_<code>". Default: none held out (the user
  decides before the one conversion). Five groups of byte-identical images carry two region
  prefixes [D]: CA/SEE x3 (CA_..._95 = SEE_..._93 in train; test CA_..._111 = train
  SEE_..._112; test SEE_..._13 = val CA_..._24) and CA/SAW x2 (both pairs in train). With a
  hold-out, a TRAIN chip whose CRC-32 equals a held-out chip's is dropped (listed in the
  notes); val/test twins are kept because they are only scored (run.py selects nothing on
  val). 18 MGRS tiles span more than one region [D]: 8 CA/SEE, 6 CA/SAW, 2 SAW/SEE,
  1 CA/SAW/SEE, 1 AK/WC. Their other-region chips stay in train as neighbours (not copies);
  the notes count them and 'overlap_train' measures any footprint overlap.
  Footprint overlap (the directory has no coordinates, so it is measured at conversion):
  meta 'overlap_train' on every non-train row (largest share of the chip covered by one
  train chip) and 'overlap_test_challenge' on train rows. Footprints in another CRS (the
  next UTM zone) are transformed into the row's CRS and clipped exactly.

Region: the file-name prefix [Z]; RGI names from [P] Table 1 are in REGIONS (matched by
  name; the chip counts agree for 13 of 18 regions, AC/ACS/AK/WC differ by 1-2 and SEE is
  613 in [D] vs 638 in Table 1). GLC names have no prefix: 'region' is taken from GLB chips
  of the same Sentinel-2 scene (1,033 chips), else of the same MGRS tile (52), when that is
  unique; otherwise "unknown" (20). 'region_source' says which ('file name', 'GLB scene',
  'GLB tile', 'none'). The code "NA" (North Asia, 261 chips) is one of pandas' default NA
  tokens: read the meta with keep_default_na=False, na_values=[""] (convert() warns when
  prepared.load_split loses it).
Date: the Sentinel-2 date in the file name, 2020-07-01 .. 2020-12-27 [D] (53 GLB chips, all
  SEE, and 200 GLC chips are from December). The SAR scene is the closest RTC acquisition
  within +-5 days [P][C]; its own date is not in the release.

Units. [P] 2.1: "the stacked images are normalized using a min-max stretch to a range of
  0-1" -- per chip, per scene or per dataset is NOT stated. [C] (later) writes linear gamma0
  by default and has a per-band min-max ("per_band", per patch) and a "per_modality" mode
  (dB, then a 2-98 % stretch). Which one the release holds is only visible in the files, so
  convert() decides it on a fixed sample of chips (s1_scale="auto") and checks every chip:
    "minmax"  max == 1 (+-1e-3) on >= 90 % of the sampled non-constant chip-bands and
              few exact ones -> values kept (units "minmax01_per_chip", NOT dB: the
              absolute calibration is gone), exact 0 -> NaN (see no data). With
              minmax_to_log=True they are stored as 10*log10(value) instead (units
              "dB_rel_chip_max"). That is dB relative to the chip maximum only when the
              stretch minimum was 0 (a 0 fill); on a fully covered chip the minimum is the
              darkest real pixel, usually open water, so 10*log10((x-lo)/(hi-lo)) distorts
              the dark lake pixels most and the darkest one becomes NaN. A choice for the
              user, off by default.
    "pct_stretch"  like minmax but >= 0.5 % of pixels at exactly 1 (a 2-98 % clip): kept
              (units "pct2_98_dB_per_chip"); zeros cannot be told from clipped pixels.
    "linear"  non-negative and some sampled value > 1 -> 10*log10, non-positive -> NaN
              (units "dB", gamma0 RTC).
    "db"      mostly negative, median within -60..10 -> kept (units "dB", gamma0 RTC).
  Stops instead of guessing: every sampled value in 0-1 but < 90 % of chip-bands reaching 1
  (a stretch over a whole scene or the whole dataset looks exactly like this, and so would
  unusually dark linear gamma0 -- inspect, then force s1_scale); one stretch over all 11
  bands or over both SAR bands; anything else. The decision, its evidence and the chip
  counts go into the manifest notes.

No data -> NaN: NaN, inf, the GeoTIFF nodata value if one is set, the PC RTC nodata -32768
  [R] in every state, and per state: non-positive values in "linear"; exact 0 in "minmax".
  How missing SAR reaches a chip is not documented for the release. In [C],
  _clip_to_patch reprojects the RTC band without dst_nodata, so by [G] uncovered pixels
  get the source nodata (-32768 per [R]; 0 only if the COG has no nodata tag), although
  the file is tagged nodata 0.0; the stack then copies the S2 patch meta, and the
  per-band min-max takes its nodata from that meta, so a -32768 fill counts as valid and
  becomes the stretch minimum. Consequences handled here:
    - exact 0 in "minmax" is a stretched fill (0 or -32768) or the chip's darkest real
      pixel (one pixel per chip and band lost);
    - a "minmax" chip-band whose non-zero values all lie above 0.5 (or have a median above
      0.99) was stretched against a far-off fill: its real pixels are squashed into
      ~0.9998..1 (exactly 1.0 in float16), so the whole band -> NaN, meta 's1_squashed' =
      number of such bands, counts in the manifest notes and in inventory().
  Pixel spacing 10 m ([P] 2.1: lakes "rasterized onto the 10 m grid"; [C] resamples the
  RTC bilinearly onto the Sentinel-2 grid); checked per chip.
Chip size 256 (the native 256 x 256 [Z][P]; 16 x 16 tokens with 16 px patches).

Checked at run time (unverified until real files are read): the value state above;
  exact 0 = no data in "minmax"; how often a -32768 fill occurs or squashed a band; 11
  bands, 256 x 256, float; masks one band 256 x 256 with values in {0, 1} and the same
  footprint as their image; band order against any band descriptions; 10 m spacing; pair
  counts of EVERY expected split; same value state in GLB and GLC.
"""

from __future__ import annotations

import errno
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import threading
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import prepared

DATASET = "glacial_lakes"

FILES = {
    "bench": {"name": "Glacial_Lake_Bench.zip", "top": "Glacial_Lake_Bench",
              "url": "https://zenodo.org/api/records/17917359/files/Glacial_Lake_Bench.zip/content",
              "size": 44268636160, "md5": "a373b0d3833126875c4b724bf75ce63a"},
    "challenge": {"name": "Glacial-Lake-Challenge.zip", "top": "Glacial-Lake-Challenge",
                  "url": "https://zenodo.org/api/records/17917359/files/Glacial-Lake-Challenge.zip/content",
                  "size": 2588538597, "md5": "f2a9a1e83eb4ca0a5d37544f282ae51f"},
}
HF = {"repo_id": "Sk-21/Cryo-Bench", "repo_type": "dataset", "filename": "data/GLB.tar.gz", "top": "GLB",
      "size": 44287419921, "sha256": "ef47a250d0bd3551c77ca1716967108c4ffa6bd56ae09f21cf02f4857ddfc085"}   # [H]
HF_SPLITS = {"train": "train", "val": "val", "validation": "val", "test": "test"}     # folder name -> split
GLB_BYTES = 56431403680         # [D] sum of the img_dir + ann_dir .tif sizes = the extracted GLB folder
DONE_MARKER = ".extraction_complete"      # written only after a full extraction
SAME_MARKER = ".zenodo_identical"         # written by check_identity() after a passing check

SOURCE = {
    "name": "Glacial-Lake-Bench (GLB) and Glacial-Lake-Challenge (GLC; 'GLBC' in the preprint)",
    "record": "https://zenodo.org/records/17917359",
    "doi": "10.5281/zenodo.17917359",
    "version": ("the only (latest) version of concept DOI 10.5281/zenodo.17917358; published 2025-12-16, "
                "record metadata last modified 2026-04-17 (Zenodo API)"),
    "licence": "CC BY 4.0",
    "creator": "Kaushik, Saurabh (University of Wisconsin-Madison)",
    "citation": ("Saurabh Kaushik, Beth Tellman, Ian Howat, Umesh Haritashya, Lalit Maurya. Glacial-Lake-Bench: "
                 "A Global Multi-Sensor Benchmark Dataset for Evaluating Deep Learning Models for Glacial Lake "
                 "Mapping. https://zenodo.org/records/17917359 (citation as given on the Zenodo record)"),
    "citation_in_archive": ("Metadata.txt in Glacial_Lake_Bench.zip cites: Kaushik, Tellman, Howat, Haritashya, "
                            "Maurya. 'Beyond Clouds: Global glacial lake mapping combining Sentinel-1 and "
                            "Sentinel-2 remote sensing data and Geo-Foundational Model.'"),
    "paper": ("Kaushik, Tellman, Howat, Haritashya: Glacial-Lake-Bench: A Global Multi-Sensor Benchmark Dataset "
              "for Evaluating Deep Learning Models for Glacial Lake Mapping. ESSD Discussions preprint, "
              "https://doi.org/10.5194/essd-2026-474 (discussion started 9 July 2026, under review)"),
    "files": {f["name"]: {"url": f["url"], "size": f["size"], "md5": f["md5"]} for f in FILES.values()},
    "labels": "Zhang et al. (2024) glacial lake inventory from 2020 Sentinel-2 images (per the preprint)",
    "sar": "Sentinel-1 RTC gamma0 from Microsoft Planetary Computer 'sentinel-1-rtc' (per the preprint)",
}

REQUIRES = [("rasterio", "rasterio==1.5.1")]      # exact version, pinned 2026-09-29

REGIONS = {                                  # code: [Z] label        # [P] Table 1 RGI region name
    "AC": "Arctic Canada",                   # Arctic Canada North
    "ACS": "Arctic Canada South",            # Arctic Canada South
    "AK": "Alaska",                          # Alaska
    "CA": "Central Asia",                    # Central Asia
    "GL": "Greenland",                       # Greenland
    "RA": "Russian Arctic",                  # Russian Arctic
    "SA": "Southern Andes",                  # Southern Andes
    "SAW": "South Asia West",                # South Asia West
    "SC": "Scandinavia",                     # Scandinavia
    "SV": "Svalbard",                        # Svalbard
    "WC": "Western Canada",                  # Western Canada and USA
    "CE": "Central Europe",                  # Central Europe
    "LL": "Lower Latitudes",                 # Low Latitudes
    "ME": "Caucasus and Middle East",        # Middle East (Caucasus)
    "NA": "North Asia",                      # North Asia
    "NZ": "New Zealand",                     # New Zealand
    "IS": "Iceland",                         # Iceland
    "SEE": "South East Asia",                # South Asia East (RGI 'South Asia East': Himalaya, e.g. 44RPU/45RXM)
}

EXPECTED = {"train": 15300, "val": 1912, "test": 1907, "challenge": 1105}       # pairs, [D]
BENCH_SPLITS = ("train", "val", "test")
CHIP, N_BANDS, S1_BANDS = 256, 11, (10, 11)                                     # 1-based bands
PIXEL_M = 10.0
CLASSES = {0: "background", 1: "lake"}
UNITS = {"db": "dB", "linear": "dB", "minmax": "minmax01_per_chip",
         "pct_stretch": "pct2_98_dB_per_chip"}
DB_RANGE = (-80.0, 40.0)
PC_NODATA = -32768.0            # [R] nodata of the PC RTC vv/vh/hh/hv assets -> no data in every state
SQUASH_MIN, SQUASH_MED = 0.5, 0.99   # "minmax" band squashed by a far-off fill (see docstring)
NEAR_KM = 10.0                  # footprints in another CRS are compared when the centres are this close

_GLB = re.compile(r"^Glacial_Lake_Bench/img_dir/(train|val|test)/"
                  r"([A-Z]+)_((S2[AB])_(\d{1,2}[A-Z]{3})_(\d{8})_\d+_L2A)_\d+\.tif$")
_GLC = re.compile(r"^Glacial-Lake-Challenge/image/((S2[AB])_(\d{1,2}[A-Z]{3})_(\d{8})_\d+_L2A)_\d+\.tif$")
_TIF = re.compile(r"^Glacial_Lake_Bench/(img_dir|ann_dir)/(train|val|test)/[^/]+\.tif$")
_NO_RETRY = {errno.ENOSPC, errno.EACCES, errno.EROFS} | {getattr(errno, "EDQUOT", errno.ENOSPC)}


def _check(ok, msg: str) -> None:
    if not ok:
        raise AssertionError(msg)


# ----------------------------------------------------------------------------- download

def download(dest, which=("bench", "challenge"), verify: bool = True, retries: int = 8,
             chunk_mb: int = 8, timeout: int = 60, bench_source: str = "zenodo") -> list[Path]:
    """Fetch the official zips into `dest` (~44.3 + 2.6 GB). `which` picks a subset.

    Resumable: a partial '<name>.part' continues with an HTTP Range request (Zenodo answers
    206; a reply whose Content-Range does not start at the resume offset restarts from 0).
    A file of the published size with a '<name>.md5ok' marker holding the published md5 is
    skipped. Before fetching, the free space in `dest` must exceed the bytes still to come;
    disk errors (full, read-only, no permission) stop at once instead of being retried. The
    md5 published by Zenodo is checked after download (verify=True); a mismatch renames the
    file to '<name>.bad' and stops.
    bench_source="hf": GLB comes from the Hugging Face copy instead (download_hf(), then
    check_identity() against the Zenodo release); the challenge zip still from Zenodo.
    """
    _check(bench_source in ("zenodo", "hf"), f"bench_source {bench_source!r}: 'zenodo' or 'hf'")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    if bench_source == "hf" and "bench" in which:
        out.append(download_hf(dest))
        check_identity(dest)
        which = tuple(k for k in which if k != "bench")
    for key in which:
        f = FILES[key]
        path, marker = dest / f["name"], dest / (f["name"] + ".md5ok")
        complete = path.is_file() and path.stat().st_size == f["size"]
        marked = marker.is_file() and marker.read_text().strip() == f["md5"]
        if complete and (marked or not verify):
            print(f"[skip] {f['name']}: complete ({f['size'] / 1e9:.2f} GB)")
            out.append(path)
            continue
        if not complete:
            part = path.with_name(path.name + ".part")
            have = part.stat().st_size if part.is_file() else 0
            need, free = f["size"] - min(have, f["size"]), shutil.disk_usage(dest).free
            _check(free > need, f"{f['name']}: {need / 1e9:.2f} GB still to download but only "
                                f"{free / 1e9:.2f} GB free in {dest}")
            _fetch(f["url"], path, f["size"], retries, chunk_mb * 2**20, timeout)
        if verify:
            print(f"[md5] {f['name']} ...", flush=True)
            got = _md5(path)
            if got != f["md5"]:
                path.replace(path.with_name(path.name + ".bad"))
                raise AssertionError(f"{f['name']}: md5 {got} != published {f['md5']}; renamed to .bad")
            marker.write_text(got)
            print(f"[ok] {f['name']} md5 {got}")
        out.append(path)
    return out


def _fetch(url: str, path: Path, size: int, retries: int, chunk: int, timeout: int) -> None:
    part = path.with_name(path.name + ".part")
    t0, last = time.time(), 0.0
    for attempt in range(retries + 1):
        have = part.stat().st_size if part.is_file() else 0
        if have > size:
            part.unlink()
            have = 0
        if have == size:
            break
        try:
            hdr = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(url, headers=hdr, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                if have and r.status_code != 206:
                    print(f"[restart] {path.name}: server ignored the range request")
                    have = 0
                elif have:
                    cr = r.headers.get("Content-Range", "")
                    m = re.match(r"bytes (\d+)-", cr)
                    if not m or int(m.group(1)) != have:
                        print(f"[restart] {path.name}: asked for bytes {have}-, got Content-Range {cr!r}; from 0")
                        part.unlink(missing_ok=True)
                        continue
                with open(part, "ab" if have else "wb") as fh:
                    for block in r.iter_content(chunk):
                        fh.write(block)
                        have += len(block)
                        if time.time() - last > 30:
                            last = time.time()
                            print(f"  {path.name}: {have / 1e9:.2f} / {size / 1e9:.2f} GB, "
                                  f"{have / 2**20 / max(last - t0, 1e-6):.1f} MB/s avg", flush=True)
        except OSError as e:                       # requests.RequestException is an OSError too
            if not isinstance(e, requests.RequestException) and e.errno in _NO_RETRY:
                raise OSError(e.errno, f"{path.name}: {e.strerror} while writing {part} (not retried; "
                                       f"{shutil.disk_usage(part.parent).free / 1e9:.2f} GB free)") from e
            wait = min(60, 2 ** attempt)
            print(f"[retry {attempt + 1}/{retries}] {path.name}: {e}; again in {wait}s", flush=True)
            time.sleep(wait)
    got = part.stat().st_size if part.is_file() else 0
    _check(got == size, f"{path.name}: {got} of {size} bytes after {retries} retries; run download() again to resume")
    part.replace(path)
    print(f"[done] {path.name}: {size / 1e9:.2f} GB in {(time.time() - t0) / 60:.1f} min")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(16 * 2**20), b""):
            h.update(block)
    return h.hexdigest()


def download_hf(dest, retries: int = 2, keep_archive: bool = False) -> Path:
    """GLB from the Hugging Face copy [H] into dest/GLB (44.3 GB download, 56.4 GB extracted, peak 100.7 GB).

    hf_hub_download(repo_id, repo_type="dataset", filename="data/GLB.tar.gz", local_dir=dest,
    token=False): with huggingface_hub >= 0.23 such a download is written under dest only [L],
    never into the HF_HOME cache (which other notebooks point at Drive). For the call the Xet
    caches go to dest/.cache/huggingface/xet and the Xet chunk cache is off. No token is sent
    (public dataset). The result must be a plain file (not a symlink) at dest/data/GLB.tar.gz.
    Before use: the size and the LFS sha256 [H] (streamed); '<name>.sha256ok' skips the hash
    next time; a mismatch renames the file to '<name>.bad' and stops. Resumable only once
    complete: huggingface_hub restarts an interrupted download from 0 [L]. Leftovers of an
    interrupted run (its '*.incomplete' temp files under dest/.cache/huggingface/download, a
    dest/.GLB_partial extraction folder) are deleted BEFORE any free-disk check, so they never
    count as used space. Free disk for the archive AND the extracted files is checked before
    the download starts.
    Extraction: the system tar (pigz if installed, else gzip), or Python's tarfile without
    tar, into dest/.GLB_partial, then renamed to dest/GLB and marked '.extraction_complete';
    a GLB folder without the marker is a half-finished extraction and is removed and redone.
    keep_archive=False deletes the archive afterwards.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out, tgz = dest / HF["top"], dest / HF["filename"]
    marker = tgz.with_name(tgz.name + ".sha256ok")
    if (out / DONE_MARKER).is_file():
        print(f"[skip] {out}: extraction complete")
        return out
    if out.exists():
        print(f"[redo] {out} has no {DONE_MARKER}: removing the half-finished extraction")
        shutil.rmtree(out)
    partial = dest / f".{HF['top']}_partial"                # before the free-disk checks: counts as used disk
    if partial.exists():
        print(f"[clean] {partial.name}: left by an interrupted extraction")
        shutil.rmtree(partial)
    _clean_incomplete(dest)                                 # the same for a killed download's temp files
    if tgz.is_file() and tgz.stat().st_size != HF["size"]:
        print(f"[redo] {tgz}: {tgz.stat().st_size:,} B, published {HF['size']:,} B; downloading again")
        tgz.unlink()
    if tgz.is_file() and marker.is_file() and marker.read_text().strip() == HF["sha256"]:
        print(f"[skip] {tgz.name}: complete, sha256 checked before")
    else:
        if not tgz.is_file():
            free = shutil.disk_usage(dest).free
            _check(free > HF["size"] + GLB_BYTES, f"{HF['filename']}: {HF['size'] / 1e9:.1f} GB archive + "
                   f"{GLB_BYTES / 1e9:.1f} GB extracted files need {(HF['size'] + GLB_BYTES) / 1e9:.1f} GB but only "
                   f"{free / 1e9:.1f} GB are free in {dest} (bench_source='zenodo' reads the zip in place)")
            got = _hf_fetch(dest, retries)
            _check(got.stat().st_size == HF["size"], f"{got}: {got.stat().st_size:,} B, published {HF['size']:,} B")
        print(f"[sha256] {tgz.name} ...", flush=True)
        got = _sha256(tgz)
        if got != HF["sha256"]:
            tgz.replace(tgz.with_name(tgz.name + ".bad"))
            raise AssertionError(f"{tgz.name}: sha256 {got} != published {HF['sha256']} [H]; renamed to .bad")
        marker.write_text(got)
        print(f"[ok] {tgz.name} sha256 {got}")
    free = shutil.disk_usage(dest).free
    _check(free > GLB_BYTES, f"extracting {tgz.name} needs {GLB_BYTES / 1e9:.1f} GB, {free / 1e9:.1f} GB free in {dest}")
    _extract_tgz(tgz, dest, HF["top"])
    if not keep_archive:
        tgz.unlink()
        marker.unlink(missing_ok=True)
        print(f"[removed] {tgz} (extraction complete) | free disk {shutil.disk_usage(dest).free / 1e9:.1f} GB")
    return out


def _clean_incomplete(dest: Path) -> None:
    """Delete huggingface_hub temp files of an interrupted download (never resumed across calls [L])."""
    work = dest / ".cache" / "huggingface" / "download"
    for p in sorted(work.rglob("*.incomplete")) if work.is_dir() else []:
        print(f"[clean] {p.name}: {p.stat().st_size / 1e9:.2f} GB of an interrupted download (not resumable)")
        p.unlink()


def _hf_fetch(dest: Path, retries: int) -> Path:
    try:
        import huggingface_hub
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError("bench_source='hf' needs huggingface_hub: pip install huggingface_hub") from e
    ver = tuple(int(x) for x in re.findall(r"\d+", huggingface_hub.__version__)[:2])
    _check(ver >= (0, 23), f"huggingface_hub {huggingface_hub.__version__}: releases before 0.23 put local_dir "
                           f"downloads of big files through the HF_HOME cache; pip install -U huggingface_hub")
    work = dest / ".cache" / "huggingface"                  # stale *.incomplete: removed by download_hf() first
    env = {"HF_XET_CACHE": str(work / "xet"), "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0"}
    saved = {k: os.environ.get(k) for k in env}
    fatal = {"RepositoryNotFoundError", "RemoteEntryNotFoundError", "RevisionNotFoundError", "GatedRepoError",
             "DisabledRepoError"}
    t0 = time.time()
    os.environ.update(env)
    try:
        for attempt in range(retries + 1):
            try:
                got = Path(hf_hub_download(repo_id=HF["repo_id"], repo_type=HF["repo_type"], filename=HF["filename"],
                                           local_dir=dest, token=False))
                break
            except Exception as e:
                if (attempt == retries or (isinstance(e, OSError) and e.errno in _NO_RETRY)
                        or {c.__name__ for c in type(e).__mro__} & fatal):
                    raise
                wait = min(60, 4 * 2 ** attempt)
                print(f"[retry {attempt + 1}/{retries}] {HF['filename']}: {type(e).__name__}: {e}; again from 0 "
                      f"in {wait}s", flush=True)
                time.sleep(wait)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    want = dest / HF["filename"]
    _check(not got.is_symlink() and got.resolve() == want.resolve(),
           f"huggingface_hub returned {got}, not a plain file at {want} (a link into a cache?)")
    print(f"[done] {HF['filename']}: {got.stat().st_size / 1e9:.2f} GB in {(time.time() - t0) / 60:.1f} min")
    return got


def _sha256(path: Path) -> str:
    h, t0, last, done = hashlib.sha256(), time.time(), time.time(), 0
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(16 * 2**20), b""):
            h.update(block)
            done += len(block)
            if time.time() - last > 30:
                last = time.time()
                print(f"  sha256 {done / 1e9:.1f} GB, {done / 2**20 / (last - t0):.0f} MB/s", flush=True)
    return h.hexdigest()


def _checked_members(t: tarfile.TarFile, root: Path):
    """The members of `t`, each checked before it is extracted (on every Python version).

    Refuses absolute names, '..', names that resolve outside `root`, links and special files:
    the archive holds plain folders and files only, so anything else stops the extraction.
    """
    from pathlib import PurePosixPath

    root = root.resolve()
    for m in t:
        name = PurePosixPath(m.name)
        target = (root / m.name).resolve()
        if (name.is_absolute() or ".." in name.parts or (target != root and root not in target.parents)
                or not (m.isfile() or m.isdir())):
            raise RuntimeError(f"unsafe member in the archive, extraction stopped: {m.name!r}")
        yield m


def _extract_tgz(tgz: Path, dest: Path, top: str) -> Path:
    """tgz -> dest/<top> through dest/.<top>_partial; the marker is written only after a complete extraction."""
    tmp, out = dest / f".{top}_partial", dest / top
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    tar, pigz, t0 = shutil.which("tar"), shutil.which("pigz"), time.time()
    if tar:                                   # relative paths: no 'C:' for a GNU tar on Windows to read as a host
        print(f"[extract] {tgz.name} with tar + {'pigz' if pigz else 'gzip'} ...", flush=True)
        # the archive's sha256 is pinned (checked before this); owners and permissions stored in it are not applied
        subprocess.run([tar, "--use-compress-program=pigz" if pigz else "-z", "-x", "--no-same-owner",
                        "--no-same-permissions", "-f", Path(os.path.relpath(tgz, dest)).as_posix(), "-C", tmp.name],
                       cwd=dest, check=True)
    else:
        print(f"[extract] {tgz.name} with Python tarfile ...", flush=True)
        with tarfile.open(tgz, "r:gz") as t:
            try:                                        # every member checked, plus the data filter
                t.extractall(tmp, members=_checked_members(t, tmp), filter="data")
            except TypeError:                           # Python < 3.12 has no filter; the check still runs
                t.extractall(tmp, members=_checked_members(t, tmp))
    got = sorted(p.name for p in tmp.iterdir())
    _check((tmp / top).is_dir(), f"{tgz.name}: no {top}/ folder after extraction (found {got})")
    if got != [top]:
        print(f"[note] {tgz.name} also holds {[g for g in got if g != top]} (not used)")
    (tmp / top).rename(out)
    (out / DONE_MARKER).write_text("ok\n")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[done] extracted to {out} in {(time.time() - t0) / 60:.1f} min")
    return out


# ----------------------------------------------------------------------------- identity with the Zenodo release

class _RangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file through HTTP Range requests, enough for zipfile.

    Every reply must be 206 with exactly the asked bytes of a file of the published size; anything
    else stops, so a server that ignores Range never streams the whole file. Timeouts, dropped
    connections, 429 and 5xx are retried with backoff.
    """

    def __init__(self, url: str, size: int, retries: int = 6, timeout: int = 60, tail: int = 1 << 17):
        super().__init__()
        self.url, self.size, self.retries, self.timeout = url, size, retries, timeout
        self.pos, self.requests, self.fetched = 0, 0, 0
        start = max(size - tail, 0)            # end record, comment, ZIP64 locator and record in one reply
        self._buf = (start, self._get(start, size))

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        pos = {0: 0, 1: self.pos, 2: self.size}[whence] + offset
        if pos < 0:
            raise OSError(errno.EINVAL, f"seek to {pos}")
        self.pos = pos
        return pos

    def read(self, n: int = -1) -> bytes:
        end = self.size if n is None or n < 0 else min(self.pos + n, self.size)
        if end <= self.pos:
            return b""
        s, b = self._buf
        if not (s <= self.pos and end <= s + len(b)):
            self._buf = s, b = self.pos, self._get(self.pos, end)
        out = b[self.pos - s:end - s]
        self.pos = end
        return out

    def readinto(self, buf) -> int:
        data = self.read(len(buf))
        buf[:len(data)] = data
        return len(data)

    def _get(self, a: int, b: int) -> bytes:
        """Bytes a .. b-1."""
        for attempt in range(self.retries + 1):
            try:
                with requests.get(self.url, headers={"Range": f"bytes={a}-{b - 1}"}, stream=True,
                                  timeout=self.timeout) as r:
                    if r.status_code == 429 or r.status_code >= 500:
                        r.raise_for_status()
                    cr = r.headers.get("Content-Range", "")
                    m = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", cr)
                    _check(r.status_code == 206 and m, f"{self.url}: HTTP {r.status_code}, Content-Range {cr!r} to "
                                                       f"a Range request; 206 expected (body not read)")
                    _check(int(m[3]) == self.size, f"{self.url}: file of {int(m[3]):,} B, published {self.size:,} B "
                                                   f"(FILES); another release?")
                    _check((int(m[1]), int(m[2])) == (a, b - 1), f"{self.url}: asked bytes {a}-{b - 1}, got {cr!r}")
                    data = r.content
                _check(len(data) == b - a, f"{self.url}: {len(data)} of {b - a} bytes in the reply")
                self.requests += 1
                self.fetched += len(data)
                return data
            except requests.RequestException as e:
                if attempt == self.retries:
                    raise
                wait = min(60, 2 ** attempt)
                print(f"[retry {attempt + 1}/{self.retries}] bytes {a}-{b - 1}: {e}; again in {wait}s", flush=True)
                time.sleep(wait)


def zenodo_directory(url: str = FILES["bench"]["url"], size: int = FILES["bench"]["size"], retries: int = 6,
                     timeout: int = 60) -> dict[str, tuple[int, int]]:
    """name -> (size, CRC-32) of every file in the official GLB zip, from its central directory only [D].

    HTTP range requests (the last 128 KB, then the directory; a few MB of 44.3 GB), parsed by
    Python's zipfile (ZIP64 included). Every reply must be 206 for a file of the published size.
    """
    t0 = time.time()
    with _RangeFile(url, size, retries, timeout) as fh, zipfile.ZipFile(fh) as z:
        out = {i.filename: (i.file_size, i.CRC) for i in z.infolist() if not i.is_dir()}
        n, mb = fh.requests, fh.fetched / 1e6
    per = pd.Series(["/".join(m.groups()) for m in map(_TIF.match, out) if m], dtype=str).value_counts()
    print(f"[zenodo] central directory: {len(out):,} files ({n} range requests, {mb:.1f} MB, {time.time() - t0:.0f} s);"
          f" .tif per folder {per.sort_index().to_dict()}")
    return out


def _crc_file(path: Path) -> int:
    c = 0
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(2**20), b""):
            c = zlib.crc32(block, c)
    return c


def _hf_stats(raw_root: Path, real: dict[str, str]) -> dict[str, tuple[int, int]]:
    """Zenodo name -> (size, st_mtime_ns) of that file in the extracted Hugging Face copy."""
    out = {}
    for n, p in real.items():
        st = (raw_root / p).stat()
        out[n] = (st.st_size, st.st_mtime_ns)
    return out


def _stamp(stats: dict[str, tuple[int, int]]) -> str:
    """sha256 over (name, size, mtime in ns) of every Zenodo-named .tif, in name order."""
    h = hashlib.sha256()
    for n in sorted(stats):
        if _TIF.match(n):
            h.update(f"{n}\t{stats[n][0]}\t{stats[n][1]}\n".encode())
    return h.hexdigest()


def _identity_record(raw_root: Path, stats: dict[str, tuple[int, int]]) -> dict | None:
    """check_identity()'s record if it still matches the files on disk, else None.

    It matches only if the file count, the total bytes AND the stamp (_stamp: every checked
    file's name, size and modification time in ns) are unchanged, so a file rewritten with the
    same size voids it too. Not seen: a change that keeps the size and restores the old
    modification time; recheck=True in check_identity() re-reads every file. A record without
    a stamp (written by older code) never matches.
    """
    try:
        rec = json.loads((raw_root / HF["top"] / SAME_MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    skip = set(rec.get("extra", []))                   # Hugging Face-only files, left out by check_identity()
    stats = {n: v for n, v in stats.items() if n not in skip}
    tif = [s for n, (s, _) in stats.items() if _TIF.match(n)]
    now = (len(tif), sum(tif), _stamp(stats))
    return rec if (rec.get("files"), rec.get("bytes"), rec.get("stamp")) == now else None


def check_identity(raw_root, reference: dict | None = None, workers: int | None = None,
                   recheck: bool = False) -> dict:
    """Require the Hugging Face GLB in raw_root/GLB to be the Zenodo release, file by file.

    Every image and mask of the Zenodo release present, with the same size and CRC-32 as in
    the central directory of Glacial_Lake_Bench.zip (`reference`: name -> (size, CRC-32);
    default zenodo_directory()). CRC-32 is computed over every extracted file (chunked,
    `workers` threads). A missing or changed file stops with examples. Files the Hugging Face
    copy has and Zenodo does not are listed under 'extra' in the record and left out: _Src
    never reads them, so the data is exactly the Zenodo release. A pass writes GLB/.zenodo_identical
    with the file count, the bytes and a stamp over every file's name, size and modification
    time (taken before the CRC pass), which convert() requires. A later change voids it: a file
    added, removed, renamed, resized, or rewritten (new modification time) -- see
    _identity_record() for the one case not seen. With a valid record the check is skipped
    unless recheck=True.
    """
    raw_root = Path(raw_root)
    real, _ = _hf_layout(raw_root)
    stats = _hf_stats(raw_root, real)
    sizes = {n: s for n, (s, _) in stats.items()}
    rec = _identity_record(raw_root, stats)
    if rec and not recheck:
        print(f"[skip] {rec['summary']} (checked {rec['checked']})")
        return rec
    (raw_root / HF["top"] / SAME_MARKER).unlink(missing_ok=True)
    ref = {n: v for n, v in (zenodo_directory() if reference is None else reference).items() if _TIF.match(n)}
    _check(ref, "the reference has no Glacial_Lake_Bench/<img_dir|ann_dir>/<split>/*.tif member")
    missing, extra = sorted(set(ref) - set(real)), sorted(set(real) - set(ref))
    same_size = sorted(n for n in set(ref) & set(real) if sizes[n] == ref[n][0])
    other_size = sorted(n for n in set(ref) & set(real) if sizes[n] != ref[n][0])
    t0, workers = time.time(), workers or min(8, os.cpu_count() or 1)
    print(f"[crc32] {len(same_size):,} files of {HF['top']}/, {workers} threads ...", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        crc = dict(zip(same_size, ex.map(_crc_file, [raw_root / real[n] for n in same_size])))
    other_crc = [n for n in same_size if crc[n] != ref[n][1]]
    ok = set(same_size) - set(other_crc)
    pairs = [n for n in ref if "/img_dir/" in n]
    good = [n for n in pairs if n in ok and n.replace("/img_dir/", "/ann_dir/", 1) in ok]
    per = pd.Series([n.split("/")[2] for n in pairs], dtype=str).value_counts()
    summary = (f"GLB from Hugging Face: {len(good):,} of {len(pairs):,} pairs byte-identical to the Zenodo release "
               f"(size + CRC-32 of every image and mask; " + ", ".join(f"{s} {per.get(s, 0):,}" for s in BENCH_SPLITS)
               + f"; {(time.time() - t0) / 60:.1f} min)")
    hf_name = lambda n: real.get(n, n)
    if extra:                                  # in the Hugging Face copy only: listed, never read
        n_x = sum("/img_dir/" in n for n in extra)
        summary += f"; left out {n_x} extra image/mask pairs of the Hugging Face copy that Zenodo does not have"
        print(f"[note] {len(extra)} files of the Hugging Face copy are not in the Zenodo release and are left out, "
              f"e.g. {', '.join(hf_name(n) for n in extra[:4])}")
    print(summary)
    problems = [f"{len(v)} {what}, e.g. {', '.join(eg)}" for v, what, eg in (
        (missing, "Zenodo files missing in the Hugging Face copy", missing[:4]),
        (other_size, "files of another size", [f"{hf_name(n)} {sizes[n]:,} B vs {ref[n][0]:,} B" for n in other_size[:4]]),
        (other_crc, "files with the same size but another CRC-32",
         [f"{hf_name(n)} 0x{crc[n]:08x} vs 0x{ref[n][1]:08x}" for n in other_crc[:4]])) if v]
    _check(not problems, "GLB from Hugging Face is NOT the Zenodo release (Zenodo names: img_dir = images/, "
                         "ann_dir = the mask folder); conversion stopped: " + "; ".join(problems))
    rec = {"summary": summary, "pairs": {s: int(per.get(s, 0)) for s in BENCH_SPLITS}, "files": len(ref),
           "bytes": int(sum(sizes[n] for n in ref)), "stamp": _stamp({n: stats[n] for n in ref}),
           "extra": extra, "extra_hf_paths": [hf_name(n) for n in extra],
           "checked": time.strftime("%Y-%m-%d %H:%M:%S"),
           "reference": "zenodo_directory()" if reference is None else "given"}
    (raw_root / HF["top"] / SAME_MARKER).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


# ----------------------------------------------------------------------------- reading

class _Src:
    """One archive: the official zip read in place, its extracted top folder, or (GLB) the Hugging Face copy."""

    def __init__(self, raw_root: Path, key: str):
        f = FILES[key]
        zp, top = raw_root / f["name"], raw_root / f["top"]
        self.key, self.zip_path, self.base, self._local, self._crc = key, None, raw_root, threading.local(), {}
        self._real, self.from_hf, self.hf_check = {}, False, None      # Zenodo name -> path in the HF copy
        if zp.is_file():
            self.zip_path, self.where = zp, str(zp)
            with zipfile.ZipFile(zp) as z:
                infos = [i for i in z.infolist() if not i.is_dir()]
            self.sizes = {i.filename: i.file_size for i in infos}
            self._crc = {i.filename: i.CRC for i in infos}
        elif top.is_dir():
            self.where = str(top)
            self.sizes = {p.relative_to(raw_root).as_posix(): p.stat().st_size
                          for p in sorted(top.rglob("*")) if p.is_file()}
        elif key == "bench" and (raw_root / HF["top"]).is_dir():
            self._real, extra = _hf_layout(raw_root)
            self.from_hf, self.where = True, f"{raw_root / HF['top']} (Hugging Face {HF['repo_id']}, Zenodo names)"
            stats = _hf_stats(raw_root, self._real)
            self.sizes = {n: s for n, (s, _) in stats.items()}
            self.sizes.update({p: (raw_root / p).stat().st_size for p in extra})
            self.hf_check = _identity_record(raw_root, stats)
            for n in (self.hf_check or {}).get("extra", []):         # not in the Zenodo release: never read
                self._real.pop(n, None)
                self.sizes.pop(n, None)
        else:
            self.where, self.sizes = None, None

    def read(self, name: str) -> bytes:
        if self.zip_path is None:
            return (self.base / self._real.get(name, name)).read_bytes()
        z = getattr(self._local, "z", None)
        if z is None:                                   # one handle per thread
            z = self._local.z = zipfile.ZipFile(self.zip_path)
        return z.read(name)

    def crc32(self, name: str) -> str:
        """CRC-32 of a member as in meta 'crc32': from the zip directory, else computed from the file."""
        c = self._crc.get(name)
        return f"0x{(zlib.crc32(self.read(name)) if c is None else c):08x}"


def _hf_layout(raw_root: Path) -> tuple[dict[str, str], list[str]]:
    """Zenodo member name -> path of that file in the extracted Hugging Face copy; + its other files.

    Split folders train, val or validation, test (any other folder stops); in each, images/
    and the ONE sibling folder whose .tif names equal those of images/ (the masks; its name
    is not assumed).
    """
    top = raw_root / HF["top"]
    _check((top / DONE_MARKER).is_file(), f"{top}: no {DONE_MARKER}, a half-finished extraction; run download_hf() again")
    rel = lambda p: p.relative_to(raw_root).as_posix()
    tifs = lambda d: {p.name for p in d.iterdir() if p.is_file() and p.name.endswith(".tif")}
    real, extra, seen = {}, [], {}
    for d in sorted(top.iterdir()):
        if not d.is_dir():
            if d.name not in (DONE_MARKER, SAME_MARKER):
                extra.append(rel(d))
            continue
        split = HF_SPLITS.get(d.name.lower())
        _check(split, f"{rel(d)}/: not a split folder (expected one of {sorted(HF_SPLITS)})")
        _check(split not in seen, f"{rel(d)}/: split '{split}' also in {HF['top']}/{seen.get(split)}/")
        seen[split] = d.name
        img = d / "images"
        _check(img.is_dir(), f"{rel(d)}/: no images/ folder [H]")
        names = tifs(img)
        subs = {s.name: tifs(s) for s in sorted(d.iterdir()) if s.is_dir() and s.name != "images"}
        hit = [s for s, v in subs.items() if v == names]
        _check(len(hit) == 1, f"{rel(d)}/: {len(hit)} folders beside images/ hold exactly its {len(names)} .tif "
               f"names (that is how the mask folder is found); other folders: "
               + ("; ".join(f"{s}/ {len(v)} .tif, missing e.g. {sorted(names - v)[:2]}, extra e.g. {sorted(v - names)[:2]}"
                            for s, v in subs.items()) or "none"))
        msk = d / hit[0]
        for n in sorted(names):
            real[f"{FILES['bench']['top']}/img_dir/{split}/{n}"] = rel(img / n)
            real[f"{FILES['bench']['top']}/ann_dir/{split}/{n}"] = rel(msk / n)
        extra += [rel(p) for p in sorted(d.rglob("*")) if p.is_file() and not (p.parent in (img, msk) and p.name in names)]
    return real, extra


def _records(src: _Src) -> list[dict]:
    """Image/mask pairs from the member names, sorted by name; unknown .tif names stop the run."""
    recs = []
    for n in sorted(src.sizes):
        if src.key == "bench":
            if not (n.startswith("Glacial_Lake_Bench/img_dir/") and n.endswith(".tif")):
                continue
            m = _GLB.match(n)
            _check(m, f"{n}: name does not follow the documented convention "
                      "<REGION>_<S2A|S2B>_<MGRS>_<YYYYMMDD>_<n>_L2A_<k>.tif (preprint 2.5)")
            split, reg, scene, _, tile, d = m.groups()
            _check(reg in REGIONS, f"{n}: region code {reg} not among the 18 codes of the Zenodo record")
            recs.append({"official": split, "image": n, "mask": n.replace("/img_dir/", "/ann_dir/", 1),
                         "region": reg, "region_source": "file name", "scene": scene, "tile": tile, "date": d})
        else:
            if not (n.startswith("Glacial-Lake-Challenge/image/") and n.endswith(".tif")):
                continue
            m = _GLC.match(n)
            _check(m, f"{n}: name does not follow <S2A|S2B>_<MGRS>_<YYYYMMDD>_<n>_L2A_<k>.tif")
            scene, _, tile, d = m.groups()
            recs.append({"official": "challenge", "image": n, "mask": n.replace("/image/", "/mask/", 1),
                         "region": "unknown", "region_source": "none", "scene": scene, "tile": tile, "date": d})
    missing = [r["mask"] for r in recs if r["mask"] not in src.sizes]
    _check(not missing, f"{len(missing)} images without a mask of the same name (preprint 2.5: "
                        f"'identical filenames'), e.g. {missing[:3]}")
    return recs


def _infer_regions(glc: list[dict], glb: list[dict]) -> None:
    """GLC names carry no region: use GLB chips of the same S2 scene, else the same MGRS tile, if unique."""
    by_scene, by_tile = {}, {}
    for r in glb:
        by_scene.setdefault(r["scene"], set()).add(r["region"])
        by_tile.setdefault(r["tile"], set()).add(r["region"])
    for r in glc:
        s, t = by_scene.get(r["scene"], set()), by_tile.get(r["tile"], set())
        if len(s) == 1:
            r["region"], r["region_source"] = next(iter(s)), "GLB scene"
        elif not s and len(t) == 1:
            r["region"], r["region_source"] = next(iter(t)), "GLB tile"


def _leaks(recs: list[dict], split_of) -> dict:
    """Name-only facts behind the choice of splits: scene sharing with train, tiles spanning regions."""
    by = {}
    for r in recs:
        by.setdefault(split_of(r), []).append(r)
    train_scenes = {r["scene"] for r in by.get("train", [])}
    tiles = {}
    for r in recs:
        if r["region_source"] == "file name":
            tiles.setdefault(r["tile"], set()).add(r["region"])
    return {"scene_shared_with_train": {sp: f"{sum(r['scene'] in train_scenes for r in rs)}/{len(rs)}"
                                        for sp, rs in sorted(by.items()) if sp != "train"},
            "tiles_in_several_regions": pd.Series(["/".join(sorted(v)) for v in tiles.values() if len(v) > 1],
                                                  dtype=str).value_counts().to_dict()}


def _read(src: _Src, rec: dict, bands=S1_BANDS) -> dict:
    """Decode one pair. bands=None reads all 11 (inventory / value-state sample)."""
    from rasterio.io import MemoryFile
    raw = src.read(rec["image"])
    with MemoryFile(raw) as mf, mf.open() as ds:
        _check(ds.count == N_BANDS, f"{rec['image']}: {ds.count} bands; documented 11 (Blue ... Elevation, VV, VH)")
        _check((ds.height, ds.width) == (CHIP, CHIP), f"{rec['image']}: {ds.height}x{ds.width}; documented 256x256")
        _check(all(np.dtype(ds.dtypes[b - 1]).kind == "f" for b in S1_BANDS),
               f"{rec['image']}: SAR bands dtype {ds.dtypes[9]}; float expected")
        img = ds.read(list(bands) if bands else None).astype(np.float32)
        info = {"crs": ds.crs.to_string() if ds.crs else "", "bounds": tuple(ds.bounds), "res": tuple(ds.res),
                "nodata": ds.nodata, "desc": tuple(ds.descriptions), "dtype": ds.dtypes[0],
                "tags": ds.tags() if bands is None else {}}
    with MemoryFile(src.read(rec["mask"])) as mf, mf.open() as ds:
        _check(ds.count == 1 and (ds.height, ds.width) == (CHIP, CHIP),
               f"{rec['mask']}: {ds.count} band(s) {ds.height}x{ds.width}; one 256x256 band expected")
        mask = ds.read(1)
        mbounds = tuple(ds.bounds) if ds.crs else None
        info["mask_dtype"] = ds.dtypes[0]
    if mbounds is not None and info["crs"]:
        _check(np.allclose(mbounds, info["bounds"], atol=1e-3),
               f"{rec['mask']}: footprint {mbounds} differs from its image {info['bounds']}")
    info["crc32"] = f"0x{zlib.crc32(raw):08x}"
    return {"img": img, "mask": mask, "info": info}


def _pol(desc: tuple, name: str) -> str:
    """Polarisation named by the band descriptions, checked against the documented order; '' if none."""
    found = [next((p for p in ("VV", "VH", "HH", "HV") if re.search(rf"\b{p}\b", (d or "").upper())), "")
             for d in desc]
    if not any(found):
        return ""
    _check(found[9] in ("VV", "HH") and found[10] in ("VH", "HV") and not any(found[:9]),
           f"{name}: band descriptions {desc} contradict the documented order (co-pol band 10, cross-pol band 11)")
    return f"{found[9]}/{found[10]}"


# ----------------------------------------------------------------------------- value state

def _squashed(v: np.ndarray) -> bool:
    """A 0-1 band stretched against a far-off fill (e.g. -32768): its non-zero values all sit near 1."""
    nz = v[v != 0]
    return bool(nz.size and v.min() >= 0 and v.max() <= 1 + 1e-6
                and (nz.min() > SQUASH_MIN or np.median(nz) > SQUASH_MED))


def _stats(rec: dict, chip: dict) -> list[dict]:
    """Per SAR band of one chip (all 11 bands read): counts and extremes over finite, non-nodata values."""
    img, nd = chip["img"], chip["info"]["nodata"]
    _check(img.shape[0] == N_BANDS, "value statistics need all 11 bands")
    fill = img == PC_NODATA
    valid = np.isfinite(img) & ~fill
    if nd is not None and not np.isnan(nd):
        valid &= img != nd
    fin = img[valid]
    stack_mn, stack_mx = (float(fin.min()), float(fin.max())) if fin.size else (np.nan, np.nan)
    idx = [b - 1 for b in S1_BANDS]
    pair = img[idx][valid[idx]]
    rows = []
    for b in S1_BANDS:
        v = img[b - 1][valid[b - 1]]
        rows.append({"image": rec["image"], "band": b, "n": int(v.size), "n_fill": int(fill[b - 1].sum()),
                     "mn": float(v.min()) if v.size else np.nan, "mx": float(v.max()) if v.size else np.nan,
                     "med": float(np.median(v)) if v.size else np.nan,
                     "n_neg": int((v < 0).sum()), "n_eq0": int((v == 0).sum()), "n_eq1": int((v == 1).sum()),
                     "squashed": _squashed(v),
                     "pair_mn": float(pair.min()) if pair.size else np.nan,
                     "pair_mx": float(pair.max()) if pair.size else np.nan,
                     "stack_mn": stack_mn, "stack_mx": stack_mx})
    return rows


def _decide(stats: list[dict]) -> tuple[str, dict]:
    """Value state of the SAR bands from sample statistics (rules in the module docstring)."""
    s = pd.DataFrame(stats)
    fill_share = round(float(s.n_fill.sum() / max(s.n_fill.sum() + s.n.sum(), 1)), 6)
    s = s[s.n > 0]
    _check(len(s), "no finite SAR value in the sample")
    v = s[s.mx > s.mn]                    # constant chip-bands (e.g. no SAR at all) say nothing about a stretch
    _check(len(v), "every sampled SAR band is constant")
    near = lambda col, x: round(float(((col - x).abs() <= 1e-3).mean()), 4)   # tolerates a (hi - lo + eps)
    ev = {"chip_bands": int(len(s)), "constant_chip_bands": int(len(s) - len(v)),
          "share_pc_nodata_-32768": fill_share, "squashed_chip_bands": int(v.squashed.sum()),
          "share_negative": round(float(s.n_neg.sum() / s.n.sum()), 6),
          "min": float(s.mn.min()), "max": float(s.mx.max()), "median_of_medians": float(v.med.median()),
          "share_band_max1": near(v.mx, 1), "share_band_min0": near(v.mn, 0),
          "share_pair_max1": near(v.pair_mx, 1), "share_stack_max1": near(v.stack_mx, 1),
          "median_share_eq1": round(float((v.n_eq1 / v.n).median()), 6),
          "median_share_eq0": round(float((v.n_eq0 / v.n).median()), 6)}
    if ev["share_negative"] > 0.5:
        _check(-60 <= ev["median_of_medians"] <= 10, f"SAR values mostly negative but median "
               f"{ev['median_of_medians']:.2f} is not a dB backscatter level: {ev}")
        return "db", ev
    if ev["min"] >= 0 and ev["max"] <= 1:
        # A per-chip stretch leaves every chip-band with a maximum of 1 (within 1e-3; real gamma0 almost
        # never); the minimum is not used because a nodata tag of 0 would hide it.
        if ev["share_band_max1"] >= 0.9:
            return ("pct_stretch" if ev["median_share_eq1"] >= 0.005 else "minmax"), ev
        _check(ev["share_pair_max1"] < 0.9, f"both SAR bands look stretched TOGETHER to 0-1 "
               f"(not handled; decide by hand): {ev}")
        _check(ev["share_stack_max1"] < 0.9, f"all 11 bands look stretched TOGETHER to 0-1, so the "
               f"SAR values are scaled by other bands (not handled; decide by hand): {ev}")
        raise AssertionError(
            f"ambiguous SAR value state: every sampled value lies in 0-1 but only {ev['share_band_max1']:.0%} of "
            f"the chip-bands reach 1. That is what a 0-1 stretch over a whole scene or the whole dataset looks "
            f"like (the units are then lost; storing 10*log10 would give dB minus an unknown constant), and "
            f"also unusually dark linear gamma0 (real RTC gamma0 over mountains normally exceeds 1 somewhere). "
            f"Inspect inventory(), then force s1_scale='linear' only if these are unstretched gamma0: {ev}")
    if ev["share_negative"] < 0.01:
        _check(1e-5 < ev["median_of_medians"] < 10, f"non-negative SAR values but median "
               f"{ev['median_of_medians']:.3g} is not a linear gamma0 level: {ev}")
        return "linear", ev
    raise AssertionError(f"SAR value state not recognised (dB / linear / min-max / percentile stretch): {ev}")


def _to_units(x: np.ndarray, state: str, nodata, to_log: bool, name: str, flags: dict | None = None) -> np.ndarray:
    """(2,H,W) raw SAR -> stored values with NaN = no data (see docstring); flags['squashed'] = bands set to NaN."""
    x = np.array(x, np.float32)
    if nodata is not None and not np.isnan(nodata):
        x[x == nodata] = np.nan
    x[x == PC_NODATA] = np.nan
    x[~np.isfinite(x)] = np.nan
    f = x[np.isfinite(x)]
    squashed = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        if state == "db":
            _check(f.size == 0 or (f.min() >= DB_RANGE[0] and f.max() <= DB_RANGE[1]),
                   f"{name}: SAR values {f.min():.2f}..{f.max():.2f} outside {DB_RANGE} (decided: dB)")
        elif state == "linear":
            x[~(x > 0)] = np.nan
            x = 10.0 * np.log10(x)
        elif state in ("minmax", "pct_stretch"):
            _check(f.size == 0 or (f.min() >= 0 and f.max() <= 1 + 1e-6),
                   f"{name}: SAR values {f.min():.4g}..{f.max():.4g} outside 0-1 (decided: {state})")
            if state == "minmax":
                x[x == 0] = np.nan
                for b in range(len(x)):
                    if _squashed(x[b][np.isfinite(x[b])]):
                        x[b] = np.nan
                        squashed += 1
                if to_log:
                    x = 10.0 * np.log10(x)
        else:
            raise ValueError(f"s1_scale {state!r} not in {sorted(UNITS)}")
    x[~np.isfinite(x)] = np.nan
    if flags is not None:
        flags["squashed"] = squashed
    return x


def _even(recs: list[dict], n: int) -> list[dict]:
    if len(recs) <= n:
        return list(recs)
    return [recs[i] for i in np.unique(np.linspace(0, len(recs) - 1, n).round().astype(int))]


def _open_all(raw_root, need_challenge: bool, need_check: bool = False) -> tuple[_Src, _Src | None]:
    raw_root = Path(raw_root)
    bench = _Src(raw_root, "bench")
    _check(bench.sizes is not None, f"neither {FILES['bench']['name']} nor {FILES['bench']['top']}/ nor the Hugging "
                                    f"Face copy {HF['top']}/ in {raw_root}")
    _check(not (need_check and bench.from_hf and bench.hf_check is None),
           f"{raw_root / HF['top']}: not (or no longer) checked against the Zenodo release; run "
           f"check_identity({str(raw_root)!r}) first (download(..., bench_source='hf') does)")
    ch = _Src(raw_root, "challenge")
    if ch.sizes is None:
        _check(not need_challenge, f"neither {FILES['challenge']['name']} nor {FILES['challenge']['top']}/ "
                                   f"in {raw_root} (pass include_challenge=False to skip it)")
        ch = None
    return bench, ch


def _check_counts(recs: list[dict], keys=tuple(EXPECTED)) -> None:
    """Every expected split (not only those present) must have its archive count."""
    got = pd.Series([r["official"] for r in recs], dtype=str).value_counts().to_dict()
    bad = [f"{k}: {got.get(k, 0)} pairs, expected {EXPECTED[k]}" for k in keys if got.get(k, 0) != EXPECTED[k]]
    _check(not bad, f"pair counts differ from the archive directory of record 17917359 (partial download or "
                    f"extraction?): {'; '.join(bad)}")


# ----------------------------------------------------------------------------- inventory

def inventory(raw_root, sample: int = 24, check_counts: bool = True) -> dict:
    """Print what is on disk and assert the documented layout; decide the SAR value state on a sample.

    Reads `sample` evenly spaced chips per archive plus two chips of every distinct file size
    (the archive holds images of 2,885,555/2,885,556 B and masks of 65,943/65,944/72,087/72,088 B).
    Also prints the name-only facts behind the split choice (scene sharing, tiles in several
    regions) and, in the "minmax" state, how many sampled SAR bands were squashed by a fill.
    """
    bench, ch = _open_all(raw_root, need_challenge=False)
    out = {"sources": {}, "splits": {}, "regions": {}, "extra_files": [], "sample": [], "state": {}}
    all_recs = {}
    for src in (bench, ch):
        if src is None:
            print(f"[missing] {FILES['challenge']['name']} / {FILES['challenge']['top']}/")
            continue
        recs = _records(src)
        all_recs[src.key] = recs
        out["sources"][src.key] = src.where
        tifs = {r["image"] for r in recs} | {r["mask"] for r in recs}
        extra = sorted(n for n in src.sizes if n not in tifs)
        out["extra_files"] += extra
        sizes = pd.Series([src.sizes[r["image"]] for r in recs]).value_counts().to_dict()
        msizes = pd.Series([src.sizes[r["mask"]] for r in recs]).value_counts().to_dict()
        print(f"{src.key}: {src.where}\n  pairs {len(recs)}; image file sizes {sizes}; mask file sizes {msizes}"
              f"\n  other files: {extra}")
        if src.from_hf:
            print("  " + (src.hf_check["summary"] if src.hf_check else "[warn] not yet checked against the Zenodo "
                          "release: check_identity(raw_root); convert() refuses until then"))
        cnt = pd.crosstab(pd.Series([r["region"] for r in recs], name="region"),
                          pd.Series([r["official"] for r in recs], name="split"))
        print(cnt.to_string())
        out["splits"].update(pd.Series([r["official"] for r in recs]).value_counts().to_dict())
        out["regions"][src.key] = cnt.to_dict()
        dates = sorted(r["date"] for r in recs)
        print(f"  S2 dates {dates[0]} .. {dates[-1]}")
    if check_counts:
        _check_counts(all_recs["bench"] + all_recs.get("challenge", []),
                      BENCH_SPLITS + (("challenge",) if "challenge" in all_recs else ()))
    if "challenge" in all_recs:
        _infer_regions(all_recs["challenge"], all_recs["bench"])
        rs = pd.Series([r["region_source"] for r in all_recs["challenge"]]).value_counts().to_dict()
        print(f"challenge region taken from: {rs}")
        out["challenge_region_source"] = rs
    out["leaks"] = _leaks(all_recs["bench"] + all_recs.get("challenge", []), lambda r: r["official"])
    print(f"chips whose Sentinel-2 scene also gives train chips: {out['leaks']['scene_shared_with_train']}\n"
          f"MGRS tiles with chips of several regions: {out['leaks']['tiles_in_several_regions']}")
    for src in (bench, ch):
        if src is None:
            continue
        recs = all_recs[src.key]
        pick = _even(recs, sample)
        seen = {r["image"] for r in pick}
        by_size = {}                                          # every distinct file size, twice
        for r in recs:
            for k in (src.sizes[r["image"]], -src.sizes[r["mask"]]):
                if len(by_size.setdefault(k, [])) < 2:
                    by_size[k].append(r)
        for v in by_size.values():
            for r in v:
                if r["image"] not in seen:
                    seen.add(r["image"])
                    pick.append(r)
        stats = []
        for i, r in enumerate(pick):
            c = _read(src, r, bands=None)
            inf, img = c["info"], c["img"]
            if i == 0:
                print(f"{src.key} first chip {r['image']}\n  dtype {inf['dtype']} crs {inf['crs']} res {inf['res']} "
                      f"nodata {inf['nodata']} mask dtype {inf['mask_dtype']}\n  band descriptions {inf['desc']}"
                      f"\n  tags {inf['tags']}")
            st = _stats(r, c)
            stats += st
            mv = np.unique(c["mask"]).tolist()
            _check(set(mv) <= {0, 1}, f"{r['mask']}: mask values {mv}; documented 0 background, 1 lake")
            _check(np.allclose(np.abs(inf["res"]), PIXEL_M, atol=1e-3),
                   f"{r['image']}: pixel size {inf['res']}; documented 10 m grid")
            finite = img[np.isfinite(img)]
            out["sample"].append({
                "archive": src.key, "file": r["image"].split("/")[-1], "split": r["official"],
                "size": src.sizes[r["image"]], "mask_size": src.sizes[r["mask"]], "mask_values": mv,
                "lake_share": float((c["mask"] == 1).mean()), "pol": _pol(inf["desc"], r["image"]),
                "nan_share_s1": float(np.isnan(img[[b - 1 for b in S1_BANDS]]).mean()),
                **{f"b{s['band']}_{k}": s[k] for s in st for k in ("mn", "mx", "med", "n_eq0", "n_fill", "squashed")},
                "all_bands_min": float(finite.min()) if finite.size else np.nan,
                "all_bands_max": float(finite.max()) if finite.size else np.nan})
        state, ev = _decide(stats)
        out["state"][src.key] = {"state": state, "evidence": ev}
        print(f"{src.key}: SAR value state '{state}' -> stored units '{UNITS[state]}'\n  evidence {ev}")
        if state == "minmax":
            print(f"  {ev['squashed_chip_bands']} of {ev['chip_bands']} sampled SAR chip-bands look squashed by a fill "
                  f"that set the stretch minimum (convert() stores them as NaN, meta s1_squashed)")
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(pd.DataFrame(out["sample"]).drop(columns=["file"]).round(4).to_string(index=False))
    if len(out["state"]) == 2:
        _check(out["state"]["bench"]["state"] == out["state"]["challenge"]["state"],
               f"GLB and GLC differ in SAR value state: {out['state']}")
    return out


# ----------------------------------------------------------------------------- convert

def convert(raw_root, out_root, s1_scale: str = "auto", minmax_to_log: bool = False,
            holdout_regions: tuple = (), ignore_no_sar: bool = True, include_challenge: bool = True,
            limit_per_split: int | None = None, check_counts: bool = True, sample: int = 120,
            workers: int | None = None, overwrite: bool = True) -> dict:
    """Write the prepared format (see module docstring). Returns the manifest.

    s1_scale "auto" decides the value state on `sample` evenly spaced chips per archive;
    or force one of "minmax", "pct_stretch", "linear", "db" (every chip is still checked).
    minmax_to_log: in the "minmax" state store 10*log10(value) (units "dB_rel_chip_max";
    see the docstring for when that is exact).
    holdout_regions: GLB region codes moved, from every official folder, to "test_region_<code>";
    train chips byte-identical (CRC-32) to a held-out chip are dropped and listed in the notes.
    limit_per_split: first N chips per output split (dry run; the count check is then skipped).
    workers: decoding threads (order and output do not depend on it).
    overwrite: replace an earlier conversion in out_root; False refuses when a finished one
    (manifest.json) exists. Files of the prepared layout left in out_root/glacial_lakes
    (split arrays, meta, hidden part files of a crashed run) are removed first either way.
    """
    for r in holdout_regions:
        _check(r in REGIONS, f"holdout region {r!r} not in {sorted(REGIONS)}")
    bench, ch = _open_all(raw_root, need_challenge=include_challenge, need_check=True)
    glb = _records(bench)
    glc = _records(ch) if include_challenge else []
    if check_counts and limit_per_split is None:
        _check_counts(glb + glc, BENCH_SPLITS + (("challenge",) if include_challenge else ()))
    _infer_regions(glc, glb)
    workers = workers or min(4, os.cpu_count() or 1)

    if s1_scale == "auto":
        decided = {}
        for src, recs in ((bench, glb), (ch, glc)):
            if recs:
                st = [row for r in _even(recs, sample) for row in _stats(r, _read(src, r, bands=None))]
                decided[src.key] = _decide(st)
                print(f"{src.key}: SAR value state '{decided[src.key][0]}' on {min(sample, len(recs))} chips")
        states = {v[0] for v in decided.values()}
        _check(len(states) == 1, f"GLB and GLC differ in SAR value state: {decided}")
        state = states.pop()
        evidence = {k: v[1] for k, v in decided.items()}
    else:
        _check(s1_scale in UNITS, f"s1_scale {s1_scale!r} not in {sorted(UNITS)} or 'auto'")
        state, evidence = s1_scale, {"forced": s1_scale}
    to_log = bool(minmax_to_log) and state == "minmax"
    units = "dB_rel_chip_max" if to_log else UNITS[state]

    out_dir = Path(out_root) / DATASET
    if (out_dir / "manifest.json").is_file():
        _check(overwrite, f"{out_dir / 'manifest.json'} exists; pass overwrite=True to replace that conversion")
        print(f"[replace] earlier conversion in {out_dir}")
    stale = sorted({p for pat in ("manifest.json", "*_images.npy", "*_labels.npy", "*_meta.csv", ".*.part*.npy")
                    for p in out_dir.glob(pat)}) if out_dir.is_dir() else []
    for p in stale:
        p.unlink()
    if stale:
        print(f"[clean] removed {len(stale)} file(s) of an earlier conversion from {out_dir}")

    def out_split(r):
        if r["official"] == "challenge":
            return "test_challenge"
        return f"test_region_{r['region']}" if r["region"] in holdout_regions else r["official"]

    held = [r for r in glb if r["region"] in holdout_regions]
    if held:
        print(f"[holdout] CRC-32 of {len(held)} held-out chips ...", flush=True)
    held_crc = {bench.crc32(r["image"]) for r in held}
    leaks = _leaks(glb + glc, out_split)

    order = ["train", "val", "test"] + [f"test_region_{h}" for h in holdout_regions] + ["test_challenge"]
    jobs = [(out_split(r), r, ch if r["official"] == "challenge" else bench) for r in glb + glc]
    jobs.sort(key=lambda j: (order.index(j[0]), j[1]["image"]))
    if limit_per_split is not None:
        kept, n = [], {}
        for j in jobs:
            n[j[0]] = n.get(j[0], 0) + 1
            if n[j[0]] <= limit_per_split:
                kept.append(j)
        jobs = kept

    per_state = {"minmax": ", exact 0 (no-data fill and the chip minimum), whole bands squashed by a fill "
                           "(s1_squashed)", "linear": ", non-positive linear values"}.get(state, "")
    notes = (f"SAR = bands 10 (co-pol, VV or HH) and 11 (cross-pol, VH or HV) of the 11-band GLB/GLC "
             f"images. Value state '{state}' ({'decided on a sample' if s1_scale == 'auto' else 'forced'}), "
             f"stored as '{units}'. Evidence: {evidence}. No data -> NaN: NaN/inf, the GeoTIFF nodata value, "
             f"exact -32768 (PC RTC nodata){per_state}. Labels 0/1 from the Zhang et al. "
             f"(2024) optical inventory; 255 where both SAR channels are no data: {ignore_no_sar}. Splits: "
             f"official GLB folders (the preprint's random 80/10/10 split; it calls leave-one-region-out the "
             f"primary evaluation), GLC -> test_challenge, held-out regions {list(holdout_regions)}. Chips whose "
             f"Sentinel-2 scene also gives train chips: {leaks['scene_shared_with_train']}. MGRS tiles with GLB "
             f"chips of several regions: {leaks['tiles_in_several_regions']}. test_challenge: the preprint trained "
             f"its GLBC models on the FULL GLB (train+val+test) minus filename matches; here only 'train' is "
             f"trained on, so test_challenge scores are not comparable with the preprint's GLBC baselines. "
             f"date = Sentinel-2 date in the file name (SAR within +-5 days). region of GLC chips inferred from "
             f"GLB (region_source; 'unknown' when not unique). overlap_* = footprint overlap between splits "
             f"(any CRS).")
    source = SOURCE
    if bench.from_hf:
        source = {**SOURCE, "glb_read_from": {"hugging_face": f"{HF['repo_id']} (dataset) {HF['filename']}",
                                              "size": HF["size"], "sha256": HF["sha256"],
                                              "check": bench.hf_check["summary"]}}
        notes += (f" GLB read from the Hugging Face copy {HF['repo_id']} {HF['filename']}, checked before conversion: "
                  f"{bench.hf_check['summary']}; source_file keeps the Zenodo names.")
    w = prepared.PreparedWriter(out_root, DATASET, CHIP, CLASSES, source, channels=("VV/HH", "VH/HV"),
                                units=units, pixel_spacing_m=PIXEL_M, notes=notes)

    def work(job):
        sp, r, src = job
        c = _read(src, r)
        inf = c["info"]
        _check(np.allclose(np.abs(inf["res"]), PIXEL_M, atol=1e-3),
               f"{r['image']}: pixel size {inf['res']}; documented 10 m grid")
        flags = {}
        img = _to_units(c["img"], state, inf["nodata"], to_log, r["image"], flags)
        mv = np.unique(c["mask"]).tolist()
        _check(set(mv) <= {0, 1}, f"{r['mask']}: mask values {mv}; documented 0 background, 1 lake")
        lab = c["mask"].astype(np.uint8)
        none = np.isnan(img).all(0)
        if ignore_no_sar:
            lab[none] = prepared.IGNORE
        b, lon, lat = inf["bounds"], np.nan, np.nan
        if inf["crs"]:
            from rasterio.warp import transform
            xs, ys = transform(inf["crs"], "EPSG:4326", [(b[0] + b[2]) / 2], [(b[1] + b[3]) / 2])
            lon, lat = round(float(xs[0]), 6), round(float(ys[0]), 6)
        d = r["date"]
        meta = {"region": r["region"], "date": f"{d[:4]}-{d[4:6]}-{d[6:]}", "region_source": r["region_source"],
                "source_file": r["image"], "official_split": r["official"], "s2_scene": r["scene"],
                "mgrs_tile": r["tile"], "pol": _pol(inf["desc"], r["image"]), "crc32": inf["crc32"],
                "s1_valid": round(float(np.isfinite(img).all(0).mean()), 6), "s1_squashed": flags.get("squashed", 0),
                "crs": inf["crs"], "x_min": round(b[0], 3), "y_min": round(b[1], 3), "x_max": round(b[2], 3),
                "y_max": round(b[3], 3), "lon": lon, "lat": lat}
        return sp, img, lab, meta, bool(none.all())

    t0, done, no_sar, squashed, dropped = time.time(), 0, {}, {}, []
    with ThreadPoolExecutor(workers) as ex:
        for s in range(0, len(jobs), 64):                        # bounded memory, order kept
            for sp, img, lab, meta, empty in ex.map(work, jobs[s:s + 64]):
                done += 1
                if sp == "train" and meta["crc32"] in held_crc:    # byte-identical twin of a held-out chip
                    dropped.append(meta["source_file"])
                    continue
                w.add(sp, img, lab, **meta)
                no_sar[sp] = no_sar.get(sp, 0) + empty
                squashed[sp] = squashed.get(sp, 0) + (meta["s1_squashed"] > 0)
            if done % 2048 < 64 or done == len(jobs):
                el = time.time() - t0
                print(f"  {done}/{len(jobs)} chips, {el / 60:.1f} min, {done / max(el, 1e-6):.1f} chips/s", flush=True)
    w.notes += f" Chips without any SAR pixel per split: {no_sar}."
    if state == "minmax":
        w.notes += f" Chips with a SAR band squashed by a fill (set to NaN) per split: {squashed}."
    dropped_names = [p.split("/")[-1] for p in dropped]
    if holdout_regions:
        held_tiles, gone = {r["tile"] for r in held}, set(dropped)
        near = [j[1] for j in jobs if j[0] == "train" and j[1]["tile"] in held_tiles and j[1]["image"] not in gone]
        w.notes += (f" Hold-out {list(holdout_regions)}: {len(dropped)} train chips dropped as byte-identical "
                    f"(CRC-32) to a held-out chip {dropped_names}; {len(near)} train chips kept on MGRS tiles that "
                    f"also hold held-out chips {sorted({r['tile'] for r in near})} (neighbours, not copies; see "
                    f"overlap_train of the held-out rows).")
    w.notes += (" Region code 'NA' is North Asia: read *_meta.csv with keep_default_na=False, na_values=[''] "
                "(pandas' default NA tokens turn it into NaN).")
    man = w.close()
    ov = _add_overlaps(out_dir, list(man["splits"]))
    man = _append_notes(out_dir, f" Chips with overlap_train > 0 per split: {ov}.")
    lost = {sp: int(prepared.load_split(out_root, DATASET, sp)["meta"]["region"].isna().sum()) for sp in man["splits"]}
    if any(lost.values()):
        print(f"[warn] prepared.load_split reads region 'NA' (North Asia) back as NaN: {lost} chips per split; "
              f"per-region code must read the meta with keep_default_na=False, na_values=['']")
    print(f"[done] {out_dir}: {man['splits']}, units {units}, chips without SAR {no_sar}"
          f"{f', squashed {squashed}' if state == 'minmax' else ''}"
          f"{f', dropped hold-out twins {dropped_names}' if holdout_regions else ''}, overlap_train > 0 {ov}")
    return man


def _append_notes(out_dir: Path, text: str) -> dict:
    """Add facts known only after close() (footprint overlaps) to the manifest notes."""
    p = out_dir / "manifest.json"
    man = json.loads(p.read_text(encoding="utf-8"))
    man["notes"] += text
    p.write_text(json.dumps(man, indent=1), encoding="utf-8")
    return man


# ----------------------------------------------------------------------------- footprints

def _rect_share(aa: np.ndarray, bb: np.ndarray) -> np.ndarray:
    """(n,4) and (m,4) boxes in one CRS -> per row of aa the largest share covered by one box of bb."""
    wx = np.clip(np.minimum(aa[:, None, 2], bb[None, :, 2]) - np.maximum(aa[:, None, 0], bb[None, :, 0]), 0, None)
    wy = np.clip(np.minimum(aa[:, None, 3], bb[None, :, 3]) - np.maximum(aa[:, None, 1], bb[None, :, 1]), 0, None)
    return (wx * wy).max(1) / ((aa[:, 2] - aa[:, 0]) * (aa[:, 3] - aa[:, 1]))


def _clip_area(poly: list, r) -> float:
    """Area of the convex polygon `poly` [(x, y), ...] inside the box r = (x0, y0, x1, y1) (Sutherland-Hodgman)."""
    for ax, lim, lower in ((0, r[0], True), (0, r[2], False), (1, r[1], True), (1, r[3], False)):
        def inside(p, ax=ax, lim=lim, lower=lower):
            return p[ax] >= lim if lower else p[ax] <= lim
        out = []
        for i, p in enumerate(poly):
            q = poly[(i + 1) % len(poly)]
            if inside(p):
                out.append(p)
            if inside(p) != inside(q):
                t = (lim - p[ax]) / (q[ax] - p[ax])
                out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
        poly = out
        if len(poly) < 3:
            return 0.0
    return 0.5 * abs(sum(poly[i - 1][0] * poly[i][1] - poly[i][0] * poly[i - 1][1] for i in range(len(poly))))


def _overlap(a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    """Per row of a: largest share of its footprint covered by ONE footprint of b.

    Same CRS: exact box intersection. Another CRS (e.g. the next UTM zone): the b footprint's
    corners are transformed into a's CRS and the quadrilateral is clipped to a's box; only
    pairs whose centres are within NEAR_KM are compared. Rows without a CRS get NaN.
    """
    from rasterio.warp import transform
    out = np.full(len(a), np.nan)
    cols = ["x_min", "y_min", "x_max", "y_max"]
    A, B = a[cols].to_numpy(float), b[cols].to_numpy(float)
    ca, cb = a["crs"].astype(str).to_numpy(), b["crs"].astype(str).to_numpy()
    num = lambda df, c: pd.to_numeric(df[c], errors="coerce").to_numpy(float)
    alon, alat, blon, blat = num(a, "lon"), num(a, "lat"), num(b, "lon"), num(b, "lat")
    for crs in sorted(set(ca) - {""}):
        ia = np.flatnonzero(ca == crs)
        out[ia] = 0.0
        jb = np.flatnonzero(cb == crs)
        jo = np.flatnonzero((cb != crs) & (cb != "") & np.isfinite(blat) & np.isfinite(blon))
        pairs = []
        for s in range(0, len(ia), 128):
            i = ia[s:s + 128]
            if len(jb):
                out[i] = _rect_share(A[i], B[jb])
            if len(jo):
                dy = (alat[i, None] - blat[None, jo]) * 111.2
                dx = ((alon[i, None] - blon[None, jo] + 180) % 360 - 180) * 111.2 * np.cos(np.radians(alat[i, None]))
                ii, jj = np.nonzero((np.abs(dy) < NEAR_KM) & (np.abs(dx) < NEAR_KM))
                pairs += list(zip(i[ii].tolist(), jo[jj].tolist()))
        corners = {}
        for src_crs in sorted({cb[j] for _, j in pairs}):
            js = sorted({j for _, j in pairs if cb[j] == src_crs})
            x0, y0, x1, y1 = B[js].T
            xs, ys = transform(src_crs, crs, np.r_[x0, x1, x1, x0].tolist(), np.r_[y0, y0, y1, y1].tolist())
            xs, ys = np.asarray(xs).reshape(4, -1), np.asarray(ys).reshape(4, -1)
            corners.update({j: [(xs[c, k], ys[c, k]) for c in range(4)] for k, j in enumerate(js)})
        for i, j in pairs:
            r = A[i]
            out[i] = max(out[i], _clip_area(corners[j], r) / ((r[2] - r[0]) * (r[3] - r[1])))
    return np.round(out, 4)


def _add_overlaps(out_dir: Path, splits: list[str]) -> dict:
    """Add overlap_train to every non-train meta and overlap_test_challenge to the train meta.

    Returns the number of chips with overlap > 0 per split (for the manifest notes).
    """
    if "train" not in splits:
        return {}
    read = lambda sp: pd.read_csv(out_dir / f"{sp}_meta.csv", dtype=str, keep_default_na=False)
    train, summary = read("train"), {}
    for sp in splits:
        if sp == "train":
            continue
        m = read(sp)
        m["overlap_train"] = _overlap(m, train)
        summary[sp] = int((m["overlap_train"] > 0).sum())
        m.to_csv(out_dir / f"{sp}_meta.csv", index=False)
        if sp == "test_challenge":
            train["overlap_test_challenge"] = _overlap(train, m)
            summary["train_vs_test_challenge"] = int((train["overlap_test_challenge"] > 0).sum())
            train.to_csv(out_dir / "train_meta.csv", index=False)
    return summary
