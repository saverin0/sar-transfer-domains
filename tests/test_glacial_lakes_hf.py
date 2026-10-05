"""Offline test of the Hugging Face path of domains.glacial_lakes (GLB from the Cryo-Bench copy).

Tiny synthetic GeoTIFFs in the documented GLB format, packed twice: as a Zenodo-like
Glacial_Lake_Bench.zip (img_dir/ann_dir) and as a Cryo-Bench-like GLB.tar.gz
(GLB/<split>/images + a mask folder whose name the code must discover; 'validation' for val).
Covers: zenodo_directory over HTTP ranges (pass, size sanity, server ignoring Range, 503
retried), download_hf with a stand-in huggingface_hub (token off, local_dir, Xet cache env,
sha256 pass/fail, size check, free disk, stale temp files, retries, old hub version),
leftovers of an interrupted run (.GLB_partial, *.incomplete) removed BEFORE the free-disk
checks (review fixes a + b), extraction with system tar and with tarfile, the extraction
marker (skip, redo), layout discovery and split mapping (+ ambiguous / missing mask folder,
unknown or doubled split), check_identity (pass, skip, changed byte, missing file, extra
file, other size, record voided by a later change incl. a same-size rewrite and a record
without a stamp, review fix c), convert refusing an unchecked copy, and convert/inventory from the
Hugging Face layout giving the same arrays and meta as from the zip (hold-out twin drop incl.).
No network: every server runs on 127.0.0.1; all files live under the temp folder of tests/_setup.py.
"""
import gc, hashlib, http.server, io, json, os, re, shutil, sys, tarfile, threading, time, types, zipfile, zlib
from pathlib import Path

import numpy as np
import pandas as pd

from _setup import TMP_BASE  # noqa: E402  (this repo's src first, import guard, temp in one folder)
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from sartransfer.domains import prepared as P
from sartransfer.domains import glacial_lakes as G

TMP = TMP_BASE / "tmp_glacial_lakes_hf"
shutil.rmtree(TMP, ignore_errors=True)
TMP.mkdir()
rng = np.random.default_rng(1)
S = 256
T0 = time.time()
SAVED = {"HF": dict(G.HF), "GLB_BYTES": G.GLB_BYTES, "FILES": G.FILES, "shutil": G.shutil, "time": G.time,
         "zenodo_directory": G.zenodo_directory, "which": shutil.which, "hub": sys.modules.get("huggingface_hub")}


def expect_fail(fn, *words, exc=AssertionError):
    try:
        fn()
    except exc as e:
        msg = str(e)
        assert all(w in msg for w in words), f"message {msg!r} lacks {words}"
        return msg
    raise SystemExit(f"expected {exc.__name__} containing {words}")


def tif(arr, epsg, x0, y0):
    with MemoryFile() as mf:
        with mf.open(driver="GTiff", width=S, height=S, count=arr.shape[0], dtype=arr.dtype.name,
                     crs=f"EPSG:{epsg}", transform=from_origin(x0, y0, 10, 10)) as ds:
            ds.write(arr)
        return mf.read()


def chip(mask):
    """11 bands, per-band min-max stretch as the preprint describes; lakes darker in SAR."""
    st = rng.random((11, S, S)).astype(np.float32)
    vv = rng.lognormal(np.log(0.08), 0.6, (S, S)).astype(np.float32)
    vh = rng.lognormal(np.log(0.015), 0.6, (S, S)).astype(np.float32)
    vv[mask == 1] *= 0.05
    vh[mask == 1] *= 0.05
    st[9], st[10] = vv, vh
    return np.stack([(b - b.min()) / (b.max() - b.min()) for b in st]).astype(np.float32)


def lake():
    m = np.zeros((S, S), np.uint8)
    r, c = rng.integers(40, 200, 2)
    m[r - 15:r + 15, c - 15:c + 15] = 1
    return m


EPSG = {"5VPJ": 32605, "6VXN": 32606, "33XVH": 32633, "33XWG": 32633, "44RPU": 32644, "45RXM": 32645}
zone = lambda n: EPSG[[p for p in n.split("_") if p in EPSG][0]]
stem = lambda f: Path(f).stem
GLB = {"train": ["AK_S2A_5VPJ_20200817_0_L2A_1", "SV_S2B_33XVH_20200801_0_L2A_5", "CA_S2B_44RPU_20200831_1_L2A_95",
                 "SEE_S2B_44RPU_20200831_1_L2A_93", "SEE_S2B_44RPU_20200831_1_L2A_120"],
       "val": ["NA_S2B_6VXN_20200828_1_L2A_231", "SV_S2A_33XWG_20200715_0_L2A_3"],
       "test": ["CA_S2B_44RPU_20200831_1_L2A_111", "SEE_S2A_45RXM_20201229_1_L2A_13"]}
GLC = ["S2B_44RPU_20200831_1_L2A_7", "S2A_5VPJ_20200817_0_L2A_9"]
TWIN = ("SEE_S2B_44RPU_20200831_1_L2A_93", "CA_S2B_44RPU_20200831_1_L2A_95")    # byte-identical, two prefixes
files = {}
for k, n in enumerate([x for v in GLB.values() for x in v] + GLC):
    m = lake()
    files[n] = (tif(chip(m), zone(n), 300000 + 3000 * k, 7e6), tif(m[None], zone(n), 300000 + 3000 * k, 7e6))
files[TWIN[0]] = files[TWIN[1]]
N_PAIRS = sum(map(len, GLB.values()))

# ---------------------------------------------------------------- the two packings of the same files
RAW_Z = TMP / "raw_zenodo"
RAW_Z.mkdir()
ZIP = RAW_Z / "Glacial_Lake_Bench.zip"
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
    for kind in ("ann_dir", "img_dir"):
        for split in ("test", "train", "val"):
            z.writestr(f"Glacial_Lake_Bench/{kind}/{split}/", b"")
            for n in sorted(GLB[split]):
                z.writestr(f"Glacial_Lake_Bench/{kind}/{split}/{n}.tif", files[n][kind == "ann_dir"])
    z.writestr("Glacial_Lake_Bench/ann_dir/test/Thumbs.db", b"\0" * 64)
    z.writestr("Glacial_Lake_Bench/Metadata.txt", b"Each image contains 11 channels: ... VV, and VH.")
CH_ZIP = RAW_Z / "Glacial-Lake-Challenge.zip"
with zipfile.ZipFile(CH_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
    for kind in ("image", "mask"):
        for n in GLC:
            z.writestr(f"Glacial-Lake-Challenge/{kind}/{n}.tif", files[n][kind == "mask"])

FOLDER = {"train": "train", "val": "validation", "test": "test"}                  # Cryo-Bench split folders
TGZ = TMP / "GLB.tar.gz"


def add(t, name, data=None):
    ti = tarfile.TarInfo(name)
    if data is None:
        ti.type, ti.mode = tarfile.DIRTYPE, 0o755
        t.addfile(ti)
    else:
        ti.size, ti.mode = len(data), 0o644
        t.addfile(ti, io.BytesIO(data))


with tarfile.open(TGZ, "w:gz", compresslevel=1) as t:
    add(t, "GLB")
    for split, folder in FOLDER.items():
        add(t, f"GLB/{folder}")
        add(t, f"GLB/{folder}/global_stats.json", b'{"mean": [0.1], "std": [0.2]}')
        for sub, which in (("images", 0), ("labels", 1)):                  # mask folder name NOT 'masks'
            add(t, f"GLB/{folder}/{sub}")
            for n in sorted(GLB[split]):
                add(t, f"GLB/{folder}/{sub}/{n}.tif", files[n][which])
    add(t, "GLB/train/quicklook")                                           # a distractor folder
    add(t, "GLB/train/quicklook/overview.tif", b"not a chip")
TGZ_BYTES, TGZ_SHA = TGZ.read_bytes(), hashlib.sha256(TGZ.read_bytes()).hexdigest()
TIF_BYTES = sum(len(files[n][0]) + len(files[n][1]) for v in GLB.values() for n in v)


# ---------------------------------------------------------------- a local HTTP server with Range support
class Handler(http.server.BaseHTTPRequestHandler):
    data, mode, gets, fail = b"", "ok", 0, 0

    def do_GET(self):
        Handler.gets += 1
        if Handler.fail:
            Handler.fail -= 1
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        d, r = Handler.data, self.headers.get("Range")
        m = re.fullmatch(r"bytes=(\d+)-(\d*)", r or "")
        if m and Handler.mode != "ignore_range":
            a = int(m[1])
            b = int(m[2]) if m[2] else len(d) - 1
            body = d[a:b + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {a}-{a + len(body) - 1}/{len(d)}")
        else:
            body = d
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except OSError:
            pass

    def log_message(self, *a):
        pass


os.environ["NO_PROXY"] = "127.0.0.1,localhost"
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()
URL = f"http://127.0.0.1:{srv.server_port}/f"

try:
    # ------------------------------------------------------------ zenodo_directory: central directory over ranges
    Handler.data = ZIP.read_bytes()
    with zipfile.ZipFile(ZIP) as z:
        truth = {i.filename: (i.file_size, i.CRC) for i in z.infolist() if not i.is_dir()}
    REF = G.zenodo_directory(URL, len(Handler.data))
    assert REF == truth, "central directory over HTTP ranges differs from zipfile's own listing"
    assert Handler.gets <= 2, Handler.gets                          # tail + directory (both in the tail here)
    expect_fail(lambda: G.zenodo_directory(URL, len(Handler.data) + 1), "published", "another release")
    Handler.mode, Handler.gets = "ignore_range", 0
    expect_fail(lambda: G.zenodo_directory(URL, len(Handler.data)), "HTTP 200", "206 expected")
    assert Handler.gets == 1
    Handler.mode, Handler.fail, Handler.gets = "ok", 1, 0          # one 503, then fine
    G.time = types.SimpleNamespace(time=time.time, sleep=lambda s: None, strftime=time.strftime)
    assert G.zenodo_directory(URL, len(Handler.data)) == truth and Handler.gets >= 2
    G.time = SAVED["time"]
    # a ZIP64 end record (more than 65,535 members, as the real zip has for its > 4 GB offsets), directory > tail
    Z64 = TMP / "z64.zip"
    with zipfile.ZipFile(Z64, "w") as z:
        for i in range(70000):
            z.writestr(f"Glacial_Lake_Bench/ann_dir/train/x{i:05d}.tif", b"")
    Handler.data, Handler.gets = Z64.read_bytes(), 0
    assert Handler.data.rfind(b"PK\x06\x06") > 0, "no ZIP64 end record in the fixture"
    with zipfile.ZipFile(Z64) as z:
        assert G.zenodo_directory(URL, len(Handler.data)) == {i.filename: (i.file_size, i.CRC) for i in z.infolist()}
    assert Handler.gets == 2, Handler.gets                                      # tail, then the directory
    print("zenodo_directory: ranges, size check, Range ignored, 503 retry, ZIP64 end record ok")

    # ------------------------------------------------------------ download_hf with a stand-in huggingface_hub
    calls = []

    def fake_hf_hub_download(**kw):
        calls.append({**kw, "env": {k: os.environ.get(k) for k in ("HF_XET_CACHE", "HF_XET_CHUNK_CACHE_SIZE_BYTES")}})
        if FAKE["raise"]:
            e = FAKE["raise"].pop(0)
            raise e
        local = Path(kw["local_dir"])
        tmp = local / ".cache" / "huggingface" / "download" / "data" / "x.1234.incomplete"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(FAKE["payload"])
        dst = local / kw["filename"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, dst)
        return str(dst)

    FAKE = {"payload": TGZ_BYTES, "raise": []}
    hub = types.ModuleType("huggingface_hub")
    hub.__version__, hub.hf_hub_download = "1.31.0", fake_hf_hub_download
    sys.modules["huggingface_hub"] = hub
    G.HF.update(size=len(TGZ_BYTES), sha256=TGZ_SHA)
    G.GLB_BYTES = TIF_BYTES
    os.environ.pop("HF_XET_CACHE", None)
    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "123"                 # the user's own value comes back after

    D1 = TMP / "dl1"
    # sha256 mismatch -> .bad, nothing extracted
    G.HF["sha256"] = "0" * 64
    expect_fail(lambda: G.download_hf(D1), "sha256", "renamed to .bad")
    assert (D1 / "data/GLB.tar.gz.bad").is_file() and not (D1 / "GLB").exists() and len(calls) == 1
    G.HF["sha256"] = TGZ_SHA
    # a stale temp file of a killed run is removed; the call itself: dataset repo, local_dir, no token, Xet env
    stale = D1 / ".cache/huggingface/download/data/old.abcd.incomplete"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"x" * 1000)
    out = G.download_hf(D1)
    c = calls[-1]
    assert out == D1 / "GLB" and (out / G.DONE_MARKER).is_file() and not stale.exists()
    assert (c["repo_id"], c["repo_type"], c["filename"], c["token"]) == ("Sk-21/Cryo-Bench", "dataset", "data/GLB.tar.gz", False)
    assert Path(c["local_dir"]) == D1
    assert c["env"] == {"HF_XET_CACHE": str(D1 / ".cache" / "huggingface" / "xet"), "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0"}
    assert "HF_XET_CACHE" not in os.environ and os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] == "123"
    assert not (D1 / "data/GLB.tar.gz").exists() and not (D1 / "data/GLB.tar.gz.sha256ok").exists()   # archive removed
    listing = sorted(p.relative_to(out).as_posix() for p in out.rglob("*"))
    assert "validation/labels/NA_S2B_6VXN_20200828_1_L2A_231.tif" in listing and "train/global_stats.json" in listing
    assert (out / "train/images" / f"{TWIN[0]}.tif").read_bytes() == files[TWIN[0]][0]
    # complete -> skipped without any hub call
    n = len(calls)
    assert G.download_hf(D1) == out and len(calls) == n
    # half-finished extraction (no marker) with the archive kept: redone from the kept archive, no new download
    D2 = TMP / "dl2"
    G.download_hf(D2, keep_archive=True)
    assert (D2 / "data/GLB.tar.gz").is_file() and (D2 / "data/GLB.tar.gz.sha256ok").read_text() == TGZ_SHA
    (D2 / "GLB" / G.DONE_MARKER).unlink()
    (D2 / "GLB/junk.txt").write_text("left by a crash")
    n = len(calls)
    G.download_hf(D2, keep_archive=True)
    assert len(calls) == n and (D2 / "GLB" / G.DONE_MARKER).is_file() and not (D2 / "GLB/junk.txt").exists()
    # the same extraction with Python's tarfile (no tar on PATH)
    D3 = TMP / "dl3"
    shutil.which = lambda name, *a, **k: None if name in ("tar", "pigz") else SAVED["which"](name, *a, **k)
    G.download_hf(D3)
    shutil.which = SAVED["which"]
    assert sorted(p.relative_to(D3 / "GLB").as_posix() for p in (D3 / "GLB").rglob("*")) == listing
    # tar + pigz, as on Colab when pigz is installed: a stand-in 'pigz' runs gzip and leaves a trace
    import subprocess
    if "GNU tar" in subprocess.run([shutil.which("tar"), "--version"], capture_output=True, text=True).stdout:
        PB, old_path = TMP / "pigz_bin", os.environ["PATH"]
        PB.mkdir()
        (PB / "pigz").write_text('#!/bin/sh\necho "$*" >> "$(dirname "$0")/used"\nexec gzip "$@"\n', newline="\n")
        os.chmod(PB / "pigz", 0o755)
        try:
            os.environ["PATH"] = str(PB) + os.pathsep + old_path
            shutil.which = lambda name, *a, **k: str(PB / "pigz") if name == "pigz" else SAVED["which"](name, *a, **k)
            G.download_hf(TMP / "dl_pigz")
        finally:
            shutil.which, os.environ["PATH"] = SAVED["which"], old_path
        assert (PB / "used").read_text().split() == ["-d"], "tar did not run pigz -d"
        assert sorted(p.relative_to(TMP / "dl_pigz/GLB").as_posix() for p in (TMP / "dl_pigz/GLB").rglob("*")) == listing
    else:
        print("[skip] tar + pigz subtest: the tar on PATH is not GNU tar")
    # a leftover partial folder of the extraction itself is cleared
    D4 = TMP / "dl4"
    (D4 / ".GLB_partial/GLB/train").mkdir(parents=True)
    G.download_hf(D4)
    assert not (D4 / ".GLB_partial").exists() and (D4 / "GLB" / G.DONE_MARKER).is_file()
    # the hub returns a file of the wrong size -> stop before hashing
    FAKE["payload"] = TGZ_BYTES[:-10]
    expect_fail(lambda: G.download_hf(TMP / "dl5"), "published")
    FAKE["payload"] = TGZ_BYTES
    # a dropped connection is retried (from 0), a missing repo is not
    FAKE["raise"] = [ConnectionError("reset by peer")]
    G.time = types.SimpleNamespace(time=time.time, sleep=lambda s: None, strftime=time.strftime)
    n = len(calls)
    G.download_hf(TMP / "dl6")
    assert len(calls) == n + 2
    RepositoryNotFoundError = type("RepositoryNotFoundError", (Exception,), {})
    FAKE["raise"] = [RepositoryNotFoundError("404")]
    n = len(calls)
    expect_fail(lambda: G.download_hf(TMP / "dl7"), "404", exc=RepositoryNotFoundError)
    assert len(calls) == n + 1
    G.time = SAVED["time"]
    # not enough free disk for archive + extracted files -> stop before any call
    G.shutil = types.SimpleNamespace(disk_usage=lambda p: types.SimpleNamespace(free=len(TGZ_BYTES)),
                                     rmtree=shutil.rmtree, which=shutil.which)
    n = len(calls)
    expect_fail(lambda: G.download_hf(TMP / "dl8"), "extracted files need", "free", "bench_source='zenodo'")
    G.shutil = SAVED["shutil"]
    assert len(calls) == n

    # leftovers of an interrupted run must not count as used disk (review fix, 2026-09-27): on a disk that
    # holds exactly what the run needs + 1,000 B, a stale .GLB_partial or *.incomplete (10,000 B) is deleted
    # BEFORE the free-disk checks; before the fix each case stopped with a false 'not enough disk'.
    def tight_disk(root, cap):
        """G.shutil stand-in: free = cap - bytes of every file under root (hidden folders included)."""
        used = lambda: sum(f.stat().st_size for f in Path(root).rglob("*") if f.is_file())
        return types.SimpleNamespace(disk_usage=lambda p: types.SimpleNamespace(free=cap - used()),
                                     rmtree=shutil.rmtree, which=shutil.which)


    def bytes_under(root):
        return sum(f.stat().st_size for f in Path(root).rglob("*") if f.is_file())


    JUNK, MARGIN = 10_000, 1_000
    STALE = ".cache/huggingface/download/data/old.abcd.incomplete"
    try:
        # (a) no archive yet: a .GLB_partial from an earlier run counted against the archive + extraction check
        D11 = TMP / "dl11"
        (D11 / ".GLB_partial/GLB/train").mkdir(parents=True)
        (D11 / ".GLB_partial/GLB/train/big.tif").write_bytes(b"\0" * JUNK)
        G.shutil = tight_disk(D11, len(TGZ_BYTES) + TIF_BYTES + MARGIN)
        n = len(calls)
        G.download_hf(D11)
        assert len(calls) == n + 1 and not (D11 / ".GLB_partial").exists() and (D11 / "GLB" / G.DONE_MARKER).is_file()
        # (a) archive kept and sha256-checked, extraction interrupted (files in .GLB_partial, no GLB/): the
        # extraction check alone -- the realistic re-run after a crash
        G.shutil = SAVED["shutil"]
        D12 = TMP / "dl12"
        G.download_hf(D12, keep_archive=True)
        (D12 / ".GLB_partial").mkdir()
        (D12 / "GLB" / G.DONE_MARKER).unlink()
        (D12 / "GLB").rename(D12 / ".GLB_partial/GLB")
        assert bytes_under(D12 / ".GLB_partial") >= TIF_BYTES
        G.shutil = tight_disk(D12, bytes_under(D12) - bytes_under(D12 / ".GLB_partial") + TIF_BYTES + MARGIN)
        n = len(calls)
        G.download_hf(D12, keep_archive=True)
        assert len(calls) == n and not (D12 / ".GLB_partial").exists() and (D12 / "GLB" / G.DONE_MARKER).is_file()
        # (b) no archive yet: a stale *.incomplete of a killed download counted against the first check
        G.shutil = SAVED["shutil"]
        D13 = TMP / "dl13"
        (D13 / STALE).parent.mkdir(parents=True)
        (D13 / STALE).write_bytes(b"\0" * JUNK)
        G.shutil = tight_disk(D13, len(TGZ_BYTES) + TIF_BYTES + MARGIN)
        n = len(calls)
        G.download_hf(D13)
        assert len(calls) == n + 1 and not (D13 / STALE).exists() and (D13 / "GLB" / G.DONE_MARKER).is_file()
        # (b) archive complete, so no hub call at all: the stale file must still go before the extraction check
        G.shutil = SAVED["shutil"]
        D14 = TMP / "dl14"
        G.download_hf(D14, keep_archive=True)
        shutil.rmtree(D14 / "GLB")
        (D14 / STALE).parent.mkdir(parents=True, exist_ok=True)
        (D14 / STALE).write_bytes(b"\0" * JUNK)
        G.shutil = tight_disk(D14, bytes_under(D14) - JUNK + TIF_BYTES + MARGIN)
        n = len(calls)
        G.download_hf(D14, keep_archive=True)
        assert len(calls) == n and not (D14 / STALE).exists() and (D14 / "GLB" / G.DONE_MARKER).is_file()
        # control: the same tight disk WITHOUT a leftover but 1 B short still stops (the checks are live)
        G.shutil = SAVED["shutil"]
        D15 = TMP / "dl15"
        G.shutil = tight_disk(D15, len(TGZ_BYTES) + TIF_BYTES)
        expect_fail(lambda: G.download_hf(D15), "extracted files need")
    finally:
        G.shutil = SAVED["shutil"]
    # huggingface_hub older than 0.23 (local_dir went through the HF_HOME cache) -> refused
    hub.__version__ = "0.22.2"
    expect_fail(lambda: G.download_hf(TMP / "dl9"), "0.23", "HF_HOME")
    hub.__version__ = "1.31.0"
    print("download_hf: token off, local_dir, Xet env set + restored, sha256 fail, skip, redo, tarfile fallback, "
          "size check, retry / no retry, free disk, leftovers (.GLB_partial, *.incomplete) removed before the "
          "free-disk checks, old hub")

    # ------------------------------------------------------------ layout discovery and split mapping
    real, extra = G._hf_layout(D1)
    assert len(real) == 2 * N_PAIRS
    assert real[f"Glacial_Lake_Bench/ann_dir/val/{GLB['val'][0]}.tif"] == f"GLB/validation/labels/{GLB['val'][0]}.tif"
    assert real[f"Glacial_Lake_Bench/img_dir/train/{GLB['train'][0]}.tif"] == f"GLB/train/images/{GLB['train'][0]}.tif"
    assert set(extra) == {"GLB/train/global_stats.json", "GLB/validation/global_stats.json", "GLB/test/global_stats.json",
                          "GLB/train/quicklook/overview.tif"}, extra


    def variant(name, change):
        """A copy of the extracted GLB (never with check_identity's record), then `change` applied."""
        root = TMP / name
        shutil.copytree(D1 / "GLB", root / "GLB")
        (root / "GLB" / G.SAME_MARKER).unlink(missing_ok=True)
        change(root / "GLB")
        return root


    r = variant("lay_val", lambda g: (g / "validation").rename(g / "val"))              # 'val' works as well
    assert G._hf_layout(r)[0] == {k: v.replace("/validation/", "/val/") for k, v in real.items()}
    r = variant("lay_amb", lambda g: shutil.copytree(g / "test/labels", g / "test/masks"))
    expect_fail(lambda: G._hf_layout(r), "GLB/test/: 2 folders beside images/")
    r = variant("lay_none", lambda g: (g / "test/labels" / f"{GLB['test'][0]}.tif").unlink())
    expect_fail(lambda: G._hf_layout(r), "GLB/test/: 0 folders beside images/", f"missing e.g. ['{GLB['test'][0]}.tif']")
    r = variant("lay_split", lambda g: shutil.copytree(g / "test", g / "holdout"))
    expect_fail(lambda: G._hf_layout(r), "GLB/holdout/: not a split folder")
    r = variant("lay_twice", lambda g: shutil.copytree(g / "validation", g / "val"))
    expect_fail(lambda: G._hf_layout(r), "split 'val' also in")
    r = variant("lay_nomark", lambda g: (g / G.DONE_MARKER).unlink())
    expect_fail(lambda: G._hf_layout(r), "half-finished extraction")
    print("layout: mask folder found by name equality, validation -> val, ambiguous / missing / unknown / doubled stop")

    # ------------------------------------------------------------ identity with the Zenodo release
    rec = G.check_identity(D1, reference=REF, workers=3)
    assert rec["summary"].startswith(f"GLB from Hugging Face: {N_PAIRS} of {N_PAIRS} pairs byte-identical to the "
                                     f"Zenodo release"), rec["summary"]
    assert rec["pairs"] == {"train": 5, "val": 2, "test": 2} and rec["files"] == 2 * N_PAIRS and rec["bytes"] == TIF_BYTES
    assert json.loads((D1 / "GLB" / G.SAME_MARKER).read_text())["summary"] == rec["summary"]
    G.zenodo_directory = lambda *a, **k: (_ for _ in ()).throw(SystemExit("must not be called"))
    assert G.check_identity(D1)["checked"] == rec["checked"]                          # valid record -> skipped
    G.zenodo_directory = lambda *a, **k: REF                                         # default reference path
    assert G.check_identity(D1, recheck=True)["reference"] == "zenodo_directory()"
    G.zenodo_directory = SAVED["zenodo_directory"]


    def flip(p, at=5000):
        b = bytearray(p.read_bytes())
        b[at] ^= 0xFF
        p.write_bytes(bytes(b))


    bad = f"GLB/train/images/{GLB['train'][1]}.tif"
    r = variant("id_byte", lambda g: flip(g.parent / bad))
    msg = expect_fail(lambda: G.check_identity(r, reference=REF), "NOT the Zenodo release",
                      "1 files with the same size but another CRC-32", bad)
    assert not (r / "GLB" / G.SAME_MARKER).exists()
    gone = GLB["val"][1]
    r = variant("id_missing", lambda g: [(g / f"validation/{s}/{gone}.tif").unlink() for s in ("images", "labels")])
    expect_fail(lambda: G.check_identity(r, reference=REF), "2 Zenodo files missing in the Hugging Face copy",
                f"Glacial_Lake_Bench/ann_dir/val/{gone}.tif")
    # a chip the Hugging Face copy has and Zenodo does not (6 test pairs on 2026-09-28): listed and left out,
    # the record stays valid, _Src never reads it; a further new file voids the record
    new = "AK_S2A_5VPJ_20200817_0_L2A_77"
    add_extra = lambda g, stem: [shutil.copy(g / f"test/{s}/{GLB['test'][0]}.tif", g / f"test/{s}/{stem}.tif")
                                 for s in ("images", "labels")]
    r = variant("id_extra", lambda g: add_extra(g, new))
    rec_x = G.check_identity(r, reference=REF)
    assert rec_x["extra"] == [f"Glacial_Lake_Bench/ann_dir/test/{new}.tif", f"Glacial_Lake_Bench/img_dir/test/{new}.tif"]
    assert rec_x["extra_hf_paths"] == [f"GLB/test/labels/{new}.tif", f"GLB/test/images/{new}.tif"]
    assert "left out 1 extra image/mask pairs" in rec_x["summary"], rec_x["summary"]
    assert rec_x["files"] == 2 * N_PAIRS and rec_x["bytes"] == TIF_BYTES and rec_x["pairs"]["test"] == 2
    src = G._Src(r, "bench")
    assert src.hf_check is not None and not any(new in n for n in list(src.sizes) + list(src._real)), "extra read"
    assert G.check_identity(r, reference=REF)["checked"] == rec_x["checked"]          # valid record -> skipped
    add_extra(r / "GLB", "AK_S2A_5VPJ_20200817_0_L2A_78")
    assert G._Src(r, "bench").hf_check is None                                       # an unlisted new file voids it
    rec_x2 = G.check_identity(r, reference=REF)
    assert len(rec_x2["extra"]) == 4 and "left out 2 extra image/mask pairs" in rec_x2["summary"]
    r = variant("id_size", lambda g: open(g / f"train/labels/{GLB['train'][2]}.tif", "ab").write(b"\0"))
    expect_fail(lambda: G.check_identity(r, reference=REF), "1 files of another size",
                f"GLB/train/labels/{GLB['train'][2]}.tif {len(files[GLB['train'][2]][1]) + 1:,} B")
    # a record voided by a later change: the file count/bytes no longer match -> convert refuses
    r = variant("id_later", lambda g: None)
    G.check_identity(r, reference=REF)
    assert G._Src(r, "bench").hf_check is not None
    (r / "GLB/test/images" / f"{GLB['test'][1]}.tif").unlink()
    (r / "GLB/test/labels" / f"{GLB['test'][1]}.tif").unlink()
    assert G._Src(r, "bench").hf_check is None
    expect_fail(lambda: G.convert(r, TMP / "out_x", check_counts=False, include_challenge=False), "check_identity(")
    # a SAME-SIZE rewrite voids the record too (review fix, 2026-09-27): the record carries a stamp over
    # every checked file's name, size and modification time (ns); count and bytes alone did not change
    same = f"GLB/train/images/{GLB['train'][4]}.tif"
    r = variant("id_same_size", lambda g: None)
    rec_s = G.check_identity(r, reference=REF)
    assert len(rec_s["stamp"]) == 64 and G._Src(r, "bench").hf_check is not None
    st = (r / same).stat()
    flip(r / same)
    if (r / same).stat().st_mtime_ns == st.st_mtime_ns:          # coarse file-system clock: make the rewrite visible
        os.utime(r / same, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert (r / same).stat().st_size == st.st_size
    assert G._Src(r, "bench").hf_check is None
    expect_fail(lambda: G.convert(r, TMP / "out_y", check_counts=False, include_challenge=False), "check_identity(")
    expect_fail(lambda: G.check_identity(r, reference=REF), "1 files with the same size but another CRC-32", same)
    # same size and an OLDER modification time (an old copy put back): the stamp differs as well
    r = variant("id_older", lambda g: None)
    G.check_identity(r, reference=REF)
    st = (r / same).stat()
    flip(r / same)
    os.utime(r / same, ns=(st.st_atime_ns, st.st_mtime_ns - 10**9))
    assert G._Src(r, "bench").hf_check is None
    # untouched files keep the record valid; a record without a stamp (older code) is not trusted
    r = variant("id_nostamp", lambda g: None)
    G.check_identity(r, reference=REF)
    assert G._Src(r, "bench").hf_check is not None
    old = json.loads((r / "GLB" / G.SAME_MARKER).read_text())
    del old["stamp"]
    (r / "GLB" / G.SAME_MARKER).write_text(json.dumps(old))
    assert G._Src(r, "bench").hf_check is None
    print("check_identity: pass + summary, skip, recheck, changed byte, missing, extra, other size, voided record "
          "(fewer files, same-size rewrite with a newer or an older time, record without a stamp)")

    # ------------------------------------------------------------ convert: Hugging Face layout == zip
    RAW_H = variant("raw_hf", lambda g: None)
    shutil.copy(CH_ZIP, RAW_H / CH_ZIP.name)
    expect_fail(lambda: G.convert(RAW_H, TMP / "out_h", check_counts=False), "not (or no longer) checked",
                "check_identity(")
    inv = G.inventory(RAW_H, check_counts=False)                                     # looking needs no check
    assert "GLB/train/global_stats.json" in inv["extra_files"] and inv["splits"]["val"] == 2
    G.check_identity(RAW_H, reference=REF)
    inv = G.inventory(RAW_H, check_counts=False)
    inv_z = G.inventory(RAW_Z, check_counts=False)
    assert inv["state"] == inv_z["state"] and inv["leaks"] == inv_z["leaks"] and inv["regions"] == inv_z["regions"]
    OUT_H, OUT_Z = TMP / "out_h", TMP / "out_z"
    man_h = G.convert(RAW_H, OUT_H, check_counts=False, holdout_regions=("CA",), workers=2)
    man_z = G.convert(RAW_Z, OUT_Z, check_counts=False, holdout_regions=("CA",), workers=2)
    assert man_h["splits"] == man_z["splits"] == {"train": 3, "val": 2, "test": 1, "test_region_CA": 2,
                                                  "test_challenge": 2}, man_h["splits"]
    hf_note = re.search(r" GLB read from the Hugging Face copy .*?source_file keeps the Zenodo names\.", man_h["notes"])
    assert hf_note and f"{N_PAIRS} of {N_PAIRS} pairs byte-identical" in hf_note[0]
    assert man_h["notes"].replace(hf_note[0], "") == man_z["notes"]
    assert f"1 train chips dropped as byte-identical (CRC-32) to a held-out chip ['{TWIN[0]}.tif']" in man_h["notes"]
    assert man_h["source"]["glb_read_from"]["sha256"] == TGZ_SHA and "glb_read_from" not in man_z["source"]
    assert {k: v for k, v in man_h["source"].items() if k != "glb_read_from"} == man_z["source"]
    for split in man_h["splits"]:
        a, b = P.load_split(OUT_H, G.DATASET, split), P.load_split(OUT_Z, G.DATASET, split)
        assert np.array_equal(np.asarray(a["images"]), np.asarray(b["images"]), equal_nan=True), split
        assert np.array_equal(np.asarray(a["labels"]), np.asarray(b["labels"])), split
        assert (OUT_H / G.DATASET / f"{split}_meta.csv").read_text() == (OUT_Z / G.DATASET / f"{split}_meta.csv").read_text()
        del a, b
    meta = pd.read_csv(OUT_H / G.DATASET / "val_meta.csv", keep_default_na=False, na_values=[""])
    assert meta.source_file.str.startswith("Glacial_Lake_Bench/img_dir/val/").all()           # Zenodo names kept
    P.validate(OUT_H, G.DATASET)
    # the same HF copy plus a chip Zenodo does not have: the output equals the zip's, the chip is nowhere
    RAW_X = variant("raw_hf_extra", lambda g: add_extra(g, new))
    shutil.copy(CH_ZIP, RAW_X / CH_ZIP.name)
    G.check_identity(RAW_X, reference=REF)
    OUT_X = TMP / "out_x_extra"
    man_x = G.convert(RAW_X, OUT_X, check_counts=False, holdout_regions=("CA",), workers=2)
    assert man_x["splits"] == man_z["splits"], man_x["splits"]
    assert "left out 1 extra image/mask pairs" in man_x["notes"]
    for split in man_x["splits"]:
        assert (OUT_X / G.DATASET / f"{split}_meta.csv").read_text() == (OUT_Z / G.DATASET / f"{split}_meta.csv").read_text()
        assert new not in (OUT_X / G.DATASET / f"{split}_meta.csv").read_text()
    print("convert: HF layout gives the same arrays, meta and hold-out twin drop as the zip; provenance in manifest; "
          "an HF-only chip is left out")

    # ------------------------------------------------------------ download(bench_source='hf'): whole flow
    ch_bytes = CH_ZIP.read_bytes()
    G.FILES = {"bench": SAVED["FILES"]["bench"],
               "challenge": {**SAVED["FILES"]["challenge"], "url": URL, "size": len(ch_bytes),
                             "md5": hashlib.md5(ch_bytes).hexdigest()}}
    Handler.data = ch_bytes
    G.zenodo_directory = lambda *a, **k: REF
    D10 = TMP / "dl10"
    got = G.download(D10, bench_source="hf")
    assert got == [D10 / "GLB", D10 / CH_ZIP.name] and (D10 / "GLB" / G.SAME_MARKER).is_file()
    n = len(calls)
    G.zenodo_directory = lambda *a, **k: (_ for _ in ()).throw(SystemExit("must not be called"))
    G.download(D10, bench_source="hf")                                               # everything skipped
    assert len(calls) == n
    man10 = G.convert(D10, TMP / "out10", check_counts=False, limit_per_split=1)
    assert man10["splits"] == {"train": 1, "val": 1, "test": 1, "test_challenge": 1}
    expect_fail(lambda: G.download(D10, bench_source="huggingface"), "'zenodo' or 'hf'")
    print("download(bench_source='hf'): HF archive + identity check + Zenodo challenge zip; second run skips all")
finally:
    G.HF.clear()
    G.HF.update(SAVED["HF"])
    G.GLB_BYTES, G.FILES, G.shutil, G.time = SAVED["GLB_BYTES"], SAVED["FILES"], SAVED["shutil"], SAVED["time"]
    G.zenodo_directory, shutil.which = SAVED["zenodo_directory"], SAVED["which"]
    if SAVED["hub"] is None:
        sys.modules.pop("huggingface_hub", None)
    else:
        sys.modules["huggingface_hub"] = SAVED["hub"]
    os.environ.pop("HF_XET_CHUNK_CACHE_SIZE_BYTES", None)
    srv.shutdown()

assert G.HF["size"] == 44287419921 and G.GLB_BYTES == 56431403680 and G.FILES["bench"]["size"] == 44268636160
meta = None
gc.collect()                                                   # memory maps hold Windows file locks
shutil.rmtree(TMP)
assert not TMP.exists()
print(f"glacial_lakes Hugging Face path verified in {time.time() - T0:.0f} s")
