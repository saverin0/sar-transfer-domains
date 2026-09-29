"""Private Hugging Face copies of the CONVERTED domain data (decided 2026-09-27: converted
files only, no raw copies).

`upload` puts one prepared dataset folder (manifest.json, <split>_images.npy, <split>_labels.npy,
<split>_meta.csv) into a PRIVATE dataset repo <owner>/sar-transfer-<dataset with '_' as '-'>, with a dataset card
that names the original source, its licence and citation. It refuses to upload into a repo that
is public, refuses an incomplete folder, and checks afterwards that every file arrived with its
local size. `fetch` downloads it back into a local folder (e.g. /content on Colab), and `stage`
brings every dataset a GPU notebook needs to local disk, from Hugging Face or else from Drive.

Tokens are passed in by the caller and never printed or written anywhere. The upload needs the
write token, which notebook 01 reads with env.secret (never into os.environ); fetching needs only
a token that can read the private repos (HF_TOKEN in notebooks 02 and 03).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import prepared
from .prepared import load_manifest

FILES = ("manifest.json", "README.md", "*_images.npy", "*_labels.npy", "*_meta.csv")
LICENCE_IDS = (("etalab", "etalab-2.0"), ("by-sa", "cc-by-sa-4.0"), ("by sa", "cc-by-sa-4.0"),
               ("cc-by-4.0", "cc-by-4.0"), ("cc by 4.0", "cc-by-4.0"))
BIBTEX = {   # from the dataset page, 2026-09-27
    "snow": """@data{IMTSFL_2025,
author = {Briand, Swann and Weissgerber, Flora and Lobry, Sylvain and Idier, J\\'{e}r\\^{o}me},
publisher = {Recherche Data Gouv},
title = {{Replication Data for : Weakly supervised learning for snow cover segmentation in mountainous areas from Sentinel-1 SAR images using interpolated NDSI time series}},
year = {2025},
version = {V1},
doi = {10.57745/IMTSFL},
url = {https://doi.org/10.57745/IMTSFL}
}""",
}


def repo_id(dataset: str, owner: str) -> str:
    return f"{owner}/sar-transfer-{dataset.replace('_', '-')}"


def _licence_id(text: str) -> str:
    t = str(text).lower()
    return next((hf for key, hf in LICENCE_IDS if key in t), "other")


def card(man: dict, rid: str) -> str:
    """Dataset card: private converted copy, original source, licence, citation, format."""
    src = man.get("source", {})
    lic = src.get("licence", "see source")
    cite = BIBTEX.get(man["dataset"]) or src.get("citation", "")
    splits = "\n".join(f"| {k} | {v:,} |" for k, v in man["splits"].items())
    classes = ", ".join(f"{k} = {v}" for k, v in man["classes"].items())
    return f"""---
license: {_licence_id(lic)}
tags: [sar, sentinel-1, segmentation, sar-transfer]
---
# {rid}: converted copy of {man['dataset']}

Private working copy for the sar-transfer project. It is NOT the original dataset: the
radar is reduced to two channels ({' / '.join(man['channels'])}, units {man['units']}),
float16, cut into {man['chip_size']} px chips; labels as class ids ({classes}; 255 = ignore).

**Original source:** {src.get('name', src.get('title', man['dataset']))}
DOI / link: {src.get('doi', src.get('landing_page', src.get('bucket', 'see manifest.json')))}
**Licence of the original:** {lic}. This copy keeps that licence and its attribution terms.

| split | chips |
|---|---|
{splits}

## Citation of the original

```
{cite}
```

Full provenance (source files, converter notes, statistics): `manifest.json`.
"""


def _local_files(d: Path) -> dict[str, int]:
    import fnmatch
    return {p.name: p.stat().st_size for p in sorted(d.iterdir())
            if p.is_file() and any(fnmatch.fnmatch(p.name, pat) for pat in FILES)}


def upload(prepared_root: str | Path, dataset: str, owner: str, token: str,
           dry_run: bool = False) -> str:
    """Upload one converted dataset to a private repo; returns the repo id."""
    d = Path(prepared_root) / dataset
    bad = prepared.check_complete(d)
    if bad:
        raise IOError(f"{d}: not a complete converted dataset, not uploaded: {bad}")
    man = load_manifest(prepared_root, dataset)
    rid = repo_id(dataset, owner)
    (d / "README.md").write_text(card(man, rid), encoding="utf-8")
    files = _local_files(d)
    missing = [f"{s}_{k}" for s in man["splits"] for k in ("images.npy", "labels.npy", "meta.csv")
               if f"{s}_{k}" not in files]
    if missing:
        raise FileNotFoundError(f"{d}: converted files missing: {missing}")
    total = sum(files.values())
    print(f"{rid} (private): {len(files)} files, {total / 1e9:.2f} GB")
    if dry_run:
        for n, s in files.items():
            print(f"  {n:<32} {s / 1e6:10.1f} MB")
        return rid

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(rid, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(rid, repo_type="dataset").private:
        raise RuntimeError(f"{rid} exists and is PUBLIC -- refusing to upload into it")
    api.upload_folder(repo_id=rid, repo_type="dataset", folder_path=str(d), allow_patterns=list(FILES),
                      commit_message=f"converted {dataset}: " + json.dumps(man["splits"]))
    remote = {f.path: getattr(f, "size", None)
              for f in api.list_repo_tree(rid, repo_type="dataset", recursive=True)}
    bad = [n for n, s in files.items() if remote.get(n) != s]
    if bad:
        raise IOError(f"{rid}: after upload these files are missing or differ in size: {bad}")
    print(f"uploaded and checked: {len(files)} files, sizes equal")
    return rid


def same_conversion(a: dict, b: dict) -> bool:
    """Two manifests describe the same conversion: same creation time and the same splits."""
    return a.get("created_utc") == b.get("created_utc") and a.get("splits") == b.get("splits")


def _into_place(tmp: Path, final: Path) -> None:
    """Give a checked, complete copy its final name (an older folder of that name is removed)."""
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)


def fetch(dataset: str, owner: str, dest: str | Path, token: str | None,
          drive_manifest: dict | None = None) -> Path:
    """Download one converted dataset from its private repo into dest/<dataset>.

    The manifest comes first. If `drive_manifest` is given and the Hugging Face copy is another
    conversion (see same_conversion), nothing more is downloaded and ValueError is raised, so a
    stale copy is never used. The files land in a temporary folder that gets the final name only
    after prepared.check_complete passes.
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    rid, dest = repo_id(dataset, owner), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / f".{dataset}.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        man = json.loads(Path(hf_hub_download(rid, "manifest.json", repo_type="dataset", token=token,
                                              local_dir=str(tmp))).read_text(encoding="utf-8"))
        if drive_manifest is not None and not same_conversion(man, drive_manifest):
            raise ValueError(f"{rid} holds another conversion (created {man.get('created_utc')}) than Drive "
                             f"(created {drive_manifest.get('created_utc')}); re-run notebook 01's upload cell")
        snapshot_download(rid, repo_type="dataset", local_dir=str(tmp), token=token)
        bad = prepared.check_complete(tmp)
        if bad:
            raise IOError(f"{rid}: download incomplete: {bad}")
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    out = dest / dataset
    _into_place(tmp, out)
    print(f"{dataset}: {len(man['splits'])} splits ready in {out}")
    return out


def stage(names, dest: str | Path, drive_root: str | Path, owner: str | None = None,
          token: str | None = None) -> list[str]:
    """Bring each converted dataset to local disk once per runtime; returns the complete ones.

    Per dataset: a complete copy already in `dest` is kept. Else the private Hugging Face copy of
    `owner` is fetched (fast), but only if it is the same conversion as the complete copy on Drive
    (or Drive has none). Else the Drive copy is used. Every copy lands in a temporary folder first
    and gets its final name only after prepared.check_complete passes, so an interrupted copy is
    never taken for a finished one. owner=None skips Hugging Face.
    """
    import time

    dest, drive_root = Path(dest), Path(drive_root)
    dest.mkdir(parents=True, exist_ok=True)
    ready = []
    for name in names:
        out, t0 = dest / name, time.perf_counter()
        if out.exists() and not prepared.check_complete(out):
            ready.append(name)
            continue
        drive = drive_root / name
        drive_bad = prepared.check_complete(drive)
        drive_man = None if drive_bad else load_manifest(drive_root, name)
        if owner is not None:
            try:
                fetch(name, owner, dest, token, drive_manifest=drive_man)
                print(f"{name}: from Hugging Face in {(time.perf_counter() - t0) / 60:.1f} min")
                ready.append(name)
                continue
            except Exception as e:                  # e.g. no read access with this token, or a stale copy
                print(f"{name}: not taken from Hugging Face ({type(e).__name__}: {str(e)[:200]})")
        if drive_bad:
            print(f"{name}: no complete copy on Drive ({drive_bad[0]}); run notebook 01")
            continue
        tmp = dest / f".{name}.partial"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(drive, tmp)
        bad = prepared.check_complete(tmp)
        if bad:
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"{name}: copy from Drive incomplete, not used: {bad}")
            continue
        _into_place(tmp, out)
        print(f"{name}: copied from Drive in {(time.perf_counter() - t0) / 60:.1f} min")
        ready.append(name)
    return ready
