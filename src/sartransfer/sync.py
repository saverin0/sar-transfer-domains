"""Pack the package source into a marked notebook cell, so Colab needs no git.

The Colab runtime cannot see local files. Cloning from GitHub works but forces a
push for every change to `src/`. This module removes that: the source travels
*inside* the `.ipynb`, which the Colab VS Code extension already sends to the
server when it runs the notebook.

Usage, always locally, never on the runtime::

    python sync.py            # pack into every notebook
    python sync.py --check    # exit 1 if a notebook is stale
    python sync.py nb.ipynb   # pack into specific notebooks
    python sync.py --clear    # empty the cells again, before publishing
    python sync.py --no-git   # pack in a folder that is not a git work tree (see below)

What travels: `src/sartransfer/**/*.py` and `pyproject.toml`, gzipped and
base64-encoded. What never travels: `.env`, anything under `.git`,
`__pycache__`, data, notebooks, results. Secrets in a notebook cell would be
committed to git, and inside base64 no search would find them, so:

- in a git work tree only files git tracks travel, and an untracked or
  git-ignored `.py` file under the package stops the sync (git add it or delete
  it), so nothing that .gitignore keeps out of the repository reaches a notebook;
- outside a git work tree, or when git fails (not installed, "dubious
  ownership", ...), the sync stops too, unless --no-git is given: then every
  `.py` under the package travels, with only the check below;
- any file with something that looks like a token or a private key stops it.

The payload is deterministic: sorted names, fixed mtime, no gzip timestamp. An
unchanged source tree produces a byte-identical cell, so `git diff` stays quiet
unless the code really changed.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path

BEGIN = "# <<<SARTRANSFER-SYNC-BEGIN>>>"
END = "# <<<SARTRANSFER-SYNC-END>>>"

PKG_DIR = Path(__file__).resolve().parent          # .../src/sartransfer
SRC_DIR = PKG_DIR.parent                           # .../src
REPO_ROOT = SRC_DIR.parent                         # repo root

EXCLUDE_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints"}

# Looks like a credential: Hugging Face, GitHub or AWS access tokens, a private key block.
SECRET_RE = re.compile(r"hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{50,}"
                       r"|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----")


# --------------------------------------------------------------------- packing

def _normalised(path: Path) -> bytes:
    """File contents with LF line endings.

    Git rewrites line endings on checkout, so reading raw bytes would make the
    fingerprint change after a clone even though no code did. Normalising here
    keeps `--check` honest across machines.
    """
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")


def _git_py_files(repo_root: Path, pkg_dir: Path) -> tuple[set[str], list[str]] | None:
    """(tracked .py files under pkg_dir, untracked or git-ignored .py files there), relative to
    repo_root; None when repo_root is not the top of a git work tree or git is missing."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True, text=True,
                              check=True).stdout
    rel = pkg_dir.relative_to(repo_root).as_posix()
    try:
        if Path(git("rev-parse", "--show-toplevel").strip()).resolve() != repo_root.resolve():
            return None
        tracked = {f for f in git("ls-files", "-z", "--", rel).split("\0") if f.endswith(".py")}
        other = sorted(f for f in git("ls-files", "-z", "--others", "--", rel).split("\0") if f.endswith(".py"))
    except (OSError, subprocess.CalledProcessError):
        return None
    return tracked, other


def collect_sources(pkg_dir: Path = PKG_DIR, repo_root: Path = REPO_ROOT,
                    allow_no_git: bool = False) -> dict[str, bytes]:
    """Return {archive path: bytes} for everything that should travel (see the module docstring)."""
    git = _git_py_files(repo_root, pkg_dir)
    if git is None and not allow_no_git:
        raise SystemExit(f"sync stopped: {repo_root} is not a git work tree, or git failed, so untracked and "
                         "git-ignored files cannot be told apart -- fix git, or run with --no-git to pack every .py")
    if git is not None and git[1]:
        raise SystemExit(f"sync stopped: untracked or git-ignored files {git[1]} -- git add them or delete them")
    files: dict[str, bytes] = {}
    paths = [p for p in sorted(pkg_dir.rglob("*.py")) if not any(part in EXCLUDE_DIRS for part in p.parts)]
    pyproject = repo_root / "pyproject.toml"
    for p in paths + ([pyproject] if pyproject.is_file() else []):
        data = _normalised(p)
        if SECRET_RE.search(data.decode("utf-8")):
            raise SystemExit(f"sync stopped: {p.relative_to(repo_root).as_posix()} contains something that "
                             "looks like a token or key")
        key = "pyproject.toml" if p == pyproject else f"sartransfer/{p.relative_to(pkg_dir).as_posix()}"
        files[key] = data
    return files


def make_payload(files: dict[str, bytes]) -> tuple[str, str]:
    """Deterministic gzipped tar, base64 encoded. Returns (payload, sha256)."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:        # uncompressed first
        for name in sorted(files):
            data = files[name]
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0                                   # no timestamps
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    import gzip

    gz = gzip.compress(raw.getvalue(), compresslevel=9, mtime=0)
    payload = base64.b64encode(gz).decode("ascii")
    return payload, hashlib.sha256(gz).hexdigest()[:16]


def source_fingerprint(files: dict[str, bytes]) -> str:
    """Hash of the source tree itself, independent of archive framing."""
    h = hashlib.sha256()
    for name in sorted(files):
        h.update(name.encode())
        h.update(hashlib.sha256(files[name]).digest())
    return h.hexdigest()[:16]


# ----------------------------------------------------------------- cell source

def build_cell(payload: str, sha: str, fingerprint: str, n_files: int) -> str:
    """The cell that unpacks the payload on the Colab runtime."""
    chunks = [payload[i:i + 96] for i in range(0, len(payload), 96)]
    literal = "\n".join(f'    "{c}"' for c in chunks)
    return f'''{BEGIN}
# Generated by `python sync.py` -- do not edit by hand.
# source fingerprint: {fingerprint}   files: {n_files}   payload sha256: {sha}
#
# The package source travels inside this notebook, so the Colab runtime needs no
# git, no clone and no GitHub token. Re-run `python sync.py` locally
# after changing anything under src/, then re-run this cell.
import base64, gzip, hashlib, io, sys, tarfile, subprocess
from pathlib import Path

_PAYLOAD = (
{literal}
)
_EXPECTED_SHA = "{sha}"
_DEST = Path("/content/sartransfer-src")

_gz = base64.b64decode(_PAYLOAD)
_got = hashlib.sha256(_gz).hexdigest()[:16]
assert _got == _EXPECTED_SHA, f"payload corrupted: {{_got}} != {{_EXPECTED_SHA}}"

if _DEST.exists():
    import shutil
    shutil.rmtree(_DEST)
_DEST.mkdir(parents=True)
with tarfile.open(fileobj=io.BytesIO(gzip.decompress(_gz))) as _tar:
    try:
        _tar.extractall(_DEST, filter="data")   # refuses paths outside _DEST
    except TypeError:
        _tar.extractall(_DEST)                  # Python < 3.12 has no filter

if str(_DEST) in sys.path:
    sys.path.remove(str(_DEST))
sys.path.insert(0, str(_DEST))
for _m in [m for m in sys.modules if m == "sartransfer" or m.startswith("sartransfer.")]:
    del sys.modules[_m]

# Colab already ships numpy and pandas. Install only what is absent (exact versions, pinned
# 2026-09-29).
for _mod, _pip in [("numpy", "numpy==2.5.3"), ("pandas", "pandas==3.0.6")]:
    try:
        __import__(_mod)
    except ImportError:
        print("installing", _pip)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", _pip], check=True)

import sartransfer
print("sartransfer", sartransfer.__version__, "from", sartransfer.__file__)
print("source fingerprint:", "{fingerprint}", "|", {n_files}, "files")
{END}'''


# ------------------------------------------------------------------- notebooks

def find_notebooks(explicit: list[str]) -> list[Path]:
    if explicit:
        return [Path(p) for p in explicit]
    return sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))


def cell_index(nb: dict) -> int | None:
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code" and BEGIN in "".join(c["source"]):
            return i
    return None


def embedded_fingerprint(nb: dict) -> str | None:
    i = cell_index(nb)
    if i is None:
        return None
    for line in "".join(nb["cells"][i]["source"]).splitlines():
        if line.startswith("# source fingerprint:"):
            return line.split()[3]
    return None


def sync(paths: list[Path], check_only: bool = False, allow_no_git: bool = False) -> int:
    files = collect_sources(allow_no_git=allow_no_git)
    fingerprint = source_fingerprint(files)
    payload, sha = make_payload(files)
    cell_src = build_cell(payload, sha, fingerprint, len(files))

    stale, missing = [], []
    for p in paths:
        nb = json.loads(p.read_text(encoding="utf-8"))
        i = cell_index(nb)
        if i is None:
            missing.append(p)
            continue
        if embedded_fingerprint(nb) == fingerprint:
            print(f"up to date  {p.name}")
            continue
        stale.append(p)
        if check_only:
            continue
        nb["cells"][i]["source"] = cell_src.splitlines(True)
        nb["cells"][i]["outputs"] = []
        nb["cells"][i]["execution_count"] = None
        p.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print(f"updated     {p.name}")

    for p in missing:
        print(f"NO SYNC CELL  {p.name} -- add one containing {BEGIN}")

    print(f"\n{len(files)} files | fingerprint {fingerprint} | payload {len(payload)/1024:.1f} kB")

    if check_only and (stale or missing):
        print("\nstale:", ", ".join(p.name for p in stale) or "none")
        print("run: python sync.py")
        return 1
    return 0


CLEARED = f"""{BEGIN}
# Empty sync cell: published notebooks carry no packed code. On your own machine,
# in the repository folder, run   python sync.py   and open the notebook again;
# this cell then holds the package source and unpacks it on the runtime.
raise RuntimeError("The sync cell is empty. On your own machine, in the repository folder, "
                   "run:  python sync.py  and reopen this notebook.")
{END}"""


def clear(paths: list[Path]) -> int:
    """Replace every sync cell's payload by a short stub that explains how to fill it."""
    for p in paths:
        nb = json.loads(p.read_text(encoding="utf-8"))
        i = cell_index(nb)
        if i is None:
            print(f"NO SYNC CELL  {p.name}")
            continue
        nb["cells"][i]["source"] = CLEARED.splitlines(True)
        nb["cells"][i]["outputs"] = []
        nb["cells"][i]["execution_count"] = None
        p.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print(f"cleared     {p.name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("notebooks", nargs="*", help="default: every notebook in notebooks/")
    ap.add_argument("--check", action="store_true",
                    help="report stale notebooks and exit 1, changing nothing")
    ap.add_argument("--clear", action="store_true",
                    help="empty the sync cells (no packed code), e.g. before publishing")
    ap.add_argument("--no-git", action="store_true",
                    help="pack every .py although this is not a git work tree (only the token check applies)")
    args = ap.parse_args(argv)
    if args.clear:
        return clear(find_notebooks(args.notebooks))
    return sync(find_notebooks(args.notebooks), check_only=args.check, allow_no_git=args.no_git)


if __name__ == "__main__":
    sys.exit(main())
