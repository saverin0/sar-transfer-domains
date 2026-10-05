"""Offline checks of the safety rules: tar members, what the sync cell may pack, which secrets reach os.environ."""
import io, os, shutil, subprocess, tarfile, tempfile
from pathlib import Path
from _setup import TMP_BASE  # noqa: E402  (this repo's src first, import guard, temp in one folder)
from sartransfer import env, sync
from sartransfer.domains import glacial_lakes as G

TMP = Path(tempfile.mkdtemp(dir=TMP_BASE))


# ---- tar: every member checked before extraction, also on the Python branch
def make_tgz(path: Path, members: list[tuple[str, str]]) -> Path:
    with tarfile.open(path, "w:gz") as t:
        for name, kind in members:
            info = tarfile.TarInfo(name)
            if kind == "dir":
                info.type = tarfile.DIRTYPE
                t.addfile(info)
            elif kind == "link":
                info.type, info.linkname = tarfile.SYMTYPE, "../../outside"
                t.addfile(info)
            else:
                info.size = 2
                t.addfile(info, io.BytesIO(b"ok"))
    return path

for bad, why in ((("GLB/../../evil.txt", "file"), ".."), (("/abs/evil.txt", "file"), "absolute"),
                 (("GLB/link", "link"), "link")):
    tgz = make_tgz(TMP / "bad.tar.gz", [("GLB", "dir"), bad])
    with tarfile.open(tgz, "r:gz") as t:
        try:
            list(G._checked_members(t, TMP / "x"))
            raise AssertionError(f"{why} member accepted")
        except RuntimeError as e:
            assert "unsafe member" in str(e), e
ok = make_tgz(TMP / "ok.tar.gz", [("GLB", "dir"), ("GLB/a.txt", "file"), ("GLB/sub", "dir"), ("GLB/sub/b.txt", "file")])
(TMP / "ex").mkdir()
which = shutil.which
shutil.which = lambda name: None                                    # force the Python tarfile branch
try:
    out = G._extract_tgz(ok, TMP / "ex", "GLB")
finally:
    shutil.which = which
assert (out / "a.txt").read_bytes() == b"ok" and (out / "sub" / "b.txt").exists() and (out / G.DONE_MARKER).exists()
print("tar member check ok")

# ---- sync: only git-tracked files travel; untracked, ignored or token-like files stop it
fake_tok = "hf_" + "A1b2C3d4" * 5
repo = TMP / "repo"
pkg = repo / "src" / "sartransfer"
pkg.mkdir(parents=True)
(pkg / "a.py").write_text("x = 1\n")
(repo / "pyproject.toml").write_text("[project]\nname = 'x'\n")
try:                                                                # not a git work tree: refused ...
    sync.collect_sources(pkg, repo)
    raise AssertionError("packed outside git without --no-git")
except SystemExit as e:
    assert "not a git work tree" in str(e), e
got = sync.collect_sources(pkg, repo, allow_no_git=True)            # ... unless asked: everything travels
assert set(got) == {"sartransfer/a.py", "pyproject.toml"}, set(got)
if shutil.which("git"):
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)  # noqa: E731
    git("init", "-q")
    (repo / ".gitignore").write_text("*token*\n")
    git("add", ".gitignore", "pyproject.toml", "src/sartransfer/a.py")   # staged only, nothing committed
    assert set(sync.collect_sources(pkg, repo)) == {"sartransfer/a.py", "pyproject.toml"}
    for name, text in (("b.py", "y = 2\n"), ("hf_token_check.py", "z = 3\n")):
        (pkg / name).write_text(text)                               # untracked, then git-ignored
        try:
            sync.collect_sources(pkg, repo)
            raise AssertionError(f"{name} was packed")
        except SystemExit as e:
            assert "untracked or git-ignored" in str(e) and name in str(e), e
        (pkg / name).unlink()
    (pkg / "c.py").write_text(f"TOKEN = '{fake_tok}'\n")
    git("add", "src/sartransfer/c.py")
    try:
        sync.collect_sources(pkg, repo)
        raise AssertionError("a token-like string was packed")
    except SystemExit as e:
        assert "looks like a token" in str(e) and fake_tok not in str(e), e
    print("sync refusals ok (git)")
else:
    print("git not found: sync git checks skipped")

# ---- secrets: load_env exports only the keys asked for; secret() never touches os.environ
keys = ("HF_TOKEN", "HF_WRITE_TOKEN", "OTHER_KEY")
saved = {k: os.environ.pop(k) for k in keys if k in os.environ}
try:
    f = TMP / ".env"
    f.write_text("HF_TOKEN=read123\nHF_WRITE_TOKEN=write456\nOTHER_KEY=other789\n")
    assert env.load_env(f, verbose=False) == ["HF_TOKEN"]
    assert os.environ.get("HF_TOKEN") == "read123"
    assert "HF_WRITE_TOKEN" not in os.environ and "OTHER_KEY" not in os.environ
    assert env.secret("HF_WRITE_TOKEN", path=f) == "write456" and "HF_WRITE_TOKEN" not in os.environ
finally:
    for k in keys:
        os.environ.pop(k, None)
    os.environ.update(saved)
print("secrets ok")
shutil.rmtree(TMP, ignore_errors=True)
print("SECURITY CHECKS PASSED")
