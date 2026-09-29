"""Offline check of domains/hub.py: card, licence ids, file selection, dry-run upload, and the
completeness check that staging and upload rely on (no network: staging from Drive only)."""
import json, shutil, tempfile
from pathlib import Path
import numpy as np
from _setup import TMP_BASE  # noqa: E402  (this repo's src first, import guard, temp in one folder)
from sartransfer.domains import hub, prepared, snow, forest, alpine, glacial_lakes
TMP = Path(tempfile.mkdtemp(dir=TMP_BASE))
want = {"snow": "etalab-2.0", "forty_v1": "cc-by-sa-4.0", "glavitu_alp": "cc-by-4.0", "glacial_lakes": "cc-by-4.0"}
for mod in (snow, forest, alpine, glacial_lakes):
    w = prepared.PreparedWriter(TMP, mod.DATASET, 32, mod.CLASSES if hasattr(mod, "CLASSES") else {0: "a", 1: "b"}, source=mod.SOURCE)
    for split in ("train", "val"):
        for _ in range(3):
            w.add(split, np.zeros((2, 32, 32), np.float32), np.zeros((32, 32), np.uint8), region="x", date="")
    w.close()
    rid = hub.upload(TMP, mod.DATASET, "saverin00", token="unused", dry_run=True)
    c = (TMP / mod.DATASET / "README.md").read_text(encoding="utf-8")
    lic = c.split("license: ", 1)[1].split("\n", 1)[0]
    assert lic == want[mod.DATASET], (mod.DATASET, lic)
    assert rid == f"saverin00/sar-transfer-{mod.DATASET.replace('_', '-')}"
    outside_code = "".join(c.split("```")[0::2])                      # the BibTeX block may hold braces
    assert "Citation of the original" in c and "{" not in outside_code and "}" not in outside_code, c
    if mod.DATASET == "snow":
        assert "IMTSFL_2025" in c
    # hidden part files and other files are never selected
    (TMP / mod.DATASET / ".train_images.part00000.npy").write_bytes(b"x")
    (TMP / mod.DATASET / "notes.txt").write_text("x")
    sel = hub._local_files(TMP / mod.DATASET)
    assert ".train_images.part00000.npy" not in sel and "notes.txt" not in sel and "README.md" in sel, sel
    print(f"  {mod.DATASET:<14} -> {rid}  licence {lic}")

# ---- completeness check, staging from Drive, upload refusal
drive, local = TMP / "drive", TMP / "local"
for name in ("toy_a", "toy_b"):
    w = prepared.PreparedWriter(drive, name, 32, {0: "a", 1: "b"}, source={"url": "synthetic"})
    for split, n in (("train", 4), ("val", 2)):
        for _ in range(n):
            w.add(split, np.ones((2, 32, 32), np.float32), np.zeros((32, 32), np.uint8), region="x", date="")
    w.close()
assert prepared.check_complete(drive / "toy_a") == []
f = drive / "toy_b" / "train_images.npy"
f.write_bytes(f.read_bytes()[: f.stat().st_size // 2])              # a copy cut off mid-way
bad = prepared.check_complete(drive / "toy_b")
assert bad and "train_images.npy" in bad[0], bad
assert prepared.check_complete(drive / "toy_missing") == ["manifest.json unusable (FileNotFoundError)"]
try:
    hub.upload(drive, "toy_b", "saverin00", token="unused", dry_run=True)
    raise AssertionError("an incomplete folder was accepted for upload")
except IOError as e:
    assert "not a complete converted dataset" in str(e), e

(local / ".toy_a.partial").mkdir(parents=True)                      # leftover of an interrupted copy
(local / ".toy_a.partial" / "junk.npy").write_bytes(b"x")
ready = hub.stage(["toy_a", "toy_b", "toy_c"], local, drive, owner=None)
assert ready == ["toy_a"], ready
assert prepared.check_complete(local / "toy_a") == [] and not (local / ".toy_a.partial").exists()
assert not (local / "toy_b").exists() and not (local / ".toy_b.partial").exists()
m = local / "toy_a" / "val_meta.csv"
m.write_text("\n".join(m.read_text().splitlines()[:-1]) + "\n")      # local copy broken later
assert prepared.check_complete(local / "toy_a")
assert hub.stage(["toy_a"], local, drive, owner=None) == ["toy_a"] and prepared.check_complete(local / "toy_a") == []

man = prepared.load_manifest(drive, "toy_a")
assert hub.same_conversion(man, json.loads(json.dumps(man)))
assert not hub.same_conversion(man, {**man, "created_utc": "2000-01-01T00:00:00Z"})
assert not hub.same_conversion(man, {**man, "splits": {**man["splits"], "train": 5}})
shutil.rmtree(TMP)
print("HUB CHECKS PASSED (card, upload refusal, check_complete, stage from Drive, same_conversion)")
