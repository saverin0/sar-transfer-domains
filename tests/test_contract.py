"""Offline test of domains.prepared (writer/reader/validate) and domains.s1input."""
import sys, tempfile
from pathlib import Path
import numpy as np
from _setup import TMP_BASE  # noqa: E402  (this repo's src first, import guard, temp in one folder)
TMP = TMP_BASE / "tmp"
TMP.mkdir(exist_ok=True)
from sartransfer.domains import prepared as P, s1input as S

rng = np.random.default_rng(0)
root = Path(tempfile.mkdtemp(dir=TMP))
w = P.PreparedWriter(root, "toy", 32, {0: "background", 1: "lake"}, source={"url": "synthetic"}, pixel_spacing_m=10.0)
P.CHUNK = 7                                               # force several part files
for split, n in (("train", 20), ("test", 9)):
    for i in range(n):
        img = rng.normal(-15, 4, (2, 32, 32)).astype(np.float32); img[1] -= 7
        img[:, :3, :3] = np.nan
        lab = (rng.random((32, 32)) < 0.3).astype(np.uint8); lab[:3, :3] = 255
        w.add(split, img, lab, region=f"r{i % 3}", date="2020-07-01")
man = w.close()
assert man["splits"] == {"test": 9, "train": 20}, man
assert not list((root / "toy").glob(".*part*")), "part files left behind"
d = P.load_split(root, "toy", "train")
assert d["images"].shape == (20, 2, 32, 32) and d["images"].dtype == np.float16 and d["labels"].dtype == np.uint8
assert list(d["meta"].chip_id[:2]) == ["train_0000000", "train_0000001"] and set(d["meta"].region) == {"r0", "r1", "r2"}
v = P.validate(root, "toy")
assert set(v.split) == {"train", "test"} and (v.nan_share > 0).all()
for bad in (dict(image=np.zeros((2, 16, 16)), label=np.zeros((32, 32))),
            dict(image=np.zeros((2, 32, 32)), label=np.full((32, 32), 7)),
            dict(image=np.full((2, 32, 32), np.inf), label=np.zeros((32, 32)))):
    try:
        P.PreparedWriter(root, "toy2", 32, {0: "a", 1: "b"}, source={}).add("train", **bad); raise SystemExit(f"accepted {bad}")
    except ValueError:
        pass
st = S.channel_stats(d["images"])
x = S.encoder_input(d["images"][:4], st)
assert x.shape == (4, 3, 32, 32) and x.dtype == np.float32 and np.isfinite(x).all() and (x[:, :, :3, :3] == 0).all()
g = S.encoder_input(d["images"][:4], st, "vv_grey")
assert np.array_equal(g[:, 0], g[:, 2])
sk = S.skip_image(d["images"][:4], st)
assert sk.shape == (4, 1, 32, 32) and sk.min() >= 0 and sk.max() <= 1
print("contract verified: writer parts + close, reader mmap, validate, bad input refused, S1 input modes, skip image")
