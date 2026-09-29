"""Offline test of domains.run on a synthetic prepared 'lake' domain (fake encoder, CPU)."""
import json, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
from _setup import TMP_BASE  # noqa: E402  (this repo's src first, import guard, temp in one folder)
TMP = TMP_BASE / "tmp"
TMP.mkdir(exist_ok=True)
from sartransfer.domains import prepared as P, run as R
from sartransfer.models import encoders as encmod

rng = np.random.default_rng(0)
root, res = Path(tempfile.mkdtemp(dir=TMP)), Path(tempfile.mkdtemp(dir=TMP))
w = P.PreparedWriter(root, "toy_lakes", 64, {0: "background", 1: "lake"}, source={"url": "synthetic"}, pixel_spacing_m=10.0)
for split, n in (("train", 40), ("val", 8), ("test", 8), ("test_challenge", 5)):
    for i in range(n):
        yy, xx = np.mgrid[:64, :64]
        cy, cx, r = rng.integers(16, 48, 2).tolist() + [rng.integers(8, 20)]
        lake = ((yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2)
        vv = np.where(lake, -22.0, -9.0) + rng.normal(0, 1.5, (64, 64))
        vh = vv - 7 + rng.normal(0, 1.0, (64, 64))
        img = np.stack([vv, vh]).astype(np.float32); img[:, :4, :4] = np.nan
        lab = lake.astype(np.uint8); lab[:4, :4] = 255
        w.add(split, img, lab, region=f"RGI{i % 3}", date="2020-08-01")
w.close()

class FakeEncoder:
    patch, dim, _n_prefix, n_layers = 16, 8, 0, 4
    spec = encmod.EncoderSpec("fake/model", "fake", "fake", batch=4)
    def __init__(self, *a, **k): pass
    def encode(self, tiles, batch=None, layer=None):
        n, _, t, _ = tiles.shape; g = t // 16
        b = tiles[:, :2].reshape(n, 2, g, 16, g, 16)
        m, s = b.mean((3, 5)), b.std((3, 5))
        return np.concatenate([m, s, m * 0.5, -m], 1).transpose(0, 2, 3, 1).astype(np.float16)   # standardised input -> O(1) features
encmod.ENCODERS["fake"] = FakeEncoder.spec
encmod.FrozenEncoder = FakeEncoder

out = R.run_domain(root, "toy_lakes", "fake", res, token=None, layer=1, seeds=(0, 1), steps=40, batch=4, device="cpu")
heads = sorted(set(zip(out["head"], out["seed"])))
assert heads == [("decoder", 0), ("decoder", 1), ("majority", -1), ("probe", -1)], heads
assert set(out.split) == {"val", "test", "test_challenge"}, set(out.split)
assert {"IoU_pos", "F1_pos", "mIoU", "accuracy"} <= set(out.columns)
probe = out[out["head"] == "probe"]
assert probe.patch_mIoU.notna().all() and (probe.mIoU > out[out["head"] == "majority"].mIoU.values).all(), probe
run = "dom_toy_lakes__fake__L1__vv_vh_diff"
for f in ("__summary.csv", "__probe.pt", "__decoder__s0.pt", "__decoder__s1.pt", "__info.json",
          "__probe__regions.csv", "__decoder__s1__regions.csv", "__majority__regions.csv"):
    assert (res / f"{run}{f}").exists(), f
reg = pd.read_csv(res / f"{run}__probe__regions.csv")
assert set(reg.region) == {"RGI0", "RGI1", "RGI2"} and set(reg.split) == {"val", "test", "test_challenge"}
info = json.loads((res / f"{run}__info.json").read_text())
assert info["train_chips_decoder"] == 40 and set(info["stats"]) == {"co", "cross", "diff"}
assert info["settings"]["steps"] == 40 and info["settings"]["batch"] == 4 and info["settings"]["data_created_utc"]
assert "torch" in info["versions"] and "numpy" in info["versions"], info["versions"]
again = R.run_domain(root, "toy_lakes", "fake", res, token=None, layer=1, seeds=(0, 1), steps=40, batch=4, device="cpu")
assert len(again) == len(out)                                       # same settings: skipped, summary returned
try:                                                                # other settings: refused, not skipped
    R.run_domain(root, "toy_lakes", "fake", res, token=None, layer=1, seeds=(0, 1), steps=80, batch=4, device="cpu")
    raise AssertionError("a finished run with other settings was skipped")
except ValueError as e:
    assert "other settings" in str(e) and "steps" in str(e), e
# a run finished before settings were recorded counts as the pre-registered settings
old_info = {k: v for k, v in info.items() if k != "settings"}
(res / f"{run}__info.json").write_text(json.dumps(old_info))
try:
    R.run_domain(root, "toy_lakes", "fake", res, token=None, layer=1, seeds=(0, 1), steps=40, batch=4, device="cpu")
    raise AssertionError("an old run was taken to match non-pre-registered settings")
except ValueError:
    pass
R._check_same_settings(res / f"{run}__info.json", {**R.PRE_REGISTERED_SETTINGS, "data_created_utc": "x"})
(res / f"{run}__info.json").write_text(json.dumps(info))
tab = R.domain_table(res)
assert set(tab["head"]) == {"decoder", "majority", "probe"} and (tab[tab["head"] == "decoder"].seeds == 2).all(), tab
assert set(tab["run"]) == {run}, set(tab["run"])
# a variant run (other layer) stays a separate row, never averaged into the first
v = pd.read_csv(res / f"{run}__summary.csv").assign(run=run.replace("__L1__", "__L2__"), layer=2)
v.to_csv(res / f"{run.replace('__L1__', '__L2__')}__summary.csv", index=False)
tab2 = R.domain_table(res)
assert len(tab2) == 2 * len(tab) and (tab2[tab2["head"] == "decoder"].seeds == 2).all(), tab2
(res / f"{run.replace('__L1__', '__L2__')}__summary.csv").unlink()
# encoder batch: unchanged for chips dividing 512 px, smaller for larger chips
assert [R.encoder_batch(32, s) for s in (64, 128, 256, 512, 992)] == [512, 512, 128, 32, 8]
assert [R.encoder_batch(16, s) for s in (128, 256, 512, 992)] == [256, 64, 16, 4]
print(tab.round(3).to_string(index=False))

# per-chip counting (the scoring path since 2026-09-28) equals the reference CPU count, for split and region sums
import torch  # noqa: E402
for k in (2, 3, 8):
    t = rng.integers(0, k, (7, 40, 40)).astype(np.uint8)
    t[rng.random(t.shape) < 0.3] = 255
    p = rng.integers(0, k, t.shape).astype(np.uint8)
    cc = R._chip_conf(torch.from_numpy(p), torch.from_numpy(t), k)
    assert cc.shape == (7, k, k) and (cc.sum(0) == R._confusion(p, t, k)).all()
    assert (cc[2:5].sum(0) == R._confusion(p[2:5], t[2:5], k)).all()
# enc_cache: one encoder load shared by two runs (forced rerun of the same data counts as a second dataset)
loads = []
FakeEncoder.__init__ = lambda self, *a, **k: loads.append(1)
cache, res2 = {}, Path(tempfile.mkdtemp(dir=TMP))
for _ in range(2):
    R.run_domain(root, "toy_lakes", "fake", res2, token=None, layer=1, seeds=(0,), steps=10, batch=4, device="cpu",
                 force=True, enc_cache=cache)
assert len(loads) == 1 and "fake" in cache, (len(loads), list(cache))
print("run_domain verified: probe + decoder seeds + majority on every eval split, region tables, info, skip, domain_table")
