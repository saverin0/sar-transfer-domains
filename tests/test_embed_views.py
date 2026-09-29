"""Offline test of domains.embed_views on synthetic prepared domains (fake encoder, CPU)."""
import json, tempfile
from pathlib import Path
import numpy as np, pandas as pd
from _setup import TMP_BASE  # noqa: E402  (this repo's src first, import guard, temp in one folder)
TMP = TMP_BASE / "tmp"
TMP.mkdir(exist_ok=True)
from sartransfer.domains import embed_views as E, prepared as P, s1input as S
from sartransfer.models import encoders as encmod


class FakeEncoder:
    patch, dim, _n_prefix, n_layers = 16, 8, 0, 4
    spec = encmod.EncoderSpec("fake/model", "fake", "fake", batch=4)
    def __init__(self, *a, **k): pass
    def encode(self, tiles, batch=None, layer=None):
        n, _, t, _ = tiles.shape; g = t // 16
        b = tiles[:, :2].reshape(n, 2, g, 16, g, 16)
        m, s = b.mean((3, 5)), b.std((3, 5))
        return np.concatenate([m, s, m * 0.5, -m], 1).transpose(0, 2, 3, 1).astype(np.float16)


encmod.ENCODERS["fake"] = FakeEncoder.spec
encmod.FrozenEncoder = FakeEncoder
E.LAYERS = {"fake": 1}
E.NAMES = {"fake": "fake encoder"}


def make(root, res, name, k, size, seed):
    rng = np.random.default_rng(seed)
    w = P.PreparedWriter(root, name, size, {i: f"c{i}" for i in range(k)}, source={"url": "synthetic"},
                         pixel_spacing_m=10.0)
    for split, n in (("train", 12), ("test", 10)):
        for i in range(n):
            yy, xx = np.mgrid[:size, :size]
            cy, cx = rng.integers(size // 4, 3 * size // 4, 2)
            lab = (((yy - cy) ** 2 + (xx - cx) ** 2) < (size // 4 + i) ** 2).astype(np.uint8)
            if k == 3:
                lab[(xx > 3 * size // 4) & (lab == 0)] = 2
            vv = np.choose(lab, [-9.0, -22.0, -15.0][:k]) + rng.normal(0, 1.0, (size, size))
            img = np.stack([vv, vv - 7 + rng.normal(0, 1.0, (size, size))]).astype(np.float32)
            img[:, :4, :4] = np.nan
            lab[:4, :4] = 255
            w.add(split, img, lab, region=f"R{i % 2}", date="2020-01-01")
    w.close()
    stats = S.channel_stats(P.load_split(root, name, "train")["images"])
    (res / f"dom_{name}__fake__L1__vv_vh_diff__info.json").write_text(json.dumps({"stats": stats}))


root, res, out = (Path(tempfile.mkdtemp(dir=TMP)) for _ in range(3))
make(root, res, "toy_two", 2, 64, 0)
make(root, res, "toy_three", 3, 96, 1)

# unit checks of the fixed rules
sh, lf = E.patch_shares(np.array([[[1] * 16 + [255] * 16] * 16 + [[0] * 32] * 16], np.uint8), 2)
assert sh.shape == (1, 2, 2, 2) and lf[0, 0, 0] == 1.0 and lf[0, 0, 1] == 0.0
pure = E.pure_classes(sh, lf)
assert pure[0, 0, 0] == 1 and pure[0, 0, 1] == -1 and pure[0, 1, 0] == 0
q = E.query_patch(np.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1]], np.int16), np.zeros((3, 3, 2)) + 0.95, 1)
assert (q["row"], q["col"], q["fallback"]) == (1, 1, False), q
assert abs(E.auc_rank([2, 3], [0, 1]) - 1.0) < 1e-12 and abs(E.auc_rank([1], [1]) - 0.5) < 1e-12
# padded chips (as alpine): the 90 % is counted within the radar-covered area (correction 2026-09-29)
labs = np.full((3, 32, 32), 255, np.uint8)
labs[:, :24, :] = 0
labs[1, :24, :12] = 1                                             # most even class mix -> chip A
labs[2, :24, :4] = 1
one_region = np.array(["ALP"] * 3)
assert E.pick_chips(labs, np.full(3, 24 * 32), one_region, 2) == [1, 2]
try:                                                              # counted on the whole chip, none would pass
    E.pick_chips(labs, np.full(3, 32 * 32), one_region, 2)
    raise SystemExit("padded chips accepted on the whole-chip count")
except ValueError:
    pass
try:                                                              # radar on less than half the chip
    E.pick_chips(labs, np.full(3, 10 * 32), one_region, 2)
    raise SystemExit("chips with less than half radar coverage accepted")
except ValueError:
    pass

table = E.run_all(root, res, out, ["toy_two", "toy_three"], token=None, device="cpu")
assert set(table.dataset) == {"toy_two", "toy_three"} and len(table) == 2 + 3, table
for d in ("toy_two", "toy_three"):
    t = table[table.dataset == d]
    assert t.chip_A.nunique() == 1 and (t.chip_A != t.chip_B).all()
    lab = np.array(P.load_split(root, d, "test")["labels"])
    meta = P.load_split(root, d, "test")["meta"]
    assert meta.region.iloc[int(t.chip_A.iloc[0])] != meta.region.iloc[int(t.chip_B.iloc[0])], "B from another region"
    assert (t.auc > 0.9).all(), t[["class", "auc"]]                      # toy classes are well separated
    assert abs(t.top_k_share.sum() - 1) < 1e-6
    for f in ("maps", "similarity", "margins"):
        assert (out / f"embed_{f}_{d}.png").stat().st_size > 5000, f
assert (table[table.dataset == "toy_two"].query_class == "c1").all()
two = table[table.dataset == "toy_two"]                     # 21 pure c1 patches -> k = 20, all c1
assert (two.top_k == 20).all() and two[two["class"] == "c1"].top_k_share.iloc[0] == 1.0, two
# view 4 (AnyUp PCA of chips A and B) with a bilinear stand-in for AnyUp: the plumbing, not AnyUp itself
import torch.nn.functional as F  # noqa: E402
from sartransfer.models import anyup_loader as AL  # noqa: E402
AL.fetch_anyup_weights = lambda p: p
AL.load_anyup = lambda p, device="cuda": "stand-in"
AL.upsample_with_value = lambda m, guide, feats, value, out_size=None, q_chunk_size=None: F.interpolate(
    value, size=out_size, mode="bilinear", align_corners=False)
full = E.run_all_anyup(root, res, out, ["toy_two", "toy_three"], token=None, anyup_weights="unused", device="cpu")
assert set(full.features) == {"raw", "anyup_quarter"} and len(full) == 2 * len(table), full
raw_again = full[full.features == "raw"].reset_index(drop=True)
pd.testing.assert_frame_equal(raw_again, table.reset_index(drop=True))          # raw rows kept unchanged
au = full[full.features == "anyup_quarter"]
for d in ("toy_two", "toy_three"):
    r, q = table[table.dataset == d].iloc[0], au[au.dataset == d]
    assert (q.chip_A == r.chip_A).all() and (q.query_row == r.query_row).all() and (q.query_class == r.query_class).all()
    assert (q.top_k > r.top_k).all() and (q.top_k <= 16 * r.top_k).all(), q.top_k
    # bilinear stand-in blurs across class borders (real AnyUp follows edges), so a lower bar than raw's 0.9
    assert (q.auc > 0.85).all() and abs(q.top_k_share.sum() - 1) < 1e-6, q[["class", "auc", "top_k_share"]]
    for f in ("maps", "similarity", "margins"):
        assert (out / f"embed_{f}_anyup_{d}.png").stat().st_size > 5000, f
    assert (out / f"embed_maps_{d}.png").exists()                              # raw figures not overwritten
out2 = Path(tempfile.mkdtemp(dir=TMP))
(out2 / "embedding_numbers.csv").write_text((out / "embedding_numbers.csv").read_text())
twice = E.run_all_anyup(root, res, out2, ["toy_two", "toy_three"], token=None, anyup_weights="unused", device="cpu")
pd.testing.assert_frame_equal(full, twice)                                       # deterministic, AnyUp rows replaced
up = E.anyup_quarter("stand-in", np.ones((4, 4, 8), np.float16), np.zeros((2, 64, 64), np.float32),
                     {"co": (0.0, 1.0), "cross": (0.0, 1.0), "diff": (0.0, 1.0)}, device="cpu")
assert up.shape == (16, 16, 8)                                   # 1/4 of the 64 px chip
assert E._block_valid(np.full((2, 16, 16), np.nan)).sum() == 0 and E._block_valid(np.zeros((2, 16, 16))).all()
again = E.run_all(root, res, Path(tempfile.mkdtemp(dir=TMP)), ["toy_two", "toy_three"], token=None, device="cpu")
pd.testing.assert_frame_equal(table, again)                               # deterministic
print(table.round(3).to_string(index=False))
print("embed_views verified: fixed rules (pure patch, chips A/B, query), maps, similarity, class margins + AUC, "
      "figures, numbers CSV, determinism")
