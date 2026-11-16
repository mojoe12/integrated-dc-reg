#!/usr/bin/env python3
"""Paired test of sphere-mean FEAT z, integrated versus TOPUP.

Each subject contributes one number per region: the mean of the first-level
z-statistic inside a 12 mm sphere. The sphere is the full ball on the z-map
grid. Voxels outside the MNI brain stay in the mean.

Two paired tests are reported for the same pairs, both two-sided:

* Student's paired t, which produced the Holm values 0.010, 0.040, and 0.071
  for 2-back versus baseline across the four task spheres.
* Wilcoxon signed-rank.

Holm correction is applied inside one contrast and one ROI family.
`group_block` in the weight-2 FEAT script is a different test: a one-sample
t of each method against zero.

    HCP_ROOT=/path/to/hcp python3 paired_roi_integrated_vs_topup.py
"""
from __future__ import annotations

import os
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

HCP_ROOT = Path(os.environ.get("HCP_ROOT", "/mnt/data/projects/jhutter/data/HCP"))
RADIUS_MM = 12.0
KEEP = [
    "100206", "100307", "100408", "100610", "101006", "101107", "101309", "101410",
    "101915", "102008", "102109", "102311", "102513", "102614", "102715", "102816",
    "103010", "103111", "103212", "103414", "103515", "103818", "104012", "104416",
    "104820", "105014", "105115", "105216", "105620", "105923", "106016", "106319",
    "106521", "106824", "107018", "107220",
]
# Power et al. 2011. IPS nodes are Fronto-parietal Task Control (191, 199).
# IPL nodes 86 and 96 are the default-mode pair used in the original seven.
ROIS = [
    ("L DLPFC", -42.0, 25.0, 30.0),
    ("R DLPFC", 48.0, 25.0, 27.0),
    ("L IPS", -28.0, -58.0, 48.0),
    ("R IPS", 33.0, -53.0, 44.0),
    ("L IPL", -44.0, -65.0, 35.0),
    ("R IPL", 52.0, -59.0, 36.0),
    ("PCC", -7.0, -55.0, 27.0),
    ("mPFC", 6.0, 54.0, 16.0),
    ("OFC", -3.0, 44.0, -9.0),
]
FAMILIES = {
    "four task spheres": ["L DLPFC", "R DLPFC", "L IPS", "R IPS"],
    "seven original spheres": [
        "L DLPFC", "R DLPFC", "L IPL", "R IPL", "PCC", "mPFC", "OFC",
    ],
}
CONTRASTS = (
    ("2-back versus baseline", "zstat9"),
    ("2-back minus 0-back", "zstat11"),
)


def holm(pvals):
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    adj = [0.0] * n
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(float(pvals[i]) * (n - rank), 1.0))
        adj[i] = running
    return adj


def zpath(sid, method, zname):
    return (HCP_ROOT / sid / "unwarp_compare/hcp_feat"
            / f"{method}_hp200_s4.feat/stats/{zname}.nii.gz")


def spheres(img):
    shape = img.shape[:3]
    grid = np.array(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"))
    xyz = nib.affines.apply_affine(img.affine, np.stack(grid, -1).reshape(-1, 3))
    xyz = xyz.reshape(*shape, 3)
    out = {}
    for name, x, y, z in ROIS:
        out[name] = (
            (xyz[..., 0] - x) ** 2
            + (xyz[..., 1] - y) ** 2
            + (xyz[..., 2] - z) ** 2
        ) <= RADIUS_MM ** 2
    return out


def subject_means(sids, masks):
    """store[contrast index][roi] = (n_subjects, 2) array, columns TOPUP, integrated."""
    store = {
        zname: {name: [] for name, *_ in ROIS}
        for _, zname in CONTRASTS
    }
    for sid in sids:
        loaded = {}
        for _, zname in CONTRASTS:
            top = np.asanyarray(nib.load(str(zpath(sid, "topup", zname))).dataobj, dtype=np.float32)
            integ = np.asanyarray(
                nib.load(str(zpath(sid, "sbrigid_l6w2", zname))).dataobj, dtype=np.float32
            )
            if top.shape != integ.shape:
                raise RuntimeError(f"{sid} {zname} shape {top.shape} != {integ.shape}")
            loaded[zname] = (top, integ)
        for name, mask in masks.items():
            for zname, (top, integ) in loaded.items():
                store[zname][name].append((float(top[mask].mean()), float(integ[mask].mean())))
    return {
        zname: {name: np.asarray(rows, dtype=np.float64) for name, rows in rois.items()}
        for zname, rois in store.items()
    }


def paired_row(name, pair):
    top = pair[:, 0]
    integ = pair[:, 1]
    diff = integ - top
    t_res = ttest_rel(integ, top)
    w_res = wilcoxon(integ, top, alternative="two-sided", zero_method="wilcox")
    return {
        "roi": name,
        "n": int(pair.shape[0]),
        "top": float(top.mean()),
        "integrated": float(integ.mean()),
        "diff": float(diff.mean()),
        "t": float(t_res.statistic),
        "p_t": float(t_res.pvalue),
        "p_w": float(w_res.pvalue),
        "n_higher": int(np.sum(diff > 0)),
    }


def main():
    missing = [
        f"{sid} {zname}"
        for sid in KEEP
        for _, zname in CONTRASTS
        for method in ("topup", "sbrigid_l6w2")
        if not zpath(sid, method, zname).exists()
    ]
    if missing:
        raise SystemExit("missing z-stat: " + " ".join(missing[:8]))
    ref = nib.load(str(zpath(KEEP[0], "topup", "zstat9")))
    masks = spheres(ref)
    means = subject_means(KEEP, masks)
    print(
        f"n={len(KEEP)}  full {RADIUS_MM:.0f} mm sphere  "
        "paired integrated minus TOPUP"
    )
    print("Holm is within the contrast and the ROI family.")
    for title, zname in CONTRASTS:
        by_roi = means[zname]
        for family, names in FAMILIES.items():
            rows = [paired_row(name, by_roi[name]) for name in names]
            t_adj = holm([row["p_t"] for row in rows])
            w_adj = holm([row["p_w"] for row in rows])
            print(f"\n## {title} | {family} | Holm m={len(names)}")
            print(
                f"{'region':8} {'TOPUP':>7} {'integr':>7} {'diff':>8} "
                f"{'t':>6} {'p_t':>9} {'Holm_t':>7} "
                f"{'p_W':>9} {'Holm_W':>7} {'n+':>5}"
            )
            for row, ht, hw in zip(rows, t_adj, w_adj):
                print(
                    f"{row['roi']:8} {row['top']:7.3f} {row['integrated']:7.3f} "
                    f"{row['diff']:+8.3f} {row['t']:6.2f} {row['p_t']:9.4g} {ht:7.3f} "
                    f"{row['p_w']:9.4g} {hw:7.3f} {row['n_higher']:2d}/{row['n']}"
                )


if __name__ == "__main__":
    main()
