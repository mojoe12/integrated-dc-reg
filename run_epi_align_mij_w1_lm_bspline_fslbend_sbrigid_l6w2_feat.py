#!/usr/bin/env python3
"""Individual FEAT for the rigid spin-echo placement and the weight-2 ramp.

Field: <sid>/unwarp_compare/u_epi_align_mij_w1_lm_bspline_fslbend_sbrigid_l6w2.nii.gz
FEAT directory: <sid>/unwarp_compare/hcp_feat/sbrigid_l6w2_hp200_s4.feat

The functional series is moved into the spin-echo frame with sb_to_se_sbrigid.mat,
unwarped with applytopup --method=jac --interp=spline, then moved back with
se_to_sb_sbrigid.mat. Both resamples are spline. The saved field is displacement
in voxels; applytopup receives it in hertz (divided by the readout in acqp.txt).
The rigid has determinant 1, so it adds no intensity modulation. Motion
correction, the shared epi_to_mni.mat, high-pass 200 s, 4 mm smoothing, and FILM
then match the other methods. Left-right only.

Does not replace w1lm or w1bsbend FEATs, the TOPUP field, the TOPUP-unwarped
images, or the raw series. The none and topup FEAT directories are not required
to run these FEATs. The comparison tables are written when those directories
have zstat9, zstat11, and zstat14.

    HCP_ROOT=/path/to/hcp python3 run_epi_align_mij_w1_lm_bspline_fslbend_sbrigid_l6w2_feat.py --jobs 2
    HCP_ROOT=/path/to/hcp python3 run_epi_align_mij_w1_lm_bspline_fslbend_sbrigid_l6w2_feat.py --group-only
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("FSLDIR", "/usr/local/fsl")
os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"

import nibabel as nib
import numpy as np
from scipy.stats import ttest_1samp, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_unwarp_pcc as C
import run_hcp_feat_after_unwarp as F

METHOD = "sbrigid_l6w2"
NATIVE = "wm_sbrigid_l6w2.nii.gz"
FIELD = "u_epi_align_mij_w1_lm_bspline_fslbend_sbrigid_l6w2.nii.gz"
HZ_NAME = "sbrigid_l6w2_hz.nii.gz"
SB_TO_SE = "sb_to_se_sbrigid.mat"
SE_TO_SB = "se_to_sb_sbrigid.mat"
F.METHODS[METHOD] = NATIVE

KEEP = [
    "100206", "100307", "100408", "100610", "101006", "101107", "101309", "101410",
    "101915", "102008", "102109", "102311", "102513", "102614", "102715", "102816",
    "103010", "103111", "103212", "103414", "103515", "103818", "104012", "104416",
    "104820", "105014", "105115", "105216", "105620", "105923", "106016", "106319",
    "106521", "106824", "107018", "107220",
]
NMI_CSV = C.HCP_ROOT / "epi_align_mij_sbrigid_l6w2_nmi_none_topup.csv"
FIELD_CSV = C.HCP_ROOT / "epi_align_mij_w1_lm_bspline_fslbend_sbrigid_l6w2_nmi.csv"
OUT_TXT = C.HCP_ROOT / "epi_align_mij_w1_lm_bspline_fslbend_sbrigid_l6w2_tables.txt"
OUT_CSV = C.HCP_ROOT / "epi_align_mij_w1_lm_bspline_fslbend_sbrigid_l6w2_roi_z.csv"
Z_NEEDED = ("zstat9", "zstat11", "zstat14")
COMPARE = (("none", "none"), ("topup", "topup"), ("rigid w2", METHOD))
BASE_ROIS = ["L DLPFC", "R DLPFC", "L IPL", "R IPL"]


def zpath(sid, method, zname):
    return (C.HCP_ROOT / sid / "unwarp_compare/hcp_feat"
            / f"{method}_hp200_s4.feat/stats/{zname}.nii.gz")


def feat_ready(sid, method=METHOD):
    return all(zpath(sid, method, z).exists() for z in Z_NEEDED)


def field_path(sid):
    return C.HCP_ROOT / sid / "unwarp_compare" / FIELD


def holm(pvals):
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, float(pvals[i]) * (n - rank))
        running = max(running, val)
        adj[i] = running
    return adj


def readout_s(work):
    line = (work / "acqp.txt").read_text().splitlines()[0].split()
    return float(line[3])


def _stem(path):
    return path.with_suffix("").with_suffix("")


def make_4d(sid):
    work = C.HCP_ROOT / sid / "unwarp_compare"
    dst = work / NATIVE
    if dst.exists():
        print(f"  reuse {dst}", flush=True)
        return dst
    for name in (FIELD, SB_TO_SE, SE_TO_SB, "se_lr.nii.gz", "acqp.txt"):
        if not (work / name).exists():
            raise FileNotFoundError(work / name)
    raw = (C.HCP_ROOT / sid / "unprocessed/3T/tfMRI_WM_LR"
           / f"{sid}_3T_tfMRI_WM_LR.nii.gz")
    se = work / "se_lr.nii.gz"
    raw_img = nib.load(str(raw))
    se_img = nib.load(str(se))
    if tuple(se_img.shape[:3]) != tuple(raw_img.shape[:3]):
        raise RuntimeError(f"{sid} spin-echo grid {se_img.shape} != functional {raw_img.shape}")
    img, u = C.load(work / FIELD)
    if u.ndim > 3:
        u = u[..., 0]
    hz_path = work / HZ_NAME
    in_se = work / "_sbrigid_l6w2_in_se.nii.gz"
    unwarped = work / "_sbrigid_l6w2_unwarped.nii.gz"
    C.save_nii((u.astype(np.float32) / readout_s(work)).astype(np.float32), img, hz_path)
    try:
        print(f"=== {sid} rigid into spin-echo frame ===", flush=True)
        F.run([
            f"{F.FSL}/flirt", "-in", str(raw), "-ref", str(se),
            "-applyxfm", "-init", str(work / SB_TO_SE),
            "-out", str(in_se), "-interp", "spline",
        ])
        print(f"=== {sid} applytopup spline+jacobian ===", flush=True)
        F.run([
            f"{F.FSL}/applytopup",
            f"--imain={in_se}",
            f"--datain={work / 'acqp.txt'}",
            "--inindex=1",
            f"--topup={_stem(hz_path)}",
            "--method=jac",
            "--interp=spline",
            "--datatype=float",
            f"--out={_stem(unwarped)}",
        ])
        in_se.unlink(missing_ok=True)
        print(f"=== {sid} rigid back to the functional grid ===", flush=True)
        sbref = (C.HCP_ROOT / sid / "unprocessed/3T/tfMRI_WM_LR"
                 / f"{sid}_3T_tfMRI_WM_LR_SBRef.nii.gz")
        F.run([
            f"{F.FSL}/flirt", "-in", str(unwarped), "-ref", str(sbref),
            "-applyxfm", "-init", str(work / SE_TO_SB),
            "-out", str(dst), "-interp", "spline",
        ])
    finally:
        in_se.unlink(missing_ok=True)
        unwarped.unlink(missing_ok=True)
        hz_path.unlink(missing_ok=True)
    if not dst.exists():
        raise FileNotFoundError(dst)
    n_out = nib.load(str(dst)).shape[3]
    if n_out != raw_img.shape[3]:
        dst.unlink(missing_ok=True)
        raise RuntimeError(f"{sid} unwarped volumes {n_out} != raw {raw_img.shape[3]}")
    return dst


def run_one(sid):
    os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "4")
    F.METHODS[METHOD] = NATIVE
    print(f"\n======== {sid} ========", flush=True)
    native = C.HCP_ROOT / sid / "unwarp_compare" / NATIVE
    if feat_ready(sid):
        print(f"  reuse FEAT {sid}", flush=True)
        if native.exists():
            native.unlink()
        return sid, "ok"
    try:
        if not field_path(sid).exists():
            return sid, "missing field"
        if not (C.HCP_ROOT / sid / "unwarp_compare" / "epi_to_mni.mat").exists():
            return sid, "missing epi_to_mni.mat"
        featdir = zpath(sid, METHOD, "zstat11").parents[1]
        if featdir.exists() and not zpath(sid, METHOD, "zstat11").exists():
            print(f"  remove incomplete {featdir}", flush=True)
            shutil.rmtree(featdir)
        make_4d(sid)
        F.run_method(sid, METHOD)
        if not feat_ready(sid):
            missing = [z for z in Z_NEEDED if not zpath(sid, METHOD, z).exists()]
            return sid, "feat missing " + " ".join(missing)
        if native.exists():
            native.unlink()
            print(f"  rm {native}", flush=True)
        return sid, "ok"
    except Exception as e:
        traceback.print_exc()
        return sid, f"FAIL {e}"


def nmi_table(lines, sids):
    rows = list(csv.DictReader(NMI_CSV.open()))
    by = {r["subject"]: r for r in rows}
    fields = {r["subject"]: r for r in csv.DictReader(FIELD_CSV.open())}
    missing = [s for s in sids if s not in by or s not in fields]
    if missing:
        raise SystemExit("NMI csv missing " + " ".join(missing))
    use = [by[s] for s in sids]
    none = np.array([float(r["nmi_none"]) for r in use])
    top = np.array([float(r["nmi_topup"]) for r in use])
    new = np.array([float(r["nmi_new"]) for r in use])
    d_top = top - none
    d_new = new - top
    mae = np.array([float(fields[s]["mae_mm"]) for s in sids])
    corr = np.array([float(fields[s]["corr"]) for s in sids])
    p_top = float(wilcoxon(top, none, alternative="two-sided").pvalue) if np.any(d_top != 0) else 1.0
    p_new = float(wilcoxon(new, top, alternative="two-sided").pvalue) if np.any(d_new != 0) else 1.0
    lines += [
        "Table 1. Fair Studholme NMI. T1 is aligned separately to each image.",
        "none = raw single-band reference.",
        "rigid w2 = single-band reference in the spin-echo frame, spline unwarp.",
        "Levels 1-5 stay at weight 1 with between-blip motion.",
        "Level 6 walks 1.25, 1.5, 1.75, 2. Levels 7-9 stay at 2.",
        f"Field agreement with TOPUP: mean |difference| {mae.mean():.3f} mm, "
        f"mean correlation {corr.mean():.3f}.",
        "",
        f"{'':12s} {'none':>8} {'TOPUP':>8} {'rigidw2':>8} "
        f"{'TOPUP-none':>11} {'p_W':>8} {'rigid-TOPUP':>12} {'p_W':>8}",
        f"{'mean':12s} {none.mean():8.4f} {top.mean():8.4f} {new.mean():8.4f} "
        f"{d_top.mean():11.4f} {p_top:8.3g} {d_new.mean():12.4f} {p_new:8.3g}",
        "",
        f"{'subject':12s} {'none':>8} {'TOPUP':>8} {'rigidw2':>8} "
        f"{'TOPUP-none':>11} {'rigid-TOPUP':>12}",
    ]
    for r, a, b in zip(use, d_top, d_new):
        lines.append(
            f"{r['subject']:12s} {float(r['nmi_none']):8.4f} "
            f"{float(r['nmi_topup']):8.4f} {float(r['nmi_new']):8.4f} "
            f"{float(a):11.4f} {float(b):12.4f}"
        )
    lines.append("")


def group_block(lines, title, contrasts, rois, sids):
    lines.append(title)
    lines.append(
        f"One-sample t of FEAT z versus 0, n={len(sids)}. "
        "Holm is within each method, across the ROIs in this table."
    )
    lines.append("Columns are mean z, t, Holm p, and how many subjects have mean z > 0.")
    lines.append("")
    ref = nib.load(str(zpath(sids[0], "topup", contrasts[0][1])))
    balls = {
        roi: C.sphere(ref, xyz, C.ROI_RADIUS_MM)
        for roi, xyz in C.ROIS.items() if roi in rois
    }
    for con, zname in contrasts:
        lines.append(f"## {con}")
        lines.append(
            f"{'roi':10s} " + " ".join(f"{lab:>22s}" for lab, _ in COMPARE)
        )
        lines.append(
            f"{'':10s} " + " ".join(f"{'mean  t   p_H  n+':>22s}" for _ in COMPARE)
        )
        per = {}
        for lab, method in COMPARE:
            vols = [np.asanyarray(nib.load(str(zpath(s, method, zname))).dataobj, dtype=np.float32)
                    for s in sids]
            vol = np.stack(vols, 0)
            per[lab] = {roi: vol[:, m].mean(axis=1) for roi, m in balls.items()}
            del vol, vols
        stored = []
        for roi in rois:
            pvals, stats = [], []
            for lab, _ in COMPARE:
                x = per[lab][roi]
                t, p = ttest_1samp(x, 0.0)
                pvals.append(float(p))
                stats.append((float(np.mean(x)), float(t), int(np.sum(x > 0))))
            stored.append((roi, pvals, stats))
        adj = [holm([row[1][m] for row in stored]) for m in range(len(COMPARE))]
        for i, (roi, _pvals, stats) in enumerate(stored):
            bits = []
            for m, (mean, t, npos) in enumerate(stats):
                bits.append(f"{mean:5.2f} {t:5.1f} {adj[m][i]:5.3f} {npos:2d}")
            lines.append(f"{roi:10s} " + "  ".join(f"{b:>22s}" for b in bits))
        lines.append("")


def priors_ready(sids):
    missing = []
    for sid in sids:
        for method in ("none", "topup"):
            if not feat_ready(sid, method):
                missing.append(f"{sid} {method}")
    return missing


def write_tables(sids):
    for sid in sids:
        for method in ("none", "topup", METHOD):
            if not feat_ready(sid, method):
                raise SystemExit(f"cannot write tables, {sid} {method} FEAT is incomplete")
    lines = [
        "Rigid spin-echo placement, mutual-information climb stopping at weight 2.",
        "Levels 1-5 at weight 1. Level 6 walks 1.25, 1.5, 1.75, 2. Levels 7-9 stay at 2.",
        f"Field {FIELD}. FEAT directory {METHOD}_hp200_s4.feat.",
        "Unwarp moves the series into the spin-echo frame, applytopup --method=jac --interp=spline,",
        "then moves it back. Both rigid resamples are spline.",
        "Official HCP FEAT: high-pass 200 s, 4 mm smoothing, FILM, left-right.",
        "Power et al. 2011, 12 mm spheres.",
        "Table 2 is 0-back minus 2-back. 2-back minus 0-back is its sign flip.",
        "",
    ]
    nmi_table(lines, sids)
    group_block(
        lines,
        "Table 2. Group activations, 0-back versus 2-back (zstat14, 0BK minus 2BK).",
        [("0BK-2BK", "zstat14")],
        list(C.ROIS),
        sids,
    )
    group_block(
        lines,
        "Table 3. Group activations, 2-back versus baseline (zstat9). DLPFC and IPL only.",
        [("2BK-baseline", "zstat9")],
        BASE_ROIS,
        sids,
    )
    text = "\n".join(lines)
    OUT_TXT.write_text(text + "\n")
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "method", "contrast", "roi", "mean_z"])
        refs = {}
        for con, zname in (("0BK-2BK", "zstat14"), ("2BK-baseline", "zstat9")):
            ref = nib.load(str(zpath(sids[0], "topup", zname)))
            refs[zname] = {
                roi: C.sphere(ref, xyz, C.ROI_RADIUS_MM)
                for roi, xyz in C.ROIS.items()
                if (con == "0BK-2BK" or roi in BASE_ROIS)
            }
        for sid in sids:
            for con, zname in (("0BK-2BK", "zstat14"), ("2BK-baseline", "zstat9")):
                balls = refs[zname]
                for lab, method in COMPARE:
                    z = np.asanyarray(
                        nib.load(str(zpath(sid, method, zname))).dataobj, dtype=np.float32
                    )
                    for roi, m in balls.items():
                        w.writerow([sid, lab, con, roi, f"{float(z[m].mean()):.6f}"])
    print("\n" + text, flush=True)
    print("wrote", OUT_TXT, flush=True)
    print("wrote", OUT_CSV, flush=True)


def missing_fields(sids):
    return [s for s in sids if not field_path(s).exists()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--group-only", action="store_true")
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()
    sids = args.subjects if args.subjects else KEEP
    unknown = [s for s in sids if s not in KEEP]
    if unknown:
        raise SystemExit("not in the KEEP 36: " + " ".join(unknown))
    absent = missing_fields(sids)
    if absent:
        raise SystemExit(
            f"{len(absent)} field file(s) still missing ({FIELD}): " + " ".join(absent)
        )
    if not args.group_only:
        todo = [s for s in sids if not feat_ready(s)]
        print(f"FEAT todo {len(todo)} / {len(sids)}  jobs={args.jobs}", flush=True)
        os.environ["OMP_NUM_THREADS"] = str(max(2, min(8, 16 // max(args.jobs, 1))))
        failed = []
        if todo and args.jobs <= 1:
            for sid in todo:
                sid, st = run_one(sid)
                print(f"  {sid} {st}", flush=True)
                if st != "ok":
                    failed.append((sid, st))
        elif todo:
            with ProcessPoolExecutor(max_workers=args.jobs) as ex:
                futs = {ex.submit(run_one, sid): sid for sid in todo}
                for fut in as_completed(futs):
                    sid = futs[fut]
                    try:
                        sid, st = fut.result()
                    except Exception as e:
                        st = f"FAIL {e}"
                        traceback.print_exc()
                    print(f"  {sid} {st}", flush=True)
                    if st != "ok":
                        failed.append((sid, st))
        if failed:
            print("FEAT failures:", flush=True)
            for sid, st in failed:
                print(f"  {sid} {st}", flush=True)
            raise SystemExit(1)
    not_ready = [s for s in sids if not feat_ready(s)]
    if not_ready:
        raise SystemExit("FEAT incomplete: " + " ".join(not_ready))
    prior = priors_ready(sids)
    if prior:
        print(
            "rigid-2 FEATs are complete. Comparison tables need "
            "none_hp200_s4.feat and topup_hp200_s4.feat.",
            flush=True,
        )
        print("missing " + " ".join(prior[:8]) + (" ..." if len(prior) > 8 else ""), flush=True)
        return
    write_tables(sids)


if __name__ == "__main__":
    main()
