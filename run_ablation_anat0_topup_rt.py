#!/usr/bin/env python3
"""Two ablations around the weight-2 field. Neither replaces that field.

(a) The same least-squares cubic B-spline solver, with the anatomical weight
    set to 0. No mutual-information ramp. Between-blip motion is still
    estimated on levels 1-5, and phase-encode translation stays at 0.
(b) FSL TOPUP's own field and movement, applied with the same rigid
    round-trip the weight-2 series uses: spline into the spin-echo frame,
    applytopup --method=jac --interp=spline, spline back onto the
    single-band reference.

Both are scored by Studholme NMI after a fresh 6-dof T1 alignment, the same
score as the cohort table. The single-band reference is one volume, so this
does not refit FEAT. Field agreement is Pearson correlation and mean absolute
difference inside the EPI brain mask.

    HCP_ROOT=/path/to/hcp python3 run_ablation_anat0_topup_rt.py --jobs 4
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ.setdefault("FSLDIR", "/usr/local/fsl")
os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_unwarp_pcc as C
import run_epi_align_mij_w1_lm_bspline_fslbend_sbrigid as SB
import run_hcp_topup_sobel as S
import run_recursive_t1_mi as Rec

KEEP = [
    "100206", "100307", "100408", "100610", "101006", "101107", "101309", "101410",
    "101915", "102008", "102109", "102311", "102513", "102614", "102715", "102816",
    "103010", "103111", "103212", "103414", "103515", "103818", "104012", "104416",
    "104820", "105014", "105115", "105216", "105620", "105923", "106016", "106319",
    "106521", "106824", "107018", "107220",
]
L6W2 = "u_epi_align_mij_w1_lm_bspline_fslbend_sbrigid_l6w2.nii.gz"
ANAT0 = "u_epi_align_lm_bspline_fslbend_sbrigid_anat0.nii.gz"
OUT_CSV = C.HCP_ROOT / "epi_align_ablation_anat0_topup_rt.csv"
FIELDS = [
    "subject",
    "nmi_topup", "nmi_l6w2", "nmi_anat0",
    "nmi_topup_rt", "nmi_l6w2_rt", "nmi_anat0_rt",
    "corr_anat0_topup", "mae_anat0_topup_mm",
    "corr_anat0_l6w2", "mae_anat0_l6w2_mm",
    "corr_l6w2_topup", "mae_l6w2_topup_mm",
]


def _stem(path):
    return path.with_suffix("").with_suffix("")


def _load_u(path):
    _, vol = C.load(path)
    if vol.ndim > 3:
        vol = vol[..., 0]
    return vol.astype(np.float32)


def _pair(a, b, mask):
    return S.corr(a, b, mask), float(np.abs(a - b)[mask].mean() * 2.0)


def _fit_anat0(sid, work, mask):
    out = work / ANAT0
    if out.exists():
        print(f"=== {sid} reuse {ANAT0} ===", flush=True)
        return
    unproc = C.HCP_ROOT / sid / "unprocessed/3T/tfMRI_WM_LR"
    sb_img, _ = C.load(unproc / f"{sid}_3T_tfMRI_WM_LR_SBRef.nii.gz")
    _, se_lr = C.load(work / "se_lr.nii.gz")
    _, se_rl = C.load(work / "se_rl.nii.gz")
    if se_lr.ndim > 3:
        se_lr = se_lr[..., 0]
    if se_rl.ndim > 3:
        se_rl = se_rl[..., 0]
    se_lr = (se_lr * (100.0 / (float(se_lr.mean()) + 1e-6))).astype(np.float32)
    se_rl = (se_rl * (100.0 / (float(se_rl.mean()) + 1e-6))).astype(np.float32)
    _, sb = C.load(work / "sbref_in_se_sbrigid.nii.gz")
    _, t1 = C.load(work / "t1_in_se_sbrigid.nii.gz")
    print(f"=== {sid} solver λ_anat=0 ===", flush=True)
    hz = S.optimize(
        se_lr, se_rl,
        S.scale01(sb.astype(np.float32), mask),
        S.scale01(t1.astype(np.float32), mask),
        mask.astype(np.float32),
        "none", 0.0,
        solver="lm", use_gate=False, init_hz=None,
        resample="bspline", bend="topup",
    )
    # Write beside the cohort field. Do not replace it.
    tmp = work / (ANAT0 + ".writing.nii.gz")
    C.save_nii((hz * S.ACQP_T).astype(np.float32), sb_img, tmp)
    tmp.replace(out)
    print(f"wrote {out}", flush=True)


def _hz_prefix(work, u, ref_img, name):
    path = work / "_ablate_rt" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    C.save_nii((u / S.ACQP_T).astype(np.float32), ref_img, path)
    return _stem(path)


def _round_trip(work, src, topup_prefix, out_path):
    if out_path.exists():
        _, vol = C.load(out_path)
        return vol.astype(np.float32)
    tmp = work / "_ablate_rt"
    tmp.mkdir(parents=True, exist_ok=True)
    in_se = tmp / "in_se.nii.gz"
    unwarped = tmp / "unwarped.nii.gz"
    se = work / "se_lr.nii.gz"
    SB._apply(src, se, work / "sb_to_se_sbrigid.mat", in_se)
    C.run([
        f"{C.FSL}/applytopup",
        f"--imain={in_se}",
        f"--datain={work / 'acqp.txt'}",
        "--inindex=1",
        f"--topup={topup_prefix}",
        "--method=jac",
        "--interp=spline",
        "--datatype=float",
        f"--out={_stem(unwarped)}",
    ])
    SB._apply(unwarped, src, work / "se_to_sb_sbrigid.mat", out_path)
    in_se.unlink(missing_ok=True)
    unwarped.unlink(missing_ok=True)
    _, vol = C.load(out_path)
    return vol.astype(np.float32)


def run_sid(sid):
    work = C.HCP_ROOT / sid / "unwarp_compare"
    unproc = C.HCP_ROOT / sid / "unprocessed/3T/tfMRI_WM_LR"
    sb_path = unproc / f"{sid}_3T_tfMRI_WM_LR_SBRef.nii.gz"
    t1_brain = work / "t1_brain.nii.gz"
    for name in (
        "se_lr.nii.gz", "se_rl.nii.gz", "acqp.txt", "tp_fieldcoef.nii.gz",
        "tp_movpar.txt", "sb_topup.nii.gz", "sbref_in_se_sbrigid.nii.gz",
        "t1_in_se_sbrigid.nii.gz", "sb_to_se_sbrigid.mat", "se_to_sb_sbrigid.mat",
        "epi_bet_mask.nii.gz", L6W2,
    ):
        if not (work / name).exists():
            raise FileNotFoundError(work / name)
    _, mask_vol = C.load(work / "epi_bet_mask.nii.gz")
    mask = mask_vol > 0.5
    sb_img, sb = C.load(sb_path)
    sb = sb.astype(np.float32)
    _fit_anat0(sid, work, mask)
    u0 = _load_u(work / ANAT0)
    u2 = _load_u(work / L6W2)
    utop = S.fsl_signed(work, sb, mask)
    _, sb_frame = C.load(work / "sbref_in_se_sbrigid.nii.gz")
    sb_frame = sb_frame.astype(np.float32)
    _, sb_top = C.load(work / "sb_topup.nii.gz")

    nmi_top = Rec.fair_nmi(sid, sb_top.astype(np.float32), f"{sid}_ab_topup", t1_brain)
    nmi_l6 = Rec.fair_nmi(
        sid, S.warp_volume_bspline(sb_frame, u2, jac=True), f"{sid}_ab_l6w2", t1_brain,
    )
    nmi_a0 = Rec.fair_nmi(
        sid, S.warp_volume_bspline(sb_frame, u0, jac=True), f"{sid}_ab_anat0", t1_brain,
    )

    hz_ref, _ = C.load(work / "se_lr.nii.gz")
    top_rt = _round_trip(work, sb_path, work / "tp", work / "sb_topup_rigid_rt.nii.gz")
    l6_prefix = _hz_prefix(work, u2, hz_ref, "l6w2_hz.nii.gz")
    l6_rt = _round_trip(work, sb_path, l6_prefix, work / "sb_l6w2_rigid_rt.nii.gz")
    a0_prefix = _hz_prefix(work, u0, hz_ref, "anat0_hz.nii.gz")
    a0_rt = _round_trip(work, sb_path, a0_prefix, work / "sb_anat0_rigid_rt.nii.gz")
    for name in ("l6w2_hz.nii.gz", "anat0_hz.nii.gz"):
        (work / "_ablate_rt" / name).unlink(missing_ok=True)

    nmi_top_rt = Rec.fair_nmi(sid, top_rt, f"{sid}_ab_topup_rt", t1_brain)
    nmi_l6_rt = Rec.fair_nmi(sid, l6_rt, f"{sid}_ab_l6w2_rt", t1_brain)
    nmi_a0_rt = Rec.fair_nmi(sid, a0_rt, f"{sid}_ab_anat0_rt", t1_brain)
    c0t, m0t = _pair(u0, utop, mask)
    c02, m02 = _pair(u0, u2, mask)
    c2t, m2t = _pair(u2, utop, mask)
    rec = {
        "subject": sid,
        "nmi_topup": nmi_top,
        "nmi_l6w2": nmi_l6,
        "nmi_anat0": nmi_a0,
        "nmi_topup_rt": nmi_top_rt,
        "nmi_l6w2_rt": nmi_l6_rt,
        "nmi_anat0_rt": nmi_a0_rt,
        "corr_anat0_topup": c0t,
        "mae_anat0_topup_mm": m0t,
        "corr_anat0_l6w2": c02,
        "mae_anat0_l6w2_mm": m02,
        "corr_l6w2_topup": c2t,
        "mae_l6w2_topup_mm": m2t,
    }
    print(
        f"{sid} NMI se-frame topup {nmi_top:.4f}  anat0 {nmi_a0:.4f}  l6w2 {nmi_l6:.4f}  "
        f"round-trip topup {nmi_top_rt:.4f}  anat0 {nmi_a0_rt:.4f}  l6w2 {nmi_l6_rt:.4f}  "
        f"corr anat0-topup {c0t:.3f}  anat0-l6w2 {c02:.3f}",
        flush=True,
    )
    return rec


def _write(rows):
    import fcntl
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    lock_path = OUT_CSV.with_suffix(".lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        by = {}
        if OUT_CSV.exists():
            with OUT_CSV.open(newline="") as f:
                for rec in csv.DictReader(f):
                    by[rec["subject"]] = rec
        for rec in rows:
            by[rec["subject"]] = {k: rec[k] for k in FIELDS}
        merged = [by[k] for k in sorted(by)]
        tmp = OUT_CSV.with_suffix(".csv.writing")
        with tmp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(merged)
        tmp.replace(OUT_CSV)
        fcntl.flock(lock, fcntl.LOCK_UN)
    return merged


def _summarize(rows):
    from scipy.stats import wilcoxon

    def col(name):
        return np.array([float(r[name]) for r in rows], dtype=np.float64)

    def report(label, a, b):
        d = a - b
        p = float(wilcoxon(a, b, alternative="two-sided").pvalue) if np.any(d != 0) else 1.0
        print(
            f"{label}: mean {a.mean():.4f} vs {b.mean():.4f}  "
            f"diff {d.mean():+.4f}  higher {int((d > 0).sum())}/{len(d)}  "
            f"Wilcoxon p={p:.4g}",
            flush=True,
        )

    print(f"\nn={len(rows)}", flush=True)
    report("SE-frame NMI anat0 minus TOPUP", col("nmi_anat0"), col("nmi_topup"))
    report("SE-frame NMI weight-2 minus TOPUP", col("nmi_l6w2"), col("nmi_topup"))
    report("SE-frame NMI weight-2 minus anat0", col("nmi_l6w2"), col("nmi_anat0"))
    report("round-trip NMI TOPUP minus native TOPUP", col("nmi_topup_rt"), col("nmi_topup"))
    report("round-trip NMI weight-2 minus TOPUP", col("nmi_l6w2_rt"), col("nmi_topup_rt"))
    report("round-trip NMI anat0 minus TOPUP", col("nmi_anat0_rt"), col("nmi_topup_rt"))
    report("round-trip NMI weight-2 minus anat0", col("nmi_l6w2_rt"), col("nmi_anat0_rt"))
    print(
        f"field |anat0-TOPUP| {col('mae_anat0_topup_mm').mean():.3f} mm  "
        f"corr {col('corr_anat0_topup').mean():.3f}",
        flush=True,
    )
    print(
        f"field |anat0-weight2| {col('mae_anat0_l6w2_mm').mean():.3f} mm  "
        f"corr {col('corr_anat0_l6w2').mean():.3f}",
        flush=True,
    )
    print(
        f"field |weight2-TOPUP| {col('mae_l6w2_topup_mm').mean():.3f} mm  "
        f"corr {col('corr_l6w2_topup').mean():.3f}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()
    sids = list(args.subjects) if args.subjects else list(KEEP)
    print(f"subjects {len(sids)} jobs {args.jobs} field {ANAT0}", flush=True)
    rows = []
    if args.jobs <= 1:
        for sid in sids:
            rows.append(run_sid(sid))
            _write(rows)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as pool:
            futs = {pool.submit(run_sid, sid): sid for sid in sids}
            for fut in as_completed(futs):
                sid = futs[fut]
                try:
                    rows.append(fut.result())
                except Exception as e:
                    traceback.print_exc()
                    print(f"FAIL {sid} {e}", flush=True)
                _write(rows)
    if rows:
        _summarize(_write(rows))
    print("wrote", OUT_CSV, flush=True)


if __name__ == "__main__":
    main()
