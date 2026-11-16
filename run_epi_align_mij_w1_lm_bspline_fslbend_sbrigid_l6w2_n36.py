#!/usr/bin/env python3
"""KEEP-36 field fit: spin-echo rigid, then the mutual-information ramp stopped at 2.

The single-band reference and the T1 are moved into the left-right spin-echo
frame by a 6-degree-of-freedom FLIRT (correlation ratio). Phase-encode
translation is free. That rigid is not updated during the spline solve.

Levels 1-5 estimate between-blip motion at weight 1. Level 6 freezes that
pose, takes its weight-1 steps, then walks 1.25, 1.5, 1.75, and 2. Levels
7-9 stay at 2. Eight steps, 25% cap on each level's own residual.

Normalized mutual information is scored on the reference in the spin-echo
frame. A second column scores the same field on the original reference.

Field: <sid>/unwarp_compare/u_epi_align_mij_w1_lm_bspline_fslbend_sbrigid_l6w2.nii.gz
CSV:   epi_align_mij_w1_lm_bspline_fslbend_sbrigid_l6w2_nmi.csv

Does not replace the weight-1 cohort field, the climb to 32, or the weight-1
rigid fits.

    HCP_ROOT=/path/to/hcp python3 run_epi_align_mij_w1_lm_bspline_fslbend_sbrigid_l6w2_n36.py --jobs 4
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

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

FIELD_NAME = "u_epi_align_mij_w1_lm_bspline_fslbend_sbrigid_l6w2.nii.gz"
COHORT_FIELD = "u_epi_align_mij_w1_lm_bspline_fslbend.nii.gz"
OUT_CSV = C.HCP_ROOT / "epi_align_mij_w1_lm_bspline_fslbend_sbrigid_l6w2_nmi.csv"
LEVEL_RAMPS = {
    6: (1.25, 1.5, 1.75, 2.0),
    7: (2.0,),
    8: (2.0,),
    9: (2.0,),
}
FIELDS = [
    "subject", "nmi_topup", "nmi_lin", "nmi_spline", "d_lin", "d_spline",
    "nmi_raw_spline", "d_raw_spline", "shift_mm", "rot_deg",
    "corr", "mae_mm", "corr_cohort", "mae_cohort_mm",
]


def _pair(u, other, mask):
    if other is None:
        return float("nan"), float("nan")
    return S.corr(u, other, mask), float(np.abs(u - other)[mask].mean() * 2.0)


def _load_u(path):
    if not path.exists():
        return None
    _, vol = C.load(path)
    if vol.ndim > 3:
        vol = vol[..., 0]
    return vol.astype(np.float32)


def _place(work, sb_path, se_path, t1_raw):
    """Move SBRef and the raw-SBRef T1 into the left-right spin-echo frame."""
    mat_se = work / "se_to_sb_sbrigid.mat"
    mat_sb = work / "sb_to_se_sbrigid.mat"
    sb_in = work / "sbref_in_se_sbrigid.nii.gz"
    t1_in = work / "t1_in_se_sbrigid.nii.gz"
    if not mat_se.exists():
        SB._flirt(se_path, sb_path, mat_se)
    if not mat_sb.exists():
        SB._save_mat(mat_sb, np.linalg.inv(SB._load_mat(mat_se)))
    if not sb_in.exists():
        SB._apply(sb_path, se_path, mat_sb, sb_in)
    if not t1_in.exists():
        SB._apply(t1_raw, se_path, mat_sb, t1_in)
    se_img, _ = C.load(se_path)
    shift, angle = SB._report(
        "spin-echo to SBRef", SB._load_mat(mat_se), se_img.affine, se_img.shape,
    )
    return sb_in, t1_in, shift, angle


def run_sid(sid):
    work = C.HCP_ROOT / sid / "unwarp_compare"
    unproc = C.HCP_ROOT / sid / "unprocessed/3T/tfMRI_WM_LR"
    t1_brain = work / "t1_brain.nii.gz"
    sb_path = unproc / f"{sid}_3T_tfMRI_WM_LR_SBRef.nii.gz"
    se_path = work / "se_lr.nii.gz"
    out = work / FIELD_NAME
    sb_img, sb_raw = C.load(sb_path)
    sb_raw = sb_raw.astype(np.float32)
    _, mask = C.load(work / "epi_bet_mask.nii.gz")
    mask_np = mask > 0.5
    t1_raw = work / "t1_to_raw_epi.nii.gz"
    if not t1_raw.exists():
        print(f"=== {sid} flirt T1 -> raw SBRef ===", flush=True)
        Rec.flirt_t1(t1_brain, sb_path, t1_raw, work / "t1_to_raw_epi.mat")
    sb_in, t1_in, shift, angle = _place(work, sb_path, se_path, t1_raw)
    if not out.exists():
        _, se_lr = C.load(se_path)
        _, se_rl = C.load(work / "se_rl.nii.gz")
        if se_lr.ndim > 3:
            se_lr = se_lr[..., 0]
        if se_rl.ndim > 3:
            se_rl = se_rl[..., 0]
        se_lr = (se_lr * (100.0 / (float(se_lr.mean()) + 1e-6))).astype(np.float32)
        se_rl = (se_rl * (100.0 / (float(se_rl.mean()) + 1e-6))).astype(np.float32)
        _, sb = C.load(sb_in)
        _, t1 = C.load(t1_in)
        print(f"=== {sid} rigid then ramp to 2, cap 25% ===", flush=True)
        hz = S.optimize(
            se_lr, se_rl,
            S.scale01(sb.astype(np.float32), mask_np),
            S.scale01(t1.astype(np.float32), mask_np),
            mask_np.astype(np.float32),
            "mij", 1.0, solver="lm", use_gate=False, init_hz=None,
            resample="bspline", bend="topup",
            mi_ramp_steps=8, mi_ssd_frac=0.25,
            mi_level_ramps=LEVEL_RAMPS, mi_level_replace=(7, 8, 9),
        )
        C.save_nii((hz * S.ACQP_T).astype(np.float32), sb_img, out)
    u = _load_u(out)
    _, sb_frame = C.load(sb_in)
    sb_frame = sb_frame.astype(np.float32)
    _, sb_top = C.load(work / "sb_topup.nii.gz")
    nmi_top = Rec.fair_nmi(sid, sb_top.astype(np.float32), f"{sid}_topup_sbw2", t1_brain)
    nmi_lin = Rec.fair_nmi(sid, C.warp_pe_np(sb_frame, u, jac=True), f"{sid}_sbw2_lin", t1_brain)
    nmi_spl = Rec.fair_nmi(
        sid, S.warp_volume_bspline(sb_frame, u, jac=True), f"{sid}_sbw2_spl", t1_brain,
    )
    nmi_raw = Rec.fair_nmi(
        sid, S.warp_volume_bspline(sb_raw, u, jac=True), f"{sid}_sbw2_rawspl", t1_brain,
    )
    fsl = S.fsl_signed(work, sb_raw, mask_np)
    corr, mae = _pair(u, fsl, mask_np)
    corr_c, mae_c = _pair(u, _load_u(work / COHORT_FIELD), mask_np)
    rec = {
        "subject": sid,
        "nmi_topup": nmi_top,
        "nmi_lin": nmi_lin,
        "nmi_spline": nmi_spl,
        "d_lin": nmi_lin - nmi_top,
        "d_spline": nmi_spl - nmi_top,
        "nmi_raw_spline": nmi_raw,
        "d_raw_spline": nmi_raw - nmi_top,
        "shift_mm": shift,
        "rot_deg": angle,
        "corr": corr,
        "mae_mm": mae,
        "corr_cohort": corr_c,
        "mae_cohort_mm": mae_c,
    }
    print(
        f"  {sid} shift={shift:.2f} mm rot={angle:.2f} deg "
        f"NMI topup={nmi_top:.4f} lin={nmi_lin:.4f} d={rec['d_lin']:+.4f} "
        f"spline={nmi_spl:.4f} d={rec['d_spline']:+.4f} "
        f"raw spline d={rec['d_raw_spline']:+.4f} "
        f"corr={corr:.3f} |d|={mae:.3f} mm "
        f"vs cohort corr={corr_c:.3f} |d|={mae_c:.3f} mm",
        flush=True,
    )
    return rec


def write_csv(rows):
    by = {}
    if OUT_CSV.exists():
        with OUT_CSV.open(newline="") as f:
            for rec in csv.DictReader(f):
                by[rec["subject"]] = {k: rec[k] for k in FIELDS}
    for rec in rows:
        by[rec["subject"]] = {k: rec[k] for k in FIELDS}
    merged = [by[k] for k in sorted(by)]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(merged)
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()
    sids = list(args.subjects) if args.subjects else list(KEEP)
    print(
        f"subjects {len(sids)} jobs {args.jobs} field {FIELD_NAME} "
        f"ladder ends at 2 cap 25%",
        flush=True,
    )
    rows = []
    if args.jobs <= 1:
        for sid in sids:
            try:
                rows.append(run_sid(sid))
            except Exception as e:
                traceback.print_exc()
                print(f"FAIL {sid} {e}", flush=True)
            write_csv(rows)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as ex:
            futs = {ex.submit(run_sid, sid): sid for sid in sids}
            for fut in as_completed(futs):
                sid = futs[fut]
                try:
                    rows.append(fut.result())
                except Exception as e:
                    traceback.print_exc()
                    print(f"FAIL {sid} {e}", flush=True)
                write_csv(rows)
    merged = write_csv(rows)
    wanted = [r for r in merged if r["subject"] in set(sids)]
    if wanted:
        ds = np.array([float(r["d_spline"]) for r in wanted])
        draw = np.array([float(r["d_raw_spline"]) for r in wanted])
        top = np.array([float(r["nmi_topup"]) for r in wanted])
        spl = np.array([float(r["nmi_spline"]) for r in wanted])
        mae = np.array([float(r["mae_mm"]) for r in wanted])
        print(
            f"n={len(wanted)} mean TOPUP {top.mean():.4f} "
            f"placed spline {spl.mean():.4f} d {ds.mean():+.4f} "
            f"n+ {int((ds > 0).sum())}/{len(ds)} "
            f"raw spline d {draw.mean():+.4f} |d| {mae.mean():.3f} mm",
            flush=True,
        )
    have = sum(
        1 for sid in sids
        if (C.HCP_ROOT / sid / "unwarp_compare" / FIELD_NAME).exists()
    )
    print("wrote", OUT_CSV, flush=True)
    if have == len(sids) and len(rows) == len(sids):
        print(f"COHORT_FIT_DONE n={have}", flush=True)
    else:
        print(f"FAIL incomplete {have}/{len(sids)}", flush=True)


if __name__ == "__main__":
    main()
