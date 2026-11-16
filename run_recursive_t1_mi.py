#!/usr/bin/env python3
"""T1 rigid used by the cohort, plus an earlier recursive field fit.

The integrated cohort imports flirt_t1 (6 dof, correlation ratio) and
fair_nmi. The weight-6 outer loop below is an earlier experiment.

Estimate the field without a TOPUP-aligned T1.

Outer loop:
  1. Rigidly align the T1 to the current EPI (raw SBRef on the first pass,
     then the Jacobian-unwarped SBRef).
  2. Re-estimate the field from the spin-echo pair plus Jacobian MI,
     weight 6, no sigmoid, warm-started from the previous field.

The spin-echo pair is the acquired LR/RL data, not TOPUP's field.
NMI is scored after a fresh rigid alignment of the T1 to each image,
so the comparison is not graded on the TOPUP registration.

    python3 run_recursive_t1_mi.py --subject 100307 --iters 3
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("FSLDIR", "/usr/local/fsl")
os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_unwarp_pcc as C
import run_hcp_topup_sobel as S

FSL = C.FSL


def flirt_t1(t1, ref, out, mat):
    C.run([
        f"{FSL}/flirt", "-in", str(t1), "-ref", str(ref),
        "-out", str(out), "-omat", str(mat), "-dof", "6",
    ])


def fair_nmi(sid, image, tag, t1_brain):
    """NMI after aligning T1 to this image, not to TOPUP."""
    work = C.HCP_ROOT / sid / "unwarp_compare"
    ref = work / f"_tmp_ref_{tag}.nii.gz"
    aligned = work / f"_tmp_t1_{tag}.nii.gz"
    mat = work / f"_tmp_t1_{tag}.mat"
    C.save_nii(image, C.load(work / "se_lr.nii.gz")[0], ref)
    flirt_t1(t1_brain, ref, aligned, mat)
    _, t1 = C.load(aligned)
    _, mask = C.load(work / "epi_bet_mask.nii.gz")
    mask = mask > 0.5
    val = S.nmi(image, t1.astype(np.float32), mask)
    for p in (ref, aligned, mat):
        p.unlink(missing_ok=True)
    return val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="100307")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--anat-w", type=float, default=6.0)
    args = ap.parse_args()
    sid = args.subject
    work = C.HCP_ROOT / sid / "unwarp_compare"
    unproc = C.HCP_ROOT / sid / "unprocessed/3T/tfMRI_WM_LR"
    t1_brain = work / "t1_brain.nii.gz"
    sb_path = unproc / f"{sid}_3T_tfMRI_WM_LR_SBRef.nii.gz"
    sb_img, sb = C.load(sb_path)
    sb = sb.astype(np.float32)
    _, se_lr = C.load(work / "se_lr.nii.gz")
    _, se_rl = C.load(work / "se_rl.nii.gz")
    if se_lr.ndim > 3:
        se_lr = se_lr[..., 0]
    if se_rl.ndim > 3:
        se_rl = se_rl[..., 0]
    se_lr = (se_lr * (100.0 / (se_lr.mean() + 1e-6))).astype(np.float32)
    se_rl = (se_rl * (100.0 / (se_rl.mean() + 1e-6))).astype(np.float32)
    fsl = S.fsl_signed(work, sb, (C.load(work / "epi_bet_mask.nii.gz")[1] > 0.5))
    _, sb_top = C.load(work / "sb_topup.nii.gz")
    nmi_raw = fair_nmi(sid, sb, "raw", t1_brain)
    nmi_top = fair_nmi(sid, sb_top.astype(np.float32), "topup", t1_brain)
    print(f"fair NMI  raw={nmi_raw:.4f}  TOPUP={nmi_top:.4f}", flush=True)

    hz = None
    for it in range(args.iters):
        if it == 0:
            ref_img = sb
            ref_path = sb_path
        else:
            ref_img = C.warp_pe_np(sb, (hz * S.ACQP_T).astype(np.float32), jac=True)
            ref_path = work / f"_rec_ref_iter{it}.nii.gz"
            C.save_nii(ref_img, sb_img, ref_path)
        t1_out = work / f"t1_to_epi_rec_iter{it}.nii.gz"
        t1_mat = work / f"t1_to_epi_rec_iter{it}.mat"
        print(f"=== iter {it} flirt T1 -> current EPI ===", flush=True)
        flirt_t1(t1_brain, ref_path, t1_out, t1_mat)
        _, t1 = C.load(t1_out)
        _, mask = C.load(work / "epi_bet_mask.nii.gz")
        mask_np = mask > 0.5
        t1_s = S.scale01(t1.astype(np.float32), mask_np)
        sb_s = S.scale01(sb, mask_np)
        print(f"=== iter {it} estimate field ===", flush=True)
        hz = S.optimize(
            se_lr, se_rl, sb_s, t1_s, mask_np.astype(np.float32),
            "mij", args.anat_w, solver="adam", use_gate=False, init_hz=hz,
        )
        u = (hz * S.ACQP_T).astype(np.float32)
        epi = C.warp_pe_np(sb, u, jac=True)
        print(
            f"  iter {it}  corr(u,FSL)={S.corr(u, fsl, mask_np):.3f}  "
            f"|u|={np.abs(u[mask_np]).mean():.3f}  |FSL|={np.abs(fsl[mask_np]).mean():.3f}  "
            f"|u-FSL|={np.abs(u - fsl)[mask_np].mean():.3f}",
            flush=True,
        )
        print(f"  iter {it}  fair NMI={fair_nmi(sid, epi, f'iter{it}', t1_brain):.4f}", flush=True)
        C.save_nii(u, sb_img, work / f"u_recursive_mij_w6_iter{it}.nii.gz")
    print("done", work / f"u_recursive_mij_w6_iter{args.iters - 1}.nii.gz", flush=True)


if __name__ == "__main__":
    main()
