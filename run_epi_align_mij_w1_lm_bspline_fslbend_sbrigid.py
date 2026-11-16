#!/usr/bin/env python3
"""Weight-1 field with a rigid between the spin echo and the single-band reference.

The rigid is a 6-degree-of-freedom FLIRT, correlation ratio, of the distorted
left-right spin echo onto the distorted single-band reference. Phase-encode
translation is free. The reference and the T1 are moved into the spin-echo
frame with that rigid; the field is then fit exactly as the weight-1
cubic-B-spline run. The spline coefficients and this rigid are not updated
in the same step.

After that field exists, the same FLIRT is repeated on the unwarped pair. A
center shift above 0.5 mm or a rotation above 0.5 degrees is composed in and
the field is fit once more. Otherwise the residual is only reported.

One subject. Does not replace the cohort field.

    python3 run_epi_align_mij_w1_lm_bspline_fslbend_sbrigid.py --subjects 105014
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback

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
import run_hcp_topup_sobel as S
import run_recursive_t1_mi as Rec

FIELD_NAME = "u_epi_align_mij_w1_lm_bspline_fslbend_sbrigid.nii.gz"
FIELD_R2 = "u_epi_align_mij_w1_lm_bspline_fslbend_sbrigid_r2.nii.gz"
BASE_FIELD = "u_epi_align_mij_w1_lm_bspline_fslbend.nii.gz"
SHIFT_MM = 0.5
ROT_DEG = 0.5


def _run(cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.check_call([str(c) for c in cmd])


def _load_mat(path):
    return np.loadtxt(path)


def _save_mat(path, mat):
    lines = [" ".join(f"{v:.10g}" for v in row) for row in np.asarray(mat)]
    path.write_text("\n".join(lines) + "\n")


def _rigid_of(mat, affine, shape):
    """Center shift in mm and rotation angle, for a world-mm FLIRT matrix."""
    center = np.array([(n - 1) / 2.0 for n in shape[:3]] + [1.0])
    moved = mat @ (affine @ center)
    here = affine @ center
    shift = moved[:3] - here[:3]
    rot = mat[:3, :3]
    # Remove scale in case the world axes are not orthonormal in the 3x3.
    u, _, vt = np.linalg.svd(rot)
    rot_u = u @ vt
    if np.linalg.det(rot_u) < 0:
        u[:, -1] *= -1
        rot_u = u @ vt
    trace = float(np.clip((np.trace(rot_u) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.degrees(np.arccos(trace)))
    return shift, angle


def _report(label, mat, affine, shape):
    shift, angle = _rigid_of(mat, affine, shape)
    norm = float(np.linalg.norm(shift))
    print(
        f"  {label} center shift {norm:.3f} mm "
        f"[{shift[0]:+.3f}, {shift[1]:+.3f}, {shift[2]:+.3f}]  "
        f"rotation {angle:.3f} deg",
        flush=True,
    )
    return norm, angle


def _flirt(src, ref, mat, out=None):
    cmd = [
        f"{C.FSL}/flirt", "-in", src, "-ref", ref,
        "-omat", mat, "-dof", "6", "-cost", "corratio",
    ]
    if out is not None:
        cmd.extend(["-out", out, "-interp", "spline"])
    _run(cmd)


def _apply(src, ref, mat, out):
    _run([
        f"{C.FSL}/flirt", "-in", src, "-ref", ref,
        "-applyxfm", "-init", mat, "-out", out, "-interp", "spline",
    ])


def _corr(a, b, mask):
    return S.corr(a, b, mask)


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


def _fit(sid, sb_path, t1_path, out, sb_img, se_lr, se_rl, mask_np):
    _, sb = C.load(sb_path)
    _, t1 = C.load(t1_path)
    print(f"=== {sid} weight-1 field, SBRef and T1 in the spin-echo frame ===", flush=True)
    hz = S.optimize(
        se_lr, se_rl,
        S.scale01(sb.astype(np.float32), mask_np),
        S.scale01(t1.astype(np.float32), mask_np),
        mask_np.astype(np.float32),
        "mij", 1.0, solver="lm", use_gate=False, init_hz=None,
        resample="bspline", bend="topup",
    )
    u = (hz * S.ACQP_T).astype(np.float32)
    C.save_nii(u, sb_img, out)
    return u


def _score(sid, tag, sb_native, sb_frame, u, mask_np, t1_brain, work, sb_raw):
    _, sb_top = C.load(work / "sb_topup.nii.gz")
    nmi_top = Rec.fair_nmi(sid, sb_top.astype(np.float32), f"{sid}_topup_{tag}", t1_brain)
    frame = sb_frame.astype(np.float32)
    raw = sb_raw.astype(np.float32)
    nmi_lin = Rec.fair_nmi(sid, C.warp_pe_np(frame, u, jac=True), f"{sid}_{tag}_lin", t1_brain)
    nmi_spl = Rec.fair_nmi(
        sid, S.warp_volume_bspline(frame, u, jac=True), f"{sid}_{tag}_spl", t1_brain,
    )
    nmi_raw = Rec.fair_nmi(
        sid, S.warp_volume_bspline(raw, u, jac=True), f"{sid}_{tag}_rawspl", t1_brain,
    )
    fsl = S.fsl_signed(work, sb_native, mask_np)
    corr, mae = _pair(u, fsl, mask_np)
    corr_b, mae_b = _pair(u, _load_u(work / BASE_FIELD), mask_np)
    print(
        f"  {sid} {tag} NMI topup={nmi_top:.4f} "
        f"lin={nmi_lin:.4f} d={nmi_lin - nmi_top:+.4f} "
        f"spline={nmi_spl:.4f} d={nmi_spl - nmi_top:+.4f} "
        f"raw-sbref spline={nmi_raw:.4f} d={nmi_raw - nmi_top:+.4f} "
        f"corr={corr:.3f} |d|={mae:.3f} mm "
        f"vs w1bend corr={corr_b:.3f} |d|={mae_b:.3f} mm",
        flush=True,
    )


def run_sid(sid):
    work = C.HCP_ROOT / sid / "unwarp_compare"
    unproc = C.HCP_ROOT / sid / "unprocessed/3T/tfMRI_WM_LR"
    t1_brain = work / "t1_brain.nii.gz"
    sb_path = unproc / f"{sid}_3T_tfMRI_WM_LR_SBRef.nii.gz"
    se_path = work / "se_lr.nii.gz"
    t1_raw = work / "t1_to_raw_epi.nii.gz"
    sb_img, sb_raw = C.load(sb_path)
    sb_raw = sb_raw.astype(np.float32)
    se_img, se_lr = C.load(se_path)
    _, se_rl = C.load(work / "se_rl.nii.gz")
    if se_lr.ndim > 3:
        se_lr = se_lr[..., 0]
    if se_rl.ndim > 3:
        se_rl = se_rl[..., 0]
    _, mask = C.load(work / "epi_bet_mask.nii.gz")
    mask_np = mask > 0.5

    mat_se = work / "se_to_sb_sbrigid.mat"
    mat_sb = work / "sb_to_se_sbrigid.mat"
    sb_in = work / "sbref_in_se_sbrigid.nii.gz"
    t1_in = work / "t1_in_se_sbrigid.nii.gz"
    _flirt(se_path, sb_path, mat_se)
    _save_mat(mat_sb, np.linalg.inv(_load_mat(mat_se)))
    _apply(sb_path, se_path, mat_sb, sb_in)
    _apply(t1_raw, se_path, mat_sb, t1_in)
    shift, angle = _report("spin-echo to SBRef", _load_mat(mat_se), se_img.affine, se_img.shape)
    _, sb_frame = C.load(sb_in)
    sb_frame = sb_frame.astype(np.float32)
    _, se_native = C.load(se_path)
    if se_native.ndim > 3:
        se_native = se_native[..., 0]
    print(
        f"  corr SE vs raw SBRef {_corr(se_native, sb_raw, mask_np):.3f}  "
        f"vs SBRef in SE frame {_corr(se_native, sb_frame, mask_np):.3f}",
        flush=True,
    )

    se_lr_s = (se_lr * (100.0 / (float(se_lr.mean()) + 1e-6))).astype(np.float32)
    se_rl_s = (se_rl * (100.0 / (float(se_rl.mean()) + 1e-6))).astype(np.float32)
    out = work / FIELD_NAME
    if out.exists():
        u = _load_u(out)
    else:
        u = _fit(sid, sb_in, t1_in, out, sb_img, se_lr_s, se_rl_s, mask_np)
    _score(sid, "sbrigid", sb_raw, sb_frame, u, mask_np, t1_brain, work, sb_raw)

    lr_u = work / "_tmp_lr_u_sbrigid.nii.gz"
    sb_u = work / "_tmp_sb_u_sbrigid.nii.gz"
    C.save_nii(S.warp_volume_bspline(se_native.astype(np.float32), u, jac=False), se_img, lr_u)
    C.save_nii(S.warp_volume_bspline(sb_frame, u, jac=False), se_img, sb_u)
    mat_res = work / "sbrigid_residual.mat"
    _flirt(sb_u, lr_u, mat_res)
    res_shift, res_angle = _report(
        "residual after the field", _load_mat(mat_res), se_img.affine, se_img.shape,
    )
    for p in (lr_u, sb_u):
        p.unlink(missing_ok=True)
    if res_shift <= SHIFT_MM and res_angle <= ROT_DEG:
        print(
            f"  residual within {SHIFT_MM:g} mm and {ROT_DEG:g} deg; no second fit",
            flush=True,
        )
        print(f"  initial rigid was {shift:.3f} mm, {angle:.3f} deg", flush=True)
        return
    sb_r2 = work / "sbref_in_se_sbrigid_r2.nii.gz"
    t1_r2 = work / "t1_in_se_sbrigid_r2.nii.gz"
    _apply(sb_in, se_path, mat_res, sb_r2)
    _apply(t1_in, se_path, mat_res, t1_r2)
    out_r2 = work / FIELD_R2
    _, sb_r2_vol = C.load(sb_r2)
    if not out_r2.exists():
        u2 = _fit(sid, sb_r2, t1_r2, out_r2, sb_img, se_lr_s, se_rl_s, mask_np)
    else:
        u2 = _load_u(out_r2)
    _score(sid, "sbrigid2", sb_raw, sb_r2_vol.astype(np.float32), u2, mask_np, t1_brain, work, sb_raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=["105014"])
    args = ap.parse_args()
    failed = False
    for sid in args.subjects:
        try:
            run_sid(sid)
        except Exception as e:
            traceback.print_exc()
            print(f"FAIL {sid} {e}", flush=True)
            failed = True
    if not failed:
        print("SBRIGID_DONE", flush=True)


if __name__ == "__main__":
    main()
