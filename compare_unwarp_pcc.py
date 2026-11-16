#!/usr/bin/env python3
"""LNCC vs Sobel vs TOPUP on HCP WM LR: PCC (0bk-2bk) and DLPFC (2bk-0bk).

PE-only warp along voxel x. T1 registered to TOPUP mean. Same native-to-MNI
affine for all methods. GPU off. One subject at a time.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import subprocess
from math import factorial
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ.setdefault("FSLDIR", "/usr/local/fsl")

import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as mm  # noqa: E402

HCP_ROOT = Path(os.environ.get("HCP_ROOT", "/mnt/data/projects/jhutter/data/HCP"))
DEFAULT_SUBJECTS = ["100206", "100307", "100408", "100610", "101006", "101309"]
FSL = os.path.join(os.environ["FSLDIR"], "bin")
CNF = os.path.join(os.environ["FSLDIR"], "etc", "flirtsch", "b02b0.cnf")
TR = 0.72
PE_SIGN = 1.0
MAX_U = 8.0
# Power et al. 2011 Neuron 72:665–678, 264-node MNI atlas
# (nilearn fetch_coords_power_2011, 1-based node). Sphere radius 12 mm.
ROIS = {
    "PCC": (-7, -55, 27),       # node 88
    "mPFC": (6, 54, 16),        # node 105
    "L DLPFC": (-42, 25, 30),   # node 201
    "R DLPFC": (48, 25, 27),    # node 175
    "OFC": (-3, 44, -9),        # node 109
    "L IPL": (-44, -65, 35),    # node 86
    "R IPL": (52, -59, 36),     # node 96
}
ROI_RADIUS_MM = 12.0


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def load(path):
    img = nib.load(str(path))
    return img, np.asanyarray(img.dataobj, dtype=np.float32)


def save_nii(data, ref_img, path):
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), ref_img.affine, ref_img.header), str(path))


def warp_pe_np(vol, u, jac=True):
    x = np.arange(vol.shape[0], dtype=np.float32)
    y = np.arange(vol.shape[1], dtype=np.float32)
    z = np.arange(vol.shape[2], dtype=np.float32)
    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
    coords = np.stack([gx + PE_SIGN * u, gy, gz], 0)
    if vol.ndim == 3:
        out = map_coordinates(vol, coords, order=1, mode="nearest")
    else:
        out = np.empty_like(vol)
        for t in range(vol.shape[3]):
            out[..., t] = map_coordinates(vol[..., t], coords, order=1, mode="nearest")
    if jac:
        j = np.clip(1.0 + PE_SIGN * np.gradient(u, axis=0), 0.0, None).astype(np.float32)
        out = out * (j[..., None] if out.ndim == 4 else j)
    return out.astype(np.float32)


def warp_pe_tf(vol, u):
    xsz, ysz, zsz = tf.shape(vol)[0], tf.shape(vol)[1], tf.shape(vol)[2]
    gx = tf.cast(tf.range(xsz)[:, None, None], tf.float32)
    gy = tf.broadcast_to(tf.range(ysz, dtype=tf.int32)[None, :, None], tf.shape(vol))
    gz = tf.broadcast_to(tf.range(zsz, dtype=tf.int32)[None, None, :], tf.shape(vol))
    src = gx + PE_SIGN * u
    x0 = tf.floor(src)
    a = src - x0
    x0i = tf.clip_by_value(tf.cast(x0, tf.int32), 0, xsz - 1)
    x1i = tf.clip_by_value(x0i + 1, 0, xsz - 1)
    v0 = tf.gather_nd(vol, tf.stack([x0i, gy, gz], -1))
    v1 = tf.gather_nd(vol, tf.stack([x1i, gy, gz], -1))
    return (1.0 - a) * v0 + a * v1


def invert01(vol, mask):
    v = vol.copy()
    m = mask > 0
    lo, hi = np.percentile(v[m], [0.5, 99.5])
    v = np.clip((v - lo) / (hi - lo + 1e-6), 0, 1)
    v[m] = 1.0 - v[m]
    v[~m] = 0
    return v.astype(np.float32)


def optimize_field(epi, t1, mask, kind, steps=250, lr=0.05):
    epi_t = tf.constant(epi)
    t1_t = tf.constant(t1)
    mask_t = tf.constant(mask.astype(np.float32))
    u = tf.Variable(tf.zeros_like(epi_t), dtype=tf.float32)
    opt = tf.keras.optimizers.Adam(lr)
    for i in range(steps):
        with tf.GradientTape() as tape:
            ui = tf.clip_by_value(u, -MAX_U, MAX_U)
            warped = warp_pe_tf(epi_t, ui)
            if kind == "lncc":
                loc = mm.local_ncc_loss(t1_t[None], warped[None], 9, mask=mask_t[None])[0]
            else:
                loc = mm.getGradientLoss(t1_t[None], warped[None])[0] / (tf.reduce_sum(mask_t) + 1e-6)
            tps = mm.getTPS(ui[None])[0]
            l2d = tf.reduce_mean(tf.square(ui[1:] - ui[:-1]))
            l2 = tf.reduce_mean(tf.square(ui))
            loss = loc + 5e-4 * tps + 2.0 * l2d + 0.5 * l2
        opt.apply_gradients(zip(tape.gradient(loss, [u]), [u]))
        if i % 25 == 0 or i == steps - 1:
            print(f"  {kind} step {i:3d} loss={float(loss):.4f} loc={float(loc):.4f} "
                  f"rms_u={float(tf.sqrt(tf.reduce_mean(tf.square(ui)))):.3f}", flush=True)
    return np.clip(u.numpy(), -MAX_U, MAX_U).astype(np.float32)


def boxcar_from_evs(evdir, names, n, tr):
    y = np.zeros(n, dtype=np.float32)
    t = np.arange(n) * tr
    for name in names:
        p = evdir / name
        if not p.exists() or p.stat().st_size == 0:
            continue
        rows = np.loadtxt(str(p), ndmin=2)
        for onset, dur, amp in rows:
            y[(t >= onset) & (t < onset + dur)] = amp
    return y


def spm_hrf(tr):
    t = np.arange(0.0, 32.0, tr, dtype=np.float64)
    h = (t ** 5) * np.exp(-t) / factorial(5) - (1.0 / 6.0) * (t ** 15) * np.exp(-t) / factorial(15)
    return (h / (h.sum() + 1e-12)).astype(np.float32)


def glm_contrasts(y4d, mask, evdir):
    n = y4d.shape[-1]
    z0 = boxcar_from_evs(evdir, ["0bk_body.txt", "0bk_faces.txt", "0bk_places.txt", "0bk_tools.txt"], n, TR)
    z2 = boxcar_from_evs(evdir, ["2bk_body.txt", "2bk_faces.txt", "2bk_places.txt", "2bk_tools.txt"], n, TR)
    h = spm_hrf(TR)
    x0 = np.convolve(z0, h)[:n]
    x2 = np.convolve(z2, h)[:n]
    drift = np.linspace(-1, 1, n, dtype=np.float32)
    X = np.stack([x0, x2, np.ones(n, np.float32), drift], 1)
    XtX_inv = np.linalg.pinv(X.T @ X)
    pix = np.where(mask)
    B = (XtX_inv @ X.T @ y4d[pix].T).T
    c_pcc = np.zeros(y4d.shape[:3], np.float32)
    c_wm = np.zeros(y4d.shape[:3], np.float32)
    c_pcc[pix] = B[:, 0] - B[:, 1]
    c_wm[pix] = B[:, 1] - B[:, 0]
    return c_pcc, c_wm


def sphere(img, xyz, r):
    sh = img.shape[:3]
    g = np.array(np.meshgrid(*[np.arange(s) for s in sh], indexing="ij"))
    xyzv = nib.affines.apply_affine(img.affine, np.stack(g, -1).reshape(-1, 3))
    return np.linalg.norm(xyzv - np.array(xyz), axis=1).reshape(sh) <= r


def roi_stats(z, mask, name, img):
    rows = []
    for roi, xyz in ROIS.items():
        m = sphere(img, xyz, ROI_RADIUS_MM) & mask
        if not m.any():
            rows.append((name, roi, 0.0, 0, 0))
            continue
        vals = z[m]
        rows.append((name, roi, float(vals.mean()), int((vals > 2.0).sum()), int(m.sum())))
    return rows


def apply_field_to_4d(src_4d, u, dst):
    img, data = load(src_4d)
    out = np.empty_like(data)
    for t0 in range(0, data.shape[3], 40):
        sl = data[..., t0:t0 + 40]
        out[..., t0:t0 + sl.shape[3]] = warp_pe_np(sl, u, jac=True)
        print(f"    warped vols {t0}-{t0+sl.shape[3]-1}", flush=True)
    save_nii(out, img, dst)
    del data, out


def prep_and_topup(sid, work, unproc, t1, mni_t1):
    work.mkdir(parents=True, exist_ok=True)
    se_lr, se_rl = work / "se_lr.nii.gz", work / "se_rl.nii.gz"
    if not se_lr.exists():
        run([f"{FSL}/fslroi", str(unproc / f"{sid}_3T_SpinEchoFieldMap_LR.nii.gz"), str(se_lr), "0", "1"])
        run([f"{FSL}/fslroi", str(unproc / f"{sid}_3T_SpinEchoFieldMap_RL.nii.gz"), str(se_rl), "0", "1"])
        run([f"{FSL}/fslmerge", "-t", str(work / "se_pair.nii.gz"), str(se_lr), str(se_rl)])
        (work / "acqp.txt").write_text("1 0 0 0.064526\n-1 0 0 0.064526\n")
    if not (work / "tp_fieldcoef.nii.gz").exists():
        run([
            f"{FSL}/topup", f"--imain={work/'se_pair.nii.gz'}", f"--datain={work/'acqp.txt'}",
            f"--config={CNF}", f"--out={work/'tp'}", f"--iout={work/'se_unwarp'}",
            f"--fout={work/'tp_field'}",
        ])
    sb = unproc / f"{sid}_3T_tfMRI_WM_LR_SBRef.nii.gz"
    wm = unproc / f"{sid}_3T_tfMRI_WM_LR.nii.gz"
    if not (work / "sb_topup.nii.gz").exists():
        run([f"{FSL}/applytopup", f"--imain={sb}", f"--datain={work/'acqp.txt'}",
             "--inindex=1", f"--topup={work/'tp'}", f"--out={work/'sb_topup'}", "--method=jac"])
    if not (work / "wm_topup.nii.gz").exists():
        run([f"{FSL}/applytopup", f"--imain={wm}", f"--datain={work/'acqp.txt'}",
             "--inindex=1", f"--topup={work/'tp'}", f"--out={work/'wm_topup'}", "--method=jac"])
    if not (work / "t1_brain.nii.gz").exists():
        run([f"{FSL}/bet", str(t1), str(work / "t1_brain"), "-R", "-f", "0.3"])
    if not (work / "t1_to_epi.nii.gz").exists():
        run([f"{FSL}/flirt", "-in", str(work / "t1_brain.nii.gz"), "-ref", str(work / "sb_topup.nii.gz"),
             "-out", str(work / "t1_to_epi.nii.gz"), "-omat", str(work / "t1_to_epi.mat"), "-dof", "6"])
    if not (work / "mean_topup.nii.gz").exists():
        run([f"{FSL}/fslmaths", str(work / "wm_topup.nii.gz"), "-Tmean", str(work / "mean_topup.nii.gz")])
    if not (work / "epi_to_mni.mat").exists():
        run([f"{FSL}/flirt", "-in", str(work / "mean_topup.nii.gz"), "-ref", str(mni_t1),
             "-out", str(work / "mean_topup_mni.nii.gz"), "-omat", str(work / "epi_to_mni.mat"), "-dof", "12"])
    if not (work / "epi_bet_mask.nii.gz").exists():
        run([f"{FSL}/bet", str(work / "sb_topup.nii.gz"), str(work / "epi_bet"), "-f", "0.3", "-m"])


def run_subject(sid, force=False):
    root = HCP_ROOT / sid
    work = root / "unwarp_compare"
    unproc = root / "unprocessed/3T/tfMRI_WM_LR"
    evdir = unproc / "LINKED_DATA/EPRIME/EVs"
    t1 = root / f"unprocessed/3T/T1w_MPR1/{sid}_3T_T1w_MPR1.nii.gz"
    mni_t1 = root / "MNINonLinear/T1w_restore.2.nii.gz"
    out_txt = work / "pcc_compare.txt"
    if out_txt.exists() and not force:
        print(f"=== {sid} already done, skip ===", flush=True)
        return parse_subject_txt(sid, out_txt)

    print(f"=== {sid} prep + TOPUP ===", flush=True)
    prep_and_topup(sid, work, unproc, t1, mni_t1)
    sb_img, sb = load(unproc / f"{sid}_3T_tfMRI_WM_LR_SBRef.nii.gz")
    _, t1epi = load(work / "t1_to_epi.nii.gz")
    _, mask = load(work / "epi_bet_mask.nii.gz")
    mask = mask > 0
    t1_inv = invert01(t1epi, mask)
    lo, hi = np.percentile(sb[mask], [0.5, 99.5])
    epi01 = np.clip((sb - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)
    epi01[~mask] = 0

    fields = {}
    for kind in ("lncc", "sobel"):
        fp = work / f"u_{kind}.nii.gz"
        if fp.exists() and not force:
            _, fields[kind] = load(fp)
            print(f"loaded {fp}", flush=True)
            continue
        print(f"=== {sid} optimize {kind} ===", flush=True)
        fields[kind] = optimize_field(epi01, t1_inv, mask, kind)
        save_nii(fields[kind], sb_img, fp)
        print(f"  {kind} |u| rms={np.sqrt((fields[kind]**2).mean()):.3f} "
              f"p95={np.percentile(np.abs(fields[kind])[mask], 95):.2f}", flush=True)

    wm_raw = unproc / f"{sid}_3T_tfMRI_WM_LR.nii.gz"
    methods_4d = {"none": wm_raw, "topup": work / "wm_topup.nii.gz"}
    for kind in ("lncc", "sobel"):
        dst = work / f"wm_{kind}.nii.gz"
        if not dst.exists() or force:
            print(f"=== {sid} apply {kind} to 4D ===", flush=True)
            apply_field_to_4d(wm_raw, fields[kind], dst)
        methods_4d[kind] = dst

    mni_img = nib.load(str(mni_t1))
    _, mni_vol = load(mni_t1)
    mni_mask = mni_vol > np.percentile(mni_vol[mni_vol > 0], 10)

    print(f"=== {sid} GLM ===", flush=True)
    all_rows = []
    for name, path in methods_4d.items():
        print(f"  GLM {name}", flush=True)
        img, y = load(path)
        c_pcc, c_wm = glm_contrasts(y, mask, evdir)
        del y
        save_nii(c_pcc, img, work / f"cope_0bk-2bk_{name}.nii.gz")
        save_nii(c_wm, img, work / f"cope_2bk-0bk_{name}.nii.gz")
        for con, native in (("0bk-2bk", work / f"cope_0bk-2bk_{name}.nii.gz"),
                            ("2bk-0bk", work / f"cope_2bk-0bk_{name}.nii.gz")):
            mni_path = work / f"cope_{con}_{name}_mni.nii.gz"
            run([f"{FSL}/flirt", "-in", str(native), "-ref", str(mni_t1), "-out", str(mni_path),
                 "-applyxfm", "-init", str(work / "epi_to_mni.mat"), "-interp", "trilinear"])
        _, pcc_m = load(work / f"cope_0bk-2bk_{name}_mni.nii.gz")
        _, wm_m = load(work / f"cope_2bk-0bk_{name}_mni.nii.gz")
        pcc_z = pcc_m / (pcc_m[mni_mask].std() + 1e-6)
        wm_z = wm_m / (wm_m[mni_mask].std() + 1e-6)
        all_rows.extend(roi_stats(pcc_z, mni_mask, f"{name}/0bk-2bk", mni_img))
        all_rows.extend(roi_stats(wm_z, mni_mask, f"{name}/2bk-0bk", mni_img))

    hdr = f"{'method/contrast':22s} {'roi':10s} {'mean':>8} {'n>2':>6} {'n_roi':>6}"
    lines = [hdr]
    print(hdr)
    parsed = []
    for r in all_rows:
        s = f"  {r[0]:22s} {r[1]:10s} {r[2]:8.3f} {r[3]:6d} {r[4]:6d}"
        print(s)
        lines.append(s)
        parsed.append((sid, r[0], r[1], r[2], r[3], r[4]))
    out_txt.write_text("\n".join(lines) + "\n")
    print("wrote", out_txt, flush=True)
    return parsed


def parse_subject_txt(sid, path):
    import re
    rows = []
    pat = re.compile(r"^\s+(\S+)\s+(.+?)\s+([+-]?\d+\.\d+)\s+(\d+)\s+(\d+)\s*$")
    for line in path.read_text().splitlines()[1:]:
        m = pat.match(line)
        if not m:
            continue
        rows.append((sid, m.group(1), m.group(2).strip(), float(m.group(3)), int(m.group(4)), int(m.group(5))))
    return rows


def write_summary(all_rows):
    csv_path = HCP_ROOT / "unwarp_compare_all.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "method_contrast", "roi", "mean", "n_gt2", "n_roi"])
        w.writerows(all_rows)
    # focus table
    focus = [("0bk-2bk", "PCC"), ("0bk-2bk", "mPFC"),
             ("2bk-0bk", "L DLPFC"), ("2bk-0bk", "R DLPFC")]
    methods = ["none", "lncc", "topup", "sobel"]
    lines = ["Unwarp comparison across HCP WM LR subjects", ""]
    for con, roi in focus:
        lines.append(f"=== {roi}  {con}  (mean contrast/brain SD) ===")
        hdr = f"{'subject':8s}" + "".join(f"{m:>10s}" for m in methods)
        lines.append(hdr)
        by = {}
        for sid, mc, r, mean, n2, nroi in all_rows:
            meth, c = mc.split("/", 1)
            if c == con and r == roi:
                by.setdefault(sid, {})[meth] = mean
        for sid in sorted(by):
            lines.append(f"{sid:8s}" + "".join(f"{by[sid].get(m, float('nan')):10.3f}" for m in methods))
        # means
        avgs = []
        for m in methods:
            vals = [by[s][m] for s in by if m in by[s]]
            avgs.append(np.mean(vals) if vals else np.nan)
        lines.append(f"{'mean':8s}" + "".join(f"{a:10.3f}" for a in avgs))
        lines.append("")
    summary = HCP_ROOT / "unwarp_compare_summary.txt"
    summary.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", csv_path)
    print("wrote", summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subjects", nargs="*", default=DEFAULT_SUBJECTS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    all_rows = []
    for sid in args.subjects:
        all_rows.extend(run_subject(sid, force=args.force))
        write_summary(all_rows)
    print("done", len(args.subjects), "subjects")


if __name__ == "__main__":
    main()
