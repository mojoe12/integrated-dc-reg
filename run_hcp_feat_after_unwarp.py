#!/usr/bin/env python3
"""After our unwarp, run HCP's official WM FEAT (hp200, s4, 8 EVs, FILM).

Does not redo HCP fMRIVolume one-step interpolation. Native 4D is mcflirt'd,
affine-resampled to T1w_restore.2, then the HCP level-1 design is pointed at it.
LR only (no level-2 LR+RL). GPU off.

Subjects without MNINonLinear Results fsf reuse 100307's design and their own
E-Prime EVs (same 3-column files HCP FEAT uses).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("FSLDIR", "/usr/local/fsl")
os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_unwarp_pcc as C

HCP = C.HCP_ROOT
FSL = C.FSL
FSF_TMPL = HCP / "100307/MNINonLinear/Results/tfMRI_WM_LR/tfMRI_WM_LR_hp200_s4_level1.fsf"
METHODS = {
    "none": "unprocessed",
    "topup": "wm_topup.nii.gz",
    "gated": "wm_nolncc_l1.nii.gz",
    "sigmoid": "wm_sig_sobel.nii.gz",
    "gate1": "wm_gate1.nii.gz",
    "mind_ung": "wm_mind_ung.nii.gz",
}
EV_NAMES = [
    "2bk_body.txt", "2bk_faces.txt", "2bk_places.txt", "2bk_tools.txt",
    "0bk_body.txt", "0bk_faces.txt", "0bk_places.txt", "0bk_tools.txt",
]
SUMMARY_KEYS = [
    ("2BK-0BK", "L DLPFC"),
    ("2BK-0BK", "R DLPFC"),
    ("2BK-0BK", "L IPL"),
    ("2BK-0BK", "R IPL"),
    ("0BK-2BK", "PCC"),
    ("0BK-2BK", "mPFC"),
    ("0BK-2BK", "OFC"),
]


def run(cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.check_call(cmd)


def native_4d(sid, method):
    root = HCP / sid
    if method == "none":
        return root / "unprocessed/3T/tfMRI_WM_LR" / f"{sid}_3T_tfMRI_WM_LR.nii.gz"
    return root / "unwarp_compare" / METHODS[method]


def ev_dir(sid):
    p = HCP / sid / "MNINonLinear/Results/tfMRI_WM_LR/EVs"
    if (p / "2bk_body.txt").exists():
        return p
    return HCP / sid / "unprocessed/3T/tfMRI_WM_LR/LINKED_DATA/EPRIME/EVs"


def fsf_template(sid):
    p = HCP / sid / "MNINonLinear/Results/tfMRI_WM_LR/tfMRI_WM_LR_hp200_s4_level1.fsf"
    return p if p.exists() else FSF_TMPL


def subject_ready(sid):
    root = HCP / sid
    work = root / "unwarp_compare"
    unproc = root / "unprocessed/3T/tfMRI_WM_LR"
    return all(p.exists() for p in [
        native_4d(sid, "none"),
        native_4d(sid, "topup"),
        work / "tp_field.nii.gz",
        work / "epi_bet_mask.nii.gz",
        work / "t1_to_epi.nii.gz",
        work / "epi_to_mni.mat",
        unproc / f"{sid}_3T_tfMRI_WM_LR_SBRef.nii.gz",
        root / "MNINonLinear" / "T1w_restore.2.nii.gz",
        ev_dir(sid) / "2bk_body.txt",
        FSF_TMPL,
    ])


def discover_subjects():
    return sorted(
        p.name for p in HCP.iterdir()
        if p.is_dir() and p.name.isdigit() and subject_ready(p.name)
    )


def write_fsf(sid, method, func_mni, out_feat):
    # FSL command-line feat5:load only allows analysis in {1, 2, 7}.
    # Analysis 3 is rewritten to 2 (stats-only), which then treats feat_files
    # as an existing .feat directory. Keep HCP's 7 (full) and point feat_files
    # at 4D data via inputtype 2. Data is already in 2 mm MNI; skip registration.
    txt = fsf_template(sid).read_text()
    func_stem = re.sub(r"\.nii(\.gz)?$", "", str(func_mni))
    evs = ev_dir(sid)
    txt = re.sub(r'set fmri\(outputdir\).*', f'set fmri(outputdir) "{out_feat}"', txt)
    txt = re.sub(r'set feat_files\(1\).*', f'set feat_files(1) "{func_stem}"', txt)
    txt = re.sub(r'set fmri\(analysis\).*', 'set fmri(analysis) 7', txt)
    txt = re.sub(r'set fmri\(inputtype\).*', 'set fmri(inputtype) 2', txt)
    txt = re.sub(r'set fmri\(poststats_yn\).*', 'set fmri(poststats_yn) 0', txt)
    txt = re.sub(r'set fmri\(reg_yn\).*', 'set fmri(reg_yn) 0', txt)
    txt = re.sub(r'set fmri\(regstandard_yn\).*', 'set fmri(regstandard_yn) 0', txt)
    txt = re.sub(r'set fmri\(featwatcher_yn\).*', 'set fmri(featwatcher_yn) 0', txt)
    for i, name in enumerate(EV_NAMES, 1):
        txt = re.sub(
            rf'set fmri\(custom{i}\).*',
            f'set fmri(custom{i}) "{evs / name}"',
            txt,
        )
    fsf = out_feat.parent / f"{method}_hp200_s4.fsf"
    fsf.write_text(txt)
    return fsf


def prep_mni(sid, method):
    work = HCP / sid / "unwarp_compare" / "hcp_feat"
    work.mkdir(parents=True, exist_ok=True)
    src = native_4d(sid, method)
    if not src.exists():
        raise FileNotFoundError(src)
    mc = work / f"wm_{method}_mc.nii.gz"
    mni = work / f"wm_{method}_mni.nii.gz"
    ref = HCP / sid / "MNINonLinear/T1w_restore.2.nii.gz"
    premat = HCP / sid / "unwarp_compare" / "epi_to_mni.mat"
    if not mc.exists():
        print(f"=== {sid} {method} mcflirt ===", flush=True)
        run([f"{FSL}/mcflirt", "-in", str(src), "-out", str(mc), "-dof", "6", "-plots"])
    if not mni.exists():
        print(f"=== {sid} {method} applywarp -> MNI ===", flush=True)
        run([
            f"{FSL}/applywarp", "-i", str(mc), "-r", str(ref), "-o", str(mni),
            "--premat=" + str(premat), "--interp=spline",
        ])
    return mni, work


def roi_zstat(sid, zpath, label):
    mni_t1 = HCP / sid / "MNINonLinear/T1w_restore.2.nii.gz"
    mni_img = nib.load(str(mni_t1))
    _, mni_vol = C.load(mni_t1)
    mni_mask = mni_vol > np.percentile(mni_vol[mni_vol > 0], 10)
    zimg = nib.load(str(zpath))
    if zimg.shape[:3] != mni_img.shape[:3]:
        tmp = Path(str(zpath) + ".r2t1.nii.gz")
        run([
            f"{FSL}/flirt", "-in", str(zpath), "-ref", str(mni_t1), "-out", str(tmp),
            "-applyxfm", "-usesqform", "-interp", "trilinear",
        ])
        _, z = C.load(tmp)
    else:
        _, z = C.load(zpath)
    return C.roi_stats(z, mni_mask, label, mni_img)


def make_mind_ung_4d(sid):
    """Apply saved ungated-MIND field to native WM 4D. Does not re-optimize."""
    work = HCP / sid / "unwarp_compare"
    dst = work / METHODS["mind_ung"]
    if dst.exists():
        return dst
    u_path = work / "u_mind_ung.nii.gz"
    if not u_path.exists():
        raise FileNotFoundError(u_path)
    _, u = C.load(u_path)
    raw = HCP / sid / "unprocessed/3T/tfMRI_WM_LR" / f"{sid}_3T_tfMRI_WM_LR.nii.gz"
    print(f"=== {sid} apply mind_ung ===", flush=True)
    C.apply_field_to_4d(raw, u, dst)
    return dst


def run_method(sid, method):
    work = HCP / sid / "unwarp_compare" / "hcp_feat"
    work.mkdir(parents=True, exist_ok=True)
    featstem = work / f"{method}_hp200_s4"
    featdir = Path(str(featstem) + ".feat")
    z11 = featdir / "stats" / "zstat11.nii.gz"  # 2BK-0BK
    z14 = featdir / "stats" / "zstat14.nii.gz"  # 0BK-2BK
    if not z11.exists():
        if method == "sigmoid" and not native_4d(sid, "sigmoid").exists():
            from run_sigmoid_vs_gated import make_sigmoid_4d
            make_sigmoid_4d(sid)
        if method == "gate1" and not native_4d(sid, "gate1").exists():
            from run_gate1_sobel import make_gate1_4d
            make_gate1_4d(sid)
        if method == "mind_ung" and not native_4d(sid, "mind_ung").exists():
            make_mind_ung_4d(sid)
        mni, work = prep_mni(sid, method)
        fsf = write_fsf(sid, method, mni, featstem)
        print(f"=== {sid} {method} feat ===", flush=True)
        run([f"{FSL}/feat", str(fsf)])
    rows = []
    if z11.exists():
        rows += roi_zstat(sid, z11, f"{method}/2BK-0BK")
        # Drop our motion-corrected / MNI 4Ds once FEAT has z-stats.
        # Stay inside this subject's hcp_feat/; do not touch feat dirs.
        drop_mni_intermediates(work, method)
    if z14.exists():
        rows += roi_zstat(sid, z14, f"{method}/0BK-2BK")
    return rows


def drop_mni_intermediates(work, method):
    """Remove wm_{method}_mc/mni 4Ds after that method's zstat11 exists."""
    for name in (f"wm_{method}_mc.nii.gz", f"wm_{method}_mni.nii.gz"):
        p = work / name
        if p.exists():
            p.unlink()
            print(f"  rm {p}", flush=True)


def official_hcp(sid):
    z11 = (HCP / sid / "MNINonLinear/Results/tfMRI_WM/"
           "tfMRI_WM_hp200_s4_level2vol.feat/cope11.feat/stats/zstat1.nii.gz")
    if not z11.exists():
        return []
    return roi_zstat(sid, z11, "HCP_official/2BK-0BK")


def write_sid_rows(sid, rows):
    work = HCP / sid / "unwarp_compare" / "hcp_feat"
    work.mkdir(parents=True, exist_ok=True)
    out = work / "roi_z.txt"
    with out.open("w") as f:
        f.write(f"{'method/contrast':24s} {'roi':10s} {'mean':>8} {'n>2':>6}\n")
        for r in rows:
            f.write(f"  {r[0]:24s} {r[1]:10s} {r[2]:8.3f} {r[3]:6d}\n")
    return out


def run_sid(sid, methods):
    print(f"\n======== {sid} ========", flush=True)
    rows = []
    try:
        rows.extend(official_hcp(sid))
    except Exception as e:
        print(f"FAIL {sid} official: {e}", flush=True)
        traceback.print_exc()
    for m in methods:
        try:
            rows.extend(run_method(sid, m))
        except Exception as e:
            print(f"FAIL {sid} {m}: {e}", flush=True)
            traceback.print_exc()
    for extra in ("gated", "sigmoid", "gate1", "mind_ung"):
        if extra in methods:
            continue
        z11e = (HCP / sid / "unwarp_compare/hcp_feat" / f"{extra}_hp200_s4.feat/stats/zstat11.nii.gz")
        z14e = z11e.parent / "zstat14.nii.gz"
        try:
            if z11e.exists():
                rows += roi_zstat(sid, z11e, f"{extra}/2BK-0BK")
            if z14e.exists():
                rows += roi_zstat(sid, z14e, f"{extra}/0BK-2BK")
        except Exception as e:
            print(f"FAIL {sid} {extra}-score: {e}", flush=True)
    write_sid_rows(sid, rows)
    print(f"{'method/contrast':24s} {'roi':10s} {'mean_z':>8} {'n>2':>6}", flush=True)
    for r in rows:
        print(f"  {r[0]:24s} {r[1]:10s} {r[2]:8.3f} {r[3]:6d}", flush=True)
    return sid, rows


def _worker(args):
    sid, methods = args
    return run_sid(sid, methods)


def holm(pvals):
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, pvals[i] * (n - rank))
        running = max(running, val)
        adj[i] = running
    return adj


def summarize(all_rows, methods):
    from collections import defaultdict
    by = defaultdict(dict)  # (sid, contrast, roi) -> {method: mean}
    for sid, methcon, roi, mean, n2, *rest in all_rows:
        if "/" not in methcon:
            continue
        meth, con = methcon.split("/", 1)
        by[(sid, con, roi)][meth] = float(mean)
    lines = ["", "Cohort mean z (paired; mind_ung vs topup / gate1 vs topup)", ""]
    lines.append(f"{'contrast':10s} {'roi':10s} {'n':>3} "
                 f"{'none':>8} {'topup':>8} {'gate1':>8} {'mind_ung':>8} "
                 f"{'mu-t':>8} {'p_mut':>8} {'g1-t':>8} {'p_g1t':>8}")
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    pairs = []
    for con, roi in SUMMARY_KEYS:
        none, top, g1, mu = [], [], [], []
        sids = sorted({s for s, c, r in by if c == con and r == roi})
        for sid in sids:
            d = by[(sid, con, roi)]
            if not all(k in d for k in ("none", "topup", "mind_ung")):
                continue
            none.append(d["none"])
            top.append(d["topup"])
            mu.append(d["mind_ung"])
            g1.append(d.get("gate1", np.nan))
        if len(mu) < 3:
            continue
        none, top, g1, mu = map(np.asarray, (none, top, g1, mu))
        p_mu = p_g1 = np.nan
        if wilcoxon is not None and np.any(mu != top):
            p_mu = wilcoxon(mu, top, alternative="two-sided").pvalue
        have_g = np.isfinite(g1)
        if wilcoxon is not None and have_g.sum() >= 3 and np.any(g1[have_g] != top[have_g]):
            p_g1 = wilcoxon(g1[have_g], top[have_g], alternative="two-sided").pvalue
        pairs.append((con, roi, len(mu), none.mean(), top.mean(),
                      np.nanmean(g1), mu.mean(),
                      (mu - top).mean(), p_mu,
                      np.nanmean(g1 - top), p_g1))
    if pairs and wilcoxon is not None:
        p_mu_adj = holm([p[8] for p in pairs])
        p_g1_adj = holm([p[10] for p in pairs])
    else:
        p_mu_adj = p_g1_adj = [np.nan] * len(pairs)
    for (con, roi, ns, nm, tm, g1m, mum, dmu, pmu, dg1, pg1), am, ag in zip(
            pairs, p_mu_adj, p_g1_adj):
        lines.append(
            f"{con:10s} {roi:10s} {ns:3d} {nm:8.3f} {tm:8.3f} {g1m:8.3f} {mum:8.3f} "
            f"{dmu:8.3f} {am:8.4f} {dg1:8.3f} {ag:8.4f}"
        )
    lines.append("p columns are Holm-adjusted Wilcoxon (mind_ung-topup, gate1-topup).")
    return "\n".join(lines)


def write_cohort(all_rows, methods):
    out = HCP / "hcp_feat_after_unwarp.txt"
    lines = [
        "HCP FEAT after our unwarp (hp200, s4, 8 EVs, FILM, LR only)",
        "HCP_official is their LR+RL level-2 zstat of 2BK-0BK (cope11), when present.",
        "",
        f"{'subject':8s} {'method/contrast':24s} {'roi':10s} {'mean':>8} {'n>2':>6}",
    ]
    for sid, a, b, c, d, *rest in all_rows:
        lines.append(f"  {sid:8s} {a:24s} {b:10s} {c:8.3f} {d:6d}")
    lines.append(summarize(all_rows, methods))
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-20:]), flush=True)
    print("wrote", out, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=["none", "topup", "gate1"])
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    sids = args.subjects if args.subjects else discover_subjects()
    missing = [s for s in sids if not subject_ready(s)]
    if missing:
        print("SKIP not ready:", " ".join(missing), flush=True)
        sids = [s for s in sids if s not in set(missing)]
    print(f"subjects {len(sids)} jobs={args.jobs}: {' '.join(sids)}", flush=True)
    all_rows = []
    if args.jobs <= 1 or len(sids) <= 1:
        for sid in sids:
            _, rows = run_sid(sid, args.methods)
            all_rows.extend((sid, *r) for r in rows)
            write_cohort(all_rows, args.methods)
    else:
        os.environ["OMP_NUM_THREADS"] = str(max(2, min(8, 80 // max(args.jobs, 1))))
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(_worker, (sid, args.methods)): sid for sid in sids}
            for fut in as_completed(futs):
                sid = futs[fut]
                try:
                    sid, rows = fut.result()
                    all_rows.extend((sid, *r) for r in rows)
                except Exception as e:
                    print(f"FAIL {sid} worker: {e}", flush=True)
                    traceback.print_exc()
                write_cohort(all_rows, args.methods)
    write_cohort(all_rows, args.methods)
    print("done", flush=True)


if __name__ == "__main__":
    main()
