# Integrated DC-Reg

Weight-2 distortion correction for HCP spin-echo fMRI, and the native FSL TOPUP comparison used for the working-memory analysis.

The solver estimates a phase-encode displacement from the left–right / right–left spin-echo pair and a T1. It does not start from TOPUP’s field, and it does not align the T1 to TOPUP’s unwarped image. Subject data, displacement fields, and FEAT directories are not in this repository.

Set `HCP_ROOT` to the directory that contains one folder per subject. The default is `/mnt/data/projects/jhutter/data/HCP`. `FSLDIR` defaults to `/usr/local/fsl`.

## Requirements

FSL (`topup`, `applytopup`, `flirt`, `bet`, `mcflirt`, `applywarp`, `feat`) and Python packages `numpy`, `scipy`, `nibabel`, `tensorflow`, and `neurite`. `model.py` imports `neurite` at load time. The mutual-information term itself is `mutual_information_loss` in that file: Shannon MI from a 32-bin soft histogram, returned as a loss (the negative of MI).

## Cohort field

```bash
HCP_ROOT=/path/to/hcp python3 run_epi_align_mij_w1_lm_bspline_fslbend_sbrigid_l6w2_n36.py --jobs 4
```

This writes, under each subject’s `unwarp_compare/`,

`u_epi_align_mij_w1_lm_bspline_fslbend_sbrigid_l6w2.nii.gz`

The saved volume is the phase-encode shift in voxels. Hertz for `applytopup` is that shift divided by the readout time 0.064526 s. The 36 subject IDs are `KEEP` in this script.

Before the solve, two placements are finished and then left fixed:

1. `bet -R -f 0.3` on the raw T1, then a 6-dof FLIRT (correlation ratio) of that brain onto the distorted single-band reference, which resamples it to 2 mm (`run_recursive_t1_mi.flirt_t1`).
2. One further 6-dof FLIRT, correlation ratio, of the distorted left–right spin echo onto that same distorted reference. Its inverse (`sb_to_se_sbrigid.mat`) resamples both the reference and the T1 into the spin-echo frame with spline interpolation. Phase-encode translation is included. `run_epi_align_mij_w1_lm_bspline_fslbend_sbrigid.py` supplies those FLIRT helpers. The solve is called with `init_hz=None` and does not update this rigid.

Inside the solve, levels 1–5 estimate a separate between-blip rigid (y and z translation and three rotations). Phase-encode translation is held at zero. Levels 6–9 freeze that rigid and still apply it. It is not stored in the saved field.

The image term is Jacobian-modulated mutual information between the single-band reference and the T1, both scaled to [0, 1]. The objective at an accepted step is

`SSD × (1 + λ_anat × MI₀ / MI) + λ_eff × bending`

SSD is the spin-echo residual after Jacobian modulation of both echoes. MI₀ is the mutual information at the start of the current Levenberg–Marquardt step, held fixed. λ_eff is the b02b0 λ times the current SSD (`ssqlambda`). Bending energy is the sum of squares of analytic cubic B-spline second derivatives, with cross terms counted twice. The gradient of the anatomical term is added at half scale in the normal equations.

Anatomical weight stays at 1 through levels 1–5, while between-blip motion is estimated. Level 6 takes its weight-1 steps, then eight steps at 1.25, 1.5, 1.75, and 2. A step is rejected when the spin-echo SSD rises more than 25% above the SSD at the start of that weight. Levels 7–9 stay at weight 2. Knot spacing, subsampling, FWHM, iteration counts, and λ follow `b02b0.cnf`. The optimizer is a TensorFlow Levenberg–Marquardt fit (`solver="lm"`, `resample="bspline"`, `bend="topup"`), implemented in `run_hcp_topup_sobel.py`.

`run_recursive_t1_mi.py` also contains an earlier weight-6 recursive fit. The cohort does not call that loop. It uses `flirt_t1` and `fair_nmi` from the same file. `fair_nmi` is Studholme normalized mutual information on the single-band reference after a fresh 6-dof T1 alignment, inside the EPI brain mask. That score is separate from the Shannon MI inside the solver.

## Conventional TOPUP

`compare_unwarp_pcc.prep_and_topup` runs FSL `topup` with `b02b0.cnf` on the spin-echo pair and applies the field with `applytopup --method=jac`. Spline is FSL’s default interpolation. The same function skull-strips the T1, writes the 6-dof `t1_to_epi.mat` onto the TOPUP-unwarped reference, the mean unwarped functional, and the shared 12-dof `epi_to_mni.mat` from that mean onto `MNINonLinear/T1w_restore.2.nii.gz`.

Running `compare_unwarp_pcc.py` as a script also fits the earlier LNCC and Sobel fields and a simple GLM. The reported comparison uses `prep_and_topup` and the FEAT scripts below.

## FEAT

Both series are motion-corrected with MCFLIRT (6 dof) on the native grid, then resampled with `applywarp --interp=spline` by the same `epi_to_mni.mat` onto `T1w_restore.2`. FEAT registration is off. The model is the HCP working-memory level-1 design on the left–right run: high-pass 200 s, 4 mm smoothing, FILM. z-statistics are written on that 2 mm grid.

```bash
# conventional series, after prep_and_topup has written wm_topup.nii.gz
HCP_ROOT=/path/to/hcp python3 run_hcp_feat_after_unwarp.py --methods topup --jobs 2

# weight-2 series: spline rigid into the spin-echo frame, applytopup
# --method=jac --interp=spline, spline rigid back, then the same FEAT
HCP_ROOT=/path/to/hcp python3 run_epi_align_mij_w1_lm_bspline_fslbend_sbrigid_l6w2_feat.py --jobs 2
```

The weight-2 script’s `--group-only` reprints the ROI tables from finished FEATs. Those tables are one-sample tests of each subject’s sphere-mean first-level z. `run_hcp_feat_after_unwarp.py` still lists older method names (`gated`, `sigmoid`, `gate1`, `mind_ung`). The comparison here is `topup` against `sbrigid_l6w2`.

## Layout

| File | Role |
| --- | --- |
| `run_epi_align_mij_w1_lm_bspline_fslbend_sbrigid_l6w2_n36.py` | Cohort weight-2 field |
| `run_epi_align_mij_w1_lm_bspline_fslbend_sbrigid.py` | Pre-solve spin-echo / reference FLIRT |
| `run_hcp_topup_sobel.py` | B-spline schedule and Levenberg–Marquardt solver |
| `model.py` | Shannon mutual information, plus other reconstruction helpers |
| `run_recursive_t1_mi.py` | `flirt_t1` and Studholme NMI scoring |
| `compare_unwarp_pcc.py` | FSL TOPUP, shared MNI affine, ROI coordinates |
| `run_hcp_feat_after_unwarp.py` | MCFLIRT, MNI resample, FEAT for the TOPUP series |
| `run_epi_align_mij_w1_lm_bspline_fslbend_sbrigid_l6w2_feat.py` | Weight-2 apply and FEAT |
