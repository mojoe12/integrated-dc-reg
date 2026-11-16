#!/usr/bin/env python3
"""TOPUP schedule on HCP, plus one extra term.

The integrated DC-Reg cohort calls optimize() with solver="lm",
resample="bspline", and bend="topup".

Matches b02b0.cnf except the optimizer and an optional anatomy term:
  cubic B-splines, knot spacing 20,16,14,12,10,6,4,4,4 mm,
  subsamp 2,2,2,2,2,1,1,1,1, fwhm 8..0 mm, miter, lambda, ssqlambda,
  bending energy, rigid movement on the first five levels, spline resampling,
  each spin-echo scaled to mean 100.
Field coefficients are hertz. PE displacement in voxels is hz * readout time.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as mm
import compare_unwarp_pcc as C

HCP = C.HCP_ROOT
ACQP_T = 0.064526
VOX_MM = 2.0
# b02b0.cnf
WARPRES = [20, 16, 14, 12, 10, 6, 4, 4, 4]
SUBSAMP = [2, 2, 2, 2, 2, 1, 1, 1, 1]
FWHM = [8, 6, 4, 3, 3, 2, 1, 0, 0]
MITER = [5, 5, 5, 5, 5, 10, 10, 20, 20]
LAMBDA = [0.005, 0.001, 0.0001, 1.5e-5, 5e-6, 5e-7, 5e-8, 5e-10, 1e-11]
ESTMOV = [1, 1, 1, 1, 1, 0, 0, 0, 0]
ADAM_LR = 1.0
# Adam's step is about the learning rate, independent of gradient size.
# 25 coarse-level updates (5 levels x 5 steps) need to be able to reach
# about a millimetre and a tenth of a radian. One shared rate cannot.
MOV_LR_MM = 0.08
MOV_LR_RAD = 0.008
ANAT_W = 0.02
# Anatomy weight vs displacement in original voxels. Center 1 vx, slope 4:
# ~0.12 at 0.5 vx, 0.5 at 1 vx, ~0.88 at 1.5 vx.
GATE_CENTER = 1.0
GATE_SLOPE = 4.0


def inhom_gate(hz):
    """Smooth weight from the field being estimated. 1 where |shift| >> 1 voxel."""
    u = tf.abs(hz) * ACQP_T
    return tf.math.sigmoid(GATE_SLOPE * (u - GATE_CENTER))


def kernel_value(x):
    ax = abs(float(x))
    if ax <= 1.0:
        return (2.0 / 3.0) + (ax * ax) * (ax / 2.0 - 1.0)
    if ax < 2.0:
        return (1.0 / 6.0) * (2.0 - ax) ** 3
    return 0.0


def kernel_deriv(x, ksp, deriv):
    """FSL cubic kernel. x is knot-spacings from the spline center.

    deriv 0 is the value. deriv 1 and 2 are per voxel, already divided by
    ksp**deriv, matching Spline1D::kernel_value order 3.
    """
    if deriv == 0:
        return kernel_value(x)
    ax = abs(float(x))
    if deriv == 1:
        if ax < 1e-6 or ax >= 2.0:
            return 0.0
        if ax <= 1.0:
            return (x / ax) * (1.5 * ax * ax - 2.0 * ax) / float(ksp)
        return (x / ax) * (-0.5 * (2.0 - ax) ** 2) / float(ksp)
    if deriv == 2:
        if ax <= 1.0:
            return (3.0 * ax - 2.0) / float(ksp * ksp)
        if ax < 2.0:
            return (2.0 - ax) / float(ksp * ksp)
        return 0.0
    raise ValueError("deriv must be 0, 1, or 2")


def kernel_size(ksp, order=3):
    if (order % 2 == 0) and (ksp % 2 == 1):
        return (order + 1) * ksp
    return (order + 1) * ksp - 1


def full_fov(n, ksp, order=3):
    """Voxels with any nonzero cubic-spline support, FSL Spline1D::FullFOV."""
    csz = n_coef(n, ksp)
    ks = kernel_size(ksp, order)
    if ks % 2:
        return (csz - 1) * ksp + ks
    return (csz - 1) * ksp + 1 + ks


def image_offset_in_fov(ksp, order=3):
    """Full-FOV row of image voxel 0. ksp==1 centers the first spline on voxel 0."""
    ks = kernel_size(ksp, order)
    coa = (ks - 1) / 2.0
    oif0 = 0 if ksp == 1 else -(order // 2) * ksp
    return int(round(coa - oif0))


def n_coef(n, ksp):
    if ksp == 1:
        return int(n)
    return int(math.ceil((n + 1) / float(ksp))) + 2


def spline_matrix(n, ksp):
    """Voxel values from cubic B-spline coefficients. Rows sum to ~1."""
    nc = n_coef(n, ksp)
    order = 3
    spsz = (order + 1) * ksp - 1
    coa = (spsz - 1) / 2.0
    M = np.zeros((n, nc), np.float64)
    for c in range(nc):
        oif = (c - order // 2) * ksp
        for i in range(spsz):
            vox = int(round(oif + (i - coa)))
            if 0 <= vox < n:
                M[vox, c] += kernel_value((vox - oif) / float(ksp))
    return M.astype(np.float32)


def spline_matrix_fov(n, ksp, deriv=0, vox_mm=1.0):
    """Map coefficients onto TOPUP's full spline FOV, including the border.

    deriv 0 is the cubic value. deriv 1 and 2 are d/d(mm) and d2/d(mm)^2:
    the kernel is per voxel, then divided by vox_mm**deriv.
    Rows are FullFOV, columns are n_coef. Coefficient c's kernel sample i
    lands at row c*ksp + i, the same spacing as spline_matrix.
    """
    nc = n_coef(n, ksp)
    ks = kernel_size(ksp)
    fov = full_fov(n, ksp)
    coa = (ks - 1) / 2.0
    scale = 1.0 / float(vox_mm) ** int(deriv)
    weights = np.empty(ks, np.float64)
    for i in range(ks):
        weights[i] = kernel_deriv((i - coa) / float(ksp), ksp, deriv) * scale
    rows = (np.arange(nc)[:, None] * int(ksp) + np.arange(ks)[None, :]).ravel()
    cols = np.repeat(np.arange(nc), ks)
    M = np.zeros((fov, nc), np.float64)
    M[rows, cols] = np.tile(weights, nc)
    return M


def eval_spline(coef, mats):
    """coef [Cx,Cy,Cz], mats are [Ni, Ci]. Returns [X,Y,Z]."""
    x = tf.tensordot(mats[0], coef, axes=[[1], [0]])  # [X, Cy, Cz]
    x = tf.transpose(x, [1, 0, 2])                    # [Cy, X, Cz]
    x = tf.tensordot(mats[1], x, axes=[[1], [0]])     # [Y, X, Cz]
    x = tf.transpose(x, [2, 1, 0])                    # [Cz, X, Y]
    x = tf.tensordot(mats[2], x, axes=[[1], [0]])     # [Z, X, Y]
    return tf.transpose(x, [1, 2, 0])                 # [X, Y, Z]


def project_coefs(field, ksps):
    """Separable least-squares cubic-spline coefficients of a dense field."""
    mats = [spline_matrix(n, k) for n, k in zip(field.shape, ksps)]
    cx, cy, cz = [m.shape[1] for m in mats]
    c = np.linalg.lstsq(mats[0], field.reshape(field.shape[0], -1), rcond=None)[0]
    c = c.reshape(cx, field.shape[1], field.shape[2])
    c = np.transpose(c, (1, 0, 2))
    c = np.linalg.lstsq(mats[1], c.reshape(c.shape[0], -1), rcond=None)[0]
    c = c.reshape(cy, cx, field.shape[2])
    c = np.transpose(c, (1, 0, 2))
    c = np.transpose(c, (2, 0, 1))
    c = np.linalg.lstsq(mats[2], c.reshape(c.shape[0], -1), rcond=None)[0]
    c = c.reshape(cz, cx, cy)
    c = np.transpose(c, (1, 2, 0))
    return c.astype(np.float32), mats


def subsample_mean(vol, ss):
    if ss == 1:
        return vol
    x, y, z = vol.shape
    v = vol[: x // ss * ss, : y // ss * ss, : z // ss * ss]
    v = v.reshape(x // ss, ss, y // ss, ss, z // ss, ss).mean(axis=(1, 3, 5))
    return v.astype(np.float32)


def smooth_fwhm(vol, fwhm_mm, vox_mm):
    if fwhm_mm <= 0:
        return vol
    sigma_mm = fwhm_mm / math.sqrt(8.0 * math.log(2.0))
    sigma_vox = sigma_mm / vox_mm
    sm = mm.gaussian_smooth3d(tf.constant(vol)[None], sigma_vox)
    return sm[0].numpy()


def ksp_of(warpres_mm, vox_mm):
    return max(1, int(round(warpres_mm / vox_mm)))


def resample_field(hz, ss_old, ss_new, new_shape):
    if ss_old == ss_new and tuple(hz.shape) == tuple(new_shape):
        return hz.astype(np.float32)
    step = float(ss_new) / float(ss_old)
    start = (step - 1.0) / 2.0
    coords = [(start + i * step) for i in range(new_shape[0])], \
             [(start + i * step) for i in range(new_shape[1])], \
             [(start + i * step) for i in range(new_shape[2])]
    grid = np.meshgrid(*coords, indexing="ij")
    from scipy.ndimage import map_coordinates
    out = map_coordinates(hz, np.stack(grid, 0), order=3, mode="nearest")
    return out.astype(np.float32)


def bend_energy(hz):
    dxx = hz[2:, :, :] - 2.0 * hz[1:-1, :, :] + hz[:-2, :, :]
    dyy = hz[:, 2:, :] - 2.0 * hz[:, 1:-1, :] + hz[:, :-2, :]
    dzz = hz[:, :, 2:] - 2.0 * hz[:, :, 1:-1] + hz[:, :, :-2]
    dxy = hz[1:, 1:, :] - hz[:-1, 1:, :] - hz[1:, :-1, :] + hz[:-1, :-1, :]
    dxz = hz[1:, :, 1:] - hz[:-1, :, 1:] - hz[1:, :, :-1] + hz[:-1, :, :-1]
    dyz = hz[:, 1:, 1:] - hz[:, :-1, 1:] - hz[:, 1:, :-1] + hz[:, :-1, :-1]
    return (tf.reduce_sum(tf.square(dxx)) + tf.reduce_sum(tf.square(dyy))
            + tf.reduce_sum(tf.square(dzz)) + 2.0 * (
                tf.reduce_sum(tf.square(dxy)) + tf.reduce_sum(tf.square(dxz))
                + tf.reduce_sum(tf.square(dyz))))


def cubic_weights(t):
    t2 = t * t
    t3 = t2 * t
    w0 = -0.5 * t3 + t2 - 0.5 * t
    w1 = 1.5 * t3 - 2.5 * t2 + 1.0
    w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
    w3 = 0.5 * t3 - 0.5 * t2
    return w0, w1, w2, w3


def sample_cubic(vol, x, y, z):
    """Catmull-Rom sample. vol [X,Y,Z], coordinates in voxels."""
    vol_p = tf.pad(vol, [[1, 2], [1, 2], [1, 2]])
    x = x + 1.0
    y = y + 1.0
    z = z + 1.0
    x0 = tf.floor(x)
    y0 = tf.floor(y)
    z0 = tf.floor(z)
    wx = cubic_weights(x - x0)
    wy = cubic_weights(y - y0)
    wz = cubic_weights(z - z0)
    ix = tf.cast(x0, tf.int32)
    iy = tf.cast(y0, tf.int32)
    iz = tf.cast(z0, tf.int32)
    nx = tf.shape(vol_p)[0] - 1
    ny = tf.shape(vol_p)[1] - 1
    nz = tf.shape(vol_p)[2] - 1
    acc = tf.zeros_like(x)
    for a in range(4):
        for b in range(4):
            for c in range(4):
                gx = tf.clip_by_value(ix + (a - 1), 0, nx)
                gy = tf.clip_by_value(iy + (b - 1), 0, ny)
                gz = tf.clip_by_value(iz + (c - 1), 0, nz)
                acc = acc + wx[a] * wy[b] * wz[c] * tf.gather_nd(
                    vol_p, tf.stack([gx, gy, gz], -1))
    return acc


def bspline3_weights(t):
    """Cubic B-spline weights for knots floor(t0)-1 .. +2. t = x - floor(x)."""
    t2 = t * t
    t3 = t2 * t
    w0 = (1.0 - 3.0 * t + 3.0 * t2 - t3) / 6.0
    w1 = (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0
    w2 = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0
    w3 = t3 / 6.0
    return w0, w1, w2, w3


def _cubic_deconv(columns, periodic, prec=1e-8):
    """Unser cubic-spline coefficients. columns is float64 (length, lines).

    The recurrence matches FSL Splinterpolator: one pole at sqrt(3)-2, gain 6.
    Periodic is TOPUP's extension. The other branch is the mirror initialisation
    FSL uses for every non-periodic mode, including applywarp's default.
    """
    z = math.sqrt(3.0) - 2.0
    n = columns.shape[0]
    c = np.array(columns, dtype=np.float64, copy=True)
    # FSL truncates (log(prec)/log(|z|) + 1.5). It does not take ceil.
    horizon = int(math.log(prec) / math.log(abs(z)) + 1.5)
    horizon = max(1, min(n, horizon))
    if periodic:
        iv = c[0].copy()
        zk = z
        idx = n - 1
        for _ in range(1, horizon):
            iv = iv + zk * c[idx]
            zk *= z
            idx -= 1
        c[0] = iv
    else:
        iv = c[0].copy()
        zk = z
        for i in range(1, horizon):
            iv = iv + zk * c[i]
            zk *= z
        c[0] = iv
    orig_last = c[-1].copy()
    for i in range(1, n):
        c[i] = c[i] + z * c[i - 1]
    if periodic:
        iv = z * c[-1]
        z2i = z * z
        for i in range(1, horizon):
            iv = iv + z2i * c[i - 1]
            z2i *= z
        c[-1] = iv / (z2i - 1.0)
    else:
        c[-1] = (-z / (1.0 - z * z)) * (2.0 * c[-1] - orig_last)
    for i in range(n - 2, -1, -1):
        c[i] = z * (c[i + 1] - c[i])
    c *= 6.0
    return c


def bspline3_prefilter(vol, periodic_axes=(True, True, True), dtype=np.float32):
    """Interpolating cubic B-spline coefficients of a 3D image.

    TOPUP sets the volume to periodic extrapolation, and the coefficient
    calculation copies that mode onto every axis. The default matches that.
    """
    out = np.asarray(vol, dtype=np.float64)
    if out.ndim != 3:
        raise ValueError("bspline3_prefilter expects a 3D volume")
    for ax, periodic in enumerate(periodic_axes):
        moved = np.moveaxis(out, ax, -1)
        flat = np.ascontiguousarray(moved.reshape(-1, moved.shape[-1]))
        coef = _cubic_deconv(flat.T, periodic).T.reshape(moved.shape)
        out = np.moveaxis(coef, -1, ax)
    return out.astype(dtype)


def _bound_index(i, n, periodic):
    """Knot index outside the array. Periodic matches TOPUP. Mirror is Thevenaz."""
    if periodic:
        return tf.math.floormod(i, n)
    period = 2 * n - 2
    i = tf.math.floormod(i, period)
    return tf.where(i >= n, period - i, i)


def sample_bspline3(coef, x, y, z):
    """Sample prefiltered cubic B-spline coefficients. Every axis is periodic.

    That is TOPUP's image model: setextrapolationmethod(periodic) is copied
    onto x, y, and z when the spline coefficients are built.
    """
    x0 = tf.floor(x)
    y0 = tf.floor(y)
    z0 = tf.floor(z)
    wx = bspline3_weights(x - x0)
    wy = bspline3_weights(y - y0)
    wz = bspline3_weights(z - z0)
    ix = tf.cast(x0, tf.int32)
    iy = tf.cast(y0, tf.int32)
    iz = tf.cast(z0, tf.int32)
    nx = tf.shape(coef)[0]
    ny = tf.shape(coef)[1]
    nz = tf.shape(coef)[2]
    acc = tf.zeros_like(x)
    for a in range(4):
        gx = _bound_index(ix + (a - 1), nx, True)
        for b in range(4):
            gy = _bound_index(iy + (b - 1), ny, True)
            for c in range(4):
                gz = _bound_index(iz + (c - 1), nz, True)
                acc = acc + wx[a] * wy[b] * wz[c] * tf.gather_nd(
                    coef, tf.stack([gx, gy, gz], -1))
    return acc


def sample_bspline3_np(coef, x, y, z, periodic_axes=(True, True, True)):
    """NumPy twin of sample_bspline3. coef is already prefiltered."""
    coef = np.asarray(coef)
    work = np.float64 if coef.dtype == np.float64 else np.float32
    x0 = np.floor(x)
    y0 = np.floor(y)
    z0 = np.floor(z)
    wx = bspline3_weights((x - x0).astype(work))
    wy = bspline3_weights((y - y0).astype(work))
    wz = bspline3_weights((z - z0).astype(work))
    nx, ny, nz = coef.shape
    periodic = tuple(bool(p) for p in periodic_axes)

    def index(i, n, wrap):
        if wrap:
            return np.mod(i, n).astype(np.int32)
        period = 2 * n - 2
        i = np.mod(i, period)
        return np.where(i >= n, period - i, i).astype(np.int32)

    ix = np.floor(x).astype(np.int32)
    iy = np.floor(y).astype(np.int32)
    iz = np.floor(z).astype(np.int32)
    acc = np.zeros(np.broadcast(x, y, z).shape, work)
    for a in range(4):
        gx = index(ix + (a - 1), nx, periodic[0])
        for b in range(4):
            gy = index(iy + (b - 1), ny, periodic[1])
            for c in range(4):
                gz = index(iz + (c - 1), nz, periodic[2])
                acc += wx[a] * wy[b] * wz[c] * coef[gx, gy, gz]
    return acc


def warp_volume_bspline(vol, u, jac=True):
    """Unwarp one 3D volume with the same cubic B-spline lookup the fitter uses."""
    coef = bspline3_prefilter(vol)
    x = np.arange(vol.shape[0], dtype=np.float32)
    y = np.arange(vol.shape[1], dtype=np.float32)
    z = np.arange(vol.shape[2], dtype=np.float32)
    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
    out = sample_bspline3_np(coef, gx + C.PE_SIGN * u, gy, gz)
    if jac:
        j = np.clip(1.0 + C.PE_SIGN * np.gradient(u, axis=0), 0.0, None).astype(np.float32)
        out = out * j
    return out.astype(np.float32)


def rigid_source(x, y, z, mov, vox):
    """Map target voxel indices to source indices. mov = tx,ty,tz mm, rx,ry,rz rad."""
    shape = tf.shape(x)
    cx = (tf.cast(shape[0], tf.float32) - 1.0) * 0.5
    cy = (tf.cast(shape[1], tf.float32) - 1.0) * 0.5
    cz = (tf.cast(shape[2], tf.float32) - 1.0) * 0.5
    px = (x - cx) * vox
    py = (y - cy) * vox
    pz = (z - cz) * vox
    rx, ry, rz = mov[3], mov[4], mov[5]
    cxr, sxr = tf.cos(rx), tf.sin(rx)
    cyr, syr = tf.cos(ry), tf.sin(ry)
    czr, szr = tf.cos(rz), tf.sin(rz)
    # R = Rz @ Ry @ Rx, small-angle compatible Euler
    r00 = czr * cyr
    r01 = czr * syr * sxr - szr * cxr
    r02 = czr * syr * cxr + szr * sxr
    r10 = szr * cyr
    r11 = szr * syr * sxr + czr * cxr
    r12 = szr * syr * cxr - czr * sxr
    r20 = -syr
    r21 = cyr * sxr
    r22 = cyr * cxr
    qx = r00 * px + r01 * py + r02 * pz + mov[0]
    qy = r10 * px + r11 * py + r12 * pz + mov[1]
    qz = r20 * px + r21 * py + r22 * pz + mov[2]
    return qx / vox + cx, qy / vox + cy, qz / vox + cz


def jac_from_hz(hz, pe_sign, ss):
    """J = 1 + pe * readout * (orig_vox/current_vox) * dHz/dindex."""
    d = tf.concat([
        hz[1:2] - hz[0:1],
        (hz[2:] - hz[:-2]) * 0.5,
        hz[-1:] - hz[-2:-1],
    ], axis=0)
    return tf.maximum(1.0 + pe_sign * ACQP_T * d / float(ss), 0.0)


def coord_grid(shape):
    x, y, z = shape
    gx, gy, gz = tf.meshgrid(
        tf.range(x, dtype=tf.float32),
        tf.range(y, dtype=tf.float32),
        tf.range(z, dtype=tf.float32),
        indexing="ij",
    )
    return gx, gy, gz


def warp_pair(lr, rl, hz, mov, ss, estmov, sampler=sample_cubic):
    """Resample the spin-echo pair. estmov only controls whether motion is updated.

    TOPUP freezes the rigid motion after the coarse levels and still applies it.
    Stopping the application lets a head roll leak into the phase-encode field.
    sampler is sample_cubic (Catmull-Rom) or sample_bspline3 (prefiltered coefficients).
    """
    gx, gy, gz = coord_grid(hz.shape)
    # displacement in voxels of this grid: hz * readout * orig_vox / current_vox
    disp = hz * ACQP_T / float(ss)
    j = jac_from_hz(hz, 1.0, ss)
    jn = jac_from_hz(hz, -1.0, ss)
    xl = gx + disp
    mov_use = mov if estmov else tf.stop_gradient(mov)
    xr, yr, zr = rigid_source(gx, gy, gz, mov_use, VOX_MM * ss)
    xr = xr - disp
    sl = sampler(lr, xl, gy, gz) * j
    sr = sampler(rl, xr, yr, zr) * jn
    return sl, sr


def valid_mask(shape):
    """Drop the non-PE border frame, matching TOPUP."""
    x, y, z = shape
    m = np.ones(shape, np.float32)
    m[:, 0, :] = 0
    m[:, -1, :] = 0
    m[:, :, 0] = 0
    m[:, :, -1] = 0
    return tf.constant(m)


def ssd_of(sl, sr, mask):
    n = tf.reduce_sum(mask)
    mean = 0.5 * (sl + sr)
    err = mask * (tf.square(mean - sl) + tf.square(mean - sr))
    return tf.reduce_sum(err) / (n + 1e-6)


def bend_terms(hz):
    """Second-difference vector whose squared norm is bend_energy(hz)."""
    dxx = hz[2:, :, :] - 2.0 * hz[1:-1, :, :] + hz[:-2, :, :]
    dyy = hz[:, 2:, :] - 2.0 * hz[:, 1:-1, :] + hz[:, :-2, :]
    dzz = hz[:, :, 2:] - 2.0 * hz[:, :, 1:-1] + hz[:, :, :-2]
    dxy = hz[1:, 1:, :] - hz[:-1, 1:, :] - hz[1:, :-1, :] + hz[:-1, :-1, :]
    dxz = hz[1:, :, 1:] - hz[:-1, :, 1:] - hz[1:, :, :-1] + hz[:-1, :, :-1]
    dyz = hz[:, 1:, 1:] - hz[:, :-1, 1:] - hz[:, 1:, :-1] + hz[:, :-1, :-1]
    rt2 = tf.sqrt(tf.constant(2.0, tf.float32))
    return tf.concat([
        tf.reshape(dxx, [-1]), tf.reshape(dyy, [-1]), tf.reshape(dzz, [-1]),
        tf.reshape(dxy * rt2, [-1]), tf.reshape(dxz * rt2, [-1]), tf.reshape(dyz * rt2, [-1]),
    ], 0)


def bend_terms_coef(coef, mats):
    """Analytic bending residual. Its squared norm is splinefield::BendEnergy.

    mats[axis] is (value, d/dmm, d2/dmm2) on the full spline FOV. Diagonal
    second derivatives are summed once and cross derivatives twice, by
    scaling the cross-term samples by sqrt(2). The samples include the
    border outside the image, which is what BendEnergy integrates.
    """
    parts = []
    rt2 = tf.sqrt(tf.cast(2.0, coef.dtype))
    for d1 in range(3):
        for d2 in range(d1, 3):
            order = [0, 0, 0]
            order[d1] += 1
            order[d2] += 1
            field = eval_spline(coef, [mats[a][order[a]] for a in range(3)])
            if d1 != d2:
                field = field * rt2
            parts.append(tf.reshape(field, [-1]))
    return tf.concat(parts, 0)


def lm_steps(coef, mov, mats, lr_t, rl_t, mask, ss, est, lam, nstep,
             sb_t=None, t1_t=None, anat_mask=None, anat_w=0.0, sampler=sample_cubic,
             bend="fd", bend_mats=None, ssd_cap=None):
    """Levenberg–Marquardt on the spin-echo residual, optional MI on the right-hand side.

    Matrix-free Gauss–Newton: (JᵀJ + μI) δ = −(Jᵀr + ½ ∇L_MI).
    L_MI is weight × SSD × MI₀ / MI. The normal matrix stays the spin-echo Jacobian.
    λ_eff = current SSD × config λ, frozen during the step, as in TOPUP ssqlambda.
    """
    coef_shape = tuple(int(s) for s in coef.shape)
    ncoef = int(np.prod(coef_shape))
    use_mi = sb_t is not None and float(anat_w) > 0.0

    def residual(theta, lam_eff):
        coef_t = tf.reshape(theta[:ncoef], coef_shape)
        hz = eval_spline(coef_t, mats)
        if est:
            mov_t = theta[ncoef:ncoef + 6]
            sl, sr = warp_pair(lr_t, rl_t, hz, mov_t, ss, True, sampler)
        else:
            sl, sr = warp_pair(lr_t, rl_t, hz, mov, ss, False, sampler)
        n = tf.reduce_sum(mask)
        data = (sl - sr) * tf.sqrt(mask / (2.0 * n + 1e-6))
        if bend == "topup":
            reg_vec = bend_terms_coef(coef_t, bend_mats)
        else:
            reg_vec = bend_terms(hz)
        reg = tf.sqrt(lam_eff + 1e-12) * reg_vec
        return tf.concat([tf.reshape(data, [-1]), reg], 0)

    def pack():
        c = tf.reshape(coef, [-1])
        if est:
            return tf.concat([c, mov], 0)
        return c

    def mi_term(theta, ref):
        """L_MI = weight × SSD × ref / MI. ref is frozen at the linearization point."""
        coef_t = tf.reshape(theta[:ncoef], coef_shape)
        hz = eval_spline(coef_t, mats)
        if est:
            sl, sr = warp_pair(lr_t, rl_t, hz, theta[ncoef:ncoef + 6], ss, True, sampler)
        else:
            sl, sr = warp_pair(lr_t, rl_t, hz, mov, ss, False, sampler)
        ssd = ssd_of(sl, sr, mask)
        disp = hz * ACQP_T / float(ss)
        gx, gy, gz = coord_grid(hz.shape)
        epi = sampler(sb_t, gx + disp, gy, gz) * jac_from_hz(hz, 1.0, ss)
        flat = tf.boolean_mask(epi, anat_mask > 0.5)
        lo = tf.stop_gradient(tf.reduce_min(flat))
        hi = tf.stop_gradient(tf.reduce_max(flat))
        epi = tf.clip_by_value((epi - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        aval = -mm.mutual_information_loss(t1_t[None], epi[None], anat_mask)[0]
        if ref is None:
            ref = tf.stop_gradient(aval)
        return float(anat_w) * ssd * ref / (aval + 1e-6), aval, ref

    def spin_echo_ssd(theta):
        coef_t = tf.reshape(theta[:ncoef], coef_shape)
        hz = eval_spline(coef_t, mats)
        if est:
            sl, sr = warp_pair(lr_t, rl_t, hz, theta[ncoef:ncoef + 6], ss, True, sampler)
        else:
            sl, sr = warp_pair(lr_t, rl_t, hz, mov, ss, False, sampler)
        return ssd_of(sl, sr, mask)

    mu = 1e-3
    n_accept = 0
    for i in range(nstep):
        theta = tf.identity(pack())
        hz0 = eval_spline(coef, mats)
        sl, sr = warp_pair(lr_t, rl_t, hz0, mov, ss, bool(est), sampler)
        ssd = float(ssd_of(sl, sr, mask))
        lam_eff = tf.constant(ssd * float(lam), tf.float32)
        r = residual(theta, lam_eff)
        cost_ls = float(tf.reduce_sum(tf.square(r)))
        with tf.GradientTape() as tape:
            tape.watch(theta)
            r_g = residual(theta, lam_eff)
        g = tape.gradient(r_g, theta, output_gradients=r_g)
        ref_fixed = None
        mi_now = 0.0
        if use_mi:
            with tf.GradientTape() as tape_m:
                tape_m.watch(theta)
                Lmi, aval, ref_fixed = mi_term(theta, None)
            g_mi = tape_m.gradient(Lmi, theta)
            g = g + 0.5 * g_mi
            mi_now = float(Lmi)
        cost = cost_ls + mi_now

        def matvec(v_np):
            v = tf.constant(v_np, tf.float32)
            with tf.autodiff.ForwardAccumulator(theta, v) as acc:
                rv = residual(theta, lam_eff)
            jv = acc.jvp(rv)
            with tf.GradientTape() as tape2:
                tape2.watch(theta)
                r2 = residual(theta, lam_eff)
            h = tape2.gradient(r2, theta, output_gradients=jv)
            return h.numpy() + mu * v_np

        b = -g.numpy()
        delta = np.zeros_like(b)
        cg_r = b.copy()
        p = cg_r.copy()
        rs = float(np.dot(cg_r, cg_r))
        bnorm = rs + 1e-12
        for _ in range(12):
            Ap = matvec(p)
            denom = float(np.dot(p, Ap)) + 1e-12
            alpha = rs / denom
            delta = delta + alpha * p
            cg_r = cg_r - alpha * Ap
            rs_new = float(np.dot(cg_r, cg_r))
            if rs_new < 1e-6 * bnorm:
                break
            p = cg_r + (rs_new / (rs + 1e-12)) * p
            rs = rs_new
        step_norm = float(np.linalg.norm(delta))
        theta_norm = float(np.linalg.norm(theta.numpy())) + 1.0
        if step_norm > 50.0 * theta_norm:
            mu = min(mu * 10.0, 1e6)
            print(f"    lm {i:3d}  reject huge step {step_norm:.2f}  μ={mu:.3g}", flush=True)
            continue
        trial = theta + tf.constant(delta, tf.float32)
        if est:
            trial = tf.concat([trial[:ncoef], tf.zeros([1], tf.float32), trial[ncoef + 1:]], 0)
        r_try = residual(trial, lam_eff)
        cost_try = float(tf.reduce_sum(tf.square(r_try)))
        if use_mi:
            Lmi_try, _, _ = mi_term(trial, ref_fixed)
            cost_try = cost_try + float(Lmi_try)
        ssd_try = float(spin_echo_ssd(trial)) if ssd_cap is not None else None
        over_cap = ssd_cap is not None and ssd_try > ssd_cap
        if (not over_cap) and cost_try < cost:
            coef.assign(tf.reshape(trial[:ncoef], coef_shape))
            if est:
                trial_mov = trial[ncoef:ncoef + 6]
                mov.assign(tf.concat([tf.zeros([1], tf.float32), trial_mov[1:]], 0))
            mu = max(mu / 3.0, 1e-8)
            flag = "accept"
            n_accept += 1
        else:
            mu = min(mu * 3.0, 1e6)
            flag = "reject"
        if i == 0 or i == nstep - 1 or flag == "reject":
            hz_n = float(tf.reduce_mean(tf.abs(eval_spline(coef, mats))))
            cap_note = ""
            if over_cap:
                cap_note = f"  ssd {ssd:.1f}->{ssd_try:.1f} over cap {ssd_cap:.1f}"
            print(
                f"    lm {i:3d}  {flag}  cost {cost:.3f}->{cost_try:.3f}  "
                f"ssd={ssd:.3f}  |hz|={hz_n:.2f}  μ={mu:.3g}  "
                f"mov={np.array2string(mov.numpy(), precision=3)}{cap_note}",
                flush=True,
            )
    return n_accept


def nmi(a, b, mask, bins=32):
    x = a[mask].astype(np.float64)
    y = b[mask].astype(np.float64)
    x = np.clip(x, *np.percentile(x, [1, 99]))
    y = np.clip(y, *np.percentile(y, [1, 99]))
    h, _, _ = np.histogram2d(x, y, bins=bins)
    p = h / max(h.sum(), 1)
    def ent(q):
        q = q[q > 1e-12]
        return float(-(q * np.log(q)).sum())
    hx, hy, hxy = ent(p.sum(1)), ent(p.sum(0)), ent(p)
    return (hx + hy) / (hxy + 1e-12)


def corr(a, b, m):
    a = a[m].astype(np.float64)
    b = b[m].astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-8))


def fsl_signed(work, sb, mask):
    _, hz = C.load(work / "tp_field.nii.gz")
    if hz.ndim > 3:
        hz = hz[..., 0]
    fsl = (hz * ACQP_T).astype(np.float32)
    _, sb_top = C.load(work / "sb_topup.nii.gz")
    c_plus = corr(C.warp_pe_np(sb, fsl, jac=True), sb_top, mask)
    c_minus = corr(C.warp_pe_np(sb, -fsl, jac=True), sb_top, mask)
    sign = 1.0 if c_plus >= c_minus else -1.0
    print(f"  FSL sign={sign:+.0f}", flush=True)
    return (sign * fsl).astype(np.float32)


def scale01(vol, mask):
    v = vol.astype(np.float32)
    lo, hi = np.percentile(v[mask], [0.5, 99.5])
    return np.clip((v - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


def optimize(lr0, rl0, sb0, t1_anat, brain, anat, anat_w, solver="adam", use_gate=True, init_hz=None,
             resample="catmull", bend="fd", mi_ramp=None, mi_ramp_steps=4, mi_ssd_frac=0.05,
             init_mov=None, ramp_only=False, mi_ramp_level=9,
             mi_level_ramps=None, mi_level_replace=(), hold_mov_after=None):
    """lr0/rl0 are full-resolution spin-echo, scaled to mean 100.

    anat is 'none', 'sobel', 'mind', 'ngf', or 'mi'. Sobel sees an inverted T1.
    MIND, NGF, and MI see the uninverted T1, scaled to 0–1 for MIND and MI.
    Those three do not multiply the Jacobian into the SBRef.
    MI is maximized (the loss uses MI_start / MI).
    ramp_only keeps init_hz and init_mov and runs mi_ramp at the final
    resolution. Omitting it follows the nine-level schedule.
    mi_ramp_level is the stage that runs mi_ramp. The default, 9, runs it
    after the last stage. An earlier frozen stage, such as 6, runs it there,
    and the remaining stages continue at anat_w.
    mi_level_ramps maps a stage number to its own weight ladder. Stages listed
    in mi_level_replace skip the anat_w iterations and start on that ladder.
    A ramp on a motion-estimation stage runs after that stage's weight-1
    steps. The pose is then held fixed for the ramp and still applied.
    hold_mov_after freezes the pose for every stage after that number.
    Zero holds it for the whole schedule. The pose still warps the pair.
    """
    coef = None
    hz_dense = None if init_hz is None else np.asarray(init_hz, np.float32)
    ss_prev = None if init_hz is None else 1
    mov0 = np.zeros(6, np.float32) if init_mov is None else np.asarray(init_mov, np.float32).reshape(6).copy()
    mov0[0] = 0.0
    mov = tf.Variable(mov0)
    schedule = list(zip(WARPRES, SUBSAMP, FWHM, MITER, LAMBDA, ESTMOV))
    shown_level = list(range(1, len(schedule) + 1))
    if ramp_only:
        if init_hz is None or init_mov is None or not mi_ramp:
            raise ValueError("ramp_only requires init_hz, init_mov, and mi_ramp")
        if solver != "lm" or anat not in ("mi", "mij"):
            raise ValueError("ramp_only requires the least-squares solver and anat mi or mij")
        schedule = schedule[-1:]
        shown_level = [len(shown_level)]
    level_ramps = {} if not mi_level_ramps else {int(k): tuple(v) for k, v in mi_level_ramps.items()}
    replace_levels = {int(x) for x in (mi_level_replace or ())}
    if hold_mov_after is not None:
        print(
            f"  motion held from level {int(hold_mov_after) + 1} at "
            f"{np.array2string(mov.numpy(), precision=4)}",
            flush=True,
        )
    # Moments persist across levels. Recreating Adam each level was resetting
    # the only state that could slow the step once the motion was close.
    mom_mm = tf.Variable(tf.zeros([3], tf.float32))
    vel_mm = tf.Variable(tf.zeros([3], tf.float32))
    mom_rad = tf.Variable(tf.zeros([3], tf.float32))
    vel_rad = tf.Variable(tf.zeros([3], tf.float32))
    mov_step = tf.Variable(0.0, tf.float32)

    def apply_mov_grad(grad):
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        mov_step.assign_add(1.0)
        t = mov_step
        b1 = 1.0 - beta1 ** t
        b2 = 1.0 - beta2 ** t

        def one(mom, vel, g, lr):
            mom.assign(beta1 * mom + (1.0 - beta1) * g)
            vel.assign(beta2 * vel + (1.0 - beta2) * tf.square(g))
            return lr * (mom / b1) / (tf.sqrt(vel / b2) + eps)

        # Index 0 is translation along phase encode. It changes the spin-echo
        # pair in the same way as a constant field, so TOPUP never estimates it.
        grad = tf.concat([tf.zeros([1], tf.float32), grad[1:]], 0)
        d_mm = one(mom_mm, vel_mm, grad[:3], MOV_LR_MM)
        d_rad = one(mom_rad, vel_rad, grad[3:], MOV_LR_RAD)
        updated = mov - tf.concat([d_mm, d_rad], 0)
        mov.assign(tf.concat([tf.zeros([1], tf.float32), updated[1:]], 0))

    raw_pair = {}

    def raw_ssd(hz_vol, mov_var, ss_now):
        """Spin-echo residual of this field on the unsmoothed full-resolution pair.

        A subsampled stage is interpolated back to the acquired grid first.
        The ramp cap stays on the smoothed residual the solver sees.
        """
        if "lr" not in raw_pair:
            vol_lr = np.asarray(lr0, np.float32)
            vol_rl = np.asarray(rl0, np.float32)
            if resample == "bspline":
                raw_pair["lr"] = tf.constant(bspline3_prefilter(vol_lr))
                raw_pair["rl"] = tf.constant(bspline3_prefilter(vol_rl))
                raw_pair["sampler"] = sample_bspline3
            elif resample == "catmull":
                raw_pair["lr"] = tf.constant(vol_lr)
                raw_pair["rl"] = tf.constant(vol_rl)
                raw_pair["sampler"] = sample_cubic
            else:
                return None
            raw_pair["mask"] = valid_mask(tuple(lr0.shape))
        hz_use = hz_vol
        if int(ss_now) != 1 or tuple(hz_vol.shape) != tuple(lr0.shape):
            hz_np = hz_vol.numpy() if tf.is_tensor(hz_vol) else np.asarray(hz_vol, np.float32)
            hz_use = tf.constant(
                resample_field(hz_np, int(ss_now), 1, tuple(int(s) for s in lr0.shape))
            )
        sl, sr = warp_pair(
            raw_pair["lr"], raw_pair["rl"], hz_use, mov_var, 1, False, raw_pair["sampler"],
        )
        return float(ssd_of(sl, sr, raw_pair["mask"]))

    def run_mi_ramp(state, level_no, fwhm_mm, weights=None):
        """Raise the mutual-information weight on one stage, motion held fixed."""
        ladder = tuple(mi_ramp if weights is None else weights)
        hz_now = eval_spline(state["coef"], state["mats"])
        sl, sr = warp_pair(
            state["lr_t"], state["rl_t"], hz_now, state["mov"], state["ss"], False, state["sampler"],
        )
        ssd0 = float(ssd_of(sl, sr, state["mask"]))
        cap = ssd0 * (1.0 + float(mi_ssd_frac))
        raw0 = raw_ssd(hz_now, state["mov"], state["ss"]) if fwhm_mm > 0 else None
        ssd_name = "smoothed ssd" if fwhm_mm > 0 else "ssd"
        raw_note = f"  unsmoothed ssd={raw0:.3f}" if raw0 is not None else ""
        print(
            f"  mi ramp level {int(level_no)} base weight {float(anat_w):g}  "
            f"{ssd_name}={ssd0:.3f}  cap={cap:.3f} ({float(mi_ssd_frac):.0%} above the starting residual)"
            f"{raw_note}",
            flush=True,
        )
        for w in ladder:
            print(f"=== mi ramp weight {float(w):g}  steps {int(mi_ramp_steps)} ===", flush=True)
            n_acc = lm_steps(
                state["coef"], state["mov"], state["mats"], state["lr_t"], state["rl_t"],
                state["mask"], state["ss"], False, state["lam"], int(mi_ramp_steps),
                sb_t=state["sb_t"], t1_t=state["t1_t"], anat_mask=state["anat_mask"],
                anat_w=float(w), sampler=state["sampler"], bend=bend,
                bend_mats=state["bend_mats"], ssd_cap=cap,
            )
            hz_now = eval_spline(state["coef"], state["mats"])
            sl, sr = warp_pair(
                state["lr_t"], state["rl_t"], hz_now, state["mov"], state["ss"], False, state["sampler"],
            )
            ssd_now = float(ssd_of(sl, sr, state["mask"]))
            raw_now = raw_ssd(hz_now, state["mov"], state["ss"]) if fwhm_mm > 0 else None
            ssd_name = "smoothed ssd" if fwhm_mm > 0 else "ssd"
            raw_note = f"  unsmoothed ssd={raw_now:.3f}" if raw_now is not None else ""
            print(
                f"  mi ramp weight {float(w):g} accepted {n_acc}  {ssd_name}={ssd_now:.3f}{raw_note}",
                flush=True,
            )
            if n_acc == 0:
                print(f"  mi ramp stop at weight {float(w):g}", flush=True)
                break
        return hz_now.numpy()

    for lev, (wr, ss, fwhm, nstep, lam, est) in enumerate(schedule):
        if hold_mov_after is not None and int(shown_level[lev]) > int(hold_mov_after):
            est = 0
        vox = VOX_MM * ss
        shape = tuple(s // ss for s in lr0.shape)
        ksp = ksp_of(wr, vox)
        ksps = (ksp, ksp, ksp)
        lr = smooth_fwhm(subsample_mean(lr0, ss), fwhm, vox)
        rl = smooth_fwhm(subsample_mean(rl0, ss), fwhm, vox)
        sb = subsample_mean(sb0, ss)
        t1 = subsample_mean(t1_anat, ss)
        bmask = subsample_mean(brain.astype(np.float32), ss) > 0.5
        if anat == "sobel":
            sb = smooth_fwhm(sb, fwhm, vox)
            t1 = smooth_fwhm(t1, fwhm, vox)
        if resample == "bspline":
            # Coefficients of the fixed images. The sampler then evaluates the
            # interpolating cubic B-spline TOPUP uses (periodic on x, y, and z).
            lr = bspline3_prefilter(lr)
            rl = bspline3_prefilter(rl)
            sb = bspline3_prefilter(sb)
            sampler = sample_bspline3
        elif resample == "catmull":
            sampler = sample_cubic
        else:
            raise ValueError("resample must be catmull or bspline")
        if hz_dense is None:
            coef_np = np.zeros([n_coef(n, ksp) for n in shape], np.float32)
        else:
            hz_s = resample_field(hz_dense, ss_prev, ss, shape)
            coef_np, _ = project_coefs(hz_s, ksps)
        mats = [tf.constant(spline_matrix(n, k)) for n, k in zip(shape, ksps)]
        if bend == "topup":
            bend_mats = [
                tuple(tf.constant(spline_matrix_fov(n, k, d, vox), tf.float32) for d in (0, 1, 2))
                for n, k in zip(shape, ksps)
            ]
        elif bend == "fd":
            bend_mats = None
        else:
            raise ValueError("bend must be fd or topup")
        coef = tf.Variable(coef_np)
        opt_c = tf.keras.optimizers.Adam(ADAM_LR)
        lr_t = tf.constant(lr)
        rl_t = tf.constant(rl)
        use_anat = anat in ("sobel", "mind", "ngf", "mi", "mij")
        sb_t = tf.constant(sb) if use_anat else None
        t1_t = tf.constant(t1) if use_anat else None
        mask = valid_mask(shape)
        anat_mask = tf.constant((bmask & (mask.numpy() > 0)).astype(np.float32))
        if anat == "sobel":
            anat0 = mm.getGradientLoss(t1_t[None], sb_t[None], weight=anat_mask[None])[0]
        elif anat == "mind":
            anat0 = mm.mind_loss(t1_t, sb_t, mask=anat_mask)[0]
        elif anat == "ngf":
            anat0 = mm.ngf_loss(t1_t, sb_t, mask=anat_mask)[0]
        elif anat in ("mi", "mij"):
            anat0 = -mm.mutual_information_loss(t1_t[None], sb_t[None], anat_mask)[0]
        else:
            anat0 = tf.constant(1.0)
        print(
            f"=== level {shown_level[lev]}/9  knots {wr:g} mm  ksp {ksp} vox  "
            f"ss {ss}  fwhm {fwhm:g}  steps {nstep}  λ {lam:g}  "
            f"mov {est}  coef {tuple(coef_np.shape)}  resample {resample}  bend {bend} ===",
            flush=True,
        )
        if ramp_only:
            recon = eval_spline(coef, mats).numpy()
            mae_mm = float(np.mean(np.abs(recon - hz_s))) * ACQP_T
            print(
                f"  continuation from the saved field, mean |projection error|={mae_mm:.4f} mm, "
                f"motion held",
                flush=True,
            )
            hz_dense = recon
            ss_prev = ss
            last = dict(
                coef=coef, mov=mov, mats=mats, bend_mats=bend_mats, lr_t=lr_t, rl_t=rl_t,
                mask=mask, ss=ss, lam=lam, sampler=sampler, sb_t=sb_t, t1_t=t1_t,
                anat_mask=anat_mask,
            )
            continue
        if solver == "lm":
            level_no = int(shown_level[lev])
            weights = level_ramps.get(level_no)
            skip_base = weights is not None and level_no in replace_levels
            mi_args = {}
            if anat in ("mi", "mij") and sb_t is not None:
                mi_args = dict(sb_t=sb_t, t1_t=t1_t, anat_mask=anat_mask, anat_w=anat_w)
            if skip_base:
                print(
                    f"  level {level_no} starts at weight {float(weights[0]):g} "
                    f"and climbs {[float(w) for w in weights]}",
                    flush=True,
                )
            else:
                lm_steps(coef, mov, mats, lr_t, rl_t, mask, ss, bool(est), lam, nstep,
                         sampler=sampler, bend=bend, bend_mats=bend_mats, **mi_args)
            hz_dense = eval_spline(coef, mats).numpy()
            ss_prev = ss
            last = dict(
                coef=coef, mov=mov, mats=mats, bend_mats=bend_mats, lr_t=lr_t, rl_t=rl_t,
                mask=mask, ss=ss, lam=lam, sampler=sampler, sb_t=sb_t, t1_t=t1_t,
                anat_mask=anat_mask,
            )
            if weights is not None and not ramp_only and anat in ("mi", "mij"):
                if est:
                    print(
                        f"  level {level_no} locks motion at "
                        f"{np.array2string(mov.numpy(), precision=4)} and ramps "
                        f"{[float(w) for w in weights]}",
                        flush=True,
                    )
                hz_dense = run_mi_ramp(last, level_no, fwhm, weights)
                ss_prev = ss
            elif (mi_ramp and not ramp_only and anat in ("mi", "mij")
                    and int(mi_ramp_level) == level_no
                    and int(mi_ramp_level) < len(WARPRES)):
                if est:
                    print(
                        f"  level {level_no} locks motion at "
                        f"{np.array2string(mov.numpy(), precision=4)} and ramps",
                        flush=True,
                    )
                hz_dense = run_mi_ramp(last, level_no, fwhm)
                ss_prev = ss
            continue
        for i in range(nstep):
            with tf.GradientTape() as tape:
                hz = eval_spline(coef, mats)
                sl, sr = warp_pair(lr_t, rl_t, hz, mov, ss, bool(est), sampler)
                ssd = ssd_of(sl, sr, mask)
                if bend == "topup":
                    reg = ssd * float(lam) * tf.reduce_sum(tf.square(bend_terms_coef(coef, bend_mats)))
                else:
                    reg = ssd * float(lam) * bend_energy(hz)
                loss = ssd + reg
                # Detach the gate so MI cannot improve by shrinking the field out of the gate.
                if not use_anat:
                    gate = None
                elif use_gate:
                    gate = tf.stop_gradient(inhom_gate(hz)) * anat_mask
                else:
                    gate = anat_mask
                if use_anat and float(tf.reduce_sum(gate)) > 200.0:
                    disp = hz * ACQP_T / float(ss)
                    gx, gy, gz = coord_grid(hz.shape)
                    if anat == "mind":
                        epi = sampler(sb_t, gx + disp, gy, gz)
                        aval = mm.mind_loss(t1_t, epi, mask=gate)[0]
                    elif anat == "ngf":
                        epi = sampler(sb_t, gx + disp, gy, gz)
                        aval = mm.ngf_loss(t1_t, epi, mask=gate)[0]
                    elif anat in ("mi", "mij"):
                        epi = sampler(sb_t, gx + disp, gy, gz)
                        if anat == "mij":
                            epi = epi * jac_from_hz(hz, 1.0, ss)
                            sel = anat_mask > 0.5
                            flat = tf.boolean_mask(epi, sel)
                            lo = tf.stop_gradient(tf.reduce_min(flat))
                            hi = tf.stop_gradient(tf.reduce_max(flat))
                            epi = tf.clip_by_value((epi - lo) / (hi - lo + 1e-6), 0.0, 1.0)
                        aval = -mm.mutual_information_loss(t1_t[None], epi[None], gate)[0]
                    else:
                        epi = sampler(sb_t, gx + disp, gy, gz) * jac_from_hz(hz, 1.0, ss)
                        aval = mm.getGradientLoss(t1_t[None], epi[None], weight=gate[None])[0]
                    # Value equals anat_w * SSD. Gradient raises MI, or lowers the other losses.
                    # stop_gradient keeps a stale early baseline from dominating once the gate grows.
                    ref = tf.stop_gradient(aval)
                    if anat in ("mi", "mij"):
                        loss = loss + float(anat_w) * ssd * ref / (aval + 1e-6)
                    else:
                        loss = loss + float(anat_w) * ssd * aval / (ref + 1e-6)
                else:
                    aval = tf.constant(0.0)
                    gate = gate if use_anat else tf.zeros_like(hz)
            grads = tape.gradient(loss, [coef, mov])
            opt_c.apply_gradients([(grads[0], coef)])
            if est and grads[1] is not None:
                apply_mov_grad(grads[1])
            if i == 0 or i == nstep - 1:
                gfrac = float(tf.reduce_sum(tf.cast(gate > 0.5, tf.float32) * anat_mask) / (tf.reduce_sum(anat_mask) + 1e-6)) if use_anat else 0.0
                print(
                    f"    step {i:3d}  ssd={float(ssd):.4f}  reg={float(reg):.4f}  "
                    f"{anat}={float(aval):.4f}  gate>0.5={gfrac:.3f}  "
                    f"|hz|={float(tf.reduce_mean(tf.abs(hz))):.2f}  "
                    f"mov={np.array2string(mov.numpy(), precision=3)}",
                    flush=True,
                )
        hz_dense = eval_spline(coef, mats).numpy()
        ss_prev = ss
        last = dict(
            coef=coef, mov=mov, mats=mats, bend_mats=bend_mats, lr_t=lr_t, rl_t=rl_t,
            mask=mask, ss=ss, lam=lam, sampler=sampler, sb_t=sb_t, t1_t=t1_t,
            anat_mask=anat_mask,
        )
    if (mi_ramp and solver == "lm" and anat in ("mi", "mij")
            and 9 not in level_ramps
            and (ramp_only or int(mi_ramp_level) >= len(WARPRES))):
        # Default ramp: after the last stage, motion already frozen.
        hz_dense = run_mi_ramp(last, 9, 0.0)
    print(f"  frozen mov={np.array2string(mov.numpy(), precision=4)}", flush=True)
    return hz_dense


def run_sid(sid, which, anat_w=None, solver="adam", use_gate=True):
    work = HCP / sid / "unwarp_compare"
    unproc = HCP / sid / "unprocessed/3T/tfMRI_WM_LR"
    _, se_lr = C.load(work / "se_lr.nii.gz")
    _, se_rl = C.load(work / "se_rl.nii.gz")
    sb_img, sb = C.load(unproc / f"{sid}_3T_tfMRI_WM_LR_SBRef.nii.gz")
    _, t1_raw = C.load(work / "t1_to_epi.nii.gz")
    _, mask = C.load(work / "epi_bet_mask.nii.gz")
    mask = mask > 0.5
    if se_lr.ndim > 3:
        se_lr = se_lr[..., 0]
    if se_rl.ndim > 3:
        se_rl = se_rl[..., 0]
    se_lr = (se_lr * (100.0 / (se_lr.mean() + 1e-6))).astype(np.float32)
    se_rl = (se_rl * (100.0 / (se_rl.mean() + 1e-6))).astype(np.float32)
    t1_raw = t1_raw.astype(np.float32)
    if which == "mind":
        t1_anat = scale01(t1_raw, mask)
        sb_anat = scale01(sb.astype(np.float32), mask)
    elif which == "ngf":
        t1_anat = t1_raw
        sb_anat = sb.astype(np.float32)
    elif which in ("mi", "mij"):
        t1_anat = scale01(t1_raw, mask)
        sb_anat = scale01(sb.astype(np.float32), mask)
    else:
        t1_anat = C.invert01(t1_raw, mask)
        sb_anat = sb.astype(np.float32)
    fsl = fsl_signed(work, sb.astype(np.float32), mask)
    if anat_w is None:
        anat_w = 0.0 if which == "b0" else ANAT_W
    print(f"=== {sid} {which}  solver={solver}  anat_w={anat_w}  adam_lr={ADAM_LR} ===", flush=True)
    hz = optimize(
        se_lr, se_rl, sb_anat, t1_anat, mask,
        which if which != "b0" else "none", anat_w, solver=solver, use_gate=use_gate,
    )
    u = (hz * ACQP_T).astype(np.float32)
    print(
        f"  corr(u, FSL)={corr(u, fsl, mask):.3f}  "
        f"|u|={np.abs(u[mask]).mean():.3f}  |FSL|={np.abs(fsl[mask]).mean():.3f}  "
        f"|u-FSL|={np.abs(u - fsl)[mask].mean():.3f}",
        flush=True,
    )
    epi = C.warp_pe_np(sb.astype(np.float32), u, jac=True)
    epi_f = C.warp_pe_np(sb.astype(np.float32), fsl, jac=True)
    print(
        f"  NMI none={nmi(sb, t1_raw, mask):.4f}  "
        f"ours={nmi(epi, t1_raw, mask):.4f}  "
        f"FSL={nmi(epi_f, t1_raw, mask):.4f}",
        flush=True,
    )
    tag = which if which == "b0" or abs(anat_w - ANAT_W) < 1e-8 else f"{which}_w{anat_w:g}"
    if which != "b0":
        tag = f"{tag}_gate" if use_gate else f"{tag}_nogate"
    if solver != "adam":
        tag = f"{tag}_{solver}"
    out = work / f"u_topuplike_{tag}.nii.gz"
    C.save_nii(u, sb_img, out)
    print("wrote", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="100307")
    ap.add_argument("--which", choices=["b0", "sobel", "mind", "ngf", "mi", "mij"], default="b0")
    ap.add_argument("--anat-w", type=float, default=None,
                    help="Extra-term weight as a fraction of the spin-echo SSD.")
    ap.add_argument("--weights", default="",
                    help="Comma-separated anat weights. Runs --which once per weight.")
    ap.add_argument("--solver", choices=["adam", "lm"], default="adam")
    ap.add_argument("--no-gate", action="store_true",
                    help="Apply the anatomy term on the whole brain, no sigmoid.")
    args = ap.parse_args()
    weights = [float(x) for x in args.weights.split(",") if x.strip()]
    if not weights:
        weights = [args.anat_w]
    M = spline_matrix(90, 5)
    print(f"spline row-sum min/max {M.sum(1).min():.4f} {M.sum(1).max():.4f}", flush=True)
    for w in weights:
        run_sid(args.subject, args.which, w, solver=args.solver, use_gate=not args.no_gate)


if __name__ == "__main__":
    main()
