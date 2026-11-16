import tensorflow as tf
import numpy as np
from os.path import expanduser
import neurite as ne
from scipy.ndimage import zoom

_layer_i = [0]


def _uniq(base):
    _layer_i[0] += 1
    return '%s_%d' % (base, _layer_i[0])

# function c2r contatenate complex input as new axis two two real inputs
c2r=lambda x:tf.stack([tf.math.real(x),tf.math.imag(x)],axis=-1)
#r2c takes the last dimension of real input and converts to complex
r2c=lambda x:tf.complex(x[...,0],x[...,1])

def myPinv(mtx, input_lam):
    ata = tf.math.conj(tf.linalg.matrix_transpose(mtx)) @ mtx
    eye_shape = [1] * (len(ata.shape) - 2) + [ata.shape[-1], ata.shape[-1]]
    eye = tf.eye(ata.shape[-1], dtype=np.complex64)
    input_lam_complex = tf.complex(input_lam, tf.constant(0., dtype=np.float32))
    to_inv = ata + tf.broadcast_to(tf.reshape(eye, eye_shape), tf.shape(ata)) * input_lam_complex
    return tf.linalg.lstsq(to_inv, tf.math.conj(tf.linalg.matrix_transpose(mtx)), fast=False)

class PInvLayer2(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    """
    This function is called to create testing model. It apply CG on each image
    in the batch.
    """
    def call(self, rhs, encodePinv, z_indices, acceleration, shapey):
        #print(b.shape, r2c(z).shape, csm.shape)
        #rhs = r2c(b) + tf.linalg.matrix_transpose(tf.math.conj(csm @ r2c(0.1 * z)))
        rhs_shape = rhs.shape.as_list()
        new_shape = [-1, rhs_shape[1], rhs_shape[2], rhs_shape[3] * rhs_shape[4], 1]
        inv = encodePinv @ tf.reshape(rhs, new_shape)
        inv_reshape = tf.reshape(inv, [-1, rhs_shape[1], rhs_shape[2], acceleration, shapey])
        #print(inv_reshape.shape)
        inv_transpose = tf.transpose(inv_reshape, (0, 1, 3, 2, 4))
        #print(inv_transpose.shape)
        inv_reshape2 = tf.reshape(inv_transpose, [-1, rhs_shape[1] * acceleration, rhs_shape[2], shapey])
        #print(acceleration, shapey)
        #print(inv_reshape2.shape)
        return c2r(tf.gather(inv_reshape2, z_indices, batch_dims=1))

class TrilinearModel(tf.keras.layers.Layer):
    def __init__(self, fsl, s=2):
        super().__init__()
        self.s = s
        self.fsl = fsl

    def build(self, input_shape):
        self.D, self.H, self.W = input_shape[1:4]
        self.Hc, self.Wc, self.Dc = self.H//self.s, self.W//self.s, self.D//self.s  # control‐grid dimensions
        print("Control grid size:", self.Hc, self.Wc, self.Dc)
        print("fsl shape", self.fsl.shape)

        # Setup the control grid as a trainable variable (per batch)
        # Here we assume batch size = 1 for simplicity. You can extend to batch >1.
        zoom_factors = [1, self.Dc / self.D, self.Hc / self.H, self.Wc / self.W]
        fsl_downsamp = zoom(self.fsl, zoom=zoom_factors, order=1)
        print("downsamp shape", fsl_downsamp.shape)
        initializer = tf.keras.initializers.Constant(fsl_downsamp)
        self.ctrl_grid = self.add_weight(shape=(1, self.Dc, self.Hc, self.Wc, 1), dtype=np.float32, trainable=True, initializer=initializer)

    def call(self, z_img_prenoise):
        batch_size = tf.shape(z_img_prenoise)[0]

        # Reshape to treat depth slices as a batch of 2D images: [D, Hc, Wc, 1]
        ctrl_y_reshaped = tf.reshape(self.ctrl_grid, [self.Dc, self.Hc, self.Wc, 1])
        print(ctrl_y_reshaped.shape)
        upsampler = tf.keras.layers.Resizing(self.H, self.W, interpolation='bilinear')
        upsampled_slices = upsampler(ctrl_y_reshaped)
        print(upsampled_slices.shape)

        # Upsample along Z (depth)
        upsampled_y = tf.transpose(upsampled_slices, [2, 0, 1, 3])
        upsampler_z = tf.keras.layers.Resizing(self.D, self.H, interpolation='bilinear')
        upsampled_y_z = upsampler_z(tf.reshape(upsampled_y, [-1, self.Dc, self.H, 1]))
        upsampled_y_z = tf.expand_dims(tf.transpose(upsampled_y_z, [1, 2, 0, 3]), 0)

        dense_y_repeat = tf.repeat(upsampled_y_z, tf.shape(z_img_prenoise)[0], axis=0)
        return dense_y_repeat

def control_grid_size(D, H, W, s):
    s = max(int(s), 1)
    if s == 1:
        return D, H, W
    return max(2, D // s), max(2, H // s), max(2, W // s)


def dense_to_ctrl_grid(dense, Dc, Hc, Wc):
    """Downsample a dense residual (1,D,H,W[,1]) onto a control grid."""
    x = np.asarray(dense, dtype=np.float32)
    while x.ndim > 4 and x.shape[-1] == 1:
        x = np.squeeze(x, axis=-1)
    if x.ndim == 3:
        x = x[None]
    if x.ndim != 4:
        raise ValueError("dense residual must be (D,H,W) or (1,D,H,W[,1]), got %s" % (x.shape,))
    if tuple(x.shape[1:4]) == (Dc, Hc, Wc):
        grid = x
    else:
        D, H, W = x.shape[1:4]
        grid = zoom(x, zoom=[1.0, Dc / float(D), Hc / float(H), Wc / float(W)], order=1)
    return np.expand_dims(grid, -1)


def upsample_ctrl_grid(ctrl, D, H, W):
    """Bilinear upsample control grid (1,Dc,Hc,Wc,1) to dense (1,D,H,W,1)."""
    Dc = tf.shape(ctrl)[1]
    xy = tf.reshape(ctrl, [Dc, tf.shape(ctrl)[2], tf.shape(ctrl)[3], 1])
    xy = tf.image.resize(xy, [H, W], method='bilinear')
    z = tf.transpose(xy, [2, 0, 1, 3])
    z = tf.image.resize(z, [D, H], method='bilinear')
    dense = tf.transpose(z, [1, 2, 0, 3])
    return tf.expand_dims(dense, 0)


class ResidualGridModel(tf.keras.layers.Layer):
    """PE residual on a control grid of stride s, bilinear-upsampled to full FOV.

    s=1 is a dense voxel residual. s>1 is TOPUP-style coarse knots: a large PE
    move is a few control-point values and has low bending after upsample.
    """
    def __init__(self, s=1, init_dense=None, **kwargs):
        super().__init__(**kwargs)
        self.s = max(int(s), 1)
        self.init_dense = init_dense

    def build(self, input_shape):
        self.D = int(input_shape[1])
        self.H = int(input_shape[2])
        self.W = int(input_shape[3])
        self.Dc, self.Hc, self.Wc = control_grid_size(self.D, self.H, self.W, self.s)
        if self.init_dense is not None:
            grid = dense_to_ctrl_grid(self.init_dense, self.Dc, self.Hc, self.Wc)
            initializer = tf.keras.initializers.Constant(grid)
        else:
            initializer = tf.keras.initializers.Zeros()
        self.ctrl_grid = self.add_weight(
            shape=(1, self.Dc, self.Hc, self.Wc, 1),
            initializer=initializer,
            dtype=np.float32,
            trainable=True,
            name='residual',
        )
        print("Residual control grid s=%d:" % self.s, self.Dc, self.Hc, self.Wc,
              "from init" if self.init_dense is not None else "zeros")

    def call(self, z_img_prenoise):
        if self.s == 1:
            dense = self.ctrl_grid
        else:
            xy = tf.reshape(self.ctrl_grid, [self.Dc, self.Hc, self.Wc, 1])
            xy = tf.image.resize(xy, [self.H, self.W], method='bilinear')
            z = tf.transpose(xy, [2, 0, 1, 3])
            z = tf.image.resize(z, [self.D, self.H], method='bilinear')
            dense = tf.expand_dims(tf.transpose(z, [1, 2, 0, 3]), 0)
        return tf.repeat(dense, tf.shape(z_img_prenoise)[0], axis=0)


class SelfModel(ResidualGridModel):
    """Back-compat alias: dense residual (s=1)."""
    def __init__(self, **kwargs):
        super().__init__(s=1, init_dense=None, **kwargs)

def compute_3d_gradients(volume):
    """
    Compute 3D gradients of a volume using Sobel filters.
    Args:
        volume: Tensor of shape [batch, depth, height, width, channels] (e.g., [1, depth, height, width, 1])
    Returns:
        grad_x, grad_y, grad_z: Gradients in x, y, and z directions
    """
    volume = tf.cast(volume, tf.float32)
    
    # Define 3D Sobel filters (3x3x3 kernels)
    sobel_x = tf.constant([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                           [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=tf.float32)
    sobel_y = tf.constant([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
                           [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=tf.float32)
    sobel_z = tf.constant([[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                           [[1, 2, 1], [2, 4, 2], [1, 2, 1]]], dtype=tf.float32)
    
    # Reshape filters to [depth, height, width, in_channels, out_channels]
    sobel_x = tf.reshape(sobel_x, [3, 3, 3, 1, 1])
    sobel_y = tf.reshape(sobel_y, [3, 3, 3, 1, 1])
    sobel_z = tf.reshape(sobel_z, [3, 3, 3, 1, 1])
    
    # Apply 3D convolution to compute gradients
    grad_x = tf.nn.conv3d(tf.expand_dims(volume, -1), sobel_x, strides=[1, 1, 1, 1, 1], padding='SAME')
    grad_y = tf.nn.conv3d(tf.expand_dims(volume, -1), sobel_y, strides=[1, 1, 1, 1, 1], padding='SAME')
    grad_z = tf.nn.conv3d(tf.expand_dims(volume, -1), sobel_z, strides=[1, 1, 1, 1, 1], padding='SAME')
    
    return grad_x, grad_y, grad_z

# Function to compute normalized cross-correlation of gradients
def gradient_correlation_loss(fixed_grads, moving_grads):
    """
    Compute negative normalized cross-correlation between gradient fields.
    Args:
        fixed_grads: Tuple of (grad_x, grad_y) for fixed image
        moving_grads: Tuple of (grad_x, grad_y) for moving image
    Returns:
        loss: Negative correlation (to minimize)
    """
    fixed_grad_x, fixed_grad_y, fixed_grad_z = fixed_grads
    moving_grad_x, moving_grad_y, moving_grad_z = moving_grads
    
    # Flatten gradients for correlation computation
    fixed_grad_x_flat = tf.reshape(fixed_grad_x, [-1])
    fixed_grad_y_flat = tf.reshape(fixed_grad_y, [-1])
    fixed_grad_z_flat = tf.reshape(fixed_grad_z, [-1])
    moving_grad_x_flat = tf.reshape(moving_grad_x, [-1])
    moving_grad_y_flat = tf.reshape(moving_grad_y, [-1])
    moving_grad_z_flat = tf.reshape(moving_grad_z, [-1])
    
    # Compute normalized cross-correlation for x and y gradients
    def normalized_cc(x, y):
        x_mean = tf.reduce_mean(x)
        y_mean = tf.reduce_mean(y)
        x_centered = x - x_mean
        y_centered = y - y_mean
        cov = tf.reduce_mean(x_centered * y_centered)
        std_x = tf.sqrt(tf.reduce_mean(tf.square(x_centered)))
        std_y = tf.sqrt(tf.reduce_mean(tf.square(y_centered)))
        return (std_x * std_y + 1e-8) / (cov + 1e-8)  # Add epsilon to avoid division by zero

    cc_x = normalized_cc(fixed_grad_x_flat, moving_grad_x_flat)
    cc_y = normalized_cc(fixed_grad_y_flat, moving_grad_y_flat)
    cc_z = normalized_cc(fixed_grad_z_flat, moving_grad_z_flat)
    
    # Combine x and y correlations (average)
    total_cc = (cc_x + cc_y + cc_z) / 2.0
    
    # Return negative correlation as loss (since we minimize)
    return total_cc

def getInvCC3D(x_a, x_b):
    input_a = x_a - tf.math.reduce_mean(x_a, axis=(1, 2, 3), keepdims=True)
    input_b = x_b - tf.math.reduce_mean(x_b, axis=(1, 2, 3), keepdims=True)
    stda = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(input_a), axis=(1, 2, 3), keepdims=True))
    stdb = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(input_b), axis=(1, 2, 3), keepdims=True))
    res = stdb * stda / tf.nn.convolution(tf.expand_dims(input_a, -1), tf.expand_dims(tf.expand_dims(input_b[0], -1), -1)) #input_b is not on the max_input-level magnitude
    return tf.squeeze(tf.squeeze(tf.squeeze(res, -1), -1), -1)

def getL2DiffZ(x_a, x_b):
    scale_a = tf.math.reduce_max(x_a, axis=(1, 2, 3), keepdims=True)
    scale_b = tf.math.reduce_max(x_b, axis=(1, 2, 3), keepdims=True)
    diff_a = (x_a[:, 1:] - x_a[:, :-1]) / scale_a
    diff_b = (x_b[:, 1:] - x_b[:, :-1]) / scale_b
    return tf.math.reduce_sum(tf.math.squared_difference(diff_a, diff_b), axis=(1, 2, 3), keepdims=False)

def getL2DiffX(x_a, x_b):
    scale_a = tf.math.reduce_max(x_a, axis=(1, 2, 3), keepdims=True)
    scale_b = tf.math.reduce_max(x_b, axis=(1, 2, 3), keepdims=True)
    diff_a = (x_a[:, :, 1:] - x_a[:, :, :-1]) / scale_a
    diff_b = (x_b[:, :, 1:] - x_b[:, :, :-1]) / scale_b
    return tf.math.reduce_sum(tf.math.squared_difference(diff_a, diff_b), axis=(1, 2, 3), keepdims=False)

def getL2DiffY(x_a, x_b):
    scale_a = tf.math.reduce_max(x_a, axis=(1, 2, 3), keepdims=True)
    scale_b = tf.math.reduce_max(x_b, axis=(1, 2, 3), keepdims=True)
    diff_a = (x_a[:, :, :, 1:] - x_a[:, :, :, :-1]) / scale_a
    diff_b = (x_b[:, :, :, 1:] - x_b[:, :, :, :-1]) / scale_b
    return tf.math.reduce_sum(tf.math.squared_difference(diff_a, diff_b), axis=(1, 2, 3), keepdims=False)

def getTPS(displacement):
    diffzx = tf.math.reduce_sum(tf.math.square(displacement[:, 1:, 1:] - displacement[:, :-1, 1:] - displacement[:, 1:, :-1] + displacement[:, :-1, :-1]), axis=(1, 2, 3), keepdims=True)
    diffzy = tf.math.reduce_sum(tf.math.square(displacement[:, 1:, :, 1:] - displacement[:, :-1, :, 1:] - displacement[:, 1:, :, :-1] + displacement[:, :-1, :, :-1]), axis=(1, 2, 3), keepdims=True)
    diffxy = tf.math.reduce_sum(tf.math.square(displacement[:, :, 1:, 1:] - displacement[:, :, :-1, 1:] - displacement[:, :, 1:, :-1] + displacement[:, :, :-1, :-1]), axis=(1, 2, 3), keepdims=True)
    diffzz = tf.math.reduce_sum(tf.math.square(displacement[:, 2:] - 2 * displacement[:, 1:-1] + displacement[:, :-2]), axis=(1, 2, 3), keepdims=True)
    diffxx = tf.math.reduce_sum(tf.math.square(displacement[:, :, 2:] - 2 * displacement[:, :, 1:-1] + displacement[:, :, :-2]), axis=(1, 2, 3), keepdims=True)
    diffyy = tf.math.reduce_sum(tf.math.square(displacement[:, :, :, 2:] - 2 * displacement[:, :, :, 1:-1] + displacement[:, :, :, :-2]), axis=(1, 2, 3), keepdims=True)
    tps = diffzz + diffxx + diffyy + 2 * (diffzx + diffzy + diffxy)
    return tf.squeeze(tf.squeeze(tf.squeeze(tps, -1), -1), -1)

def getGradientLoss(fixed, moving, weight=None):
    """Sobel-gradient L2. If weight is set, return a weighted mean (O(1)); else a sum."""
    fixed_grad_x, fixed_grad_y, fixed_grad_z = compute_3d_gradients(fixed)
    moving_grad_x, moving_grad_y, moving_grad_z = compute_3d_gradients(moving)
    max_fixed  = tf.math.reduce_max(tf.expand_dims(fixed,  -1), axis=(1, 2, 3, 4), keepdims=True)
    max_moving = tf.math.reduce_max(tf.expand_dims(moving, -1), axis=(1, 2, 3, 4), keepdims=True)
    x_diff = tf.math.squared_difference(fixed_grad_x / max_fixed, moving_grad_x / max_moving)
    y_diff = tf.math.squared_difference(fixed_grad_y / max_fixed, moving_grad_y / max_moving)
    z_diff = tf.math.squared_difference(fixed_grad_z / max_fixed, moving_grad_z / max_moving)
    if weight is None:
        mask_fixed = tf.expand_dims(fixed, -1) > 1e-4
        x_sum = tf.math.reduce_sum(tf.where(mask_fixed, x_diff, 0.0), axis=(1, 2, 3, 4), keepdims=False)
        y_sum = tf.math.reduce_sum(tf.where(mask_fixed, y_diff, 0.0), axis=(1, 2, 3, 4), keepdims=False)
        z_sum = tf.math.reduce_sum(tf.where(mask_fixed, z_diff, 0.0), axis=(1, 2, 3, 4), keepdims=False)
        return x_sum + y_sum + z_sum
    w = tf.cast(weight, tf.float32)
    while len(w.shape) < 5:
        w = tf.expand_dims(w, -1)
    w = tf.broadcast_to(w, tf.shape(x_diff))
    denom = tf.reduce_sum(w, axis=(1, 2, 3, 4)) + 1e-6
    tot = (tf.reduce_sum(w * x_diff, axis=(1, 2, 3, 4))
           + tf.reduce_sum(w * y_diff, axis=(1, 2, 3, 4))
           + tf.reduce_sum(w * z_diff, axis=(1, 2, 3, 4)))
    return tot / denom


def mind_ssc(vol, radius=2, dilation=2):
    """MIND-SSC descriptor (Heinrich MICCAI 2013). vol [X,Y,Z] or [B,X,Y,Z] -> [B,X,Y,Z,12].

    Fully differentiable (shifts, avg-pool SSD, variance norm, exp). Fixed-image
    descriptors do not depend on the warp; Adam on a PE residual is valid.
    """
    v = tf.cast(vol, tf.float32)
    if len(v.shape) == 3:
        v = v[None]
    img = v[..., None]  # [B,X,Y,Z,1]
    six = np.array([[0, 1, 1], [1, 1, 0], [1, 0, 1],
                    [1, 1, 2], [2, 1, 1], [1, 2, 1]], dtype=np.int32)
    dist2 = ((six[:, None, :] - six[None, :, :]) ** 2).sum(-1)
    pairs = [(i, j) for i in range(6) for j in range(i)]
    pairs = [(i, j) for i, j in pairs if dist2[i, j] == 2]
    assert len(pairs) == 12, pairs
    k1 = np.zeros((3, 3, 3, 1, 12), dtype=np.float32)
    k2 = np.zeros((3, 3, 3, 1, 12), dtype=np.float32)
    for c, (i, j) in enumerate(pairs):
        k1[six[i, 0], six[i, 1], six[i, 2], 0, c] = 1.0
        k2[six[j, 0], six[j, 1], six[j, 2], 0, c] = 1.0
    img_p = tf.pad(img, [[0, 0], [dilation] * 2, [dilation] * 2, [dilation] * 2, [0, 0]],
                   mode="SYMMETRIC")
    dils = [1, dilation, dilation, dilation, 1]
    s1 = tf.nn.conv3d(img_p, k1, strides=[1, 1, 1, 1, 1], padding="VALID", dilations=dils)
    s2 = tf.nn.conv3d(img_p, k2, strides=[1, 1, 1, 1, 1], padding="VALID", dilations=dils)
    sq = tf.square(s1 - s2)
    ksz = radius * 2 + 1
    # SAME avg-pool via pad so descriptor grid matches the image
    pad = radius
    sq_p = tf.pad(sq, [[0, 0], [pad] * 2, [pad] * 2, [pad] * 2, [0, 0]], mode="SYMMETRIC")
    ssd = tf.nn.avg_pool3d(sq_p, ksize=[1, ksz, ksz, ksz, 1], strides=[1, 1, 1, 1, 1],
                           padding="VALID")
    mind = ssd - tf.reduce_min(ssd, axis=-1, keepdims=True)
    var = tf.reduce_mean(mind, axis=-1, keepdims=True)
    var_mean = tf.reduce_mean(var, axis=[1, 2, 3], keepdims=True)
    var = tf.clip_by_value(var, var_mean * 0.001, var_mean * 1000.0 + 1e-6)
    return tf.exp(-mind / (var + 1e-6))


def ngf_loss(fixed, moving, mask=None, eps=1e-2):
    """Haber–Modersitzki normalized gradient fields. Mean of 1-(n_f·n_m)^2.

    Sobel gradients are unit-normalized per voxel so edge *strength* does not
    dominate (unlike getGradientLoss). Squared inner product is sign-invariant,
    so T1 invert is not required. mask is a voxel weight.
    """
    f = tf.cast(fixed, tf.float32)
    mv = tf.cast(moving, tf.float32)
    if len(f.shape) == 3:
        f = f[None]
        mv = mv[None]
    gx_f, gy_f, gz_f = compute_3d_gradients(f)
    gx_m, gy_m, gz_m = compute_3d_gradients(mv)

    def unit(gx, gy, gz):
        s = tf.sqrt(gx * gx + gy * gy + gz * gz + eps * eps)
        return gx / s, gy / s, gz / s

    nfx, nfy, nfz = unit(gx_f, gy_f, gz_f)
    nmx, nmy, nmz = unit(gx_m, gy_m, gz_m)
    loc = 1.0 - tf.square(nfx * nmx + nfy * nmy + nfz * nmz)
    if mask is None:
        return tf.reduce_mean(loc, axis=[1, 2, 3, 4])
    w = tf.cast(mask, tf.float32)
    while len(w.shape) < 4:
        w = w[None]
    while len(w.shape) < 5:
        w = w[..., None]
    w = tf.broadcast_to(w, tf.shape(loc))
    return tf.reduce_sum(loc * w, axis=[1, 2, 3, 4]) / (tf.reduce_sum(w, axis=[1, 2, 3, 4]) + 1e-6)


def mind_loss(fixed, moving, mask=None, eps=1e-6):
    """Mean SSD of MIND-SSC descriptors. fixed/moving [X,Y,Z] or [B,X,Y,Z]."""
    f = mind_ssc(fixed)
    m = mind_ssc(moving)
    diff = tf.square(f - m)
    if mask is None:
        return tf.reduce_mean(diff, axis=[1, 2, 3, 4])
    w = tf.cast(mask, tf.float32)
    while len(w.shape) < 4:
        w = w[None]
    while len(w.shape) < 5:
        w = w[..., None]
    w = tf.broadcast_to(w, tf.shape(diff))
    return tf.reduce_sum(diff * w, axis=[1, 2, 3, 4]) / (tf.reduce_sum(w, axis=[1, 2, 3, 4]) + eps)


def distortion_blend_weight(fsl_u, brain_mask, lo_pct=80.0, hi_pct=99.0):
    """Stop-grad weight in [0,1] from |FSL PE field|: 0 below lo_pct, 1 at hi_pct.

    Uses the TOPUP field only (not residual) so Sobel cannot grow its own support.
    """
    sev = tf.abs(tf.cast(fsl_u, tf.float32))
    while len(sev.shape) > 4:
        sev = tf.squeeze(sev, axis=-1)
    m = tf.ones_like(sev) if brain_mask is None else tf.cast(brain_mask, tf.float32)
    while len(m.shape) > 4:
        m = tf.squeeze(m, axis=-1)
    m = tf.broadcast_to(m, tf.shape(sev))

    def _pcts(v, mask, lo, hi):
        v = np.asarray(v)
        mask = np.asarray(mask)
        sel = v[mask.reshape(v.shape) > 0.5]
        if sel.size == 0:
            return np.array([0.0, 1.0], dtype=np.float32)
        return np.percentile(sel, [float(lo), float(hi)]).astype(np.float32)

    bounds = tf.numpy_function(
        _pcts, [sev, m, tf.constant(lo_pct, tf.float32), tf.constant(hi_pct, tf.float32)],
        tf.float32)
    bounds.set_shape([2])
    bounds = tf.stop_gradient(bounds)
    lo, hi = bounds[0], bounds[1]
    w = tf.clip_by_value((sev - lo) / (hi - lo + 1e-6), 0.0, 1.0) * m
    return tf.stop_gradient(w)


def distortion_sigmoid_weight(fsl_u, brain_mask, center=2.0, scale=0.75):
    """Stop-grad logistic weight of |FSL PE displacement| in voxels.

    w = sigmoid((|u| - center) / scale) * mask

    Raw sigmoid(|u|) is 0.5 at zero shift, so it still spends half its mass
    on undistorted brain. This shifted form is ~0.06 at 0 vx, ~0.3 at 1.4 vx
    (pooled p80), 0.5 at 2 vx, ~0.97 at 4.7 vx (pooled p99).
    Uses TOPUP only (not residual), same as the percentile gate.
    """
    sev = tf.abs(tf.cast(fsl_u, tf.float32))
    while len(sev.shape) > 4:
        sev = tf.squeeze(sev, axis=-1)
    m = tf.ones_like(sev) if brain_mask is None else tf.cast(brain_mask, tf.float32)
    while len(m.shape) > 4:
        m = tf.squeeze(m, axis=-1)
    m = tf.broadcast_to(m, tf.shape(sev))
    c = tf.cast(center, tf.float32)
    s = tf.cast(scale, tf.float32)
    w = tf.math.sigmoid((sev - c) / (s + 1e-6)) * m
    return tf.stop_gradient(w)


def distortion_vx_weight(fsl_u, brain_mask, lo=1.0, hi=2.0):
    """Hard zero below lo EPI voxels, linear ramp to 1 at hi. Physical, not percentile."""
    sev = tf.abs(tf.cast(fsl_u, tf.float32))
    while len(sev.shape) > 4:
        sev = tf.squeeze(sev, axis=-1)
    m = tf.ones_like(sev) if brain_mask is None else tf.cast(brain_mask, tf.float32)
    while len(m.shape) > 4:
        m = tf.squeeze(m, axis=-1)
    m = tf.broadcast_to(m, tf.shape(sev))
    w = tf.clip_by_value((sev - float(lo)) / (float(hi) - float(lo) + 1e-6), 0.0, 1.0) * m
    return tf.stop_gradient(w)

def local_ncc_loss(fixed, moving, win, mask=None, eps=1e-5):
    """VoxelMorph-style local NCC. Returns 1 - mean(cc) over mask, shape [B]."""
    I = tf.expand_dims(tf.cast(fixed, tf.float32), -1)
    J = tf.expand_dims(tf.cast(moving, tf.float32), -1)
    win = int(win)
    filt = tf.ones([win, win, win, 1, 1], dtype=tf.float32)

    def conv(x):
        return tf.nn.conv3d(x, filt, strides=[1, 1, 1, 1, 1], padding='SAME')

    I_sum = conv(I)
    J_sum = conv(J)
    I2_sum = conv(I * I)
    J2_sum = conv(J * J)
    IJ_sum = conv(I * J)
    win_size = tf.cast(win * win * win, tf.float32)
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_I * J_sum - u_J * I_sum + u_I * u_J * win_size
    I_var = I2_sum - 2.0 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2.0 * u_J * J_sum + u_J * u_J * win_size
    cc = (cross * cross) / (I_var * J_var + eps)
    cc = tf.squeeze(cc, -1)
    if mask is None:
        w = tf.cast(fixed > 1e-4, tf.float32)
    else:
        w = tf.cast(mask, tf.float32)
        while len(w.shape) > 4:
            w = tf.squeeze(w, axis=-1)
    denom = tf.reduce_sum(w, axis=[1, 2, 3]) + eps
    cc_mean = tf.reduce_sum(cc * w, axis=[1, 2, 3]) / denom
    return 1.0 - cc_mean


def gaussian_smooth3d(vol, sigma):
    """Separable 3D Gaussian. sigma<=0 is identity."""
    sigma = float(sigma) if sigma is not None else 0.0
    vol = tf.cast(vol, tf.float32)
    if sigma <= 0:
        return vol
    radius = max(int(3.0 * sigma + 0.5), 1)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k = (k / k.sum()).astype(np.float32)
    vol5 = tf.expand_dims(vol, -1)
    kd = tf.constant(k.reshape(-1, 1, 1, 1, 1))
    kh = tf.constant(k.reshape(1, -1, 1, 1, 1))
    kw = tf.constant(k.reshape(1, 1, -1, 1, 1))
    vol5 = tf.nn.conv3d(vol5, kd, strides=[1, 1, 1, 1, 1], padding='SAME')
    vol5 = tf.nn.conv3d(vol5, kh, strides=[1, 1, 1, 1, 1], padding='SAME')
    vol5 = tf.nn.conv3d(vol5, kw, strides=[1, 1, 1, 1, 1], padding='SAME')
    return tf.squeeze(vol5, -1)


def masked_winsorize01_tf(vol, mask, lo=0.005, hi=0.995):
    """Clip to mask percentiles; percentile edges have stop_gradient."""
    vol = tf.cast(vol, tf.float32)
    w = tf.cast(mask, tf.float32)
    while len(w.shape) > 4:
        w = tf.squeeze(w, axis=-1)
    w = tf.broadcast_to(w, tf.shape(vol))

    def _pct(v, m):
        v = np.asarray(v)
        m = np.asarray(m)
        sel = v[m > 0.5]
        if sel.size == 0:
            return np.array([0.0, 1.0], dtype=np.float32)
        return np.percentile(sel, [100.0 * lo, 100.0 * hi]).astype(np.float32)

    bounds = tf.numpy_function(_pct, [vol, w], tf.float32)
    bounds.set_shape([2])
    bounds = tf.stop_gradient(bounds)
    p_lo, p_hi = bounds[0], bounds[1]
    return tf.clip_by_value((vol - p_lo) / (p_hi - p_lo + 1e-6), 0.0, 1.0)


def mutual_information_loss(fixed, moving, mask, n_bins=32, sigma=None, eps=1e-8):
    """Soft-histogram MI on masked voxels. Returns -MI, shape [B]."""
    n_bins = int(n_bins)
    if sigma is None:
        sigma = 0.5 / n_bins
    f = tf.reshape(tf.cast(fixed, tf.float32), [tf.shape(fixed)[0], -1])
    m = tf.reshape(tf.cast(moving, tf.float32), [tf.shape(moving)[0], -1])
    w = tf.reshape(tf.cast(mask, tf.float32), [-1])
    idx = tf.where(w > 0.05)[:, 0]
    f = tf.gather(f, idx, axis=1)
    m = tf.gather(m, idx, axis=1)
    ww = tf.gather(w, idx)
    ww = ww / (tf.reduce_sum(ww) + eps)
    ww = tf.reshape(ww, [1, -1, 1])
    centers = tf.linspace(0.5 / n_bins, 1.0 - 0.5 / n_bins, n_bins)

    def soft_assign(vals):
        d = (vals[:, :, None] - centers[None, None, :]) / sigma
        return tf.nn.softmax(-0.5 * tf.square(d), axis=-1)

    pf = soft_assign(f)
    pm = soft_assign(m)
    p_f = tf.reduce_sum(ww * pf, axis=1)
    p_m = tf.reduce_sum(ww * pm, axis=1)
    p_joint = tf.matmul(tf.transpose(pf * ww, [0, 2, 1]), pm)
    p_f = p_f / (tf.reduce_sum(p_f, axis=-1, keepdims=True) + eps)
    p_m = p_m / (tf.reduce_sum(p_m, axis=-1, keepdims=True) + eps)
    p_joint = p_joint / (tf.reduce_sum(p_joint, axis=[1, 2], keepdims=True) + eps)
    h_f = -tf.reduce_sum(p_f * tf.math.log(p_f + eps), axis=-1)
    h_m = -tf.reduce_sum(p_m * tf.math.log(p_m + eps), axis=-1)
    h_j = -tf.reduce_sum(p_joint * tf.math.log(p_joint + eps), axis=[1, 2])
    return -(h_f + h_m - h_j)

def sample_trilinear(vol, coords):
    """Trilinear sample. vol [B,Z,X,Y], coords [N,3] (z,x,y) voxels -> [B,N]."""
    vol = tf.cast(vol, tf.float32)
    coords = tf.cast(coords, tf.float32)
    zsz = tf.cast(tf.shape(vol)[1], tf.float32)
    xsz = tf.cast(tf.shape(vol)[2], tf.float32)
    ysz = tf.cast(tf.shape(vol)[3], tf.float32)
    z = tf.clip_by_value(coords[:, 0], 0.0, zsz - 1.001)
    x = tf.clip_by_value(coords[:, 1], 0.0, xsz - 1.001)
    y = tf.clip_by_value(coords[:, 2], 0.0, ysz - 1.001)
    z0 = tf.floor(z)
    x0 = tf.floor(x)
    y0 = tf.floor(y)
    zd = z - z0
    xd = x - x0
    yd = y - y0
    v = tf.transpose(vol, [1, 2, 3, 0])

    def corner(zi, xi, yi):
        idx = tf.stack([tf.cast(zi, tf.int32), tf.cast(xi, tf.int32), tf.cast(yi, tf.int32)], axis=1)
        return tf.transpose(tf.gather_nd(v, idx), [1, 0])

    c000 = corner(z0, x0, y0)
    c001 = corner(z0, x0, y0 + 1)
    c010 = corner(z0, x0 + 1, y0)
    c011 = corner(z0, x0 + 1, y0 + 1)
    c100 = corner(z0 + 1, x0, y0)
    c101 = corner(z0 + 1, x0, y0 + 1)
    c110 = corner(z0 + 1, x0 + 1, y0)
    c111 = corner(z0 + 1, x0 + 1, y0 + 1)
    zd = zd[None, :]
    xd = xd[None, :]
    yd = yd[None, :]
    c00 = c000 * (1.0 - zd) + c100 * zd
    c01 = c001 * (1.0 - zd) + c101 * zd
    c10 = c010 * (1.0 - zd) + c110 * zd
    c11 = c011 * (1.0 - zd) + c111 * zd
    c0 = c00 * (1.0 - xd) + c10 * xd
    c1 = c01 * (1.0 - xd) + c11 * xd
    return c0 * (1.0 - yd) + c1 * yd


def boundary_based_loss(vol, coords_wm, coords_gm, temperature=0.25):
    """Greve/FSL BBR on T2*-like EPI: WM darker than GM.

    Minimizing sigmoid((I_wm - I_gm)/t) wants I_gm > I_wm at the T1 WM sheet.
    Sample locations are fixed in T1/EPI grid; gradients flow through vol.
    Returns [B].
    """
    n = tf.shape(coords_wm)[0]
    def _empty():
        return tf.zeros([tf.shape(vol)[0]], dtype=tf.float32)
    def _ok():
        i_wm = sample_trilinear(vol, coords_wm)
        i_gm = sample_trilinear(vol, coords_gm)
        return tf.reduce_mean(tf.nn.sigmoid((i_wm - i_gm) / temperature), axis=1)
    return tf.cond(n > 0, _ok, _empty)


def makePhysicsModels(b, pre_x, smri, fsl, b0up, b0dn, senseLam, tpsLam, l2dULam, l2ULam, b0Lam, locLam, locWin, noCorrection, csm, encodePinv, encode, z_indices, acceleration, shapex, shapey, readout_ratio, residual_s=1, residual_init=None, brain_mask=None, anat_metric='mi', mi_bins=32, smooth_sigma=0.0, bbr_lam=0.0, bbr_wm=None, bbr_gm=None, sobel_lam=0.0, sobel_lo_pct=80.0, sobel_hi_pct=99.0):
    with tf.name_scope('myModel'):
        cg = PInvLayer2()
        encode_exp = tf.expand_dims(tf.expand_dims(r2c(encode), 0), 0)
        inv = cg(r2c(b), r2c(encodePinv), z_indices, acceleration, shapey)
        inv = tf.math.abs(r2c(inv))
        if noCorrection:
            return inv, pre_x, inv, pre_x, None
        residualModel = ResidualGridModel(s=residual_s, init_dense=residual_init, name=_uniq('self_model'))
        r = tf.squeeze(residualModel(pre_x), -1)
        fsl_u = tf.cast(fsl, tf.float32)
        while len(fsl_u.shape) > 4:
            fsl_u = tf.squeeze(fsl_u, axis=-1)
        u = fsl_u + r

        dy = tf.constant(1.0, dtype=u.dtype)
        assert shapey > 1
        inner_du_dy = (u[:, :, :, 2:] - u[:, :, :, :-2]) / (2 * dy)
        du_dy = tf.concat([((u[:, :, :, 1] - u[:, :, :, 0]) / dy)[:, :, :, None], inner_du_dy, ((u[:, :, :, -1] - u[:, :, :, -2]) / dy)[:, :, :, None]], axis=3)
        # B0 pair is in the same voxel units as u; fMRI warp is scaled by readout_ratio.
        J1 = tf.maximum(1.0 + du_dy, 0.0)
        J2 = tf.maximum(1.0 - du_dy, 0.0)
        J_fmri = tf.maximum(1.0 + readout_ratio * du_dy, 0.0)

        tps = getTPS(r)
        y_source = SpatialTransformer(interp_method='linear', fill_value=None, name=_uniq('transformer'))
        uexp = tf.expand_dims(u, -1) 
        u3d = tf.concat([tf.zeros_like(uexp), tf.zeros_like(uexp), uexp], -1)
        x_warp = tf.squeeze(y_source([tf.expand_dims(pre_x, -1), readout_ratio * u3d]), -1)
        x = x_warp * J_fmri

        b0up_repeat = tf.repeat(b0up, tf.shape(pre_x)[0], axis=0)
        b0dn_repeat = tf.repeat(b0dn, tf.shape(pre_x)[0], axis=0)
        b0up_adj = tf.squeeze(y_source([tf.expand_dims(b0up_repeat, -1), u3d]), -1) * J1
        b0dn_adj = tf.squeeze(y_source([tf.expand_dims(b0dn_repeat, -1), -u3d]), -1) * J2
        b0_err = tf.math.square(b0up_adj - b0dn_adj)
        if brain_mask is not None:
            m = tf.cast(brain_mask, b0_err.dtype)
            l2b0 = tf.math.reduce_sum(b0_err * m, axis=(1, 2, 3)) / (tf.math.reduce_sum(m) + 1e-6)
        else:
            l2b0 = tf.math.reduce_sum(b0_err, axis=(1, 2, 3)) / (shapex * shapey)

        l2U = tf.math.reduce_sum(tf.math.square(r), axis=(1, 2, 3), keepdims=True)
        l2U = tf.squeeze(tf.squeeze(tf.squeeze(l2U, -1), -1), -1)

        du = r[:, :, :, 1:] - r[:, :, :, :-1]  # residual smoothness along PE
        l2dU = tf.math.reduce_sum(tf.math.square(du), axis=(1, 2, 3), keepdims=False)

        if anat_metric == 'mi':
            smri_s = masked_winsorize01_tf(gaussian_smooth3d(smri, smooth_sigma), brain_mask)
            mov_s = masked_winsorize01_tf(gaussian_smooth3d(x_warp, smooth_sigma), brain_mask)
            loc = mutual_information_loss(smri_s, mov_s, brain_mask, n_bins=mi_bins)
        else:
            # Geometry only: Jacobian intensity modulation is not T1 contrast.
            loc = local_ncc_loss(smri, x_warp, locWin, mask=brain_mask)
        bbr = tf.zeros([tf.shape(pre_x)[0]], dtype=tf.float32)
        if bbr_lam and bbr_wm is not None and bbr_gm is not None:
            x_bbr = masked_winsorize01_tf(x_warp, brain_mask)
            bbr = boundary_based_loss(x_bbr, bbr_wm, bbr_gm)
        sobel = tf.zeros([tf.shape(pre_x)[0]], dtype=tf.float32)
        if sobel_lam:
            w_sev = distortion_blend_weight(
                fsl_u, brain_mask, lo_pct=sobel_lo_pct, hi_pct=sobel_hi_pct)
            sobel = getGradientLoss(smri, x_warp, weight=w_sev)
        loss = (b0Lam * l2b0 + locLam * loc + float(bbr_lam) * bbr
                + float(sobel_lam) * sobel
                + tpsLam * tps + l2dULam * l2dU + l2ULam * l2U)
        return inv, x, loss, tps, residualModel

def cc3d(x, smri):
    return tf.math.reciprocal(getInvCC3D(x, smri))

def transform(vol, loc_shift, interp_method='linear', fill_value=None,
              shift_center=True):
    """Apply affine or dense transforms to images in N dimensions.
    Essentially interpolates the input ND tensor at locations determined by
    loc_shift. The latter can be an affine transform or dense field of location
    shifts in the sense that at location x we now have the data from x + dx, so
    we moved the data.
    Args:
        vol: tensor or array-like structure  of size vol_shape or
            (*vol_shape, C), where C is the number of channels.
        loc_shift: A shift
            volume of shape (*new_vol_shape, D) or (*new_vol_shape, C, D),
            where C is the number of channels, and D is the dimentionality
            D = len(vol_shape). If the shape is (*new_vol_shape, D), the same
            transform applies to all channels of of the input tensor.
        interp_method: 'linear' or 'nearest'.
        fill_value: Value to use for points sampled outside the domain. If
            None, the nearest neighbors will be used.
        shift_center: Shift grid to image center when converting affine
            transforms to dense transforms.
    Returns:
        Tensor whose voxel values are the values of the input tensor
        interpolated at the locations defined by the transform.
    Keywords:
        interpolation, sampler, resampler, linear, bilinear
    """
    # convert data type if needed
    ftype = tf.float32
    if not tf.is_tensor(vol) or not vol.dtype.is_floating:
        vol = tf.cast(vol, ftype)
    if not tf.is_tensor(loc_shift) or not loc_shift.dtype.is_floating:
        loc_shift = tf.cast(loc_shift, ftype)

    # parse spatial location shape, including channels if available
    loc_volshape = loc_shift.shape[:-1]

    # volume dimensions
    nb_dims = len(vol.shape) - 1
    is_channelwise = len(loc_volshape) == (nb_dims + 1)
    assert loc_shift.shape[-1] == nb_dims, \
        'Dimension check failed for ne.utils.transform(): {}D volume (shape {}) called ' \
        'with {}D transform'.format(nb_dims, vol.shape[:-1], loc_shift.shape[-1])

    # location should be mesh and delta
    mesh = ne.utils.volshape_to_meshgrid(loc_volshape, indexing='ij')  # volume mesh
    for d, m in enumerate(mesh):
        if m.dtype != loc_shift.dtype:
            mesh[d] = tf.cast(m, loc_shift.dtype)
    loc = [mesh[d] + loc_shift[..., d] for d in range(nb_dims)]

    # if channelwise location, then append the channel as part of the location lookup
    if is_channelwise:
        loc.append(mesh[-1])

    # test single
    return ne.utils.interpn(vol, loc, interp_method=interp_method, fill_value=fill_value)

from keras.layers import Layer
class SpatialTransformer(Layer):
    """
    ND spatial transformer layer
    Applies affine and dense transforms to images. A dense transform gives
    displacements (not absolute locations) at each voxel.
    If you find this layer useful, please cite:
      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.
    Originally, this code was based on voxelmorph code, which
    was in turn transformed to be dense with the help of (affine) STN code
    via https://github.com/kevinzakka/spatial-transformer-network.
    Since then, we've re-written the code to be generalized to any
    dimensions, and along the way wrote grid and interpolation functions.
    """

    def __init__(self,
                 interp_method='linear',
                 fill_value=None,
                 shape=None,
                 **kwargs):
        """
        Parameters:
            interp_method: Interpolation method. Must be 'linear' or 'nearest'.
            fill_value: Value to use for points sampled outside the domain.
                If None, the nearest neighbors will be used.
            shape: ND output shape used when converting affine transforms to dense
                transforms. Includes only the N spatial dimensions. If None, the
                shape of the input image will be used.
        """
        self.interp_method = interp_method
        self.fill_value = fill_value
        self.shape = shape
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'interp_method': self.interp_method,
            'fill_value': self.fill_value,
            'shape': self.shape,
        })
        return config

    def build(self, input_shape):

        # sanity check on input list
        if len(input_shape) > 2:
            raise ValueError('Spatial Transformer must be called on a list of length 2: '
                             'first argument is the image, second is the transform.')

        # set up number of dimensions
        self.ndims = len(input_shape[0]) - 2
        self.imshape = input_shape[0][1:]
        self.trfshape = input_shape[1][1:]

        # make sure inputs are reasonable shapes
        image_shape = tuple(self.imshape[:-1])
        dense_shape = tuple(self.trfshape[:-1])
        if image_shape != dense_shape:
            raise ValueError(f'Dense transform shape {dense_shape} does not match image shape {image_shape}.')

        # confirm built
        self.built = True

    def call(self, inputs):
        """
        Parameters
            inputs: List of [img, trf], where img is the ND moving image and trf
            is either a dense warp of shape [B, D1, ..., DN, N] or an affine matrix
            of shape [B, N, N+1].
        """

        # necessary for multi-gpu models
        vol = tf.keras.backend.reshape(inputs[0], (-1, *self.imshape))
        trf = tf.keras.backend.reshape(inputs[1], (-1, *self.trfshape))

        # map transform across batch
        return tf.map_fn(self._single_transform, [vol, trf], fn_output_signature=vol.dtype)

    def _single_transform(self, inputs):
        return transform(inputs[0], inputs[1], interp_method=self.interp_method,
                               fill_value=self.fill_value)


