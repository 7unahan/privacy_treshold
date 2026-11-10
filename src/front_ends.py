"""
front_ends.py - the two evaluation front ends, P_dir and P_ali.

Companion code for:
  Wrong Input, Wrong Threshold: Re-evaluating Adversarial Facial Privacy.

P_dir  resizes the WHOLE photograph to the recognizer's input resolution.
       This is what the surveyed literature evaluates on.
P_ali  takes five landmarks, solves a similarity transform (Umeyama) onto
       the ArcFace template, warps and crops. This is what a deployed
       recognition pipeline does.

Both take a float tensor in [0, 1], shape (B, 3, H, W), RGB, and return a
tensor of shape (B, 3, S, S) ready for `normalise` and the recognizer.

Provenance
----------
The geometry below (ARC_SRC, umeyama_sim, M112, the theta construction in
`warp_theta`, the crop mask) is taken from the measurement kernel that
produced the numbers in the paper. The arithmetic is unchanged. Two things
were ported, not copied: `crop_mask` was torch and is numpy here, and the
landmark cache was dropped. The two torch calls at the end of `P_ali`
(affine_grid, grid_sample) and the one in `P_dir` (interpolate) are verbatim.

`python src/front_ends.py` runs a self-test that needs no dataset and no
model weights. The numpy geometry is always tested. The torch sampling is
tested against an independent numpy bilinear sampler when torch is installed.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ARC_SRC", "umeyama_sim", "M112", "warp_theta", "crop_mask",
    "P_dir", "P_ali", "normalise", "detect_landmarks",
]

# ArcFace five-point template for a 112x112 crop (left eye, right eye, nose,
# left mouth corner, right mouth corner). Published with ArcFace/InsightFace.
ARC_SRC = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                    [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)


def umeyama_sim(src, dst):
    """Least-squares similarity transform taking `src` points onto `dst`.

    Returns a 2x3 matrix [s*R | t] (Umeyama 1991), reflection excluded.
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    ms, md = src.mean(0), dst.mean(0)
    s0, d0 = src - ms, dst - md
    U, S, Vt = np.linalg.svd(d0.T @ s0 / len(src))
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    sc = np.trace(np.diag(S) @ D) / s0.var(0).sum()
    return np.hstack([sc * R, (md - sc * R @ ms)[:, None]])


def M112(lm):
    """Five landmarks (in image pixels) -> 2x3 matrix onto the 112 template."""
    return umeyama_sim(np.asarray(lm, float), ARC_SRC)


def warp_theta(M, size, H, W, shift=0.0):
    """The 2x3 `theta` that torch's affine_grid needs for this warp.

    M      2x3, source image pixels -> 112 template pixels
    size   output side S (112 or 160)
    shift  offset of the output grid in OUTPUT pixels (half-pixel test only)

    Convention: pixel INDEX coordinates on both sides, the convention of
    cv2.warpAffine, with align_corners=False in affine_grid / grid_sample.
    """
    k = size / 112.0
    Ai = np.linalg.inv(np.vstack([k * np.asarray(M, float), [0, 0, 1]]))
    P = np.array([[size / 2, 0, size / 2 - 0.5 + shift],
                  [0, size / 2, size / 2 - 0.5 + shift], [0, 0, 1]])
    N_ = np.array([[2 / W, 0, 1 / W - 1], [0, 2 / H, 1 / H - 1], [0, 0, 1]])
    return (N_ @ Ai @ P)[:2]


def crop_mask(lms, size=112, H=256, W=256):
    """1 where a source pixel lands inside the aligned crop, else 0.

    Shape (B, 1, H, W), float64. Used to ask how much of a perturbation the
    crop keeps, which is how "perturbation lost outside the crop" is tested
    as an explanation rather than assumed.
    """
    v, u = np.meshgrid(np.arange(H, dtype=np.float64),
                       np.arange(W, dtype=np.float64), indexing="ij")
    P = np.stack([u, v, np.ones_like(u)])
    out = []
    for lm in lms:
        M = M112(lm) * (size / 112.0)
        q = np.einsum("ij,jhw->ihw", M, P)
        out.append(((q[0] >= 0) & (q[0] <= size - 1) &
                    (q[1] >= 0) & (q[1] <= size - 1)).astype(np.float64))
    return np.stack(out)[:, None]


# --------------------------------------------------------------------------
# the two front ends (torch)
# --------------------------------------------------------------------------
def P_dir(x, size):
    """Whole-image front end: bilinear resize of the full photograph."""
    import torch.nn.functional as F
    return F.interpolate(x, (size, size), mode="bilinear", align_corners=False)


def P_ali(x, lms, size, shift=0.0):
    """Aligned front end: similarity warp onto the ArcFace template, crop.

    Differentiable in x, so the same operator serves as the pipeline-aware
    remedy when a protection is optimised through it.
    """
    import torch
    import torch.nn.functional as F
    B_, _, H, W = x.shape
    th = [warp_theta(M112(lm), size, H, W, shift) for lm in lms]
    g = F.affine_grid(torch.tensor(np.stack(th), dtype=torch.float32, device=x.device),
                      (B_, 3, size, size), align_corners=False)
    return F.grid_sample(x, g, mode="bilinear", padding_mode="zeros", align_corners=False)


def normalise(x, name):
    """Recognizer input normalisation. InsightFace-lineage models take BGR."""
    if name == "FaceNet":
        return (x - 0.5) / 0.5
    return (x.flip(1) - 0.5) / 0.5


def detect_landmarks(x, device="cpu"):
    """Five landmarks per image with MTCNN (largest face). None if no face.

    x: float tensor in [0, 1], (B, 3, H, W). Needs facenet-pytorch.
    """
    from facenet_pytorch import MTCNN
    mt = MTCNN(keep_all=False, select_largest=True, device=device)
    u8 = (x.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    _, _, l = mt.detect(u8, landmarks=True)
    return [None if a is None else np.asarray(a[0], float) for a in l]


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------
def _np_bilinear(img, xs, ys):
    """Reference sampler: bilinear, zeros outside. img (C,H,W), xs/ys (S,S)."""
    C, H, W = img.shape
    x0 = np.floor(xs).astype(int); y0 = np.floor(ys).astype(int)
    out = np.zeros((C,) + xs.shape)
    for dy in (0, 1):
        for dx in (0, 1):
            xi, yi = x0 + dx, y0 + dy
            w = (1 - np.abs(xs - xi)) * (1 - np.abs(ys - yi))
            ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            out += np.where(ok, w, 0.0) * img[:, np.clip(yi, 0, H - 1), np.clip(xi, 0, W - 1)]
    return out


def _source_coords(M, size, shift=0.0):
    """Where each output pixel reads from, by direct inversion of the warp."""
    k = size / 112.0
    Ai = np.linalg.inv(np.vstack([k * M, [0, 0, 1]]))
    jj, ii = np.meshgrid(np.arange(size) + shift, np.arange(size) + shift)
    q = np.einsum("ij,jhw->ihw", Ai, np.stack([jj, ii, np.ones_like(jj)]).astype(float))
    return q[0], q[1]


def _selftest():
    rng = np.random.default_rng(12345)

    # 1. Umeyama recovers a known similarity transform exactly.
    ang, sc, t = 0.07, 0.56, np.array([-14.0, -21.0])   # a face ~200 px wide in a 256 frame
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    src = rng.uniform(0, 256, (5, 2))
    M = umeyama_sim(src, (sc * R @ src.T).T + t)
    assert np.allclose(M[:, :2], sc * R, atol=1e-10) and np.allclose(M[:, 2], t, atol=1e-9)
    print("  1. Umeyama recovers a known transform            ok")

    # 2. Landmarks that ARE a transformed template map back onto the template.
    Rinv = R.T / sc
    lm = (Rinv @ (ARC_SRC - t).T).T
    back = (M112(lm) @ np.vstack([lm.T, np.ones(5)])).T
    assert np.allclose(back, ARC_SRC, atol=1e-9)
    print("  2. M112 sends the landmarks onto the template    ok")

    # 3. theta agrees with direct inversion of the warp, for both input sizes
    #    and for the half-pixel shift. This is the coordinate convention that
    #    affine_grid(align_corners=False) + grid_sample(align_corners=False) use.
    H = W = 256
    for size in (112, 160):
        for shift in (0.0, 0.5):
            th = warp_theta(M112(lm), size, H, W, shift)
            c = (2 * np.arange(size) + 1) / size - 1          # affine_grid base grid
            xn, yn = np.meshgrid(c, c)
            g = np.einsum("ij,jhw->ihw", th, np.stack([xn, yn, np.ones_like(xn)]))
            xs = ((g[0] + 1) * W - 1) / 2                     # grid_sample unnormalise
            ys = ((g[1] + 1) * H - 1) / 2
            rx, ry = _source_coords(M112(lm), size, shift)
            assert np.abs(xs - rx).max() < 1e-9 and np.abs(ys - ry).max() < 1e-9
    print("  3. theta matches direct inversion (112, 160; shift 0, 0.5)   ok")

    # 4. The crop mask and the warp agree on what is inside the crop.
    m = crop_mask([lm], 112, H, W)[0, 0]
    rx, ry = _source_coords(M112(lm), 112)
    inside = (rx >= 0) & (rx <= W - 1) & (ry >= 0) & (ry <= H - 1)
    probe = m[np.clip(np.rint(ry).astype(int), 0, H - 1), np.clip(np.rint(rx).astype(int), 0, W - 1)]
    assert probe[inside].mean() > 0.97
    print(f"  4. crop mask consistent with the warp (covers {m.mean()*100:.1f}% of the frame)   ok")

    # 5. torch sampling against the numpy reference (only if torch is present).
    try:
        import torch
    except ImportError:
        print("  5. torch not installed: sampling check SKIPPED (run it where torch is)")
        return
    x = torch.tensor(rng.uniform(0, 1, (1, 3, H, W)), dtype=torch.float32)
    for size in (112, 160):
        got = P_ali(x, [lm], size)[0].numpy()
        rx, ry = _source_coords(M112(lm), size)
        ref = _np_bilinear(x[0].numpy().astype(float), rx, ry)
        err = np.abs(got - ref).max()
        assert err < 1e-4, err
    print("  5. P_ali equals the numpy reference sampler      ok")


if __name__ == "__main__":
    print("front_ends self-test")
    _selftest()
