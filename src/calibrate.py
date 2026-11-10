"""
calibrate.py - verification thresholds, calibrated per front end.

Companion code for:
  Wrong Input, Wrong Threshold: Re-evaluating Adversarial Facial Privacy.

A threshold is a quantile of an impostor similarity distribution, and that
distribution depends on what the recognizer is shown. So a threshold belongs
to a (recognizer, front end) pair, not to a recognizer.

Input is a matrix of L2-normalised embeddings E, shape (N, D), one row per
clean image, all embedded through ONE front end. Every pair of rows must be
an impostor pair: one photograph per identity. On a set without identity
labels (FFHQ) that is an assumption; on a labelled set (CelebA-HQ, one
photograph per subject) it holds by construction. The paper reports both.

The sample size is N, the number of images. The N(N-1)/2 pairs they form are
not independent observations.

Provenance: `threshold` is the quantile used in the measurement kernel that
produced the paper's thresholds, unchanged. `far_at` is its inverse and is
how the false-accept rate of an inherited constant is read off.

`python src/calibrate.py` runs a self-test on synthetic embeddings. It needs
no dataset and no model weights.
"""

from __future__ import annotations

import numpy as np

__all__ = ["impostor_scores", "threshold", "far_at", "calibrate"]

FAR = 0.01


def impostor_scores(E):
    """All pairwise cosines between distinct rows of E (upper triangle)."""
    E = np.asarray(E, float)
    S = E @ E.T
    return S[np.triu_indices(len(E), 1)]


def threshold(E, far=FAR):
    """The (1 - far) quantile of the impostor cosines."""
    return float(np.quantile(impostor_scores(E), 1 - far))


def far_at(E, tau):
    """False-accept rate a given threshold produces on these impostors."""
    return float((impostor_scores(E) >= tau).mean())


def calibrate(G, far=FAR):
    """Thresholds for a dict of embedding matrices.

    G maps a key such as 'IR152|direct' or 'IR152|align' to its (N, D)
    embeddings. Returns the same keys mapped to thresholds.
    """
    return {k: threshold(E, far) for k, E in G.items()}


def _selftest():
    rng = np.random.default_rng(12345)

    def emb(n, d, shared):
        """Unit vectors with a common component, as shared background gives."""
        v = rng.normal(size=(n, d)) + shared * rng.normal(size=(1, d))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    ali = emb(500, 128, shared=0.0)     # aligned crops: little in common
    dire = emb(500, 128, shared=0.35)   # whole images: shared background

    t_ali, t_dir = threshold(ali), threshold(dire)
    assert abs(far_at(ali, t_ali) - FAR) < 2e-4 and abs(far_at(dire, t_dir) - FAR) < 2e-4
    print(f"  1. a calibrated threshold gives the nominal rate       ok  "
          f"(tau_ali {t_ali:.3f}, tau_dir {t_dir:.3f})")

    assert t_dir > t_ali
    leak = far_at(dire, t_ali)
    assert leak > 2 * FAR
    print(f"  2. a threshold from one front end leaks on the other   ok  "
          f"(nominal {FAR:.1%}, measured {leak:.1%})")

    T = calibrate({"m|direct": dire, "m|align": ali})
    assert T["m|direct"] == t_dir and T["m|align"] == t_ali
    print("  3. calibrate() keys thresholds by recognizer and front end   ok")


if __name__ == "__main__":
    print("calibrate self-test (synthetic embeddings, not paper numbers)")
    _selftest()
