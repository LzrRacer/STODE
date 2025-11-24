# src/utils/ot_utils.py
import torch
import numpy as np

def _to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

def barycentric_map_entropic_ot(source, target, reg=0.05, scale_vec=None, metric='euclidean'):
    """
    Compute entropic OT transport plan between 'source' (n x d) and 'target' (m x d),
    return barycentric map y_tilde for each source sample. Uses POT (ot.sinkhorn).

    Args:
        source, target: torch.Tensor [n,d] and [m,d], on any device.
        reg: entropic regularization (epsilon).
        scale_vec: optional [1,d] tensor to reweight feature dimensions during cost computation.
        metric: POT distance metric (default 'euclidean' -> squared cost for W2).

    Returns:
        y_tilde: torch.Tensor [n,d], on source.device
    """
    try:
        import ot  # pip install pot
    except Exception as e:
        raise ImportError("POT (package 'ot') is required for barycentric_map_entropic_ot. pip install pot") from e

    dev = source.device
    src = source if scale_vec is None else source * scale_vec
    tgt = target if scale_vec is None else target * scale_vec

    Xa = _to_numpy(src)
    Xb = _to_numpy(tgt)

    n, m = Xa.shape[0], Xb.shape[0]
    a = np.ones(n) / n
    b = np.ones(m) / m

    # Cost matrix (squared Euclidean if metric='euclidean')
    C = ot.dist(Xa, Xb, metric=metric) ** 2
    P = ot.sinkhorn(a, b, C, reg=reg, numItermax=2000)  # returns (n x m)

    y_bar = (P @ Xb) / (P.sum(axis=1, keepdims=True) + 1e-12)  # (n x d)
    y_bar_t = torch.tensor(y_bar, dtype=source.dtype, device=dev)

    # unscale back if we scaled features for the cost
    if scale_vec is not None:
        y_bar_t = y_bar_t / (scale_vec + 1e-12)

    return y_bar_t
