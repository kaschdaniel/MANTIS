import numpy as np
from mesh import U, U_theta, U_phi


def forward(E_in, layers):
    """Forward propagation through given layers."""
    E = np.asarray(E_in, dtype=np.complex128)
    for Ml in layers:
        E = Ml @ E
    return E


def backward(lam_out, layers):
    """Backward propagation through reversed layers."""
    return forward(lam_out, adjoint_layers(layers))


def adjoint_layers(layers):
    """Transposing layers and reversal of their order."""
    return [Ml.T for Ml in reversed(layers)]


def forward_history(E_in, layers):
    """Like forward(), but returns electric field a all positions."""
    E = np.asarray(E_in, dtype=np.complex128)
    hist = [E.copy()]
    for Ml in layers:
        E = Ml @ E
        hist.append(E.copy())
    return np.array(hist)


def backward_history(lam_out, layers):
    """Wie backward(), but returns electric field a all positions."""
    hist = forward_history(lam_out, adjoint_layers(layers))
    return hist[::-1]


def gradient(fh, bh, plan):
    """Calculation of gradient for all values of theta and phi based on forward and backward
3 propagating fields"""
    grad_th, grad_ph = [], []
    for l, (kind, ks) in enumerate(plan):
        if kind != "mzi":
            grad_th.append(np.zeros(0))
            grad_ph.append(np.zeros(0))
            continue
        gt = np.zeros(len(ks))
        gp = np.zeros(len(ks))
        for i, k in enumerate(ks):
            # phase_shift applied to upper mode -> only channel k contributes
            gt[i] = -2*np.imag(np.sum(bh[2*l+1][k] * fh[2*l+1][k]))
            gp[i] = -2*np.imag(np.sum(bh[2*l+2][k] * fh[2*l+2][k]))
        grad_th.append(gt)
        grad_ph.append(gp)
    return grad_th, grad_ph
