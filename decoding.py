import numpy as np


def detection(E):
    '''
    Calculate detected signal

    Parameters
    ----------
    E : 1D array of type complex
        Electrical field output of chip.

    Returns
    -------
    1D array of type float
        Measured intensity of all output ports.

    '''
    return np.abs(E)**2


def predict(E, det1=0, det7=-1, classes=(4, 9)):
    '''
    Returns winner following >>winner takes it all<< comparing detectors of index det1 and det7

    Parameters
    ----------
    E : (channel, batch) or (L+1, channel, batch) complex
        (Batch of) Electrical field output of chip.
    det1 : int, optional
        Position of detector for 1st digit. The default is 0.
    det2 : int, optional
        Position of detector for 2nd digit. The default is -1.

    Returns
    -------
    1D ndarray
        Labels of detectors with highest detected power, 1 for detector 1 and 7 for detector 7.

    '''
    Efield = E[-1] if E.ndim == 3 else E
    I = detection(Efield)
    return np.where(I[det1] >= I[det7], classes[0], classes[1])


def _detector_scores(E, y, det1, det7, classes=(4, 9)):
    """Extract detector intensities and one-hot targets.

    Returns
    -------
    Efield : (channel, batch) complex, the output field
    s1, s7 : (batch,) real, detected intensity |E|^2 at each detector
    t1, t7 : (batch,) float, one-hot targets for class 1 and class 7
    """
    # Handling E either in form (Layer, Channel, Batch) or (Channel, Batch)
    if E.ndim == 3:
        Efield = E[-1]
    elif E.ndim == 2:
        Efield = E
    elif E.ndim == 1:
        Efield = E[:, None]            # (N,) -> (N, 1)
    else:
        raise ValueError("E must have ndim 1, 2 or 3!")

    I = detection(Efield)
    s1, s7 = I[det1], I[det7]
    y = np.atleast_1d(y)
    t1 = (y == classes[0]).astype(float)
    t7 = (y == classes[1]).astype(float)
    if len(t1) != Efield.shape[1]:
        raise ValueError(f"{len(t1)} Labels, but {Efield.shape[1]} Samples")
    return Efield, s1, s7, t1, t7


def loss_function(E, y, det1, det7, kind="mse", classes=(4, 9)):
    """Scalar loss and per-sample loss for a batch.

    Parameters
    ----------
    E : (channel, batch) or (L+1, channel, batch) complex
    y : (batch,) labels, values 1 or 7
    det1, det7 : int, channel indices of the two detectors
    kind : one of LOSS_KINDS

    Returns
    -------
    loss       : float, mean over the batch
    per_sample : (batch,) loss of each sample, for learning curves
    """
    if kind not in LOSS_KINDS:
        raise ValueError(f"unknown loss kind: {kind!r}")
    # evaluating intensities at chosen detectors
    _, s1, s7, t1, t7 = _detector_scores(E, y, det1, det7, classes)
    per_sample, _, _ = LOSS_KINDS[kind](s1, s7, t1, t7)
    return per_sample  # returns loss per batch using given LOSS kinds.


def adjoint_source(E, y, det1, det7, kind="mse_norm"):
    """Gamma_L: Initial vector for backward propagation (Hughes convention)."""
    Efield, s1, s7, t1, t7 = _detector_scores(E, y, det1, det7)
    _, dL_ds1, dL_ds7 = LOSS_KINDS[kind](s1, s7, t1, t7)
    B = Efield.shape[1] if Efield.ndim == 2 else 1
    Gam = np.zeros_like(Efield)
    Gam[det1] = dL_ds1 * np.conj(Efield[det1]) / B
    Gam[det7] = dL_ds7 * np.conj(Efield[det7]) / B
    return Gam


def mse(s1, s7, t1, t7):
    """Squared distance of the detector intensities to the targets."""
    d1, d7 = s1 - t1, s7 - t7
    per_sample = d1**2 + d7**2
    return per_sample, 2.0 * d1, 2.0 * d7


def softmax_ce(s1, s7, t1, t7, eps=1e-12):
    """Cross-entropy over the two detectors, using the intensities as logits."""

    # group inputs and targets into arrays instead of stacking
    z = np.array([s1, s7])
    tgt = np.array([t1, t7])

    # shift logits by max value for numerical stability before exp
    max_z = np.max(z, axis=0, keepdims=True)
    z = z - max_z

    # get softmax probabilities
    ez = np.exp(z)
    ez_sum = np.sum(ez, axis=0, keepdims=True)
    soft = ez / ez_sum

    # calculate cross entropy loss
    loss_matrix = tgt * np.log(soft + eps)
    per_sample = -np.sum(loss_matrix, axis=0)

    # gradients for ce loss are simply prediction minus target
    grad_s1 = soft[0] - t1
    grad_s7 = soft[1] - t7

    return per_sample, grad_s1, grad_s7


LOSS_KINDS = {"mse": mse, "softmax": softmax_ce}
