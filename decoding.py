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


def detection_and_determine_winner(E, det1=0, det2=-1):
    '''
    Calculate power of electrical field output at detector ports

    Parameters
    ----------
    E : 1D array of type complex
        Electrical field output of chip.
    det1 : int, optional
        Position of detector for 1st digit. The default is 0.
    det2 : int, optional
        Position of detector for 2nd digit. The default is -1.

    Returns
    -------
    int
        Highest detected power, 0 for 1st detector and 1 for 2nd given detector position.

    '''
    I1=np.abs(E[det1])**2
    I2=np.abs(E[det2])**2
    
    if I1 > I2:
        return 0
    else:
        return 1

def _detector_scores(E, y, det1, det7):
    """Extract detector intensities and one-hot targets.

    Returns
    -------
    Efield : (channel, batch) complex, the output field
    s1, s7 : (batch,) real, detected intensity |E|^2 at each detector
    t1, t7 : (batch,) float, one-hot targets for class 1 and class 7
    """
    #Handling E either in form (Channel, Layer, Batch) or (Channel, Batch)
    if E.ndim == 3:
        Efield=E[-1]
    elif E.ndim == 2:
        Efield=E
    else:
        raise ValueError("E must have ndim 2 or 3")

    I=detection(E)
    s1, s7 = I[det1], I[det7]
    t1 = (y == 1).astype(float)
    t7 = (y == 7).astype(float)
    return Efield, s1, s7, t1, t7



def loss_function(E, y, det1, det7, kind="mse"):
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
    _, s1, s7, t1, t7 = _detector_scores(E, y, det1, det7) #evaluating intensities at chosen detectors 
    print("Huhu")
    print(s1)
    print(s7)
    per_sample, _, _ = LOSS_KINDS[kind](s1, s7, t1, t7)
    return per_sample.mean(), per_sample #returns scalar (average of loss over all samples), (loss per channel, per batch -> size (channel, batch))


    

# ------------------------------------------------------------ loss kinds
# Each returns (per_sample, dL_ds1, dL_ds7), all of shape (batch,).

def mse(s1, s7, t1, t7):
    """Squared distance of the raw detector intensities to the targets.

    Simplest choice. Note that it is sensitive to the absolute energy
    arriving at the detectors, not only to their ratio.
    """
    d1, d7 = s1 - t1, s7 - t7
    per_sample = d1**2 + d7**2
    return per_sample, 2.0 * d1, 2.0 * d7


def mse_norm(s1, s7, t1, t7, eps=1e-12):
    """MSE on the normalized scores p = s / (s1 + s7).

    Invariant to overall loss/energy, since only the ratio of the two
    detectors matters. Derivatives follow from the quotient rule:
        dp1/ds1 =  s7/tot^2,   dp1/ds7 = -s1/tot^2   (and vice versa for p7)
    """
    tot = s1 + s7 + eps
    p1, p7 = s1 / tot, s7 / tot
    d1, d7 = p1 - t1, p7 - t7
    per_sample = d1**2 + d7**2

    dL_dp1, dL_dp7 = 2.0 * d1, 2.0 * d7
    dL_ds1 = dL_dp1 * ( s7 / tot**2) + dL_dp7 * (-s7 / tot**2)
    dL_ds7 = dL_dp1 * (-s1 / tot**2) + dL_dp7 * ( s1 / tot**2)
    return per_sample, dL_ds1, dL_ds7


def softmax_ce(s1, s7, t1, t7, eps=1e-12):
    """Cross-entropy over the two detectors, using the intensities as logits.

    Penalizes only the relative assignment. Because the logits are raw
    intensities, the effective sharpness depends on the absolute scale of
    s1, s7 -- worth keeping in mind when tuning theta.
    """
    z = np.stack([s1, s7], axis=0)                  # (2, batch)
    z = z - z.max(axis=0, keepdims=True)            # numerical stability
    ez = np.exp(z)
    soft = ez / ez.sum(axis=0, keepdims=True)
    tgt = np.stack([t1, t7], axis=0)

    per_sample = -(tgt * np.log(soft + eps)).sum(axis=0)
    # standard result: dL/dz = softmax - target
    return per_sample, soft[0] - t1, soft[1] - t7


LOSS_KINDS = {"mse": mse, "mse_norm": mse_norm, "softmax": softmax_ce}


