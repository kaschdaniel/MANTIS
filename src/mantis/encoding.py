"""Encoding layer: map raw samples (images, 2D points) onto input fields of the mesh.

Every encoder returns complex fields of shape (B, N), one row per sample, where
N is the number of mesh channels.
"""

import numpy as np
from skimage.transform import resize


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
def down_sample(image, m_side=10):
    """Down-sample a square image to (m_side, m_side) with skimage's resize.

    Parameters
    ----------
    image : 2D array
        Square image, e.g. a 28x28 MNIST digit.
    m_side : int
        Target side length, giving N = m_side**2 channels.

    Returns
    -------
    2D array of float, shape (m_side, m_side)
    """
    image = np.asarray(image, dtype=float)
    assert image.shape[0] == image.shape[1], "Image not quadratic"
    out = resize(image, (m_side, m_side), anti_aliasing=True, preserve_range=True)
    assert out.shape == (m_side, m_side)
    return out


def reshape_and_normalize(image):
    """Flatten an 8-bit image and scale its pixel values to [0, 1]."""
    return image.reshape(-1).astype(np.float64) / 255


# ---------------------------------------------------------------------------
# Field encodings
# ---------------------------------------------------------------------------
def amplitude_encoding(x, theta_enc=1):
    """Encode values in the field amplitude: E = theta_enc * x."""
    return (x * theta_enc).astype(complex)


def phase_encoding(x, theta_enc=1):
    """Encode values in the field phase at constant amplitude: E = exp(i * theta_enc * pi * x)."""
    return np.exp(1j * theta_enc * np.pi * x)


def normalize_fields(E):
    """Scale every row of E to unit energy, sum |E|^2 = 1. All-zero rows stay zero."""
    norm = np.linalg.norm(E, axis=-1, keepdims=True)
    return E / np.where(norm > 0, norm, 1)


# ---------------------------------------------------------------------------
# Batch encoders
# ---------------------------------------------------------------------------
def encode_batch(images, m_side=10, theta_enc=1, normalize_energy=False, encoding="amplitude"):
    """Down-sample, flatten and encode a batch of images into input fields.

    Parameters
    ----------
    images : 2D or 3D array
        A single image (H, W) or a batch (B, H, W).
    m_side : int
        Side length after down-sampling, giving N = m_side**2 channels.
    theta_enc : float
        Encoding strength: amplitude factor ("amplitude") or phase factor in
        units of pi ("phase").
    normalize_energy : bool
        Scale every field to unit energy.
    encoding : {"amplitude", "phase"}
        Whether the pixel values go into the amplitude or the phase.

    Returns
    -------
    E : 2D complex array, shape (B, N)
    """
    if encoding not in ("amplitude", "phase"):
        raise ValueError(f"unknown encoding {encoding!r}")
    images = np.asarray(images)
    if images.ndim == 2:  # single image
        images = images[None, ...]

    X = np.array([reshape_and_normalize(down_sample(img, m_side)) for img in images])
    E = phase_encoding(X, theta_enc) if encoding == "phase" else amplitude_encoding(X, theta_enc)
    return normalize_fields(E) if normalize_energy else E


def encode_points(
    XY, channels=100, padding=0, theta_enc=1, normalize_energy=False, encoding="amplitude"
):
    """Encode 2D points as fields of length `channels`.

    The coordinates are repeated over the channels with period 2*(padding+1):
        padding=0 : (a_x, a_y, a_x, a_y, ...)
        padding=1 : (a_x, 0, a_y, 0, a_x, 0, ...)
        padding=2 : (a_x, 0, 0, a_y, 0, 0, a_x, ...)
    with a = theta_enc * x                 (encoding="amplitude")
      or a = exp(i * theta_enc * pi * x)   (encoding="phase").

    Parameters
    ----------
    XY : 2D array, shape (B, 2)
        Point coordinates.
    channels : int
        Field length N = number of mesh channels.
    padding : int
        Number of zero channels after each x and each y value.
    theta_enc : float
        Encoding strength: amplitude factor ("amplitude") or phase factor in
        units of pi ("phase").
    normalize_energy : bool
        Scale every field to unit energy.
    encoding : {"amplitude", "phase"}

    Returns
    -------
    E : 2D complex array, shape (B, channels)
    """
    if encoding == "amplitude":
        feat = amplitude_encoding(XY, theta_enc)
    elif encoding == "phase":
        feat = phase_encoding(XY, theta_enc)
    else:
        raise ValueError(f"unknown encoding {encoding!r}")

    k = np.arange(channels)
    filled = k % (padding + 1) == 0  # positions that carry a value
    which = (k // (padding + 1)) % 2  # 0 -> x, 1 -> y
    E = np.zeros((len(XY), channels), complex)
    E[:, filled] = feat[:, which[filled]]
    return normalize_fields(E) if normalize_energy else E
