# ENCODING

import numpy as np
from skimage import data, color
from skimage.transform import downscale_local_mean
from skimage.transform import resize
from keras.datasets import mnist

def load_mnist(values):
    '''
    Loads MNIST data of given digits

    Parameters
    ----------
    values : list
        Digits to be loaded.

    Returns
    -------
    X_train : 3D array
        Training images of MNIST digits.
    y_train : 1D array
        Labels for training data, given as presented digit (e.g. 7).
    X_test : 3D array
        Testing images of MNIST digits.
    y_test : 1D array
        Labels for testing data, given as presented digit (e.g. 7).

    '''
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train, X_test = X_train[np.isin(y_train, values)], X_test[np.isin(y_test, values)]
    y_train, y_test = y_train[np.isin(y_train, values)], y_test[np.isin(y_test, values)]
    return X_train, y_train, X_test, y_test


def down_sample(image, m_side=10):
    '''
    Returns down-sampled image by factor "image length/m_side" on both axis

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    m : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    image_downsampled : TYPE
        DESCRIPTION.

    '''
    image = np.asarray(image, dtype=float)
    assert image.shape[0] == image.shape[1], "Image not quadratic"
    factor = int(len(image)/m_side)+1
    out = resize(image, (m_side, m_side), anti_aliasing=True,
                 preserve_range=True)
    assert out.shape == (m_side, m_side) #assert output image is quadratic
    return out


def reshape_and_normalize(image):
    '''
    Returns normalized vector of image.

    Parameters
    ----------
    image : 2D array
        DESCRIPTION.

    Returns
    -------
    1D array
        Normalized vector.

    '''
    return image.reshape(-1).astype(np.float64) / 255

def encode_batch(images, m_side=10, theta_enc=1, normalize_energy=False, encoding_type="amplitude"):
    '''
    Reshapes, normalizes and encodes batch of images into electrical field amplitude.
    Works for single images as well.

    Parameters
    ----------
    images : list or 2D array of float
        DESCRIPTION.
    m_side : TYPE, optional
        DESCRIPTION. The default is 10.
    theta : TYPE, optional
        Hyperparameter controlling the strength of the encoding. The default is 1.
    normalize_energy : bool, optional
        Whether the images should be normalized for total energy. The default is False.
    encoding_type : string, optional
        Declares whether the information is decoded in the amplitude of the E-field or its phase.

    Returns
    -------
    Array
        Array of normalized amplitude-encoded electrical fields as vectors.

    '''
    images = np.asarray(images)
    if images.ndim == 2:              #For the case of single or array of pictures
        images = images[None, ...]
    fields = []
    for img in images:
        small = down_sample(img, m_side=m_side)
        vec = reshape_and_normalize(small)
        if encoding_type == "phase":
            E = phase_encoding(vec, theta_enc)
        else:
            E = amplitude_encoding(vec, theta_enc)
        if normalize_energy:
            nrm = np.linalg.norm(E)
            if nrm > 0:
                E = E / nrm
        fields.append(E)
    return np.array(fields)

def amplitude_encoding(image, theta_enc=1):
    '''
    Encode image information in amplitude of electric field

    Parameters
    ----------
    image : 1D numpy array
        Vector of image to be encoded
    theta : float
        Hyperparameter controlling the strength of the encoding

    Returns
    -------
    E : 1D complex numpy array
        Amplitude-encoded electric field

    '''
    # Convert image to normalized vector
    ampl_mod = image
    
    E = np.ones_like(ampl_mod, dtype = complex)
    # Add amplitude modulation
    E = E * ampl_mod * theta_enc
    return E

def phase_encoding(image, theta_enc=1):
    """Encode pixel values in the phase, constant amplitude."""
    return np.exp(1j * theta_enc * np.pi * image)

def _balanced_indices(y, values, number, rng):
    '''
    Indices of an equal number of samples per class, shuffled.

    Parameters
    ----------
    y : 1D array
        Labels of the full pool.
    values : sequence
        Class values to balance over, e.g. (1, 7).
    number : int or None
        Total number of samples wanted across all classes. None -> use as
        many as the smallest class allows.
    rng : np.random.Generator

    Returns
    -------
    1D array of int
        Positions into y, class-balanced and shuffled.
    '''
    per_class = [np.flatnonzero(y == v) for v in values]
    avail = min(len(idx) for idx in per_class)      # limited by smallest class

    if number is None:
        n_each = avail
    else:
        n_each = number // len(values)
        if n_each > avail:
            print(f"Only {avail} samples per class available "
                  f"({len(values)*avail} total, requested {number}).")
            n_each = avail

    picked = np.concatenate([rng.permutation(idx)[:n_each] for idx in per_class])
    return rng.permutation(picked)   # mix classes, otherwise minibatches
                                     # would each contain only one class

def get_data(values, mode: str, number=None, m_side=10, theta_enc=1,
             normalize_energy=False, seed=None, balanced=True, verbose=True):
    """Load, encode and sample MNIST data.

    Parameters
    ----------
    values : array-like of int
        Wanted MNIST classes, e.g. np.array([1, 7]).
    mode : str
        "Testing" or "Training".
    number : int or None
        Total number of samples requested across all classes. With
        balanced=True this is split evenly, so the effective count is
        len(values) * (number // len(values)). None -> take as many as
        possible.
    m_side : int
        New edge size after down-sampling (pixels).
    theta_enc : float
        Hyperparameter (amplitude factor).
    normalize_energy : bool
        Whether each field is normalized to unit energy.
    seed : int or None
        Seed for the shuffle (reproducible splits for the report).
    balanced : bool
        Draw an equal number of samples per class. The MNIST pool is
        slightly imbalanced (6742 ones vs 6265 sevens in the training
        split), which would put the majority-class baseline at 52.5%
        instead of 50% and make accuracies harder to interpret.
    verbose : bool
        Print sample count and class distribution.

    Returns
    -------
    E, y : encoded fields (N, num) and labels (num,)
    """
    X_train, y_train, X_test, y_test = load_mnist(values)
    X, y = (X_test, y_test) if mode == "Testing" else (X_train, y_train)
    rng = np.random.default_rng(seed)

    if balanced:
        indices = _balanced_indices(y, values, number, rng)
    else:
        indices = rng.permutation(len(y))
        if number is not None:
            if number > len(y):
                print(f"Just {len(y)} images found. (Requested {number})")
            indices = indices[:number]

    # encode ONLY the selected images -- down_sample is the expensive part
    E = encode_batch(X[indices], m_side, theta_enc, normalize_energy)   # (B, N)
    y_sel = y[indices]
    assert len(E) == len(y_sel), f"{len(E)} fields but {len(y_sel)} labels"

    if verbose:
        counts = {int(v): int(np.sum(y_sel == v)) for v in values}
        print(f"{mode}: {len(y_sel)} samples, class counts {counts}")
    return E.T, y_sel                       # -> (N, B)

