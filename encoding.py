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
    Returns down-sampled image using resize function from skimage

    Parameters
    ----------
    image : 3D array
        Training images of MNIST digits
    m_side : int
        Sidelength to sample image to, default is 10.

    Returns
    -------
    image_downsampled : 3D Array
        DESCRIPTION.

    '''
    image = np.asarray(image, dtype=float)
    assert image.shape[0] == image.shape[1], "Image not quadratic"
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
    return (image * theta_enc).astype(complex)

def phase_encoding(image, theta_enc=1):
    """Encode pixel values in the phase, constant amplitude."""
    return np.exp(1j * theta_enc * np.pi * image)

def _balanced_indices(y, values, number):
    '''
    Positions of an equal number of samples per class.

    The pool is expected to be shuffled already, so taking the first n of
    each class is a random draw and no further shuffling is needed here.

    Parameters
    ----------
    y : 1D array
        Labels of the (already shuffled) pool.
    values : sequence
        Class values to balance over, e.g. (1, 7).
    number : int or None
        Total number of samples wanted across all classes. None -> as many
        as the smallest class allows.

    Returns
    -------
    1D array of int
        Positions into y, grouped by class.
    '''
    per_class = [np.flatnonzero(y == v) for v in values]
    avail = min(len(idx) for idx in per_class)          # smallest class caps it

    if number is None:
        n_each = avail
    else:
        n_each = number // len(values)
        if n_each > avail:
            print(f"Only {avail} samples per class available "
                  f"({len(values)*avail} total, requested {number}).")
            n_each = avail
    return np.concatenate([idx[:n_each] for idx in per_class])


def split(X, y, values, split_ratio=0.8, rng=None):
    '''
    Split a data set into training and testing parts, balanced per class.

    The split is applied within each class separately, so both parts keep
    the class balance of the input. Assumes X, y are already shuffled.

    Parameters
    ----------
    X, y : arrays
        Images and labels, first axis = sample.
    values : sequence
        Class values present in y.
    split_ratio : float
        Fraction going to the training part, e.g. 0.8 for a 4:1 ratio.
    rng : np.random.Generator or None
        Used only to mix the classes again after splitting.

    Returns
    -------
    X_train, y_train, X_test, y_test
    '''
    rng = np.random.default_rng() if rng is None else rng
    train_i, test_i = [], []
    for v in values:
        idx = np.flatnonzero(y == v)
        cut = int(round(split_ratio * len(idx)))
        train_i.append(idx[:cut])
        test_i.append(idx[cut:])
    # mix the classes, otherwise minibatches would be single-class
    train_i = rng.permutation(np.concatenate(train_i))
    test_i = rng.permutation(np.concatenate(test_i))
    return X[train_i], y[train_i], X[test_i], y[test_i]


def get_data(values, number=None, m_side=10, theta_enc=1,
             normalize_energy=False, seed=None, balanced=True,
             split_ratio=0.8, verbose=True, encoding="amplitude"):
    """Load, sample, split and encode MNIST data.

    Only MNIST's own training split is used as the pool; its test split is
    discarded. Training and testing data are cut from the same `number`
    samples, as specified for the project.

    Parameters
    ----------
    values : array-like of int
        Wanted MNIST classes, e.g. np.array([1, 7]).
    number : int or None
        Total sample count across all classes. With balanced=True this is
        divided evenly, so the effective count is
        len(values) * (number // len(values)). None -> as many as possible.
    m_side : int
        Edge size after down-sampling, giving N = m_side**2 channels.
    theta_enc : float
        Hyperparameter (amplitude factor).
    normalize_energy : bool
        Whether each field is normalized to unit energy.
    seed : int or None
        Seed for the shuffle. Fix it so the same data is used throughout.
    balanced : bool
        Draw an equal number of samples per class. The MNIST pool is
        slightly imbalanced (6742 ones vs 6265 sevens), which would put the
        majority-class baseline above 50% and make accuracies harder to read.
    split_ratio : float
        Fraction of the data used for training, e.g. 0.8 for a 4:1 ratio.
    verbose : bool
        Print sample counts and class distributions.

    Returns
    -------
    E_train, y_train, E_test, y_test
        Fields as (N, B) complex, labels as (B,).
    """
    X_all, y_all, _, _ = load_mnist(values)     # discard MNIST's own test split
    rng = np.random.default_rng(seed)

    # shuffle the whole pool once
    perm = rng.permutation(len(y_all))
    X_all, y_all = X_all[perm], y_all[perm]

    # keep `number` samples, balanced if requested
    keep = _balanced_indices(y_all, values, number) if balanced \
        else np.arange(len(y_all))[:number]
    X_all, y_all = X_all[keep], y_all[keep]

    # split into training and testing, both balanced per class
    X_train, y_train, X_test, y_test = split(X_all, y_all, values,
                                             split_ratio, rng)

    # encode ONLY the selected images -- down_sample is the expensive part
    E_train = encode_batch(X_train, m_side, theta_enc, normalize_energy,
                           encoding_type=encoding).T
    E_test  = encode_batch(X_test,  m_side, theta_enc, normalize_energy,
                           encoding_type=encoding).T

    if verbose:
        for name, y in (("Train", y_train), ("Test ", y_test)):
            counts = {int(v): int(np.sum(y == v)) for v in values}
            print(f"{name}: {len(y):5d} samples, class counts {counts}")
    return E_train, y_train, E_test, y_test