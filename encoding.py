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

def encode_batch(images, m_side=10, theta=1, normalize_energy=False):
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
        E = amplitude_encoding(vec, theta)
        if normalize_energy:
            nrm = np.linalg.norm(E)
            if nrm > 0:
                E = E / nrm
        fields.append(E)
    return np.array(fields)

def amplitude_encoding(image, theta=1):
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
    E = E * ampl_mod * theta
    return E

def get_data(values, mode: str, number=None, m_side=10, theta=1,
             normalize_energy=False, seed=None):
    """Load, encode and sample MNIST data.
 
    Parameters
    ----------
    values : array-like of int
        Wanted MNIST classes, e.g. np.array([1, 7]).
    mode : str
        "Testing" or "Training".
    number : int
        Amount of data samples requested.
    m : int
        New edge size after down-sampling (pixels).
    theta : float
        Hyperparameter (amplitude factor).
    normalize_energy : bool
        Whether each field is normalized to unit energy.
    seed : int or None
        Seed for the shuffle (reproducible splits for the report).
 
    Returns
    -------
    E, y : encoded fields (N,num) and labels (num,)
    """
    X_train, y_train, X_test, y_test = load_mnist(values)
 
    # --- select split ---
    if mode == "Testing":
        X, y = X_test, y_test
    else:
        X, y = X_train, y_train
 
    E = encode_batch(X, m_side, theta, normalize_energy)   # (B, N): Line corresponds to image

    # consistency check on image axis
    assert len(E) == len(y), \
        f"Mismatch: {len(E)} fields but {len(y)} labels"

    k = len(y) #number of samples

    # Case 1: fewer images available than requested
    if number == None:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(k)
        return E[indices].T, y[indices]        # choose on (B,N), then -> (N, number)
    elif (number > k):
        print(f"Just {k} images found. (Requested number = {number})")
        return E.T, y                      # first here -> (N, k)
    else:
        # Case 2: more images available than requested -> random subset
        rng = np.random.default_rng(seed)
        indices = rng.permutation(k)[:number]
        return E[indices].T, y[indices]        # choose on (B,N), then -> (N, number)
    raise ValueError("Handling of variable 'number' threw exception!")

