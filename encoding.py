# ENCODING

import numpy as np
from skimage import data, color
from skimage.transform import downscale_local_mean
from keras.datasets import mnist

def load_mnist(values): #values: desired numbers (in our case 7 and 1)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train, X_test = X_train[np.isin(y_train, values)], X_test[np.isin(y_test, values)]
    y_train, y_test = y_train[np.isin(y_train, values)], y_test[np.isin(y_test, values)]
    return X_train, y_train, X_test, y_test


def down_sample(image, m=10):
    '''
    Returns down-sampled image by factor "image length/m" on both axis

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
    factor = int(len(image)/m)+1
    image_downsampled = downscale_local_mean(image, (factor, factor))
    return image_downsampled

def reshape_and_normalize(image):
    return image.reshape(-1).astype(np.float64) / 255 #suggested by claude instead of previous way
    #return image.reshape(image.shape[0]*image.shape[0]).astype(np.float) / 255

def encode_batch(images, m=10, theta=1, normalize_energy=False):
    images = np.asarray(images)
    if images.ndim == 2:              #For the case of single or array of pictures
        images = images[None, ...]
    fields = []
    for img in images:
        small = down_sample(img, m=m)
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
        vector of image to be encoded
    theta : float
        hyperparameter controlling the strength of the encoding

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

def get_data(values, mode: str, number, m=10, theta=1,
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
 
    # --- select split (FIX: Training must use X_train, not X_test) ---
    if mode == "Testing":
        X, y = X_test, y_test
    else:
        X, y = X_train, y_train
 
    E = encode_batch(X, m, theta, normalize_energy)   # (B, N): Zeile = Bild

    # consistency check auf der Bild-Achse
    assert len(E) == len(y), \
        f"Mismatch: {len(E)} fields but {len(y)} labels"

    k = len(y) #number of samples

    # Case 1: fewer images available than requested
    if number > k:
        print(f"Just {k} images found. (Requested number = {number})")
        return E.T, y                      # erst hier -> (N, k)

    # Case 2: more available than requested -> random subset
    rng = np.random.default_rng(seed)
    indices = rng.permutation(k)[:number]
    return E[indices].T, y[indices]        # auswählen auf (B,N), dann -> (N, number)

