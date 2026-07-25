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




