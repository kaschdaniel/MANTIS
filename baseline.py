import numpy as np
from metrics import confusion_matrix, accuracy

def linear_regression(X_train, y_train, X_test, y_test):
    '''
    Least-squares linear regression as a linear baseline classifier.

    Fits a linear model from flattened pixel values to a binary target
    (0 for digit 1, 1 for digit 7) by solving the normal equations via
    the Moore-Penrose pseudoinverse, then thresholds the continuous
    output at 0.5 to obtain class labels. The achieved test accuracy is
    the linear baseline the optical classifier is compared against.

    Parameters
    ----------
    X_train : 2D array of float, shape (n_train, n_features)
        Training images, already down-sampled and flattened to one row
        per image. Feature values are expected to be normalized to [0, 1].
    y_train : 1D array of int, shape (n_train,)
        Training labels as raw MNIST digits (1 or 7). Internally mapped
        to 0 and 1; the input array is not modified.
    X_test : 2D array of float, shape (n_test, n_features)
        Test images, same feature dimension and preprocessing as X_train.
    y_test : 1D array of int, shape (n_test,)
        Test labels as raw MNIST digits (1 or 7).

    Returns
    -------
    theta : 1D array of float, shape (n_features + 1,)
        Fitted weights. theta[0] is the intercept (bias), theta[1:] are
        the per-pixel weights.
    y_pred : 1D array of float, shape (n_test,)
        Continuous model output on the test set, before thresholding.
        Values are not restricted to [0, 1].
    accuracy : float
        Fraction of correctly classified test samples, computed from the
        diagonal of the confusion matrix.'''
    # Change y labels from 1 and 7 to 0 and 1
    y_train = (y_train == 7).astype(int)
    y_test = (y_test == 7).astype(int)
    
    # MODEL TRAINING
    F = np.vstack([np.ones(len(X_train.T)), X_train]).T
    # Calculate Moore-Penrose pseudoinverse
    theta = np.linalg.pinv(F) @ y_train

    # MODEL TESTING
    y_pred = theta[0] + np.sum(theta[1:] * X_test.T, axis = 1)
    y_pred_label = (y_pred >= 0.5).astype(np.float64)

    conf_matrix = confusion_matrix(y_test, y_pred_label, classes=(0, 1))
    accuracy_val = accuracy(conf_matrix)
    return theta, y_pred, accuracy_val, conf_matrix