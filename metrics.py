import numpy as np


def confusion_matrix(y_true, y_pred, classes=(4, 9)):
    '''
    Confusion matrix for arbitrary label values.

    Parameters
    ----------
    y_true, y_pred : 1D array
        True and predicted labels, same encoding.
    classes : sequence
        Label values in row/column order. Use (1, 7) for the raw MNIST
        digits, (0, 1) for the linear baseline encoding.

    Returns
    -------
    conf : 2D array of int
        conf[i, j] = samples of true class classes[i] predicted as
        classes[j]. Correct predictions on the diagonal.
    '''
    y_true, y_pred = np.ravel(y_true), np.ravel(y_pred)
    conf = np.zeros((len(classes), len(classes)), dtype=int)
    for i, t in enumerate(classes):
        for j, p in enumerate(classes):
            conf[i, j] = np.sum((y_true == t) & (y_pred == p))

    # every sample must land in exactly one cell -> catches mixing up
    # the 0/1 and 1/7 encodings, which would silently give a near-empty matrix
    if conf.sum() != y_true.size:
        raise ValueError(f"only {conf.sum()} of {y_true.size} samples matched "
                         f"classes={list(classes)} -- wrong label encoding?")
    return conf


def accuracy(conf):
    '''Fraction of correct predictions from a confusion matrix.'''
    print(conf)
    return np.trace(conf) / conf.sum()


def print_confusion(conf, classes=(1, 7)):
    '''Print the confusion matrix with labelled rows and columns.'''
    print("            predicted")
    print("        " + "".join(f"{c:>7}" for c in classes))
    for i, c in enumerate(classes):
        print(f"true {c:>3} " + "".join(f"{v:>7}" for v in conf[i]))
    print(f"accuracy: {accuracy(conf):.4f}")
