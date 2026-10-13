import numpy as np
from keras.datasets import mnist
from skimage.transform import resize


def load_mnist(values):
    """
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

    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train, X_test = X_train[np.isin(y_train, values)], X_test[np.isin(y_test, values)]
    y_train, y_test = y_train[np.isin(y_train, values)], y_test[np.isin(y_test, values)]
    return X_train, y_train, X_test, y_test


def down_sample(image, m_side=10):
    """
    Returns down-sampled image using resize function from skimage.

    Parameters
    ----------
    image : 3D array
        Training images of MNIST digits
    m_side : int
        Sidelength to sample image to, default is 10.

    Returns
    -------
    image_downsampled : 3D Array
    """
    image = np.asarray(image, dtype=float)
    assert image.shape[0] == image.shape[1], "Image not quadratic"
    out = resize(image, (m_side, m_side), anti_aliasing=True, preserve_range=True)
    assert out.shape == (m_side, m_side)  # assert output image is quadratic
    return out


def reshape_and_normalize(image):
    """
    Returns normalized vector of image.

    Parameters
    ----------
    image : 2D array

    Returns
    -------
    1D array
        Normalized vector.

    """
    return image.reshape(-1).astype(np.float64) / 255


def encode_batch(images, m_side=10, theta_enc=1, normalize_energy=False, encoding_type="amplitude"):
    """
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

    """
    images = np.asarray(images)
    if images.ndim == 2:  # For the case of single or array of pictures
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
    """
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

    """
    return (image * theta_enc).astype(complex)


def phase_encoding(image, theta_enc=1):
    """Encode pixel values in the phase, constant amplitude."""
    return np.exp(1j * theta_enc * np.pi * image)


def _balanced_indices(y, values, number):
    """
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
    """
    per_class = [np.flatnonzero(y == v) for v in values]
    avail = min(len(idx) for idx in per_class)  # smallest class caps it

    if number is None:
        n_each = avail
    else:
        n_each = number // len(values)
        if n_each > avail:
            print(
                f"Only {avail} samples per class available ({len(values) * avail} total, requested {number})."
            )
            n_each = avail
    return np.concatenate([idx[:n_each] for idx in per_class])


def split(X, y, values, split_ratio=0.8, rng=None):
    """
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
    """
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


def get_MNIST_data(
    values,
    number=None,
    m_side=10,
    theta_enc=1,
    normalize_energy=False,
    seed=None,
    balanced=True,
    split_ratio=0.8,
    verbose=True,
    encoding="amplitude",
):
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
    X_all, y_all, _, _ = load_mnist(values)  # discard MNIST's own test split
    rng = np.random.default_rng(seed)

    # shuffle the whole pool once
    perm = rng.permutation(len(y_all))
    X_all, y_all = X_all[perm], y_all[perm]

    # keep `number` samples, balanced if requested
    keep = _balanced_indices(y_all, values, number) if balanced else np.arange(len(y_all))[:number]
    X_all, y_all = X_all[keep], y_all[keep]

    # split into training and testing, both balanced per class
    X_train, y_train, X_test, y_test = split(X_all, y_all, values, split_ratio, rng)

    # encode ONLY the selected images
    E_train = encode_batch(X_train, m_side, theta_enc, normalize_energy, encoding_type=encoding).T
    E_test = encode_batch(X_test, m_side, theta_enc, normalize_energy, encoding_type=encoding).T

    if verbose:
        for name, y in (("Train", y_train), ("Test ", y_test)):
            counts = {int(v): int(np.sum(y == v)) for v in values}
            print(f"{name}: {len(y):5d} samples, class counts {counts}")
    return E_train, y_train, E_test, y_test

def generate_spirals(theta_max, points_per_spiral=250, spirals_number=4,
                     max_noise=0, unit_radius=True, rng=None):
    """Interleaved Archimedean spirals r = phi / (theta_max*pi).
 
    Spiral i is rotated by 2*pi*i/spirals_number; its points get label i.
 
    Parameters
    ----------
    theta_max : float
        Total spiral angle in units of pi (theta_max/2 turns).
    rng : int, None or np.random.Generator
        Source of the noise.
 
    Returns
    -------
    XY : (S*P, 2) coordinates
    y  : (S*P,) int spiral labels 0 ... S-1
    """
    rng = np.random.default_rng(rng)
    S, P = spirals_number, points_per_spiral
    th = float(theta_max) * np.pi
    t = th * np.arange(1, P + 1) / P                       # (P,) angle along a spiral
    offset = 2 * np.pi * np.arange(S)[:, None] / S         # (S,1) rotation per spiral
 
    r = t / th if unit_radius else t                       # outer end at r=1 if unit_radius
    scale = 1 / th if unit_radius else 1                   # noise scaled like the radius
    noise = rng.uniform(-max_noise, max_noise, size=(2, S, P)) * scale
 
    x = r * np.cos(t + offset) + noise[0]                  # (S,P)
    y = r * np.sin(t + offset) + noise[1]
    labels = np.repeat(np.arange(S), P)                    # spiral i -> label i
    return np.c_[x.ravel(), y.ravel()], labels
 
 
def _stratified_split(y, split_ratio, rng):
    """Index arrays (train, test): every class split with the same ratio, then shuffled."""
    train, test = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        n_train = int(round(split_ratio * len(idx)))
        train.append(idx[:n_train])
        test.append(idx[n_train:])
    # shuffle across classes
    return rng.permutation(np.concatenate(train)), rng.permutation(np.concatenate(test))
 
 
def encode_points(XY, channels=100, padding=0, theta_enc=1, normalize_energy=False,
                  encoding="amplitude"):
    """Encode 2D points as fields of length `channels`.
 
    Pattern (period 2*(padding+1), repeated until `channels` is reached):
        padding=0 : (a_x, a_y, a_x, a_y, ...)
        padding=1 : (a_x, 0, a_y, 0, a_x, 0, ...)
        padding=2 : (a_x, 0, 0, a_y, 0, 0, a_x, ...)
    with a = theta_enc * x            (encoding="amplitude")
      or a = exp(i * theta_enc * x)   (encoding="phase").
 
    XY : (B, 2) coordinates. Returns (B, channels) complex.
    """
    if encoding == "amplitude":
        feat = theta_enc * XY.astype(complex)
    elif encoding == "phase":
        feat = np.exp(1j * theta_enc * XY)
    else:
        raise ValueError(f"unknown encoding {encoding!r}")
 
    k = np.arange(channels)
    filled = k % (padding + 1) == 0                   # positions that carry a value
    which = (k // (padding + 1)) % 2                  # 0 -> x, 1 -> y
    E = np.zeros((len(XY), channels), complex)
    E[:, filled] = feat[:, which[filled]]
 
    if normalize_energy:                              # sum |E|^2 = 1 per sample
        norm = np.linalg.norm(E, axis=1, keepdims=True)
        E = E / np.where(norm > 0, norm, 1)
    return E
 
 
def get_spiral_data(
    theta_max,
    points_per_spiral=250,
    spirals_number=4,
    max_noise=0,
    unit_radius=True,
    channels=100,
    padding=0,
    theta_enc=1,
    normalize_energy=False,
    seed=None,
    balanced=True,
    split_ratio=0.8,
    verbose=True,
    encoding="amplitude",
    return_coords=False,
):
    """Generate, split and encode spiral data.
 
    Task: assign a point (x, y) to the spiral it lies on.
 
    Parameters
    ----------
    theta_max : float
        Total spiral angle in units of pi (theta_max/2 turns). Sets the
        difficulty: neighbouring spirals are 2/(theta_max*spirals_number)
        apart. For a sweep, call once per value.
    points_per_spiral : int
        Points per spiral.
    spirals_number : int
        Number of interleaved spirals = number of classes.
    max_noise : float
        Uniform noise amplitude per coordinate, in units of the unscaled
        spiral (neighbouring spirals are 2*pi/spirals_number apart).
    unit_radius : bool
        Scale the spirals so their outer end lies at radius 1, independent
        of theta_max. (Geometry only; field energy is set by normalize_energy.)
    channels : int
        Field length N = number of mesh channels.
    padding : int
        Number of zero channels after each x and each y value.
    theta_enc : float
        Hyperparameter: amplitude factor (amplitude) or phase factor (phase).
    normalize_energy : bool
        Whether each field is normalized to unit energy, sum |E|^2 = 1.
    seed : int or None
        Seed for noise, split and shuffle. Fix it so the same data is used throughout.
    balanced : bool
        Split every class with the same ratio, so all spirals have equal
        shares in train and test. The pool itself is balanced by construction.
    split_ratio : float
        Fraction of the data used for training, e.g. 0.8 for a 4:1 ratio.
    verbose : bool
        Print sample counts and class distributions.
    encoding : "amplitude" or "phase"
    return_coords : bool
        Additionally return the (B, 2) coordinates, e.g. for plot_spirals.
 
    Returns
    -------
    E_train, y_train, E_test, y_test  (+ U_train, U_test if return_coords)
        Fields as (N, B) complex, labels as (B,) int = spiral index 0 ... S-1.
    """
    if np.ndim(theta_max) != 0:
        raise ValueError("theta_max must be a single value; call once per value for a sweep")
    rng = np.random.default_rng(seed)
 
    # generate all spirals, label = spiral index
    XY, y_all = generate_spirals(theta_max, points_per_spiral, spirals_number,
                                 max_noise, unit_radius, rng)
    
    # split into training and testing, balanced per class if requested
    if balanced:
        i_train, i_test = _stratified_split(y_all, split_ratio, rng)
    else:
        perm = rng.permutation(len(y_all))
        n_train = int(round(split_ratio * len(y_all)))
        i_train, i_test = perm[:n_train], perm[n_train:]
 
    # encode, fields as (N, B) like the MNIST data
    E_train = encode_points(XY[i_train], channels, padding, theta_enc,
                            normalize_energy, encoding).T
    E_test = encode_points(XY[i_test], channels, padding, theta_enc,
                           normalize_energy, encoding).T
    y_train, y_test = y_all[i_train], y_all[i_test]
 
    if verbose:
        for name, y in (("Train", y_train), ("Test ", y_test)):
            counts = {int(c): int(np.sum(y == c)) for c in range(spirals_number)}
            print(f"{name}: {len(y):5d} samples, class counts {counts}")
    
    if return_coords:
        return E_train, y_train, E_test, y_test, XY[i_train], XY[i_test]
    return E_train, y_train, E_test, y_test