"""Data sources: load or generate samples, draw balanced subsets, split, and encode.

The two pipelines `get_MNIST_data` and `get_spiral_data` return fields of
shape (N, B) and labels of shape (B,), ready for the mesh.
"""

import urllib.request

import numpy as np

from .encoding import encode_batch, encode_points
from .paths import DATA_DIR

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------
def _mnist_arrays():
    """Full MNIST as (X_train, y_train, X_test, y_test); downloaded once to DATA_DIR."""
    path = DATA_DIR / "mnist.npz"
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MNIST_URL, path)
    with np.load(path) as f:
        return f["x_train"], f["y_train"], f["x_test"], f["y_test"]


def load_mnist(values):
    """Load the MNIST images of the given digits.

    Parameters
    ----------
    values : sequence of int
        Digits to keep, e.g. (1, 7).

    Returns
    -------
    X_train : 3D array, shape (B, 28, 28)
    y_train : 1D array
        Labels as the shown digit (e.g. 7).
    X_test, y_test
        Same for MNIST's own test split.
    """
    X_train, y_train, X_test, y_test = _mnist_arrays()
    tr, te = np.isin(y_train, values), np.isin(y_test, values)
    return X_train[tr], y_train[tr], X_test[te], y_test[te]


def generate_spirals(
    theta_max, points_per_spiral=250, spirals_number=4, max_noise=0, unit_radius=True, rng=None
):
    """Interleaved Archimedean spirals r = phi / (theta_max*pi).

    Spiral i is rotated by 2*pi*i/spirals_number; its points get label i.

    Parameters
    ----------
    theta_max : float
        Total spiral angle in units of pi (theta_max/2 turns).
    points_per_spiral : int
    spirals_number : int
    max_noise : float
        Uniform noise amplitude per coordinate, in units of the unscaled spiral.
    unit_radius : bool
        Scale the spirals so their outer end lies at radius 1.
    rng : int, None or np.random.Generator
        Source of the noise.

    Returns
    -------
    XY : 2D array, shape (S*P, 2)
    y : 1D int array, shape (S*P,), spiral labels 0 ... S-1
    """
    rng = np.random.default_rng(rng)
    S, P = spirals_number, points_per_spiral
    th = float(theta_max) * np.pi
    t = th * np.arange(1, P + 1) / P  # (P,) angle along a spiral
    offset = 2 * np.pi * np.arange(S)[:, None] / S  # (S, 1) rotation per spiral

    r = t / th if unit_radius else t  # outer end at r=1 if unit_radius
    scale = 1 / th if unit_radius else 1  # noise scaled like the radius
    noise = rng.uniform(-max_noise, max_noise, size=(2, S, P)) * scale

    x = r * np.cos(t + offset) + noise[0]  # (S, P)
    y = r * np.sin(t + offset) + noise[1]
    labels = np.repeat(np.arange(S), P)  # spiral i -> label i
    return np.c_[x.ravel(), y.ravel()], labels


# ---------------------------------------------------------------------------
# Sampling and splitting
# ---------------------------------------------------------------------------
def _balanced_indices(y, values, number):
    """Positions of an equal number of samples per class.

    The pool is expected to be shuffled already, so taking the first n of
    each class is a random draw.

    Parameters
    ----------
    y : 1D array
        Labels of the (already shuffled) pool.
    values : sequence
        Class values to balance over, e.g. (1, 7).
    number : int or None
        Total number of samples across all classes. None -> as many as the
        smallest class allows.

    Returns
    -------
    1D int array
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
                f"Only {avail} samples per class available "
                f"({len(values) * avail} total, requested {number})."
            )
            n_each = avail
    return np.concatenate([idx[:n_each] for idx in per_class])


def _stratified_split(y, classes, split_ratio, rng, shuffle_within=True):
    """Index arrays (train, test) with every class split at the same ratio.

    Parameters
    ----------
    y : 1D array
        Labels.
    classes : sequence
        Class values present in y.
    split_ratio : float
        Fraction going to the training part, e.g. 0.8 for a 4:1 ratio.
    rng : np.random.Generator
    shuffle_within : bool
        Shuffle each class before cutting. Not needed if y is already shuffled.

    Returns
    -------
    train, test : 1D int arrays
        Shuffled across classes, so minibatches are not single-class.
    """
    train, test = [], []
    for c in classes:
        idx = np.flatnonzero(y == c)
        if shuffle_within:
            idx = rng.permutation(idx)
        cut = int(round(split_ratio * len(idx)))
        train.append(idx[:cut])
        test.append(idx[cut:])
    return rng.permutation(np.concatenate(train)), rng.permutation(np.concatenate(test))


def _print_counts(y_train, y_test, classes):
    for name, y in (("Train", y_train), ("Test ", y_test)):
        counts = {int(c): int(np.sum(y == c)) for c in classes}
        print(f"{name}: {len(y):5d} samples, class counts {counts}")


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
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
        Encoding strength, see `encoding.encode_batch`.
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
    encoding : {"amplitude", "phase"}

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

    # split per class; the pool is already shuffled
    i_train, i_test = _stratified_split(y_all, values, split_ratio, rng, shuffle_within=False)

    # encode only the selected images
    E_train = encode_batch(X_all[i_train], m_side, theta_enc, normalize_energy, encoding).T
    E_test = encode_batch(X_all[i_test], m_side, theta_enc, normalize_energy, encoding).T
    y_train, y_test = y_all[i_train], y_all[i_test]

    if verbose:
        _print_counts(y_train, y_test, values)
    return E_train, y_train, E_test, y_test


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
        Encoding strength, see `encoding.encode_points`.
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
    encoding : {"amplitude", "phase"}
    return_coords : bool
        Additionally return the (B, 2) coordinates, e.g. for plot_spirals.

    Returns
    -------
    E_train, y_train, E_test, y_test  (+ XY_train, XY_test if return_coords)
        Fields as (N, B) complex, labels as (B,) int = spiral index 0 ... S-1.
    """
    if np.ndim(theta_max) != 0:
        raise ValueError("theta_max must be a single value; call once per value for a sweep")
    rng = np.random.default_rng(seed)

    # generate all spirals, label = spiral index
    XY, y_all = generate_spirals(
        theta_max, points_per_spiral, spirals_number, max_noise, unit_radius, rng
    )

    # split into training and testing, balanced per class if requested
    if balanced:
        i_train, i_test = _stratified_split(y_all, range(spirals_number), split_ratio, rng)
    else:
        perm = rng.permutation(len(y_all))
        cut = int(round(split_ratio * len(y_all)))
        i_train, i_test = perm[:cut], perm[cut:]

    # encode, fields as (N, B) like the MNIST data
    E_train = encode_points(XY[i_train], channels, padding, theta_enc, normalize_energy, encoding).T
    E_test = encode_points(XY[i_test], channels, padding, theta_enc, normalize_energy, encoding).T
    y_train, y_test = y_all[i_train], y_all[i_test]

    if verbose:
        _print_counts(y_train, y_test, range(spirals_number))
    if return_coords:
        return E_train, y_train, E_test, y_test, XY[i_train], XY[i_test]
    return E_train, y_train, E_test, y_test
