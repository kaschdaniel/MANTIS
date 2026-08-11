from encoding import *
from processing import *
from decoding import *
from visualize import *
from Trainer import *
from metrics import *
from mesh import *
from baseline import *

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist


def linear_regression_sweep(values=np.array([1, 7]), number=2000, theta_enc=1,
                            seed=1550, balanced=True, split_ratio=0.8):
    def accuracy_of_linear_regression(E_train, Y_train, E_test, Y_test, classes=(1,7)):
        return linear_regression(E_train, Y_train, E_test, Y_test, classes)[2]

    ms = [1, 2, 4, 6, 8, 10, 16, 20, 28]

    # with energy normalization
    norm_energy = True
    acc_normed = []
    for m_side in ms:
        E_train, Y_train, E_test, Y_test = get_data(
            values, number, m_side, theta_enc, norm_energy,
            seed, balanced, split_ratio)
        acc_normed.append(accuracy_of_linear_regression(
            E_train, Y_train, E_test, Y_test, values))
    print(acc_normed)
    plt.plot(ms, acc_normed, marker="o", markersize=2,
             label="w/ energy normalization")

    # without energy normalization
    norm_energy = False
    acc_not_normed = []
    for m_side in ms:
        E_train, Y_train, E_test, Y_test = get_data(
            values, number, m_side, theta_enc, norm_energy,
            seed, balanced, split_ratio)
        acc_not_normed.append(accuracy_of_linear_regression(
            E_train, Y_train, E_test, Y_test, values))
    print(acc_not_normed)
    plt.plot(ms, acc_not_normed, marker="o", markersize=2,
             label="w/o energy normalization")

    plt.xlabel("Side length of MNIST Datasets")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.savefig("results/linear_regression/m_sweep_with_and_without_energy_normalization.png",
                dpi=600, bbox_inches='tight')
    print("Accuracy results of linear regression")
    print(
        f"Without energy normalization: {np.max(acc_not_normed):.3f} @ m={ms[np.argmax(acc_not_normed)]}")
    print(
        f"With energy normalization: {np.max(acc_normed):.3f} @ m={ms[np.argmax(acc_normed)]}")


def standard_training(values, number, m, theta_enc, normalize_energy,
                      param_init_seed, balanced, split_ratio, encoding,
                      detectors, loss_kind, learning_rate, batch_size,
                      init, max_epochs, patience, min_delta,
                      eta_bs, alpha_fiber, verbose, save_path=None):

    N = m**2

    # Load and encode data (identical for all tests)
    E_train, Y_train, E_test, Y_test = get_data(
        values, number, m, theta_enc, normalize_energy,
        param_init_seed, balanced, split_ratio, verbose)

    # CONFIG + MESH
    cfg = TrainConfig(m, theta_enc, normalize_energy, encoding,
                      detectors, loss_kind, learning_rate, batch_size,
                      init, max_epochs, patience, min_delta,
                      param_init_seed, eta_bs, alpha_fiber)
    mesh = MZIMesh(cfg.N, plan_rectangular(cfg.N, cfg.N))

    print(cfg)

    trainer = Trainer(mesh, cfg)
    history = trainer.fit(E_train, Y_train)

    test_loss, test_acc = trainer.evaluate(E_test, Y_test)
    trainer.test_acc = test_acc
    print(f"validation acc {history['acc'][-1]:.4f} | test acc {test_acc:.4f} | "
          f"{len(history['loss'])} epochs, {trainer.train_time:.1f}s")
    print(f"inference: {trainer.inference_time(E_test)*1e3:.2f} ms/sample")

    # Save the run as a reloadable file, like the sweeps do (skipped if
    # save_path is None). Trainer.load(save_path) restores it later.
    if save_path is not None:
        trainer.save(save_path, test_acc=test_acc)

    fig, _ = plot_training(trainer)
    plt.show()
    return trainer


def training_compare_initialization(values, number, m, theta_enc, normalize_energy,
                                    param_init_seed, balanced, split_ratio, encoding,
                                    detectors, loss_kind, learning_rate, batch_size,
                                    init, max_epochs, patience, min_delta,
                                    eta_bs, alpha_fiber, verbose):

    sweep = ["haar", "random"]
    sweep_label = "initialization"

    # Load and encode data (identical for all runs)
    E_tr, y_tr, E_te, y_te = get_data(
        values, number, m, theta_enc, normalize_energy,
        param_init_seed, balanced, split_ratio, verbose)

    # Perform training
    trainers = []
    for s in sweep:
        cfg = TrainConfig(m, theta_enc, normalize_energy, encoding,
                          detectors, loss_kind, learning_rate, batch_size,
                          s, max_epochs, patience, min_delta,  # s for init
                          param_init_seed, eta_bs, alpha_fiber)
        t = Trainer(MZIMesh(cfg.N, plan_rectangular(cfg.N, cfg.N)), cfg)
        t.fit(E_tr, y_tr)
        t.test_acc = t.evaluate(E_te, y_te)[1]
        t.save(f"results/{sweep_label}/{s}.json", test_acc=t.test_acc)
        trainers.append(t)

    # Print results
    for s, t in zip(sweep, trainers):
        print(f"{sweep_label}={str(s):>8}  epochs={len(t.history['loss']):3d}  "
              f"loss={t.history['loss'][-1]:.5f}  "
              f"train acc={t.history['acc'][-1]:.4f}  "
              f"test acc={t.test_acc:.4f}  {t.train_time:.0f}s")

    # Plot training results
    fig, _ = plot_training(trainers, sweep, keys=(
        "batch_loss", "loss", "acc", "grad_norm"), sweep_label=sweep_label,
        x_axis="epoch")
    fig.savefig(f"results/{sweep_label}/plot_training_epoch.png",
                dpi=600, bbox_inches='tight')

    fig, _ = plot_training(trainers, sweep, keys=(
        "batch_loss", "loss", "acc", "grad_norm"), sweep_label=sweep_label,
        x_axis="time")
    fig.savefig(f"results/{sweep_label}/plot_training_time.png",
                dpi=600, bbox_inches='tight')


def training_sweep_batch_size(sweep_param, values, number, m, theta_enc, normalize_energy,
                              param_init_seed, balanced, split_ratio, encoding,
                              detectors, loss_kind, learning_rate, batch_size,
                              init, max_epochs, patience, min_delta,
                              eta_bs, alpha_fiber, verbose):

    sweep = sweep_param[0]
    sweep_label = sweep_param[1]+" 2"

    # Load and encode data (identical for all runs)
    E_tr, y_tr, E_te, y_te = get_data(
        values, number, m, theta_enc, normalize_energy,
        param_init_seed, balanced, split_ratio, verbose)

    # Perform training
    trainers = []
    for s in sweep:
        cfg = TrainConfig(m, theta_enc, normalize_energy, encoding,
                          detectors, loss_kind, learning_rate, s,  # s for batch_size
                          init, max_epochs, patience, min_delta,
                          param_init_seed, eta_bs, alpha_fiber)
        t = Trainer(MZIMesh(cfg.N, plan_rectangular(cfg.N, cfg.N)), cfg)
        t.fit(E_tr, y_tr)
        t.test_acc = t.evaluate(E_te, y_te)[1]
        t.save(f"results/{sweep_label}/{s}.json", test_acc=t.test_acc)
        trainers.append(t)

    # Print results
    for s, t in zip(sweep, trainers):
        print(f"m={m:3d}  N={m*m:4d}  epochs={len(t.history['loss']):3d}  "
              f"loss={t.history['loss'][-1]:.5f}  "
              f"train acc={t.history['acc'][-1]:.4f}  "
              f"test acc={t.test_acc:.4f}  {t.train_time:.0f}s")

    # Plot training results (epoch axis + wall-clock time axis)
    fig, _ = plot_training(trainers, sweep, keys=(
        "batch_loss", "loss", "acc", "grad_norm"), sweep_label=sweep_label,
        x_axis="epoch")
    fig.savefig(f"results/{sweep_label}/plot_training_epoch.png",
                dpi=600, bbox_inches='tight')

    fig, _ = plot_training(trainers, sweep, keys=(
        "batch_loss", "loss", "acc", "grad_norm"), sweep_label=sweep_label,
        x_axis="time")
    fig.savefig(f"results/{sweep_label}/plot_training_time.png",
                dpi=600, bbox_inches='tight')


def training_sweep_learning_rate(sweep_param, values, number, m, theta_enc, normalize_energy,
                                 param_init_seed, balanced, split_ratio, encoding,
                                 detectors, loss_kind, learning_rate, batch_size,
                                 init, max_epochs, patience, min_delta,
                                 eta_bs, alpha_fiber, verbose):

    sweep = sweep_param[0]
    sweep_label = sweep_param[1]

    # Load and encode data (identical for all runs)
    E_tr, y_tr, E_te, y_te = get_data(
        values, number, m, theta_enc, normalize_energy,
        param_init_seed, balanced, split_ratio, verbose)

    # Perform training
    trainers = []
    for s in sweep:
        cfg = TrainConfig(m, theta_enc, normalize_energy, encoding,
                          detectors, loss_kind, s, batch_size,  # s for learning_rate
                          init, max_epochs, patience, min_delta,
                          param_init_seed, eta_bs, alpha_fiber)
        t = Trainer(MZIMesh(cfg.N, plan_rectangular(cfg.N, cfg.N)), cfg)
        t.fit(E_tr, y_tr)
        t.test_acc = t.evaluate(E_te, y_te)[1]
        t.save(f"results/{sweep_label}/{s:.3f}.json", test_acc=t.test_acc)
        trainers.append(t)

    # Print results
    for s, t in zip(sweep, trainers):
        print(f"{sweep_label}={s:>8}  epochs={len(t.history['loss']):3d}  "
              f"loss={t.history['loss'][-1]:.5f}  "
              f"train acc={t.history['acc'][-1]:.4f}  "
              f"test acc={t.test_acc:.4f}  {t.train_time:.0f}s")

    # Plot training results (epoch axis + wall-clock time axis)
    fig, _ = plot_training(trainers, sweep, keys=(
        "batch_loss", "loss", "acc", "grad_norm"), sweep_label=sweep_label,
        x_axis="epoch")
    fig.savefig(f"results/{sweep_label}/plot_training_epoch.png",
                dpi=600, bbox_inches='tight')

    fig, _ = plot_training(trainers, sweep, keys=(
        "batch_loss", "loss", "acc", "grad_norm"), sweep_label=sweep_label,
        x_axis="time")
    fig.savefig(f"results/{sweep_label}/plot_training_time.png",
                dpi=600, bbox_inches='tight')


def training_sweep_loss_funct(sweep_param, values, number, m, theta_enc, normalize_energy,
                              param_init_seed, balanced, split_ratio, encoding,
                              detectors, loss_kind, learning_rate, batch_size,
                              init, max_epochs, patience, min_delta,
                              eta_bs, alpha_fiber, verbose):

    sweep = sweep_param[0]
    sweep_label = sweep_param[1]

    # Load and encode data (identical for all runs)
    E_tr, y_tr, E_te, y_te = get_data(
        values, number, m, theta_enc, normalize_energy,
        param_init_seed, balanced, split_ratio, verbose)

    # Perform training
    trainers = []
    for s in sweep:
        cfg = TrainConfig(m, theta_enc, normalize_energy, encoding,
                          detectors, s, learning_rate, batch_size,  # s for loss_kind
                          init, max_epochs, patience, min_delta,
                          param_init_seed, eta_bs, alpha_fiber)
        t = Trainer(MZIMesh(cfg.N, plan_rectangular(cfg.N, cfg.N)), cfg)
        t.fit(E_tr, y_tr)
        t.test_acc = t.evaluate(E_te, y_te)[1]
        t.save(f"results/{sweep_label}/{s}.json", test_acc=t.test_acc)
        trainers.append(t)

    # Print results
    for s, t in zip(sweep, trainers):
        print(f"{sweep_label}={str(s):>8}  epochs={len(t.history['loss']):3d}  "
              f"loss={t.history['loss'][-1]:.5f}  "
              f"train acc={t.history['acc'][-1]:.4f}  "
              f"test acc={t.test_acc:.4f}  {t.train_time:.0f}s")

    # Plot training results (epoch axis + wall-clock time axis)
    fig, _ = plot_training(trainers, sweep, keys=(
        "batch_loss", "loss", "acc", "grad_norm"), sweep_label=sweep_label,
        x_axis="epoch")
    fig.savefig(f"results/{sweep_label}/plot_training_epoch.png",
                dpi=600, bbox_inches='tight')

    fig, _ = plot_training(trainers, sweep, keys=(
        "batch_loss", "loss", "acc", "grad_norm"), sweep_label=sweep_label,
        x_axis="time")
    fig.savefig(f"results/{sweep_label}/plot_training_time.png",
                dpi=600, bbox_inches='tight')


def training_sweep_m_sidelength(sweep_param, values, number, m, theta_enc, normalize_energy,
                                param_init_seed, balanced, split_ratio, encoding,
                                detectors, loss_kind, learning_rate, batch_size,
                                init, max_epochs, patience, min_delta,
                                eta_bs, alpha_fiber, verbose):

    sweep = sweep_param[0]
    sweep_label = sweep_param[1]

    # NOTE: unlike the other sweeps, the data cannot be loaded once -- m sets
    # the image size, so get_data runs inside the loop. The detectors also
    # scale with N = m^2, placed at 1/3 and 2/3 of the channels.
    trainers = []
    for s in sweep:
        detectors = (s**2 // 3, 2 * s**2 // 3)  # scale with N, s for m_side
        cfg = TrainConfig(s, theta_enc, normalize_energy, encoding,
                          detectors, loss_kind, learning_rate, batch_size,
                          init, max_epochs, patience, min_delta,
                          param_init_seed, eta_bs, alpha_fiber)
        E_tr, y_tr, E_te, y_te = get_data(
            values, number, s, theta_enc, normalize_energy,
            param_init_seed, balanced, split_ratio, verbose)
        t = Trainer(MZIMesh(cfg.N, plan_rectangular(cfg.N, cfg.N)), cfg)
        t.fit(E_tr, y_tr)
        t.test_acc = t.evaluate(E_te, y_te)[1]
        t.save(f"results/{sweep_label}/{s}.json", test_acc=t.test_acc)
        trainers.append(t)

    # Print results
    for s, t in zip(sweep, trainers):
        print(f"{sweep_label}={str(s):>8}  N={s*s:4d}  epochs={len(t.history['loss']):3d}  "
              f"loss={t.history['loss'][-1]:.5f}  "
              f"train acc={t.history['acc'][-1]:.4f}  "
              f"test acc={t.test_acc:.4f}  {t.train_time:.0f}s")

    # Plot training results (epoch axis + wall-clock time axis)
    fig, _ = plot_training(trainers, sweep, keys=(
        "batch_loss", "loss", "acc", "grad_norm"), sweep_label=sweep_label,
        x_axis="epoch")
    fig.savefig(f"results/{sweep_label}/plot_training_epoch.png",
                dpi=600, bbox_inches='tight')

    fig, _ = plot_training(trainers, sweep, keys=(
        "batch_loss", "loss", "acc", "grad_norm"), sweep_label=sweep_label,
        x_axis="time")
    fig.savefig(f"results/{sweep_label}/plot_training_time.png",
                dpi=600, bbox_inches='tight')


def training_sweep_layer_count(sweep_param, values, number, m, theta_enc, normalize_energy,
                               param_init_seed, balanced, split_ratio, encoding,
                               detectors, loss_kind, learning_rate, batch_size,
                               init, max_epochs, patience, min_delta,
                               eta_bs, alpha_fiber, verbose):

    sweep = sweep_param[0]
    sweep_label = sweep_param[1]

    # Load and encode data (identical for all runs -- m is constant, only the
    # number of mesh layers L changes via plan_rectangular(N, L))
    E_tr, y_tr, E_te, y_te = get_data(
        values, number, m, theta_enc, normalize_energy,
        param_init_seed, balanced, split_ratio, verbose)

    # Perform training
    trainers = []
    for s in sweep:
        cfg = TrainConfig(m, theta_enc, normalize_energy, encoding,
                          detectors, loss_kind, learning_rate, batch_size,
                          init, max_epochs, patience, min_delta,
                          param_init_seed, eta_bs, alpha_fiber)
        t = Trainer(MZIMesh(cfg.N, plan_rectangular(cfg.N, s)),
                    cfg)  # s for layer count
        t.fit(E_tr, y_tr)
        t.test_acc = t.evaluate(E_te, y_te)[1]
        t.save(f"results/{sweep_label}/{s}.json", test_acc=t.test_acc)
        trainers.append(t)

    # Print results
    for s, t in zip(sweep, trainers):
        print(f"{sweep_label}={str(s):>8}  N={m*m:4d}  epochs={len(t.history['loss']):3d}  "
              f"loss={t.history['loss'][-1]:.5f}  "
              f"train acc={t.history['acc'][-1]:.4f}  "
              f"test acc={t.test_acc:.4f}  {t.train_time:.0f}s")

    # Plot training results (epoch axis + wall-clock time axis)
    fig, _ = plot_training(trainers, sweep, keys=(
        "batch_loss", "loss", "acc", "grad_norm"), sweep_label=sweep_label,
        x_axis="epoch")
    fig.savefig(f"results/{sweep_label}/plot_training_epoch.png",
                dpi=600, bbox_inches='tight')

    fig, _ = plot_training(trainers, sweep, keys=(
        "batch_loss", "loss", "acc", "grad_norm"), sweep_label=sweep_label,
        x_axis="time")
    fig.savefig(f"results/{sweep_label}/plot_training_time.png",
                dpi=600, bbox_inches='tight')


def training_sweep_theta_enc_phase(sweep_param, values, number, m, theta_enc, normalize_energy,
                                   param_init_seed, balanced, split_ratio, encoding,
                                   detectors, loss_kind, learning_rate, batch_size,
                                   init, max_epochs, patience, min_delta,
                                   eta_bs, alpha_fiber, verbose):

    sweep = sweep_param[0]
    sweep_label = sweep_param[1]

    # PHASE encoding: information sits in the phase of E, |E_n| is constant.
    # The swept theta_enc controls how much phase span the pixel values cover,
    # so the data must be re-encoded for every value -> get_data in the loop.
    trainers = []
    for s in sweep:
        cfg = TrainConfig(m, s, normalize_energy, "phase",  # s for theta_enc
                          detectors, loss_kind, learning_rate, batch_size,
                          init, max_epochs, patience, min_delta,
                          param_init_seed, eta_bs, alpha_fiber)
        E_tr, y_tr, E_te, y_te = get_data(
            values, number, m, s, normalize_energy,
            param_init_seed, balanced, split_ratio, verbose, encoding="phase")
        t = Trainer(MZIMesh(cfg.N, plan_rectangular(cfg.N, cfg.N)), cfg)
        t.fit(E_tr, y_tr)
        t.test_acc = t.evaluate(E_te, y_te)[1]
        t.save(f"results/{sweep_label}/{s}.json", test_acc=t.test_acc)
        trainers.append(t)

    # Print results
    for s, t in zip(sweep, trainers):
        print(f"{sweep_label}={s:>8}  epochs={len(t.history['loss']):3d}  "
              f"loss={t.history['loss'][-1]:.5f}  "
              f"train acc={t.history['acc'][-1]:.4f}  "
              f"test acc={t.test_acc:.4f}  {t.train_time:.0f}s")

    # Plot training results (epoch axis + wall-clock time axis)
    fig, _ = plot_training(trainers, sweep, keys=(
        "batch_loss", "loss", "acc", "grad_norm"), sweep_label=sweep_label,
        x_axis="epoch")
    fig.savefig(f"results/{sweep_label}/plot_training_epoch.png",
                dpi=600, bbox_inches='tight')

    fig, _ = plot_training(trainers, sweep, keys=(
        "batch_loss", "loss", "acc", "grad_norm"), sweep_label=sweep_label,
        x_axis="time")
    fig.savefig(f"results/{sweep_label}/plot_training_time.png",
                dpi=600, bbox_inches='tight')


# maps a sweep value to a layer plan; keeps the sweep generic over architectures
def _architecture_plan(name, N):
    if name == "rectangular":
        return plan_rectangular(N, N)
    if name == "triangular":
        return plan_triangular(N)
    if name == "redundant":
        # N extra layers on top of the mesh
        return plan_redundant(N, N)
    if name == "permuting":
        return plan_permuting(N)
    raise ValueError(f"unknown architecture '{name}'")


def training_sweep_architecture(sweep_param, values, number, m, theta_enc, normalize_energy,
                                param_init_seed, balanced, split_ratio, encoding,
                                detectors, loss_kind, learning_rate, batch_size,
                                init, max_epochs, patience, min_delta,
                                eta_bs, alpha_fiber, verbose):

    sweep = sweep_param[0]
    sweep_label = sweep_param[1]

    # AMPLITUDE encoding, data identical for all runs -- only the mesh topology
    # (the plan passed to MZIMesh) changes between runs.
    E_tr, y_tr, E_te, y_te = get_data(
        values, number, m, theta_enc, normalize_energy,
        param_init_seed, balanced, split_ratio, verbose, encoding="amplitude")

    trainers = []
    for s in sweep:
        cfg = TrainConfig(m, theta_enc, normalize_energy, "amplitude",
                          detectors, loss_kind, learning_rate, batch_size,
                          init, max_epochs, patience, min_delta,
                          param_init_seed, eta_bs, alpha_fiber)
        plan = _architecture_plan(s, cfg.N)  # s for architecture name
        t = Trainer(MZIMesh(cfg.N, plan), cfg)
        t.fit(E_tr, y_tr)
        t.test_acc = t.evaluate(E_te, y_te)[1]
        t.save(f"results/{sweep_label}/{s}.json", test_acc=t.test_acc)
        trainers.append(t)

    # Print results (n_layers differs per architecture -> show it)
    for s, t in zip(sweep, trainers):
        print(f"{sweep_label}={str(s):>12}  layers={t.mesh.n_layers:3d}  "
              f"MZIs={t.mesh.n_mzis:4d}  epochs={len(t.history['loss']):3d}  "
              f"loss={t.history['loss'][-1]:.5f}  "
              f"train acc={t.history['acc'][-1]:.4f}  "
              f"test acc={t.test_acc:.4f}  {t.train_time:.0f}s")

    # Plot training results (epoch axis + wall-clock time axis)
    fig, _ = plot_training(trainers, sweep, keys=(
        "batch_loss", "loss", "acc", "grad_norm"), sweep_label=sweep_label,
        x_axis="epoch")
    fig.savefig(f"results/{sweep_label}/plot_training_epoch.png",
                dpi=600, bbox_inches='tight')

    fig, _ = plot_training(trainers, sweep, keys=(
        "batch_loss", "loss", "acc", "grad_norm"), sweep_label=sweep_label,
        x_axis="time")
    fig.savefig(f"results/{sweep_label}/plot_training_time.png",
                dpi=600, bbox_inches='tight')


def read_sweep(sweep_param):
    """Load a saved sweep and reproduce its printout and plots.

    Reads the JSON files written by a training sweep and rebuilds the
    trainers, so results can be inspected without retraining. Same printout
    and plots as the sweep functions, but nothing is written to disk.

    sweep_param : (sweep, sweep_label)
        sweep       : the values used, e.g. [8, 16, 32] or ["mse", "softmax"]
        sweep_label : the results sub-folder, e.g. "batch_size"
    """
    sweep = sweep_param[0]
    sweep_label = sweep_param[1]

    # Load the trainers back from disk
    trainers = []
    for s in sweep:
        # learning_rate was saved with 3 decimals, everything else as-is
        name = f"{s}"
        t = Trainer.load(f"results/{sweep_label}/{name}.json")
        trainers.append(t)

    # Print results
    for s, t in zip(sweep, trainers):
        print(f"{sweep_label}={str(s):>8}  epochs={len(t.history['loss']):3d}  "
              f"loss={t.history['loss'][-1]:.5f}  "
              f"train acc={t.history['acc'][-1]:.4f}  "
              f"test acc={t.test_acc:.4f}  {t.train_time:.0f}s")

    # Plot training results (no saving)
    plot_training(trainers, sweep, keys=(
        "batch_loss", "loss", "acc", "grad_norm"), sweep_label=sweep_label,
        x_axis="epoch")
    plot_training(trainers, sweep, keys=(
        "batch_loss", "loss", "acc", "grad_norm"), sweep_label=sweep_label,
        x_axis="time")

    plot_training(trainers, sweep, keys=(
        "loss", "acc", "grad_norm"), sweep_label=sweep_label,
        x_axis="epoch")
    plot_training(trainers, sweep, keys=(
        "loss", "acc", "grad_norm"), sweep_label=sweep_label,
        x_axis="time")

    return trainers


###############################################################################

def main():
    # Set standard parameters
    values = np.array([4, 9])  # digits you want from the mnist dataset
    number = 2000  # Sets number of samples you want to get in total
    m = 10  # side length (pixel) of mnist image after downsampling
    norm_energy = True  # Bool for if energy of encoded image should be normalized or not
    seed = 1550  # Random seed to control random choice of number -pictures out of the available ones,
    # seed = None leads to random results for each iteration
    balanced = True  # Enforcing equality of classes
    split_ratio = 0.8  # Sets training-sample proportion
    verbose = False

    # Standard parameters specific to OML system
    theta_enc = 1
    encoding = 'phase'
    detectors = (33, 66)
    loss_kind = 'mse'
    learning_rate = 1
    batch_size = 64
    init = "haar"
    max_epochs = 40
    patience = 10
    min_delta = 1e-10
    eta_bs = 1.0
    alpha_fiber = 0.0

    # Linear regression
    linear_regression_sweep(values, number, theta_enc,
                            seed, balanced, split_ratio)

    # Training with standard_parameters
    # standard_training(values, number, m, theta_enc, norm_energy,
    #                   seed, balanced, split_ratio, encoding,
    #                   detectors, loss_kind, learning_rate, batch_size,
    #                   init, max_epochs, patience, min_delta,
    #                   eta_bs, alpha_fiber, verbose)

    # can also save single runs
    trainer = standard_training(values, number, m, theta_enc, norm_energy,
                                seed, balanced, split_ratio, encoding,
                                detectors, loss_kind, learning_rate, batch_size,
                                init, max_epochs, patience, min_delta,
                                eta_bs, alpha_fiber, verbose,
                                save_path="results/final/best_model_digits4_9.json")

    # trainer = Trainer.load("results/final/best_model.json")
    # print(trainer.extra["test_acc"])

    # training_compare_initialization(values, number, m, theta_enc, norm_energy,
    #                       seed, balanced, split_ratio, encoding,
    #                       detectors, loss_kind, learning_rate, batch_size,
    #                       init, max_epochs, patience, min_delta,
    #                       eta_bs, alpha_fiber, verbose)

    # sweep = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    # training_sweep_batch_size((sweep, sweep_label), values, number, m, theta_enc, norm_energy,
    #                       seed, balanced, split_ratio, encoding,
    #                       detectors, loss_kind, learning_rate, batch_size,
    #                       init, max_epochs, patience, min_delta,
    #                       eta_bs, alpha_fiber, verbose)

    # sweep = [0.01, 0.03, 0.1, 0.3, 1.0]
    # sweep_label = "learning_rate"
    # training_sweep_learning_rate((sweep, sweep_label), values, number, m, theta_enc, norm_energy,
    #                       seed, balanced, split_ratio, encoding,
    #                       detectors, loss_kind, learning_rate, batch_size,
    #                       init, max_epochs, patience, min_delta,
    #                       eta_bs, alpha_fiber, verbose)

    # sweep = ["mse", "softmax"]
    # sweep_label = "loss_functions"
    # training_sweep_loss_funct((sweep, sweep_label), values, number, m, theta_enc, norm_energy,
    #                       seed, balanced, split_ratio, encoding,
    #                       detectors, loss_kind, learning_rate, batch_size,
    #                       init, max_epochs, patience, min_delta,
    #                       eta_bs, alpha_fiber, verbose)

    # sweep = [6, 8, 10, 14, 18, 22]
    # sweep_label = "m_sidelength"
    # training_sweep_m_sidelength((sweep, sweep_label), values, number, m, theta_enc, norm_energy,
    #                       seed, balanced, split_ratio, encoding,
    #                       detectors, loss_kind, learning_rate, batch_size,
    #                       init, max_epochs, patience, min_delta,
    #                       eta_bs, alpha_fiber, verbose)

    # sweep = [10, 50, 100, 150, 200]
    # sweep_label = "layer_count"
    # training_sweep_layer_count((sweep, sweep_label), values, number, m, theta_enc, norm_energy,
    #                       seed, balanced, split_ratio, encoding,
    #                       detectors, loss_kind, learning_rate, batch_size,
    #                       init, max_epochs, patience, min_delta,
    #                       eta_bs, alpha_fiber, verbose)

    # sweep = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    # sweep_label = "theta_enc_phase"
    # training_sweep_theta_enc_phase((sweep, sweep_label), values, number, m, theta_enc, norm_energy,
    #                       seed, balanced, split_ratio, encoding,
    #                       detectors, loss_kind, learning_rate, batch_size,
    #                       init, max_epochs, patience, min_delta,
    #                       eta_bs, alpha_fiber, verbose)

    # sweep = ["rectangular", "triangular", "redundant", "permuting"]
    # sweep_label = "architecture"
    # training_sweep_architecture((sweep, sweep_label), values, number, m, theta_enc, norm_energy,
    #                       seed, balanced, split_ratio, encoding,
    #                       detectors, loss_kind, learning_rate, batch_size,
    #                       init, max_epochs, patience, min_delta,
    #                       eta_bs, alpha_fiber, verbose)

    # %%
    # Read a saved sweep back without retraining
    # sweep = [6,8,10,14,18,22]
    # sweep_label = "m_sidelength2"
    # read_sweep((sweep, sweep_label))


if __name__ == "__main__":
    main()
