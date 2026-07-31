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
    def accuracy_of_linear_regression(E_train, Y_train, E_test, Y_test):
        return linear_regression(E_train, Y_train, E_test, Y_test)[2]
    
    # Linear regression for different values of m with and without normalization
    norm_energy = True
    # Sweep over ms
    ms = [1, 2, 4, 6, 8, 10, 16, 20, 28]
    acc_normed = []
    for m_side in ms:
        E_train, Y_train, E_test, Y_test = get_data(values, number, m_side,
                                                    theta_enc, norm_energy, seed, balanced, split_ratio)
        acc_normed.append(accuracy_of_linear_regression(
            E_train, Y_train, E_test, Y_test))

    print(acc_normed)
    plt.plot(ms, acc_normed, marker="o", markersize=2,
             label="w/ energy normalization")

    # Without normalized Energy
    norm_energy = False

    # Sweep over ms
    ms = [1, 2, 4, 6, 8, 10, 16, 20, 28]
    acc_not_normed = []
    for m_side in ms:
        E_train, Y_train, E_test, Y_test = get_data(values, number, m_side,
                                                    theta_enc, norm_energy, seed, balanced, split_ratio)
        acc_not_normed.append(accuracy_of_linear_regression(
            E_train, Y_train, E_test, Y_test))

    print(acc_not_normed)
    plt.plot(ms, acc_not_normed, marker="o", markersize=2,
             label="w/o energy normalization")
    plt.xlabel("Side length of MNIST Datasets")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.savefig("results/linear_regression/m_sweep_with_and_without_energy_normalization.png",
                dpi=600, bbox_inches='tight')
    # %%% print max. accuracy for both linear regression methods
    print("Accuracy results of linear regression")
    print(
        f"Without energy normalization: {np.max(acc_not_normed):.3f} @ m={ms[np.argmax(acc_not_normed)]}")
    print(
        f"With energy normalization: {np.max(acc_normed):.3f} @ m={ms[np.argmax(acc_normed)]}")


def standard_training(values, number, m, theta_enc, normalize_energy,
                      param_init_seed, balanced, split_ratio, encoding,
                      detectors, loss_kind, learning_rate, batch_size,
                      init, max_epochs, patience, min_delta,
                      eta_bs, alpha_fiber, verbose):

    N = m**2

    # Load and encode data (identical for all tests)
    E_train, Y_train, E_test, Y_test = get_data(values, number, m,
                                                theta_enc, normalize_energy, param_init_seed, balanced, split_ratio, verbose)

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
    print(f"validation acc {history['acc'][-1]:.4f} | test acc {test_acc:.4f} | "
          f"{len(history['loss'])} epochs, {trainer.train_time:.1f}s")
    print(f"inference: {trainer.inference_time(E_test)*1e3:.2f} ms/sample")

    fig, _ = plot_training(trainer)


def training_compare_initialization(values, number, m, theta_enc, normalize_energy,
                      param_init_seed, balanced, split_ratio, encoding,
                      detectors, loss_kind, learning_rate, batch_size,
                      init, max_epochs, patience, min_delta,
                      eta_bs, alpha_fiber, verbose):
    
    N = m**2

    initial = ["haar", "random"]
    trainers = []
    for i in initial:
        cfg = TrainConfig(m, theta_enc, normalize_energy, encoding,
                          detectors, loss_kind, learning_rate, batch_size,
                          i, max_epochs, patience, min_delta,
                          param_init_seed, eta_bs, alpha_fiber)
        E_tr, y_tr, E_te, y_te = get_data(values, number, m,
                                          theta_enc, normalize_energy,
                                          param_init_seed, verbose)
        t = Trainer(MZIMesh(cfg.N, plan_rectangular(cfg.N, cfg.N)), cfg)
        t.fit(E_tr, y_tr)
        t.test_acc = t.evaluate(E_te, y_te)[1]
        t.save(f"results/initialization/{i}.json", test_acc=t.test_acc)
        trainers.append(t)

    fig, _ = plot_training(trainers, initial, keys=(
        "batch_loss", "loss", "acc", "grad_norm"), sweep_label="Init.")


def training_sweep_batch_size(sweep_param, values, number, m, theta_enc, normalize_energy,
                      param_init_seed, balanced, split_ratio, encoding,
                      detectors, loss_kind, learning_rate, batch_size,
                      init, max_epochs, patience, min_delta,
                      eta_bs, alpha_fiber, verbose):
    
    sweep = sweep_param[0]
    sweep_label = sweep_param[1]

    # Perform training
    trainers = []
    for s in sweep:
        cfg = TrainConfig(m, theta_enc, normalize_energy, encoding,
                          detectors, loss_kind, learning_rate, s, #s for batch_size
                          init, max_epochs, patience, min_delta,
                          param_init_seed, eta_bs, alpha_fiber)
        E_tr, y_tr, E_te, y_te = get_data(values, number, m,
                                          theta_enc, normalize_energy,
                                          param_init_seed, verbose)
        t = Trainer(MZIMesh(cfg.N, plan_rectangular(cfg.N, cfg.N)), cfg)
        t.fit(E_tr, y_tr)
        t.test_acc = t.evaluate(E_te, y_te)[1]
        t.save(f"results/{sweep_label}/{s}.json", test_acc=t.test_acc)
        trainers.append(t)

    # Plot trainingresults
    for s, t in zip(sweep, trainers):
        print(f"m={m:3d}  N={m*m:4d}  epochs={len(t.history['loss']):3d}  "
              f"loss={t.history['loss'][-1]:.5f}  "
              f"train acc={t.history['acc'][-1]:.4f}  "
              f"test acc={t.test_acc:.4f}  {t.train_time:.0f}s")

    fig, _ = plot_training(trainers, sweep, keys=(
    "batch_loss", "loss", "acc", "grad_norm"), sweep_label=sweep_label, x_axis="epoch")

    fig.savefig(f"results/{sweep_label}/plot_training_epoch.png",
                dpi=600, bbox_inches='tight')

    fig, _ = plot_training(trainers, sweep, keys=(
    "batch_loss", "loss", "acc", "grad_norm"), sweep_label=sweep_label, x_axis="time")
    fig.savefig(f"results/{sweep_label}/plot_training_time.png",
                dpi=600, bbox_inches='tight')


def training_sweep_learning_rate(sweep_param, values, number, m, theta_enc, normalize_energy,
                      param_init_seed, balanced, split_ratio, encoding,
                      detectors, loss_kind, learning_rate, batch_size,
                      init, max_epochs, patience, min_delta,
                      eta_bs, alpha_fiber, verbose):
    
    sweep = sweep_param[0]
    sweep_label = sweep_param[1]

    # Perform training
    trainers = []
    for s in sweep:
        cfg = TrainConfig(m, theta_enc, normalize_energy, encoding,
                          detectors, loss_kind, s, batch_size,
                          init, max_epochs, patience, min_delta,
                          param_init_seed, eta_bs, alpha_fiber)
        E_tr, y_tr, E_te, y_te = get_data(values, number, m,
                                                    theta_enc, normalize_energy, param_init_seed, balanced, split_ratio, verbose)
        t = Trainer(MZIMesh(cfg.N, plan_rectangular(cfg.N, cfg.N)), cfg)
        t.fit(E_tr, y_tr)
        t.test_acc = t.evaluate(E_te, y_te)[1]
        t.save(f"results/{sweep_label}/{s:.3f}.json", test_acc=t.test_acc)
        trainers.append(t)

    # Plot training results
    fig, _ = plot_training(trainers, sweep, keys=(
        "batch_loss", "loss", "acc", "grad_norm"), sweep_label=sweep_label)

    for s, t in zip(sweep, trainers):
        print(f"{sweep_label}={s}  epochs={len(t.history['loss']):3d}"
              f"loss={t.history['loss'][-1]:.5f}  "
              f"train acc={t.history['acc'][-1]:.4f}  "
              f"test acc={t.test_acc:.4f}  {t.train_time:.0f}s")

    fig.savefig(f"results/{sweep_label}/plot_training.png",
                dpi=600, bbox_inches='tight')

def training_sweep_loss_funct(sweep_param, values, number, m, theta_enc, normalize_energy,
                      param_init_seed, balanced, split_ratio, encoding,
                      detectors, loss_kind, learning_rate, batch_size,
                      init, max_epochs, patience, min_delta,
                      eta_bs, alpha_fiber, verbose):
    
    sweep = sweep_param[0]
    sweep_label = sweep_param[1]

    # Perform training
    trainers = []
    for s in sweep:
        cfg = TrainConfig(m, theta_enc, normalize_energy, encoding,
                          detectors, s, learning_rate, batch_size, #s for loss_kind
                          init, max_epochs, patience, min_delta,
                          param_init_seed, eta_bs, alpha_fiber)
        E_tr, y_tr, E_te, y_te = get_data(values, number, m,
                                          theta_enc, normalize_energy,
                                          param_init_seed, verbose)
        t = Trainer(MZIMesh(cfg.N, plan_rectangular(cfg.N, cfg.N)), cfg)
        t.fit(E_tr, y_tr)
        t.test_acc = t.evaluate(E_te, y_te)[1]
        t.save(f"results/{sweep_label}/{s}.json", test_acc=t.test_acc)
        trainers.append(t)

    # Plot trainingresults
    fig, _ = plot_training(trainers, sweep, keys=(
        "batch_loss", "loss", "acc", "grad_norm"), sweep_label=sweep_label)

    for s, t in zip(sweep, trainers):
        print(f"m={m:3d}  N={m*m:4d}  epochs={len(t.history['loss']):3d}  "
              f"loss={t.history['loss'][-1]:.5f}  "
              f"train acc={t.history['acc'][-1]:.4f}  "
              f"test acc={t.test_acc:.4f}  {t.train_time:.0f}s")

    fig.savefig(f"results/{sweep_label}/plot_training.png",
                dpi=600, bbox_inches='tight')



###############################################################################

def main():
    # Set standard parameters
    values = np.array([1, 7])  # digits you want from the mnist dataset
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
    encoding = 'amplitude'
    detectors = (33, 66)
    loss_kind = 'mse'
    learning_rate = 1.3
    batch_size = 1
    init = "haar"
    max_epochs = 1
    patience = 1
    min_delta = 1e-3
    eta_bs = 1.0
    alpha_fiber = 0.0

    # Linear regression
    # linear_regression_sweep(values, number, theta_enc,
    #                         seed, balanced, split_ratio)

    # Training with standard_parameters
    standard_training(values, number, m, theta_enc, norm_energy,
                      seed, balanced, split_ratio, encoding,
                      detectors, loss_kind, learning_rate, batch_size,
                      init, max_epochs, patience, min_delta,
                      eta_bs, alpha_fiber, verbose)


    # training_compare_initialization(values, number, m, theta_enc, norm_energy,
    #                       seed, balanced, split_ratio, encoding,
    #                       detectors, loss_kind, learning_rate, batch_size,
    #                       init, max_epochs, patience, min_delta,
    #                       eta_bs, alpha_fiber, verbose)
    
    # sweep = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    # sweep_label = "batch_size"
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

if __name__ == "__main__":
    main()
