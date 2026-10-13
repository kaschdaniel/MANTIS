import matplotlib.pyplot as plt
import numpy as np

from baseline import *
from decoding import *
from encoding import *
from mesh import *
from metrics import *
from processing import *
from Trainer import *
from visualize import *


def linear_regression_sweep_spirals(
    theta_max, #accepts array as well
    points_per_spiral=250, 
    spirals_number=2, 
    max_noise=0,
    normalization=True, 
    seed=1550,
    channels=100, 
    padding=0, 
    theta_enc=1,
    balanced=True,
    split_ratio=0.8,
    verbose=True,
):
    def accuracy_of_linear_regression(E_train, Y_train, E_test, Y_test, classes=(0, 1)):
        return linear_regression(E_train, Y_train, E_test, Y_test, classes)[2]

    ms = [1, 2, 4, 6, 8, 10, 16, 20, 28]
    ms = [20]
    # with energy normalization
    norm_energy = True
    acc_normed = []
    for m_side in ms:
        E_train, Y_train, E_test, Y_test = get_spiral_data(
            theta_max,
            points_per_spiral, 
            spirals_number, 
            max_noise,
            normalization, 
            seed,
            channels, 
            padding, 
            theta_enc, 
            norm_energy, 
            balanced,
            split_ratio,
            verbose,
            encoding="amplitude",
        )
        acc_normed.append(accuracy_of_linear_regression(E_train, Y_train, E_test, Y_test, [0,1]))
    print(acc_normed)
    plt.plot(ms, acc_normed, marker="o", markersize=2, label="w/ energy normalization")

    # without energy normalization
    norm_energy = False
    acc_not_normed = []
    for m_side in ms:
        E_train, Y_train, E_test, Y_test = get_spiral_data(
            theta_max,
            points_per_spiral, 
            spirals_number, 
            max_noise,
            normalization, 
            seed,
            channels, 
            padding, 
            theta_enc, 
            norm_energy, 
            balanced,
            split_ratio,
            verbose,
            encoding,
        )
        acc_not_normed.append(
            accuracy_of_linear_regression(E_train, Y_train, E_test, Y_test, [0,1])
        )
    print(acc_not_normed)
    plt.plot(ms, acc_not_normed, marker="o", markersize=2, label="w/o energy normalization")
    plt.xlabel("Side length of Datasets")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.savefig(
        "results/linear_regression/spirals_m_sweep_with_and_without_energy_normalization.png",
        dpi=600,
        bbox_inches="tight",
    )
    print("Accuracy results of linear regression")
    print(
        f"Without energy normalization: {np.max(acc_not_normed):.3f} @ m={ms[np.argmax(acc_not_normed)]}"
    )
    print(f"With energy normalization: {np.max(acc_normed):.3f} @ m={ms[np.argmax(acc_normed)]}")



def training_with_spirals(
    values,
    number,
    m,
    theta_enc,
    normalize_energy,
    param_init_seed,
    balanced,
    split_ratio,
    encoding,
    detectors,
    loss_kind,
    learning_rate,
    batch_size,
    init,
    max_epochs,
    patience,
    min_delta,
    eta_bs,
    alpha_fiber,
    padding = 0,
    theta_max=np.pi,
    points_per_spiral=250,
    spirals_number=2,
    max_noise=0,
    normalization=True,
    channels=100,
    verbose=False,
    save_path=None,
):
    
    
    # Load and encode data (identical for all tests)
    E_train, Y_train, E_test, Y_test = get_spiral_data(
        theta_max, #accepts array as well
        points_per_spiral, 
        spirals_number, 
        max_noise,
        normalization, 
        param_init_seed,
        channels,
        padding, 
        theta_enc,
        normalize_energy, 
        balanced,
        split_ratio,
        verbose,
        encoding,
    )

    # CONFIG + MESH
    cfg = TrainConfig(
        m,
        theta_enc,
        normalize_energy,
        encoding,
        detectors,
        loss_kind,
        learning_rate,
        batch_size,
        init,
        max_epochs,
        patience,
        min_delta,
        param_init_seed,
        eta_bs,
        alpha_fiber,
    )

    mesh = MZIMesh(cfg.N, plan_rectangular(cfg.N, cfg.N))
    print(cfg)
    trainer = Trainer(mesh, cfg)
    history = trainer.fit(E_train, Y_train, values)
    test_loss, test_acc = trainer.evaluate(E_test, Y_test, values)
    trainer.test_acc = test_acc
    print(
        f"validation acc {history['acc'][-1]:.4f} | test acc {test_acc:.4f} | {len(history['loss'])} epochs, {trainer.train_time:.1f}s"
    )
    print(f"inference: {trainer.inference_time(E_test) * 1e3:.2f} ms/sample")

    # Save the run as a reloadable file, like the sweeps do (skipped if
    # save_path is None). Trainer.load(save_path) restores it later.
    if save_path is not None:
        trainer.save(save_path, test_acc=test_acc)
    fig, _ = plot_training(trainer)
    plt.show()
    return trainer    
    return trainers
