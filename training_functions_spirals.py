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
    verbose,
    save_path=None,
):
    padding = 0
    theta_max=np.pi
    points_per_spiral=250
    spirals_number=2
    max_noise=0
    normalization=True
    channels=m**2
    
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
