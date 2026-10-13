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
from training_functions_MNIST import *



###############################################################################

def main():
    # Set standard parameters
    values = np.array([1, 7])  # digits you want from the mnist dataset
    number = 2000  # Sets number of samples you want to get in total
    m = 10  # side length (pixel) of mnist image after downsampling
    norm_energy = True  # Bool for if energy of encoded image should be normalized or not
    seed = (
        1550  # Random seed to control random choice of number -pictures out of the available ones,
    )
    # seed = None leads to random results for each iteration
    balanced = True  # Enforcing equality of classes
    split_ratio = 0.8  # Sets training-sample proportion
    verbose = False
    # Standard parameters specific to OML system
    theta_enc = 1
    encoding = "amplitude"
    detectors = (33, 66)
    loss_kind = "mse"
    learning_rate = 1
    batch_size = 64
    init = "haar"
    max_epochs = 40
    patience = 10
    min_delta = 1e-10
    eta_bs = 1.0
    alpha_fiber = 0.0
    
    
    #### SPIRALS ################################################################
    theta_max = np.pi
    points_per_spiral=250
    spirals_number=2
    max_noise=0
    normalization=True
    channels = 100
    padding = 0
    
    # Linear regression
    linear_regression_sweep_spirals(theta_max, points_per_spiral, spirals_number, 
        max_noise, normalization, seed, channels, padding, theta_enc,
        balanced, split_ratio, verbose)
    
    # trainer = training_with_spirals(values, number, m, theta_enc, norm_energy,
    #                   seed, balanced, split_ratio, encoding,
    #                   detectors, loss_kind, learning_rate, batch_size,
    #                   init, max_epochs, patience, min_delta,
    #                   eta_bs, alpha_fiber, padding, theta_max, points_per_spiral, 
    #                   spirals_number, max_noise, normalization,channels,verbose, 
    #                   save_path="results/spirals_test.json")
    
    #### MNIST ################################################################
    # Linear regression
    # linear_regression_sweep_MNIST(values, number, theta_enc, seed, balanced, split_ratio)
    
    # Training with standard_parameters
    # standard_training(values, number, m, theta_enc, norm_energy,
    #                   seed, balanced, split_ratio, encoding,
    #                   detectors, loss_kind, learning_rate, batch_size,
    #                   init, max_epochs, patience, min_delta,
    #                   eta_bs, alpha_fiber, verbose)

    # can also save single runs
    # trainer = standard_training(values, number, m, theta_enc, norm_energy,
    #                             seed, balanced, split_ratio, encoding,
    #                             detectors, loss_kind, learning_rate, batch_size,
    #                             init, max_epochs, patience, min_delta,
    #                             eta_bs, alpha_fiber, verbose,
    #                             save_path="results/final/best_model_digits79.json")
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
    # training_with_animation(values, number, m, theta_enc, norm_energy,
    #                       seed, balanced, split_ratio, encoding,
    #                       detectors, loss_kind, learning_rate, batch_size,
    #                       init, max_epochs, patience, min_delta,
    #                       eta_bs, alpha_fiber, verbose,
    #                       anim_path="results/animation/class7_hist.mp4",
    #                       show_class=7, with_histogram=True)
    # %%
    # Read a saved sweep back without retraining
    # sweep = [6,8,10,14,18,22]
    # sweep_label = "m_sidelength2"
    # read_sweep((sweep, sweep_label))
    
    
    
    


# %%
if __name__ == "__main__":
    main()

# %%
