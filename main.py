from encoding import *
from processing import *
from decoding import *
from visualize import *
from Trainer import *
from mesh import *

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist



def linear_regression(X_train, y_train, X_test, y_test):
    '''
    Linear regression, calculation of confusion matrix and accuracy

    Parameters
    ----------
    X_train : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.

    Returns
    -------
    theta : TYPE
        DESCRIPTION.
    y_predict : TYPE
        DESCRIPTION.
    accuracy : TYPE
        DESCRIPTION.

    '''
    # Change y labels from 1 and 7 to 0 and 1
    y_train = (y_train == 7).astype(int)
    y_test = (y_test == 7).astype(int)
    
    # MODEL TRAINING
    F = np.vstack([np.ones(len(X_train)), X_train.T]).T
    # Calculate Moore-Penrose pseudoinverse
    theta = np.linalg.pinv(F) @ y_train

    # MODEL TESTING
    y_pred = theta[0] + np.sum(theta[1:] * X_test, axis = 1)
    y_pred_label = (y_pred >= 0.5).astype(np.float64)
    
    conf_matrix = confusion_matrix(y_test, y_pred_label)
    
    accuracy = (conf_matrix[0,0]+ conf_matrix[1,1])/np.sum(conf_matrix)
    
    return theta, y_pred, accuracy


def confusion_matrix(y_true, y_pred_label):
    '''
    Calculate confusion matrix for binary classification

    Parameters
    ----------
    y_true : 1D Array of int
        True labels.
    y_pred_label : 1D Array of int
        Predicted labels.

    Returns
    -------
    conf_matrix : 2x2 Array of int
        Confusion matrix.

    '''
    conf_matrix = np.zeros((2, 2), dtype=int)
    for t in (0, 1):
        for p in (0, 1):
            conf_matrix[t, p] = np.sum((y_true == t) & (y_pred_label == p))
    print(conf_matrix)
    return conf_matrix
    


### PREPARE TRAINING DATA

values = np.array([1,7]) #figures you want from the mnist dataset
mode = "Training" #Selects if you want the test-set ("Testing") or the training-set ("Training")
number = 1000 #Sets number of samples you want to get in total
m_side = 10 #side length (pixel) of mnist image after downsampling
theta = 1 #mysterious hyper parameter for amplitude scaling
norm_energy = True #Bool for if energy of encoded image should be normalized or not
seed = None #Random seed to control random choice of number -pictures out of the available ones, 
            #seed = None leads to random results for each iteration

E_X, Y = get_data(values, mode, number, m_side, theta, norm_energy, seed) #E_X are the flattened arrays of the encoded mnist images (complex valued), 
                                                                  #Y are the referring labels

N=100 #number of channels (number of pixels in mnist image)
prop_mesh = MZIMesh(N, plan_rectangular(N, N)) #creating mesh object
thetas, phis = prop_mesh.init_random() #initializing random weights

layers = prop_mesh.layer_matrices_separate(thetas, phis)   #build transfer matrices (layers) from parameters

### PREPARE TESTING DATA

values = np.array([1,7]) #figures you want from the mnist dataset
mode = "Testing" #Selects if you want the test-set ("Testing") or the training-set ("Training")
number_t = 1000 #Sets number of samples you want to get in total
m_side = 10 #side length (pixel) of mnist image after downsampling
theta = 1 #mysterious hyper parameter for amplitude scaling
norm_energy = True #Bool for if energy of encoded image should be normalized or not
seed = None #Random seed to control random choice of number -pictures out of the available ones, 
            #seed = None leads to random results for each iteration
E_X_test, Y_test = get_data(values, mode, number_t, m_side, theta, norm_energy, seed) #E_X are the flattened arrays of the encoded mnist images (complex valued), 



# TRAINING (with testing)
learning_rate = 1e-2

for j in range(number):
    
    if j % 100 == 0: 
        E_out_test = forward(E_X_test, layers)
        print(accuracy(E_out_test, Y_test, 33, 66))
    
    #print(j)
    # 1. Forward propagation
    E_out_f1 = forward_history(E_X[:,j], layers)
    I1 = np.abs(E_out_f1[:,:])**2
    # if j%50 ==0:
    #     fig, _ = plot_intensity_map(I1, detectors=[33,66])
    
    # 2. Calculate error vector
    E_in_b1 = adjoint_source(E_out_f1[-1], Y[j], 33, 66, kind="mse_norm")[:,0]
    
    # 3. Propagate backwards
    E_out_b1 = backward_history(E_in_b1, layers)
    I2 = np.abs(E_out_b1[:,:])**2
    # if j%50 ==0:
    #     fig, _ = plot_intensity_map(I2, detectors=[33,66])
    
    # 4. Calculate gradient
    gradient = -2*np.imag(E_out_f1 * E_out_b1)
    
    # 5. Update weights
    for i in range(len(thetas)):
        #print(i)
        if i%2 == 0:
            thetas[i] = thetas[i] - learning_rate * gradient[1+2*i,::2]
            phis[i] = phis[i] - learning_rate * gradient[2+2*i,::2]
        else:
            thetas[i] = thetas[i] - learning_rate * gradient[1+2*i,1:-1:2]
            phis[i] = phis[i] - learning_rate * gradient[2+2*i,1:-1:2]
    layers = prop_mesh.layer_matrices_separate(thetas, phis)   #build transfer matrices (layers) from parameters

    
    
### ACTUAL TESTING
E_out_test = forward(E_X_test, layers)
print(accuracy(E_out_test, Y_test, 33, 66))

### EXAMPLE OUTPUTS
for c in range(10):
    E_out_test_plot = forward_history(E_X_test[:,c], layers)
    I1 = np.abs(E_out_test_plot[:,:])**2
    fig, ax = plot_intensity_map(I1, detectors=[33,66])
    ax.set_title(Y_test[c])




#%%
#def main():    


# if __name__ == "__main__":
#     main()

# #%%%
# # ----------------------------------------------------------------------
# # 1. Daten vorbereiten: Train / Val / Test getrennt
# # ----------------------------------------------------------------------
# # get_data liefert (N, k) und Labels. Encoding-Parameter wie gehabt.
# E_train, y_train = get_data(values, "Training", number=1400, m_side=10,
#                             theta=1, normalize_energy=True, seed=0)
# E_pool,  y_pool  = get_data(values, "Testing",  number=None,  m_side=10,
#                             theta=1, normalize_energy=True, seed=0)

# # Testpool in Validierung und Test aufteilen (einmalig, fester Seed)
# rng = np.random.default_rng(0)
# perm = rng.permutation(E_pool.shape[1])
# half = len(perm) // 2
# val_idx, test_idx = perm[:half], perm[half:]
# E_val,  y_val  = E_pool[:, val_idx],  y_pool[val_idx]
# E_test, y_test = E_pool[:, test_idx], y_pool[test_idx]

# # ----------------------------------------------------------------------
# # 2. Ein einzelner Trainingslauf
# # ----------------------------------------------------------------------
# N = 100
# mesh = MZIMesh(N, plan_rectangular(N, N))

# cfg = TrainConfig(
#     loss_kind="mse_norm",
#     learning_rate=1e-2,
#     detectors=(33, 66),
#     init="haar",
#     seed=0,
# )

# trainer = Trainer(mesh, cfg)
# history = trainer.fit(E_train, y_train, E_val, y_val)

# # Test genau EINMAL, mit den besten (nicht den letzten) Parametern
# test_loss, test_acc = trainer.evaluate(E_test, y_test)
# print(f"Val acc {trainer.best['val_acc']:.3f} | "
#       f"Test acc {test_acc:.3f} | "
#       f"stopped at epoch {trainer.best['epoch']} | "
#       f"{trainer.train_time:.1f}s")

# # Lernkurve fürs Protokoll
# fig, _ = plot_learning_curves(history)   # train_loss, val_loss, val_acc
# # %%
