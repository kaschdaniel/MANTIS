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
  
    
def target_output(label, labels=[1,7], detectors=[33,66], N=100):
    '''
    Returns target output based on given label digit, e.g. 1 or 7

    Parameters
    ----------
    label : TYPE
        DESCRIPTION.
    labels : TYPE, optional
        DESCRIPTION. The default is [1,7].
    detectors : TYPE, optional
        DESCRIPTION. The default is [33,66].
    N : int, optional
        Number of channels. The default is 100.

    Returns
    -------
    vector : TYPE
        DESCRIPTION.

    '''
    vector = np.zeros(N)
    index = int(np.argwhere(np.array(labels) == label))
    vector[detectors[index]] = 1
    return vector
        
    


#%% Loading the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#printing the shapes of the vectors 
print('X_train: ' + str(X_train.shape))
print('Y_train: ' + str(y_train.shape))
print('X_test:  '  + str(X_test.shape))
print('Y_test:  '  + str(y_test.shape))


for i in range(3):  
    plt.subplot(330 + 1 + i)
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
    plt.show()
    


#%% Selecting only images of class 1 and 7

X_train = X_train[(y_train == 1) | (y_train == 7)]
y_train = y_train[(y_train == 1) | (y_train == 7)]
print("Size of Training-Set (X-vals): "+str(np.shape(X_train)))

X_test = X_test[(y_test == 1) | (y_test == 7)]
print("Size of Test-Set (X-vals): "+str(np.shape(X_test)))
y_test = y_test[(y_test == 1) | (y_test == 7)]

#%% Preprocessing of images to normalized vector

### Test for 1 image ###
image = X_train[0]
#print(len(image))

image_downsampled = down_sample(image, m_side=10)

#fig, ax = plt.subplots()
#ax.imshow(image_downsampled, cmap=plt.get_cmap('gray'))
    
#print(len(image_downsampled))

image_vector = reshape_and_normalize(image_downsampled)

# For all training and test data

# Downsample
X_train_ds = [down_sample(X, m_side=10) for X in X_train]
X_test_ds = [down_sample(X, m_side=10) for X in X_test]
#print(np.shape(X_train_ds))

# Resize to vector and normalize
X_train_vec = np.array([reshape_and_normalize(X) for X in X_train_ds])
X_test_vec = np.array([reshape_and_normalize(X) for X in X_test_ds])
#print(np.shape(X_train_vec))




#%% Linear regression as baseline

theta, y_predict, acc = linear_regression(X_train_vec, y_train, X_test_vec, y_test)
print(acc)



#%% Optical Machine Learning Solution

#%%% Encoding


E = amplitude_encoding(image_vector, theta=1)
print(np.shape(E))

#%%% Processing

values = np.array([1,7]) #figures you want from the mnist dataset
mode = "Training" #Selects if you want the test-set ("Testing") or the training-set ("Training")
number = 10000 #Sets number of samples you want to get in total
m_side = 10 #side length (pixel) of mnist image after downsampling
theta = 1 #mysterious hyper parameter for amplitude scaling
norm_energy = True #Bool for if energy of encoded image should be normalized or not
seed = None #Random seed to control random choice of number -pictures out of the available ones, 
            #seed = None leads to random results for each iteration

E_X, Y = get_data(values, mode, number, m_side, theta, norm_energy, seed) #E_X are the flattened arrays of the encoded mnist images (complex valued), 
                                                                  #Y are the referring labels
#Example of the first entry
plt.matshow(np.real(E_X[:,0]).reshape(1, -1), ) #reshaping it just for plotting with plt.matshow
plt.matshow(np.real(E_X[:,0]).reshape(m_side,m_side))
print(Y[0])

print("Number of '7' in this set: " + str(np.sum(Y == 7))+", number of '1' in this set: " + str(np.sum(Y == 1)))


N=4 #Number of channels
L=4 #Number of layers
test_plan = plan_rectangular(N, L) #other plans are plan_redundant and plan_triangular
test_mesh = MZIMesh(N, test_plan) #Object containing the geometry of the MZI-Mesh

#Each mesh object carries its properties:
n_layers = test_mesh.n_layers
n_slots = test_mesh.slot_counts
n_mzis = test_mesh.n_mzis
print(f"Number of layers in the setup: {n_layers}") #layers of setup
print(f"MZIs along the layers in the setup: {n_slots}") #distribution of MZIs along the layers
print(f"Total number of (trainable) MZIs in the setup: {n_mzis}") #number of (trainable MZIs in this setup)


plot_mesh(test_mesh, color_by=None, detectors=[0, 3], label_step=None) #function of visualize.py to show the geometry (and detectors)


thetas, phis = test_mesh.init_random() # ist jetzt korrigiert, ist in deinem Jupyter Notebook aber noch falsch
print(f"Phis: {phis}")
print(f"Thetas: {thetas}")

layers = test_mesh.layer_matrices(thetas, phis)   #building layers (transfer matrices) out of the parameters
fig, _ = plot_layers(layers, mode="Abs")


N=100 #number of channels (number of pixels in mnist image)
prop_mesh = MZIMesh(N, plan_rectangular(N, N)) #creating mesh object
thetas, phis = prop_mesh.init_random() #initializing random weights

layers = prop_mesh.layer_matrices_separate(thetas, phis)   #build transfer matrices (layers) from parameters

# E_out  = forward(E_X, layers)         #propagation that gives end result directly
# print(E_out)                          #format: [channels, batches]
# print(np.shape(E_out))

# E_hist_in = forward_history(E_X, layers) #propagation that gives all E-fields for every layer
# #print(E_hist)                         #format: [layers, channels, batches]
# I = np.abs(E_hist_in[:,:,2])**2
# fig, _ = plot_intensity_map(I, detectors=[33,66])


# # Backward propagation
# E_in = backward(E_out*np.exp(1j*np.pi/3), layers)
# print(np.sum(np.abs(E_in)-np.abs(E_X)))



# E_hist = backward_history(E_out, layers)
# E_hist_phase = backward_history(E_out*np.exp(1j*np.pi/3), layers) 

# I = np.abs(E_hist_in[:,:,2] + E_hist[:,:,2])**2
# I_phase = np.abs(E_hist_in[:,:,2] + E_hist_phase[:,:,2])**2

# fig, _ = plot_intensity_map(I, detectors=[33,66])
# fig, _ = plot_intensity_map(I_phase, detectors=[33,66])

# print(np.sum(I[0,:]-I_phase[0,:]))

# diff = I[0,:]-I_phase[0,:]

values = np.array([1,7]) #figures you want from the mnist dataset
mode = "Testing" #Selects if you want the test-set ("Testing") or the training-set ("Training")
number_t = 1000 #Sets number of samples you want to get in total
m_side = 10 #side length (pixel) of mnist image after downsampling
theta = 1 #mysterious hyper parameter for amplitude scaling
norm_energy = True #Bool for if energy of encoded image should be normalized or not
seed = None #Random seed to control random choice of number -pictures out of the available ones, 
            #seed = None leads to random results for each iteration
E_X_test, Y_test = get_data(values, mode, number_t, m_side, theta, norm_energy, seed) #E_X are the flattened arrays of the encoded mnist images (complex valued), 



# TEST FOR TRAINING
learning_rate = 1e-2

for j in range(number):
    #print(j)
    # 1. Forward propagation
    E_out_f1 = forward_history(E_X[:,j], layers)
    I1 = np.abs(E_out_f1[:,:])**2
    # if j%50 ==0:
    #     fig, _ = plot_intensity_map(I1, detectors=[33,66])
    
    # 2. Calculate error vector
    #E_in_b1 = np.conj(E_out_f1[-1] - target_output(Y[j]))
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
            #thetas[i+1] = thetas[i+1] - learning_rate * gradient[i+3,1:-1:2]
            #phis[i+1] = phis[i+1] - learning_rate * gradient[i+4,1:-1:2]
            
    
    if j % 100 == 0: 
        E_out_test = forward(E_X_test, layers)
        print(accuracy(E_out_test, Y_test, 33, 66))
# Testing





#%%% Decoding

detected_signal = detection(E)
y_pred = detection_and_determine_winner(detected_signal)
print(y_pred)
    
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
