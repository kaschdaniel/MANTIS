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



def linear_regression(X_train, y_train, X_test, y_test):
    '''
    Linear regression, calculation of confusion matrix and accuracy
    '''
    # Change y labels from 1 and 7 to 0 and 1
    y_train_norm = (y_train == 7).astype(int)
    y_test_norm = (y_test == 7).astype(int)

    # MODEL TRAINING
    F = np.vstack([np.ones(len(X_train.T)), X_train]).T
    # Calculate Moore-Penrose pseudoinverse
    theta = np.linalg.pinv(F) @ y_train_norm

    # MODEL TESTING
    y_pred = theta[0] + np.sum(theta[1:] * X_test.T, axis = 1)
    y_pred_label = (y_pred >= 0.5).astype(int)#(np.float64)
    
    
    y_pred_label[y_pred_label == 1] = 7
    y_pred_label[y_pred_label == 0] = 1

    conf_matrix = confusion_matrix(y_test, y_pred_label)

    accuracy = (conf_matrix[0,0]+ conf_matrix[1,1])/np.sum(conf_matrix)

    return theta, y_pred, accuracy



#%% Begin of relevant code

#%%% Load and encode data (identical for all tests)
values = np.array([1,7])#figures you want from the mnist dataset
number = 2000            #Sets number of samples you want to get in total
m_side = 10             #side length (pixel) of mnist image after downsampling
theta_enc = 1           #mysterious hyper parameter for amplitude scaling
norm_energy = True      #Bool for if energy of encoded image should be normalized or not
seed = 1550             #Random seed to control random choice of number -pictures out of the available ones, 
                        #seed = None leads to random results for each iteration
balanced = True         #Enforcing equality of classes
split_ratio = 0.8       #Sets training-sample proportion

E_train, Y_train, E_test, Y_test = get_data(values, number, m_side, 
                        theta_enc, norm_energy, seed, balanced, split_ratio)    
#E_* are the flattened arrays of the encoded mnist images (complex valued), 
#Y_* are the referring labels


#%%% Linear regression
def accuracy_of_linear_regression(E_train, Y_train, E_test, Y_test):
    return linear_regression(E_train, Y_train, E_test, Y_test)[2]


#With normalized Energy
norm_energy=True
print(accuracy_of_linear_regression(E_train, Y_train, E_test, Y_test))
#_, _, acc = linear_regression(E_train, Y_train, E_test, Y_test)
#print(acc)

#Sweep over ms
ms = [1,2,4,6,8,10,16,20,28]
acc_normed=[]
for m_side in ms:
    E_train, Y_train, E_test, Y_test = get_data(values, number, m_side, 
                        theta_enc, norm_energy, seed, balanced, split_ratio) 
    acc_normed.append(accuracy_of_linear_regression(E_train, Y_train, E_test, Y_test))

print(acc_normed)
plt.plot(ms,acc_normed,marker="o", markersize=2, label="w/ energy normalization")

#Without normalized Energy
norm_energy = False      #Bool for if energy of encoded image should be normalized or not
E_train, Y_train, E_test, Y_test = get_data(values, number, m_side, 
                        theta_enc, norm_energy, seed, balanced, split_ratio)
print(accuracy_of_linear_regression(E_train, Y_train, E_test, Y_test))
#_, _, acc = linear_regression(E_train, Y_train, E_test, Y_test)
#print(acc)

#Sweep over ms
ms = [1,2,4,6,8,10,16,20,28]
acc_not_normed=[]
for m_side in ms:
    E_train, Y_train, E_test, Y_test = get_data(values, number, m_side, 
                        theta_enc, norm_energy, seed, balanced, split_ratio) 
    acc_not_normed.append(accuracy_of_linear_regression(E_train, Y_train, E_test, Y_test))

print(acc_not_normed)
plt.plot(ms,acc_not_normed,marker="o", markersize=2, label="w/o energy normalization")
plt.xlabel("Side length of MNIST Datasets")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(alpha=0.2)
plt.savefig("results/linear_regression/m_sweep_with_and_without_energy_normalization.png", dpi=600, bbox_inches='tight')
#%%% print max. accuracy for both linear regression methods
print("Accuracy results of linear regression")
print(f"Without energy normalization: {np.max(acc_not_normed):.3f} @ m={ms[np.argmax(acc_not_normed)]}")
print(f"With energy normalization: {np.max(acc_normed):.3f} @ m={ms[np.argmax(acc_normed)]}")

#%%% Standard Training
m = 10
detector_positions = (33, 66)


### CONFIG + MESH
cfg = TrainConfig(m_side=m, theta_enc=1, normalize_energy=True, encoding='amplitude', 
        detectors=detector_positions, loss_kind='mse', learning_rate=0.1, batch_size=64,
                  init="haar", max_epochs=10, patience=20, min_delta=1e-4, 
                  param_init_seed=1550, eta_bs=1.0, alpha_fiber=0.0)
mesh = MZIMesh(cfg.N, plan_rectangular(cfg.N, cfg.N))

print(cfg)

trainer = Trainer(mesh, cfg)
history = trainer.fit(E_train, Y_train)

test_loss, test_acc = trainer.evaluate(E_test, Y_test)
print(f"validation acc {history['acc'][-1]:.4f} | test acc {test_acc:.4f} | "
      f"{len(history['loss'])} epochs, {trainer.train_time:.1f}s")
print(f"inference: {trainer.inference_time(E_test)*1e3:.2f} ms/sample")

fig, _ = plot_training(trainer)

#%%Haar vs random
m = 10
detector_positions = (33, 66)

initial=["haar", "random"]
trainers = []
for i in initial:
    cfg = TrainConfig(m_side=m, theta_enc=1, normalize_energy=True, encoding='amplitude', 
            detectors=detector_positions, loss_kind='mse', learning_rate=0.1, batch_size=64,
                      init=i, max_epochs=35, patience=2, min_delta=1e-4, 
                      param_init_seed=1550, eta_bs=1.0, alpha_fiber=0.0)
    E_tr, y_tr, E_te, y_te = get_data(values, number=2000, m_side=m,
                                      theta_enc=1, normalize_energy=True,
                                      seed=1550, balanced=True, verbose=True)
    t = Trainer(MZIMesh(cfg.N, plan_rectangular(cfg.N, cfg.N)), cfg)
    t.fit(E_tr, y_tr)
    t.test_acc = t.evaluate(E_te, y_te)[1]  
    t.save(f"results/initialization/{i}.json", test_acc=t.test_acc)
    trainers.append(t)

#%%%
fig, _ = plot_training(trainers, initial, keys=("batch_loss", "loss", "acc", "grad_norm"), sweep_label="Init.")


#%%% Test for batch sizes

batch_sizes = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])

m = 10
detector_positions = (33, 66)

trainers = []
for size in batch_sizes:
    cfg = TrainConfig(m_side=m, theta_enc=1, normalize_energy=True, encoding='amplitude', 
            detectors=detector_positions, loss_kind='mse', learning_rate=0.1, batch_size=size,
                      init="haar", max_epochs=35, patience=2, min_delta=1e-4, 
                      param_init_seed=1550, eta_bs=1.0, alpha_fiber=0.0)
    E_tr, y_tr, E_te, y_te = get_data(values, number=2000, m_side=m,
                                      theta_enc=1, normalize_energy=True,
                                      seed=1550, balanced=True, verbose=True)
    t = Trainer(MZIMesh(cfg.N, plan_rectangular(cfg.N, cfg.N)), cfg)
    t.fit(E_tr, y_tr)
    t.test_acc = t.evaluate(E_te, y_te)[1]  
    t.save(f"results/batch_size/{size}.json", test_acc=t.test_acc)
    trainers.append(t)
    
#%%%
fig, _ = plot_training(trainers, batch_sizes, keys=("batch_loss", "loss", "acc", "grad_norm"))
#%%% Sweep over batch sizes
#Standard variables
init="haar"
m = 10
detector_positions = (33, 66)

#Sweep Array:
sweep=np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])                        #<----------------------------------------
sweep_label="batch_size"                  #<----------------------------------------

#Perform training
trainers = []
for s in sweep:
    cfg = TrainConfig(m_side=m, theta_enc=1, normalize_energy=True, encoding='amplitude', 
            detectors=detector_positions, loss_kind='mse', learning_rate=0.1, batch_size=s,
                      init=init, max_epochs=35, patience=2, min_delta=1e-4,     #patience=2 und min_delta=1e-4 hat sich für mich jetzt gut ergeben, vielleicht sogar nur 1e-3
                      param_init_seed=1550, eta_bs=1.0, alpha_fiber=0.0)
    E_tr, y_tr, E_te, y_te = get_data(values, number=2000, m_side=m,
                                      theta_enc=1, normalize_energy=True,
                                      seed=1550, balanced=True, verbose=True)
    t = Trainer(MZIMesh(cfg.N, plan_rectangular(cfg.N, cfg.N)), cfg)
    t.fit(E_tr, y_tr)
    t.test_acc = t.evaluate(E_te, y_te)[1]  
    t.save(f"results/{sweep_label}/{s}.json", test_acc=t.test_acc)
    trainers.append(t)

#Plot trainingresults
fig, _ = plot_training(trainers, sweep, keys=("batch_loss", "loss", "acc", "grad_norm"), sweep_label=sweep_label)

for s, t in zip(sweep, trainers):
    print(f"m={m:3d}  N={m*m:4d}  epochs={len(t.history['loss']):3d}  "
          f"loss={t.history['loss'][-1]:.5f}  "
          f"train acc={t.history['acc'][-1]:.4f}  "
          f"test acc={t.test_acc:.4f}  {t.train_time:.0f}s")

#%%%
plt.savefig("results/{sweep_label}/plot_training.png", dpi=600, bbox_inches='tight')





#%%% Fixed ratio of batch_size and learning_rate
learning_rate_fixed = batch_sizes*0.1
batch_sizes = [16]


trainers_fixed = []
for size, rate in zip(batch_sizes, learning_rate_fixed):
    cfg = TrainConfig(m_side=m, theta_enc=1, normalize_energy=True, encoding='amplitude', 
            detectors=detector_positions, loss_kind='mse', learning_rate=rate, batch_size=size,
                      init="random", max_epochs=35, patience=2, min_delta=1e-4, 
                      param_init_seed=1550, eta_bs=1.0, alpha_fiber=0.0)
    E_tr, y_tr, E_te, y_te = get_data(values, number=200, m_side=m,
                                      theta_enc=1, normalize_energy=True,
                                      seed=1550, balanced=True, verbose=True)
    t = Trainer(MZIMesh(cfg.N, plan_rectangular(cfg.N, cfg.N)), cfg)
    t.fit(E_tr, y_tr)
    t.test_acc_fixed = t.evaluate(E_te, y_te)[1]      
    trainers_fixed.append(t)

#%%%
fig, _ = plot_training(trainers_fixed, batch_sizes, keys=("batch_loss", "loss", "acc", "grad_norm"))



# runs = {"history": history, "test_acc": test_acc,
#                "train_time": trainer.train_time, "cfg": cfg} #for loop over multiple 
# runss={}
# runss[0]=runs
# plot_runs(runss, keys=("loss", "acc", "grad_norm"), label="m")

# ms = [4, 6, 8, 10, 14,20,26]
# trainers = []
# for m in ms:
#     cfg = TrainConfig(m_side=m, learning_rate=0.3, batch_size=64, param_init_seed=0, max_epochs=3)
#     E_tr, y_tr, E_te, y_te = get_data(values, number=200, m_side=m,
#                                       theta_enc=1, normalize_energy=True,
#                                       seed=1550, balanced=True, verbose=True)
#     t = Trainer(MZIMesh(cfg.N, plan_rectangular(cfg.N, cfg.N)), cfg)
#     t.fit(E_tr, y_tr)
#     t.test_acc = t.evaluate(E_te, y_te)[1]      
#     trainers.append(t)

# fig, _ = plot_training(trainers, ms, "m")

# # Einzelner Lauf
# fig, _ = plot_training(trainers[0])











# #%% To be deleted when cleaned up
# ### PREPARE TRAINING DATA

# values = np.array([1,7]) #figures you want from the mnist dataset
# mode = "Training" #Selects if you want the test-set ("Testing") or the training-set ("Training")
# number = 1000 #Sets number of samples you want to get in total
# m_side = 10 #side length (pixel) of mnist image after downsampling
# theta_enc = 1 #mysterious hyper parameter for amplitude scaling
# norm_energy = True #Bool for if energy of encoded image should be normalized or not
# seed = None #Random seed to control random choice of number -pictures out of the available ones, 
#             #seed = None leads to random results for each iteration

# E_X, Y = get_data(values, mode, number, m_side, theta_enc, norm_energy, seed) #E_X are the flattened arrays of the encoded mnist images (complex valued), 
#                                                                   #Y are the referring labels

# N=100 #number of channels (number of pixels in mnist image)
# prop_mesh = MZIMesh(N, plan_rectangular(N, N)) #creating mesh object
# thetas, phis = prop_mesh.init_random() #initializing random weights

# layers = prop_mesh.layer_matrices_separate(thetas, phis)   #build transfer matrices (layers) from parameters

# ### PREPARE TESTING DATA

# values = np.array([1,7]) #figures you want from the mnist dataset
# mode = "Testing" #Selects if you want the test-set ("Testing") or the training-set ("Training")
# number_t = 1000 #Sets number of samples you want to get in total
# m_side = 10 #side length (pixel) of mnist image after downsampling
# theta_enc = 1 #mysterious hyper parameter for amplitude scaling
# norm_energy = True #Bool for if energy of encoded image should be normalized or not
# seed = None #Random seed to control random choice of number -pictures out of the available ones, 
#             #seed = None leads to random results for each iteration
# E_X_test, Y_test = get_data(values, mode, number_t, m_side, theta_enc, norm_energy, seed) #E_X are the flattened arrays of the encoded mnist images (complex valued), 



# # TRAINING (with testing)
# learning_rate = 1e-2

# for j in range(number):
    
#     if j % 100 == 0: 
#         E_out_test = forward(E_X_test, layers)
#         print(accuracy(E_out_test, Y_test, 33, 66))
    
#     #print(j)
#     # 1. Forward propagation
#     E_out_f1 = forward_history(E_X[:,j], layers)
#     I1 = np.abs(E_out_f1[:,:])**2
#     # if j%50 ==0:
#     #     fig, _ = plot_intensity_map(I1, detectors=[33,66])
    
#     # 2. Calculate error vector
#     E_in_b1 = adjoint_source(E_out_f1[-1], Y[j], 33, 66, kind="mse_norm")[:,0]
    
#     # 3. Propagate backwards
#     E_out_b1 = backward_history(E_in_b1, layers)
#     I2 = np.abs(E_out_b1[:,:])**2
#     # if j%50 ==0:
#     #     fig, _ = plot_intensity_map(I2, detectors=[33,66])
    
#     # 4. Calculate gradient
#     gradient = -2*np.imag(E_out_f1 * E_out_b1)
    
#     # 5. Update weights
#     for i in range(len(thetas)):
#         #print(i)
#         if i%2 == 0:
#             thetas[i] = thetas[i] - learning_rate * gradient[1+2*i,::2]
#             phis[i] = phis[i] - learning_rate * gradient[2+2*i,::2]
#         else:
#             thetas[i] = thetas[i] - learning_rate * gradient[1+2*i,1:-1:2]
#             phis[i] = phis[i] - learning_rate * gradient[2+2*i,1:-1:2]
#     layers = prop_mesh.layer_matrices_separate(thetas, phis)   #build transfer matrices (layers) from parameters

    
    
# ### ACTUAL TESTING
# E_out_test = forward(E_X_test, layers)
# print(accuracy(E_out_test, Y_test, 33, 66))

# ### EXAMPLE OUTPUTS
# for c in range(10):
#     E_out_test_plot = forward_history(E_X_test[:,c], layers)
#     I1 = np.abs(E_out_test_plot[:,:])**2
#     fig, ax = plot_intensity_map(I1, detectors=[33,66])
#     ax.set_title(Y_test[c])




# #%%
# #def main():    


# # if __name__ == "__main__":
# #     main()

# # #%%%
# # # ----------------------------------------------------------------------
# # # 1. Daten vorbereiten: Train / Val / Test getrennt
# # # ----------------------------------------------------------------------
# # # get_data liefert (N, k) und Labels. Encoding-Parameter wie gehabt.
# # E_train, y_train = get_data(values, "Training", number=1400, m_side=10,
# #                             theta=1, normalize_energy=True, seed=0)
# # E_pool,  y_pool  = get_data(values, "Testing",  number=None,  m_side=10,
# #                             theta=1, normalize_energy=True, seed=0)

# # # Testpool in Validierung und Test aufteilen (einmalig, fester Seed)
# # rng = np.random.default_rng(0)
# # perm = rng.permutation(E_pool.shape[1])
# # half = len(perm) // 2
# # val_idx, test_idx = perm[:half], perm[half:]
# # E_val,  y_val  = E_pool[:, val_idx],  y_pool[val_idx]
# # E_test, y_test = E_pool[:, test_idx], y_pool[test_idx]

# # # ----------------------------------------------------------------------
# # # 2. Ein einzelner Trainingslauf
# # # ----------------------------------------------------------------------
# # N = 100
# # mesh = MZIMesh(N, plan_rectangular(N, N))

# # cfg = TrainConfig(
# #     loss_kind="mse_norm",
# #     learning_rate=1e-2,
# #     detectors=(33, 66),
# #     init="haar",
# #     seed=0,
# # )

# # trainer = Trainer(mesh, cfg)
# # history = trainer.fit(E_train, y_train, E_val, y_val)

# # # Test genau EINMAL, mit den besten (nicht den letzten) Parametern
# # test_loss, test_acc = trainer.evaluate(E_test, y_test)
# # print(f"Val acc {trainer.best['val_acc']:.3f} | "
# #       f"Test acc {test_acc:.3f} | "
# #       f"stopped at epoch {trainer.best['epoch']} | "
# #       f"{trainer.train_time:.1f}s")

# # # Lernkurve fürs Protokoll
# # fig, _ = plot_learning_curves(history)   # train_loss, val_loss, val_acc
# # # %%

# %%
