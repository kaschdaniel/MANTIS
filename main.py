from encoding import *
#from processing import *
from decoding import *

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist



def linear_regression(X_train, y_train, X_test, y_test):
    # MODEL TRAINING
    F = np.vstack([np.ones(len(X_train)), X_train.T]).T
    # Calculate Moore-Penrose pseudoinverse
    theta = np.linalg.pinv(F) @ y_train

    # MODEL TESTING
    y_predict = theta[0] + np.sum(theta[1:] * X_test)
        
    mse = calc_mse(y_test, y_predict)
    
    return theta, y_predict, mse

def calc_mse(y, y_predict):
    return np.sum((y-y_predict)**2)/len(y)
    
    

def main():
    #%% Loading the dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    #printing the shapes of the vectors 
    print('X_train: ' + str(X_train.shape))
    print('Y_train: ' + str(y_train.shape))
    print('X_test:  '  + str(X_test.shape))
    print('Y_test:  '  + str(y_test.shape))
    
    
    for i in range(9):  
        plt.subplot(330 + 1 + i)
        plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
        
    
    
    
    
    #%% Selecting only images of class 1 and 7
    
    X_train = X_train[(y_train == 1) | (y_train == 7)]
    y_train = y_train[(y_train == 1) | (y_train == 7)]
    print(np.shape(X_train))
    
    X_test = X_test[(y_test == 1) | (y_test == 7)]
    print(np.shape(X_test))
    y_test = y_test[(y_test == 1) | (y_test == 7)]
    
    
    #%% Preprocessing of images to normalized vector
    
    ### Test for 1 image ###
    image = X_train[0]
    #print(len(image))
    
    image_downsampled = down_sample(image, m=10)
    
    #fig, ax = plt.subplots()
    #ax.imshow(image_downsampled, cmap=plt.get_cmap('gray'))
        
    #print(len(image_downsampled))
    
    image_vector = reshape_and_normalize(image_downsampled)
    
    # For all training and test data
    
    # Downsample
    X_train_ds = [down_sample(X, m=10) for X in X_train]
    X_test_ds = [down_sample(X, m=10) for X in X_test]
    #print(np.shape(X_train_ds))
    
    # Resize to vector and normalize
    X_train_vec = np.array([reshape_and_normalize(X) for X in X_train_ds])
    X_test_vec = np.array([reshape_and_normalize(X) for X in X_test_ds])
    #print(np.shape(X_train_vec))
    
    
    
    
    #%% Linear regression as baseline
    
    theta, y_predict, mse = linear_regression(X_train_vec, y_train, X_test_vec, y_test)
    print(mse)
    
    
    
    #%% Optical Machine Learning Solution
    
    #%%% Encoding
    
    
    E = amplitude_encoding(image_vector, theta=1)
    print(np.shape(E))
    
    #%%% Processing
    
    
    
    
    #%%% Decoding
    
    detected_signal = detection(E)
    y_pred = determine_winner(detected_signal)
    print(y_pred)
    
    


if __name__ == "__main__":
    main()
