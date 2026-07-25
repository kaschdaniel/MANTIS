# PROCESSING
import numpy as np

#################################################################################################
#----------------------------Basic functions-----------------------------------------------------

def beamsplitter():
    return (1/np.sqrt(2)) * np.array([[1, 1j],
                                    [1j, 1]], dtype=np.complex128) #Datatype not optimal for GPU Acceleration

def phase_shift(alpha): #Phaseshift for upper mode
    return np.array([[np.exp(1j*alpha), 0],
                     [0, 1]], dtype=np.complex128)

def U(theta, phi): #2x2 Transfer-matrix of MZI: R_phi @ B @ R_theta @ B 
    B = beamsplitter()
    return phase_shift(phi) @ B @ phase_shift(theta) @ B

#################################################################################################
#----------------------------MZI Array Logic-----------------------------------------------------

def build_layer(thetas, phis, pos=0):
    # Guard-logic
    n=len(thetas)
    if n != len(phis):
        raise ValueError("Arrays must have the same length!")
    pos_is_even: bool = pos % 2 == 0 #check if position of layer is even or not
    #Builds matrix for one layer of MZI
    M = np.zeros((2*n, 2*n), dtype=np.complex128)
    if pos_is_even:
        for i in range(n):
            M[2*i:2*i+2, 2*i:2*i+2] = U(thetas[i], phis[i])
    else:
        for i in range(n-1):
            M[2*i+1:2*i+3, 2*i+1:2*i+3] = U(thetas[i], phis[i])
    return M 

def apply_layer(E_in, M): #Applies one Mesh-layer
    E = E_in.copy()
    E_out = M@E
    return E_out

#################################################################################################
#----------------------------Prepare/set weights-------------------------------------------------

#random generate for length l and width w


#set given values (in Matrix form)


