# PROCESSING
import numpy as np

#################################################################################################
#----------------------------Basic functions-----------------------------------------------------
#################################################################################################

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
#################################################################################################

def build_single_layer(thetas, phis, pos=0):
    """returns full N x N matrix of the layer of position "pos"
    pos even  -> pairs (0,1),(2,3),...   -> w   MZIs
    pos odd -> pairs (1,2),(3,4),...     -> w-1 MZIs
    """
    if len(thetas) != len(phis):
        raise ValueError("Arrays must have the same length!")
    N = 2 * len(thetas)                       
    M = np.eye(N, dtype=np.complex128)        
    start = 0 if pos % 2 == 0 else 1
    for i, k in enumerate(range(start, N - 1, 2)):
        M[k:k+2, k:k+2] = U(thetas[i], phis[i])
    return M

def forward_single_layer(E_in, M): #Applies one Mesh-layer
    E_out = M@E_in
    return E_out

def forward_all_layers_with_history(E_in, params):
    """params = (phis, thetas), beide Form (w, L) mit w = N//2.
    Gibt Array (L+1, N) aller Zwischenzustaende zurueck."""
    phis, thetas = params
    N = len(E_in)
    L = phis.shape[1]
    E = np.asarray(E_in, dtype=np.complex128).copy()
    hist = [E.copy()]
    for l in range(L):
        start = 0 if l % 2 == 0 else 1
        for i, k in enumerate(range(start, N - 1, 2)):
            E[k:k+2] = U(thetas[i, l], phis[i, l]) @ E[k:k+2]
        hist.append(E.copy())
    return np.array(hist)


