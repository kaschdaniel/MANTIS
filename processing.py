# PROCESSING
import numpy as np

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

def forward(E_in, layers): #for propagation through given layers, returns only last E-Values
    """E_in: (N,) fuer ein Sample oder (N, B) fuer einen Batch.
    layers: Liste der N x N Layer-Matrizen (aus mesh.layer_matrices()).
    Batch funktioniert ohne Sonderfall, weil M @ E in NumPy
    fuer Vektoren und Matrizen dasselbe tut."""
    E = np.asarray(E_in, dtype=np.complex128)
    for Ml in layers:
        E = Ml @ E
    return E


def forward_history(E_in, layers): #for propagation through given layers, returns all E-Values after each layer
    """Wie forward(), gibt aber alle Zwischenzustaende zurueck.
    Rueckgabe: (L+1, N) bzw. (L+1, N, B), Eintrag l ist das Feld VOR Layer l."""
    E = np.asarray(E_in, dtype=np.complex128)
    hist = [E.copy()]
    for Ml in layers:
        E = Ml @ E
        hist.append(E.copy())
    return np.array(hist)

