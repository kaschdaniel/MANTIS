# PROCESSING
import numpy as np
from mesh import U, U_theta, U_phi

#################################################################################################
#----------------------------MZI Array Logic-----------------------------------------------------
#################################################################################################

def forward(E_in, layers): #for propagation through given layers, returns only last E-Values
    """E_in: (N,) fuer ein Sample oder (N, B) fuer einen Batch.
    layers: Liste der N x N Layer-Matrizen (aus mesh.layer_matrices()).
    Batch funktioniert ohne Sonderfall, weil M @ E in NumPy
    fuer Vektoren und Matrizen dasselbe tut."""
    E = np.asarray(E_in, dtype=np.complex128)
    for Ml in layers:
        E = Ml @ E
    return E


def backward(lam_out, layers):
    """Adjoint-Feld rueckwaerts durch das Mesh.

    lam_out : Adjoint-Quelle am Ausgang (lam_out aus loss_function),
              (N,) oder (N, B).  NICHT zusaetzlich konjugieren -- die
              Konjugation steckt schon in der Definition dL/dE*.
    layers  : Layer-Matrizen in VORWAERTS-Reihenfolge, so wie
              mesh.layer_matrices() sie liefert.

    Rueckgabe: Adjoint-Feld am Mesh-Eingang, gleiche Form wie lam_out.
    """
    return forward(lam_out, adjoint_layers(layers))

def adjoint_layers(layers):
    """Transponiert (NICHT hermitesch!) und umgekehrte Reihenfolge.

    Hughes-Konvention: das Adjoint-Feld wird als Gamma = conj(dL/dE*)
    definiert und mit M^T propagiert.  Grund: ein reziprokes optisches
    System, rueckwaerts durchleuchtet, realisiert physikalisch M^T.
    Wegen  M^T conj(x) = conj(M^H x)  ist das aequivalent zu
    lambda = dL/dE* mit M^H -- ABER die beiden Konventionen duerfen
    nicht gemischt werden.  Der Fehlervektor MUSS also konjugiert
    hereinkommen (siehe adjoint_source in decoding.py).
    """
    return [Ml.T for Ml in reversed(layers)]

def forward_history(E_in, layers): #for propagation through given layers, returns all E-Values after each layer
    """Wie forward(), gibt aber alle Zwischenzustaende zurueck.
    Rueckgabe: (L+1, N) bzw. (L+1, N, B), Eintrag l ist das Feld VOR Layer l."""
    E = np.asarray(E_in, dtype=np.complex128)
    hist = [E.copy()]
    for Ml in layers:
        E = Ml @ E
        hist.append(E.copy())
    return np.array(hist)

def backward_history(lam_out, layers):
    """Wie backward(), gibt aber alle Adjoint-Felder zurueck.

    Rueckgabe: (L+1, N) bzw. (L+1, N, B) mit Eintrag l = lambda_l,
    also in DERSELBEN Indizierung wie forward_history.  Ohne das
    Umdrehen am Ende laeuft der Index rueckwaerts und passt nicht
    zu den Vorwaertsfeldern -- die haeufigste Fehlerquelle hier.
    """
    hist = forward_history(lam_out, adjoint_layers(layers))
    return hist[::-1]

def gradient(fh, bh, plan):
    """Gradienten aller theta und phi aus Vorwaerts- und Adjoint-Historie.

    fh, bh : Historien aus forward_history / backward_history, gebaut mit
             layer_matrices_separate -> Laenge 2L+1.
             Position 2l+1 = nach dem theta-Shifter von Layer l
             Position 2l+2 = nach dem phi-Shifter   von Layer l
    plan   : mesh.plan

    Rueckgabe: (grad_thetas, grad_phis), gleiche Struktur wie thetas/phis.
    Batch (N,B) wird ueber die Sample-Achse summiert.
    """
    grad_th, grad_ph = [], []
    for l, (kind, ks) in enumerate(plan):
        if kind != "mzi":
            grad_th.append(np.zeros(0)); grad_ph.append(np.zeros(0))
            continue
        gt = np.zeros(len(ks)); gp = np.zeros(len(ks))
        for i, k in enumerate(ks):
            # phase_shift wirkt auf den OBEREN Mode -> nur Kanal k traegt bei
            gt[i] = -2*np.imag(np.sum(bh[2*l+1][k] * fh[2*l+1][k]))
            gp[i] = -2*np.imag(np.sum(bh[2*l+2][k] * fh[2*l+2][k]))
        grad_th.append(gt); grad_ph.append(gp)
    return grad_th, grad_ph