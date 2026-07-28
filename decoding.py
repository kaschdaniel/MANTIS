import numpy as np


def detection(E):
    '''
    

    Parameters
    ----------
    E : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return np.abs(E)**2


def determine_winner(E, det1=0, det2=-1):
    '''
    

    Parameters
    ----------
    E : TYPE
        DESCRIPTION.
    det1 : TYPE, optional
        DESCRIPTION. The default is 0.
    det2 : TYPE, optional
        DESCRIPTION. The default is -1.

    Returns
    -------
    int
        DESCRIPTION.

    '''
    if E[det1] > E[det2]:
        return 0
    else:
        return 1
    
# def loss_function(E, y, det1, det7):
#     """
#     Accepts batch of evaluations at once
#     """
#     dim = E.ndim
#     if dim==3:#Full history E-given
#         Efinal = E[-1,:,:] #form: [channel, batch]
#     elif dim==2:
#         Efinal = E
#     else:
#         raise ValueError("Dimension of E faulty, ndim=2 or ndim=3 expected!")
#     I = np.abs(Efinal)**2 #shape (channel, batch)
#     scores = I[[det1, det7], :]
#     target = np.zeros_like(scores)   # (2, batch)
#     target[0, y == 1] = 1
#     target[1, y == 7] = 1

#     #calc difference to target1
#     print(np.shape(I[det1,:]))
#     print("")
#     diff1=target[0]- I[det1,:]
#     diff7=target[1]-I[det7,:]

#     loss = (diff1**2+diff7**2)

#     return loss

def loss_function(E, y, det1, det7, kind="mse"):
    """Loss + Adjoint-Quelle fuer den Rueckwaertspass.
 
    E    : (channel, batch) oder (L+1, channel, batch)
    y    : Labels (batch,), Werte 1 oder 7
    kind : "mse" | "softmax" | "mse_norm"  (Loss-Wahl, s.u.)
 
    Rueckgabe
    ---------
    loss     : Skalar, Mittel ueber den Batch
    per_samp : (batch,) Loss je Sample (fuer Kurven/Debugging)
    lam_out  : (channel, batch) komplexe Adjoint-Quelle dL/dE*  am Ausgang.
               Nur die Detektorkanaele sind ungleich null. Das ist das
               lambda_L, mit dem der Rueckwaertspass startet.
    """

    def _outputs(E):
        """Ausgangsfeld (channel, batch) aus (channel,batch) oder (L+1,channel,batch)."""
        if E.ndim == 3:
            return E[-1]
        elif E.ndim == 2:
            return E
        raise ValueError("E muss ndim 2 oder 3 haben")

    Efield = _outputs(E)                       # (N, B) komplex
    N, B = Efield.shape
    I = np.abs(Efield)**2                       # (N, B) reell
 
    s1 = I[det1]                                # (B,) Score Detektor 1
    s7 = I[det7]                                # (B,) Score Detektor 7
    t1 = (y == 1).astype(float)                 # Ziel an det1
    t7 = (y == 7).astype(float)                 # Ziel an det7
 
    # dL/dI an den beiden Detektoren -> spaeter via Kettenregel auf E*
    # Fuer I = |E|^2 gilt dI/dE* = E, also dL/dE* = (dL/dI) * E  am Detektor.
    lam_out = np.zeros_like(Efield)
 
    # ============================ Loss-Wahl ============================
    if kind == "mse":
        # quadratischer Abstand der ROHEN Detektorintensitaeten zum Ziel.
        # Einfachste Wahl, Adjoint-Quelle direkt ablesbar.
        d1, d7 = s1 - t1, s7 - t7
        per_samp = d1**2 + d7**2
        # dL/dI = 2*(s - t)  am jeweiligen Detektor
        lam_out[det1] = 2.0 * d1 * Efield[det1]
        lam_out[det7] = 2.0 * d7 * Efield[det7]
 
    # ---- weitere Optionen: einkommentieren und kind="..." waehlen ----
    elif kind == "mse_norm":
        # MSE auf normierten Scores p = s / (s1+s7). Verlust-/energieinvariant,
        # weil nur das VERHAELTNIS der beiden Detektoren zaehlt.
        tot = s1 + s7 + 1e-12
        p1, p7 = s1 / tot, s7 / tot
        d1, d7 = p1 - t1, p7 - t7
        per_samp = d1**2 + d7**2
        # dL/ds1 und dL/ds7 ueber Quotientenregel:
        #   dp1/ds1 =  s7/tot^2,  dp1/ds7 = -s1/tot^2  (analog p7)
        dL_dp1, dL_dp7 = 2*d1, 2*d7
        dL_ds1 = dL_dp1 * ( s7/tot**2) + dL_dp7 * (-s7/tot**2)
        dL_ds7 = dL_dp1 * (-s1/tot**2) + dL_dp7 * ( s1/tot**2)
        lam_out[det1] = dL_ds1 * Efield[det1]
        lam_out[det7] = dL_ds7 * Efield[det7]
 
    elif kind == "softmax":
        # Cross-Entropy ueber die zwei Detektoren als Logits z = (s1, s7).
        # Robust, bestraft nur die relative Zuordnung.
        z = np.stack([s1, s7], axis=0)          # (2, B)
        z = z - z.max(axis=0, keepdims=True)    # numerisch stabil
        ez = np.exp(z)
        soft = ez / ez.sum(axis=0, keepdims=True)   # (2, B)
        tgt = np.stack([t1, t7], axis=0)            # (2, B) one-hot
        per_samp = -(tgt * np.log(soft + 1e-12)).sum(axis=0)
        # dL/dz = softmax - target   (Standard-Resultat)
        dL_ds1 = soft[0] - t1
        dL_ds7 = soft[1] - t7
        lam_out[det1] = dL_ds1 * Efield[det1]
        lam_out[det7] = dL_ds7 * Efield[det7]
 
    else:
        raise ValueError(f"unbekannte Loss-Wahl: {kind!r}")
        #
    
    loss = per_samp.mean()
    lam_out = lam_out / B #for backprop.
    return loss, per_samp, lam_out #returns shape(?, batch, (channel, batch))