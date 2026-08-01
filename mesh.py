"""
mesh.py -- Aufbau und Propagation von MZI-Meshes.

Grundidee:
Ein Mesh ist nichts weiter als eine LISTE VON LAYERN. Ein Layer ist ein Tupel

    ("mzi", ks)     ks = Liste der oberen Wellenleiter-Indizes, z.B. [0, 2, 4]
                    -> MZIs sitzen auf den Paaren (0,1), (2,3), (4,5)

    ("perm", perm)  perm = Index-Array, feste Verdrahtung ohne Parameter
                    -> nur fuer die permuting-Architektur (PRM)

Die verschiedenen Architekturen (rectangular, triangular, redundant, permuting)
unterscheiden sich AUSSCHLIESSLICH in dieser Liste. Der Code, der daraus
Matrizen macht, ist fuer alle derselbe. Eine neue Architektur = eine neue
plan_*() Funktion, sonst nichts.
"""

import numpy as np


#################################################################################################
#----------------------------Basic functions-----------------------------------------------------
#################################################################################################

def beamsplitter(eta=1.0):
    """eta in [0,1]: Leistungstransmission des Strahlteilers.
    eta=1 -> verlustfrei (unitaerer Standard-BS)."""
    B_ideal = (1/np.sqrt(2)) * np.array([[1, 1j],
                                         [1j, 1]], dtype=np.complex128)
    return np.sqrt(eta) * B_ideal

def phase_shift(alpha): #Phaseshift for upper mode
    return np.array([[np.exp(1j*alpha), 0],
                     [0, 1]], dtype=np.complex128)

def U(theta, phi, eta=1.0):  # 2x2 Transfer-matrix of MZI: R_phi @ B @ R_theta @ B
    """eta: Leistungstransmission je Strahlteiler. Zwei BS pro MZI
    -> Gesamt-MZI-Transmission eta**2. eta=1 -> verlustfrei/unitaer."""
    B = beamsplitter(eta)
    return phase_shift(phi) @ B @ phase_shift(theta) @ B

def U_theta(theta, eta=1.0):
    B = beamsplitter(eta)
    return phase_shift(theta) @ B

def U_phi(phi, eta=1.0):
    B = beamsplitter(eta)
    return phase_shift(phi) @ B

#################################################################################################
# ----------------------------Bauplaene (Architekturen)------------------------------------------
#################################################################################################

def plan_rectangular(N, L):
    """Clements / rectangular mesh: L Layer, abwechselnd Offset 0 und 1.
    Universell fuer L = N. Rueckgabe: Liste von ("mzi", ks)."""
    plan = []
    for l in range(L):
        start = l % 2
        ks = list(range(start, N - 1, 2))
        plan.append(("mzi", ks))
    return plan


def plan_redundant(N, extra_layers):
    """Redundant rectangular mesh (RRM, Pai Sec. IV A):
    identisch zum rectangular mesh, nur mit mehr Layern als noetig.
    Mehr Parameter -> deutlich bessere Konvergenz."""
    return plan_rectangular(N, N + extra_layers)


def plan_triangular(N):
    """Reck / triangular mesh: 2N-3 Layer, insgesamt N(N-1)/2 MZIs.

    Ein MZI sitzt auf Paar (k, k+1) im Layer l genau dann, wenn
        (1) k und l dieselbe Paritaet haben   -> MZIs ueberlappen nicht
        (2) k >= |l - (N-2)|                  -> schneidet das Dreieck aus

    Bedingung (2) laesst die Layer von der Breite 1 auf N/2 in der Mitte
    (Layer l = N-2, dort sitzt der 'apex' bei k=0) anwachsen und wieder
    schrumpfen. Damit ist jedes Layer unterschiedlich breit -- deshalb die
    Speicherung der Gewichte als Liste statt als (w, L)-Array.
    """
    plan = []
    for l in range(2*N - 3):
        k_min = abs(l - (N - 2))
        ks = [k for k in range(k_min, N - 1) if k % 2 == l % 2]
        plan.append(("mzi", ks))
    return plan


def _rect_permutation(N, k):
    """Hilfsfunktion: eine feste Verdrahtung P_k, die Wellenleiter um 2^k
    verschiebt (Pai Sec. IV B). Innerhalb jedes Blocks der Groesse 2^(k+1)
    werden die beiden Haelften getauscht.

    Achtung: das Paper legt P_k nicht eindeutig fest, das hier ist eine
    konkrete, plausible Wahl.
    """
    shift = 2**k
    block = 2*shift
    perm = np.arange(N)
    for start in range(0, N - block + 1, block):
        idx = perm[start:start+block].copy()
        perm[start:start+block] = np.concatenate([idx[shift:], idx[:shift]])
    return perm


def plan_permuting(N, K=None):
    """Permuting rectangular mesh (PRM, Pai Sec. IV B):
    K Bloecke aus je ceil(N/K) rectangular-Layern, dazwischen feste
    Permutationen P_1 ... P_(K-1). Die Permutationen lassen weit entfernte
    Wellenleiter miteinander wechselwirken -> kein banded unitary."""
    if K is None:
        K = int(np.ceil(np.log2(N)))
    per_block = int(np.ceil(N / K))
    plan = []
    for k in range(K):
        plan += plan_rectangular(N, per_block)
        if k < K - 1:
            plan.append(("perm", _rect_permutation(N, k + 1)))
    return plan


#################################################################################################
# ----------------------------Mesh-Class---------------------------------------------------------
#################################################################################################

class MZIMesh:
    """Haelt die ARCHITEKTUR (N + Bauplan), NICHT die Gewichte.

    Die Gewichte werden bewusst von aussen uebergeben. Das kostet ein paar
    Zeichen mehr beim Aufruf, macht aber Gradientenchecks und
    Finite-Differenzen trivial: man ruft die Funktion einfach mit gestoerten
    Parametern auf, statt Objektzustand hin- und herzukopieren.

    Gewichte-Konvention:
        thetas, phis = Listen der Laenge L (= Anzahl Layer).
        thetas[l] ist ein 1D-Array mit einem Eintrag pro MZI in Layer l.
        Fuer perm-Layer ist der Eintrag ein leeres Array.
    """

    def __init__(self, N, plan):
        if N % 2 != 0:
            raise ValueError("N sollte gerade sein.")
        self.N = N
        self.plan = plan

    # -------------------------------------------------- Struktur-Infos
    @property
    def n_layers(self):
        return len(self.plan)

    @property
    def slot_counts(self):
        """Anzahl MZIs pro Layer, z.B. [4, 3, 4, 3, ...]."""
        return [len(data) if kind == "mzi" else 0 for kind, data in self.plan]

    @property
    def n_mzis(self):
        """Anzahl trainierbarer MZIs (jedes hat theta UND phi)."""
        return sum(self.slot_counts)

    # -------------------------------------------------- Initialisierung
    def init_random(self, rng=None):
        """Uniform initialization: theta in [0, pi], phi in [0, 2pi).
        """
        if rng is None:
            rng = np.random.default_rng()
        thetas = [rng.uniform(0, np.pi, n) for n in self.slot_counts] #creates as many random values for theta as there are needed in each layer
        phis = [rng.uniform(0, 2*np.pi, n) for n in self.slot_counts] #... for phi ..

        return thetas, phis #Returns python lists containing numpy arrays

    def init_haar(self, rng=None):
        """Haar initialization for uniformly random global transfer matrices.

        Sets the phase shifts using the sensitivity index alpha: 
        t = xi^(1/alpha), where xi ~ U(0,1). This ensures that input light 
        spreads evenly across all spatial modes instead of localizing along 
        the diagonal.
        """
        if rng is None:
            rng = np.random.default_rng()
        
        alphas = self.sensitivity_index()
        thetas, phis = [], []
        
        for a in alphas:
            xi = rng.uniform(0, 1, len(a))
            t = xi ** (1.0 / np.maximum(a, 1))
            thetas.append(2 * np.arccos(np.sqrt(t)))
            phis.append(rng.uniform(0, 2*np.pi, len(a)))
            
        return thetas, phis

    def sensitivity_index(self):
        """Calculates the structural sensitivity index for each MZI.
        
        The index indicates how much global influence a component has on the 
        optical network: |I| + |O| - N - 1. |I| and |O| are the number of 
        physical inputs and outputs connected to the MZI.
        
        Determine connectivity topologically by propagating boolean reachability forward 
        and backward. This works universally for any mesh design.
        """
        N = self.N

        # forward pass: trace reachable inputs per MZI
        reach = np.eye(N, dtype=bool)
        I_counts = []
        for kind, data in self.plan:
            if kind == "perm":
                reach = reach[data]
                I_counts.append(np.zeros(0, dtype=int))
                continue
            counts = []
            for k in data:
                both = reach[k] | reach[k+1]
                counts.append(int(both.sum()))
            for i, k in enumerate(data):          # count before mixing the paths
                both = reach[k] | reach[k+1]
                reach[k] = both
                reach[k+1] = both
            I_counts.append(np.array(counts))

        # backward pass: trace reachable outputs per MZI
        reach = np.eye(N, dtype=bool)
        O_counts = []
        for kind, data in reversed(self.plan):
            if kind == "perm":
                inv = np.argsort(data)
                reach = reach[inv]
                O_counts.append(np.zeros(0, dtype=int))
                continue
            counts = []
            for k in data:
                both = reach[k] | reach[k+1]
                counts.append(int(both.sum()))
            for k in data:
                both = reach[k] | reach[k+1]
                reach[k] = both
                reach[k+1] = both
            O_counts.append(np.array(counts))
        O_counts = O_counts[::-1]

        return [Ic + Oc - N - 1 for Ic, Oc in zip(I_counts, O_counts)]

    # -------------------------------------------------- Matrizen
    def layer_matrices(self, thetas, phis, eta_bs=1.0, alpha_fiber=0.0):
        """Liste der N x N Matrizen, eine pro Layer.

        eta_bs      : Leistungstransmission je Strahlteiler (BS-Verlust).
                      1.0 -> verlustfrei.
        alpha_fiber : Wellenleiterverlust in dB pro Layer-Abschnitt.
                      0.0 -> verlustfrei. Wird als diagonaler Faktor an
                      JEDES Layer multipliziert (betrifft auch Kanaele
                      ohne MZI), getrennt vom BS-Verlust.

        Bewusst NICHT nur das Produkt: die adjungierte Backpropagation
        braucht die Felder an jedem einzelnen Layer, und die Feld-
        visualisierung fuer den Bericht ebenfalls.
        """
        N = self.N
        # dB (Leistung) -> Amplitudenfaktor: 10^(-alpha/20)
        fiber_amp = 10.0 ** (-alpha_fiber / 20.0)
        mats = []
        for (kind, data), th, ph in zip(self.plan, thetas, phis):
            if kind == "perm":
                M = np.eye(N, dtype=np.complex128)[data]
            else:
                M = np.eye(N, dtype=np.complex128)
                for i, k in enumerate(data):
                    M[k:k+2, k:k+2] = U(th[i], ph[i], eta_bs)
            mats.append(fiber_amp * M)   # Fiber-Verlust: getrennter Diagonalfaktor
        return mats
    
    # -------------------------------------------------- Matrizen
    def layer_matrices_separate(self, thetas, phis, eta_bs=1.0, alpha_fiber=0.0):
        """Generates the N x N optical transfer matrices for each phase shifter layer separately.

        eta_bs      : Power transmission coefficient per beamsplitter (1.0 = lossless).
        alpha_fiber : Waveguide propagation loss in dB per layer section. Applied
                    uniformly as an amplitude reduction factor across all channels.
        """
        N = self.N
        
        # Convert power loss in dB to field amplitude factor
        fiber_amp = 10.0 ** (-alpha_fiber / 20.0)
        mats = []
        
        for (kind, data), th, ph in zip(self.plan, thetas, phis):
            if kind == "perm":
                M_theta = np.eye(N, dtype=np.complex128)[data]
                M_phi = M_theta.copy()
            else:
                M_theta = np.eye(N, dtype=np.complex128)
                M_phi = M_theta.copy()
                for i, k in enumerate(data):
                    M_theta[k:k+2, k:k+2] = U_theta(th[i], eta_bs)
                    M_phi[k:k+2, k:k+2] = U_phi(ph[i], eta_bs)
            
            # Apply uniform waveguide propagation loss to each sub-layer
            mats.append(fiber_amp * M_theta)
            mats.append(fiber_amp * M_phi)
            
        return mats