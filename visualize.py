import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def plot_intensity_map(intensities, detectors=None, ax=None, title=None,
                       cmap="inferno", log=False):
    """Intensitaetsverteilung |E|^2 als Heatmap.
 
    Parameter
    ---------
    intensities : array (L+1, N)
        z.B. np.abs(forward_all_layers_with_history(E_in, params))**2
        Zeile s = Feldzustand nach s Schichten (s=0 ist der Eingang).
    detectors : list[int], optional
        Kanalindizes, die am rechten Rand markiert werden.
    log : bool
        Logarithmische Farbskala. Sinnvoll, wenn das Feld stark
        lokalisiert ist (grosser Dynamikbereich durch den Lichtkegel).
    """
    I = np.asarray(intensities, dtype=float)
    L1, N = I.shape                            # L1 = L+1
 
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, 0.09*L1 + 4), max(3, 0.05*N + 3)))
    else:
        fig = ax.figure
 
    data = I.T                                 # -> (N, L+1): Kanal auf y-Achse
    if log:
        data = np.log10(np.maximum(data, 1e-12))
 
    im = ax.imshow(data, aspect="auto", origin="upper", cmap=cmap,
                   interpolation="nearest",
                   extent=[-0.5, L1 - 0.5, N - 0.5, -0.5])
 
    if detectors is not None:
        for d in detectors:
            ax.plot(L1 - 0.5, d, marker="<", ms=10, color="lime",
                    clip_on=False, zorder=5)
            ax.text(L1 - 0.2, d, f"det {d}", va="center", fontsize=8,
                    color="lime", clip_on=False)
 
    ax.set_xlabel("layer  (0 = input field)")
    ax.set_ylabel("channel $k$")
    ax.set_title(title or r"field intensity $|E|^2$")
 
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(r"$\log_{10}|E|^2$" if log else r"$|E|^2$", fontsize=9)
 
    fig.tight_layout()
    return fig, ax