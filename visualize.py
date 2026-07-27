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


def plot_mesh(mesh, ax=None, color_by=None, detectors=None,
              label_step=None, figsize=None, title=None):
    """Zeichnet die Struktur eines MZIMesh.
 
    mesh       : MZIMesh-Objekt (nur .N und .plan werden gebraucht)
    color_by   : None    -> alle MZIs gleich eingefaerbt
                 "alpha" -> nach Sensitivitaetsindex (Pai Sec. III A)
                 "layer" -> nach Layer-Index, hilft beim Abzaehlen
    detectors  : Liste von Kanal-Indizes, die als Detektoren markiert werden
    label_step : Beschriftung jedes n-ten Kanals. None -> automatisch
    """
    N = mesh.N
    plan = mesh.plan
    L = len(plan)
 
    if ax is None:
        if figsize is None:
            figsize = (max(6.0, 0.32*L), max(3.0, 0.16*N))
        fig, ax = plt.subplots(figsize=figsize)
 
    # --- 1) Wellenleiter als Hintergrundlinien --------------------------------
    for i in range(N):
        ax.plot([-0.6, L - 0.4], [i, i], color="0.88", lw=0.8, zorder=0)
 
    # --- 2) Farbskala vorbereiten --------------------------------------------
    cmap = plt.get_cmap("viridis")
    if color_by == "alpha":
        alphas = mesh.sensitivity_index()
        vmax = max((a.max() for a in alphas if len(a) > 0), default=1)
        vmin = min((a.min() for a in alphas if len(a) > 0), default=0)
    else:
        alphas = None
 
    # --- 3) Layer durchgehen --------------------------------------------------
    for l, (kind, data) in enumerate(plan):
 
        if kind == "perm":
            # perm ist so definiert, dass (P @ E)[j] = E[perm[j]].
            # Kanal i landet also auf Position inv[i].
            inv = np.argsort(data)
            for i in range(N):
                j = inv[i]
                moved = (i != j)
                ax.plot([l - 0.42, l + 0.42], [i, j],
                        color="tab:orange" if moved else "0.88",
                        lw=1.0 if moved else 0.8,
                        zorder=2 if moved else 0)
            continue
 
        for i_slot, k in enumerate(data):
            if color_by == "alpha":
                c = cmap((alphas[l][i_slot] - vmin) / max(vmax - vmin, 1))
            elif color_by == "layer":
                c = cmap(l / max(L - 1, 1))
            else:
                c = "tab:red"
 
            # Verbinder zwischen Kanal k und k+1 ...
            ax.plot([l, l], [k, k + 1], color=c, lw=1.4, zorder=2)
            # ... plus Punkt in der Mitte als eigentliches MZI
            ax.plot([l], [k + 0.5], marker="o", ms=4.5,
                    color=c, zorder=3)
 
    # --- 4) Detektoren markieren ---------------------------------------------
    if detectors is not None:
        for d in detectors:
            ax.plot([L - 0.4], [d], marker="s", ms=7,
                    color="tab:blue", zorder=4, clip_on=False)
            ax.annotate(f"det {d}", (L - 0.25, d), va="center",
                        fontsize=8, color="tab:blue", annotation_clip=False)
 
    # --- 5) Achsen ------------------------------------------------------------
    if label_step is None:
        label_step = 1 if N <= 20 else max(1, N // 10)
    ticks = list(range(0, N, label_step))
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks, fontsize=8)
    ax.set_ylabel("Channel")
 
    x_step = 1 if L <= 20 else max(1, L // 12)
    ax.set_xticks(list(range(0, L, x_step)))
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("Layer")
 
    ax.set_xlim(-0.9, L - 0.1)
    ax.set_ylim(N - 0.5, -0.5)          # Kanal 0 oben, wie im Setup-Bild
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
 
    n_mzi = sum(len(d) for kind, d in plan if kind == "mzi")
    ax.set_title(title if title is not None
                 else f"N={N}, {L} Layer, {n_mzi} MZIs", fontsize=10)
    return ax