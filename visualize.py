import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def plot_intensity_map(intensities, detectors=None, ax=None, title=None,
                       cmap="inferno", log=False):
    """Feldintensitaet |E|^2 als Heatmap (Kanal ueber Layer).

    intensities : (L+1, N), z.B. np.abs(forward_history(E_in, layers))**2
                  Zeile s = Feld nach s Layern (s=0 = Eingang).
    detectors   : Kanalindizes, am rechten Rand markiert.
    log         : logarithmische Farbskala bei stark lokalisiertem Feld.
    """
    I = np.asarray(intensities, dtype=float)
    if I.ndim != 2:
        raise ValueError(f"erwarte (L+1, N), bekommen {I.shape} "
                         "- bei Batch erst ein Sample waehlen: hist[:, :, b]")
    L1, N = I.shape

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, 0.09*L1 + 4), max(3, 0.05*N + 3)))
    else:
        fig = ax.figure

    data = np.log10(np.maximum(I.T, 1e-12)) if log else I.T
    im = ax.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest",
                   extent=[-0.5, L1 - 0.5, N - 0.5, -0.5])

    for d in (detectors or []):
        ax.plot(L1 - 0.5, d, marker="<", ms=10, color="forestgreen", clip_on=False, zorder=5)
        ax.text(L1 +0.5, d, f"det {d}", va="center",rotation='vertical', fontsize=8, color="forestgreen", clip_on=False)

    ax.set_xlabel("layer  (0 = input field)")
    ax.set_ylabel(r"channel $k$")
    ax.set_title(title or r"field intensity $|E|^2$")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(r"$\log_{10}|E|^2$" if log else r"$|E|^2$", fontsize=9)
    fig.tight_layout()
    return fig, ax


def plot_intensity_map_with_histogram(I, det1, det7, Y,
                                      cmap="inferno", log=False,
                                      class_colors=("tab:cyan", "tab:orange"),
                                      target_color="red", title=None):
    """Intensitaetskarte + gedrehtes Histogramm der Ausgangsintensitaeten.
 
    I    : (L+1, N) fuer ein Sample oder (L+1, N, B) fuer einen Batch.
           Zeile s = Feldintensitaet nach s Layern (s=0 = Eingang).
    det1 : Kanalindex des Detektors fuer Klasse "1".
    det7 : Kanalindex des Detektors fuer Klasse "7".
    Y    : Label(s). Skalar/int fuer ein Sample, sonst Array (B,).
 
    Heatmap  = Mittelwert ueber alle Samples.
    Histogram = Ausgangsintensitaet je Kanal, pro Klasse ueberlagert
                halbtransparent, plus schraffiertes Ziel-Overlay.
    """
    I = np.asarray(I, dtype=float)
    Y = np.atleast_1d(np.asarray(Y))
 
    # ---------- Formen vereinheitlichen: immer (L+1, N, B) ----------
    if I.ndim == 2:
        I = I[:, :, None]
    elif I.ndim != 3:
        raise ValueError(f"erwarte (L+1, N) oder (L+1, N, B), bekommen {I.shape}")
    L1, N, B = I.shape
    if len(Y) != B:
        raise ValueError(f"Y hat {len(Y)} Labels, I aber {B} Samples")
 
    heat = I.mean(axis=2)                    # (L+1, N): Mittel ueber Samples
    I_out = I[-1]                            # (N, B): Ausgangsintensitaeten
 
    # ---------- pro Klasse mitteln und normieren ----------
    classes = np.unique(Y)
    bars = {}
    for c in classes:
        mask = (Y == c)
        prof = I_out[:, mask].mean(axis=1)   # mittlere Intensitaet je Kanal
        total = prof.sum()
        bars[c] = prof / total if total > 0 else prof
 
    # ---------- Layout: Colorbar links, Heatmap mitte, Histogram rechts ----------
    fig, (cax, ax, hax) = plt.subplots(
        1, 3, figsize=(max(9, 0.09*L1 + 6), max(3.5, 0.05*N + 3)),
        gridspec_kw={"width_ratios": [0.035, 1.0, 0.42], "wspace": 0.15})
 
    data = np.log10(np.maximum(heat.T, 1e-12)) if log else heat.T
    im = ax.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest",
                   extent=[-0.5, L1 - 0.5, N - 0.5, -0.5])
    ax.set_xlabel("layer  (0 = input field)")
    ax.set_ylabel(r"channel $k$")
    ax.set_title(title or r"mean field intensity $|E|^2$")
 
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$\log_{10}|E|^2$" if log else r"$|E|^2$", fontsize=9)
    cax.yaxis.set_ticks_position("left")
    cax.yaxis.set_label_position("left")
 
    # ---------- Histogram: barh = um 90 Grad gedreht ----------
    channels = np.arange(N)
    for c, col in zip(classes, class_colors):
        hax.barh(channels, bars[c], height=0.85, color=col, alpha=0.55,
                 label=f"class {c}", zorder=2)
 
    # ---------- Ziel-Overlay ----------
    # Option 3a (gewaehlt): Ziel ist die literale 1.0 -> gesamte Energie am Detektor
    target_scale = 1.0
    # --- alternative Skalierungen, bei Bedarf einkommentieren -------------
    # Option 3b: auf die hoechste vorkommende Balkenhoehe
    # target_scale = max(b.max() for b in bars.values())
    # Option 3c: auf die Summe der beiden Detektorbalken
    # target_scale = max(sum(b[[det1, det7]].sum() for b in bars.values()), 1e-12)
    # ---------------------------------------------------------------------
    # Detectors
    detectors=np.array([det1, det7])
    labels=["1","7"]
    for d,l in zip(detectors,labels):
        ax.plot(L1 - 0.5, d, marker="<", ms=10, color="forestgreen", clip_on=False, zorder=5)
        ax.text(L1 + 0.75, d, f"det {l}", va="center",rotation='vertical', fontsize=8, color="forestgreen", clip_on=False)

 

    #----------------------------------------------------------------------
    for c in classes:
        d = det1 if c == classes[0] else det7
        hax.barh([d], [target_scale], height=0.85,
                 facecolor=target_color, linewidth=1.2,
                 alpha=0.3, zorder=3,
                 label="target" if c == classes[0] else None)
 
    hax.set_ylim(N - 0.5, -0.5)              # gleiche Kanalachse wie Heatmap
    hax.set_yticks([])
    hax.set_xlim(0, target_scale * 1.05)     # bei 3b/3c ebenfalls passend
    # --- falls die Balken zu flach wirken, stattdessen autoskalieren: -----
    # hax.set_xlim(0, 1.05 * max(b.max() for b in bars.values()))
    # ---------------------------------------------------------------------
    hax.set_xlabel(r"norm. $|E|^2$")
    hax.axhline(det1, color="0.5", lw=0.6, ls=":", zorder=1)
    hax.axhline(det7, color="0.5", lw=0.6, ls=":", zorder=1)
    hax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    for s in ("top", "right"):
        hax.spines[s].set_visible(False)
 
    return fig, (ax, hax)


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

def plot_layers(layers, mode: str = "Abs"):
    """ Plots the layers as transfer-matrices
    mode: Either "Abs" or "Complex" for np.abs(layers) or np.imag(layers) and np.real(layers)
    """
    L = len(layers)

    if mode == "Complex":
        # zwei Zeilen: Realteil oben, Imaginaerteil unten
        fig, axes = plt.subplots(2, L, figsize=(2.2*L, 4.6))
        for l, M in enumerate(layers):
            im = axes[0, l].matshow(np.real(M), vmin=-1, vmax=1, cmap="RdBu_r")
            axes[1, l].matshow(np.imag(M), vmin=-1, vmax=1, cmap="RdBu_r")
            axes[0, l].set_title(f"L{l}", fontsize=9)
            for row in (0, 1):
                axes[row, l].set_xticks([]); axes[row, l].set_yticks([])
        axes[0, 0].set_ylabel("Re", fontsize=10)
        axes[1, 0].set_ylabel("Im", fontsize=10)
        fig.colorbar(im, ax=axes, shrink=0.6)
        return fig, axes

    # Standard: Betrag, eine Zeile
    fig, axes = plt.subplots(1, L, figsize=(2.2*L, 2.4))
    for l, (ax, M) in enumerate(zip(axes, layers)):
        im = ax.matshow(np.abs(M), vmin=0, vmax=1)
        ax.set_title(f"L{l}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.7)
    return fig, axes