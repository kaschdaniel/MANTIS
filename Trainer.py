from dataclasses import dataclass
from processing import *
from decoding import *
from metrics import *
import numpy as np, time

@dataclass
class TrainConfig:
    """Everything that defines one run. Dumped verbatim into the results."""
    # --- data / encoding ---
    m_side: int = 10                 # -> N = m_side**2 input channels
    theta_enc: float = 1.0
    encoding: str = "amplitude"      # "amplitude" | "phase"
    normalize_energy: bool = True
    n_samples: int | None = 2000     # total, balanced over both classes

    # --- readout ---
    detectors: tuple | None = None   # None -> thirds of N, see __post_init__

    # --- optimization ---
    loss_kind: str = "mse_norm"
    learning_rate: float = 1e-2
    batch_size: int = 32
    init: str = "haar"               # "haar" | "random"
    max_epochs: int = 300
    patience: int = 20
    min_delta: float = 1e-4
    seed: int | None = 0

    # --- hardware imperfections (fixed, not trained) ---
    eta_bs: float = 1.0
    alpha_fiber: float = 0.0

    @property
    def N(self):
        return self.m_side ** 2

    def __post_init__(self):
        # detectors at one third and two thirds of the channels, so they
        # scale automatically when m_side is swept
        if self.detectors is None:
            self.detectors = (self.N // 3, 2 * self.N // 3)


def _copy(params):
    """Deep copy of a weight list -- needed because step() updates in place."""
    return [p.copy() for p in params]
 
 
class Trainer:
    def __init__(self, mesh, cfg: TrainConfig):
        if mesh.N != cfg.N:
            raise ValueError(f"mesh has N={mesh.N} but cfg.m_side={cfg.m_side} "
                             f"implies N={cfg.N} -- encoding and mesh disagree")
        if max(cfg.detectors) >= mesh.N:
            raise ValueError(f"detector {max(cfg.detectors)} outside N={mesh.N}")
        self.mesh, self.cfg = mesh, cfg
        self.rng = np.random.default_rng(cfg.seed)
        init = mesh.init_haar if cfg.init == "haar" else mesh.init_random
        self.thetas, self.phis = init(self.rng)
        self.history = {"train_loss": [], "val_loss": [], "val_acc": [],
                        "grad_norm": []}
        self.best = {"val_acc": -np.inf, "params": None, "epoch": -1}
 
    def _layers(self):
        """Transfer matrices for the current weights. Rebuilt every step
        because the weights change; this is the runtime bottleneck."""
        return self.mesh.layer_matrices_separate(
            self.thetas, self.phis, self.cfg.eta_bs, self.cfg.alpha_fiber)
 
    # one gradient step on one minibatch
    def step(self, E_batch, y_batch):
        cfg = self.cfg
        d1, d7 = cfg.detectors
        layers = self._layers()
 
        fh = forward_history(E_batch, layers)                    # forward fields
        Gam = adjoint_source(fh, y_batch, d1, d7, cfg.loss_kind)  # Gamma_L
        bh = backward_history(Gam, layers)                       # adjoint fields
        g_th, g_ph = gradient(fh, bh, self.mesh.plan)
 
        # descent: minus the gradient. In-place, so self.thetas stays a
        # list of arrays (never use "self.thetas -= ...", that extends the list)
        for l in range(len(self.thetas)):
            self.thetas[l] -= cfg.learning_rate * g_th[l]
            self.phis[l]   -= cfg.learning_rate * g_ph[l]
 
        loss, _ = loss_function(fh, y_batch, d1, d7, cfg.loss_kind)
        grad_norm = np.sqrt(sum(np.sum(g**2) for g in g_th + g_ph))
        return loss, grad_norm
 
    # returns (loss, accuracy) without touching the parameters
    def evaluate(self, E, y):
        d1, d7 = self.cfg.detectors
        E_out = forward(E, self._layers())
        loss, _ = loss_function(E_out, y, d1, d7, self.cfg.loss_kind)
        conf = confusion_matrix(y, predict(E_out, d1, d7), classes=(1, 7))
        return loss, accuracy(conf)
 
    def _run_epoch(self, E, y):
        """One pass over the training set in shuffled minibatches."""
        idx = self.rng.permutation(E.shape[1])
        losses, norms = [], []
        for s in range(0, len(idx), self.cfg.batch_size):
            b = idx[s:s + self.cfg.batch_size]
            loss, gn = self.step(E[:, b], y[b])
            losses.append(loss); norms.append(gn)
        return float(np.mean(losses)), float(np.mean(norms))
 
    def fit(self, E_train, y_train, E_val, y_val):
        t0 = time.perf_counter()
        wait = 0
        # start from the initial weights, so best["params"] is never None
        self.best = {"val_acc": -np.inf, "epoch": -1,
                     "params": (_copy(self.thetas), _copy(self.phis))}
 
        for epoch in range(self.cfg.max_epochs):
            train_loss, grad_norm = self._run_epoch(E_train, y_train)
            val_loss, val_acc = self.evaluate(E_val, y_val)
            for key, val in zip(self.history,
                                (train_loss, val_loss, val_acc, grad_norm)):
                self.history[key].append(val)
 
            # early stopping on validation accuracy
            if val_acc > self.best["val_acc"] + self.cfg.min_delta:
                self.best = {"val_acc": val_acc, "epoch": epoch,
                             "params": (_copy(self.thetas), _copy(self.phis))}
                wait = 0
            else:
                wait += 1
                if wait >= self.cfg.patience:
                    break
 
        # restore the best weights, not the last ones
        self.thetas, self.phis = self.best["params"]
        self.train_time = time.perf_counter() - t0
        return self.history
 
    def inference_time(self, E, repeats=10):
        """Seconds per single classification, for the performance section."""
        layers = self._layers()          # a real chip has this in hardware
        one = E[:, :1]
        t0 = time.perf_counter()
        for _ in range(repeats):
            forward(one, layers)
        return (time.perf_counter() - t0) / repeats
