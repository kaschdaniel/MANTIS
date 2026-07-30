from dataclasses import dataclass
import numpy as np, time

from processing import forward, forward_history, backward_history, gradient
from decoding import loss_function, adjoint_source, predict
from metrics import confusion_matrix, accuracy


@dataclass
class TrainConfig:
    """Everything that defines one run. Dumped verbatim into the results."""
    # --- data / encoding ---
    m_side: int = 10                 # -> N = m_side**2 input channels
    theta_enc: float = 1.0
    normalize_energy: bool = True
    encoding: str = "amplitude"      # "amplitude" | "phase"

    # --- readout ---
    detectors: tuple | None = None   # None -> thirds of N, see __post_init__

    # --- optimization ---
    loss_kind: str = "mse_norm"
    learning_rate: float = 1e-2
    batch_size: int = 32
    init: str = "haar"               # "haar" | "random"
    max_epochs: int = 300
    patience: int = 20               # epochs without loss improvement
    min_delta: float = 1e-4          # smallest loss drop counted as progress
    seed: int | None = 0

    # --- hardware imperfections (fixed, not trained) ---
    eta_bs: float = 1.0              # power transmission per beam splitter
    alpha_fiber: float = 0.0         # waveguide loss in dB per layer

    @property
    def N(self):
        return self.m_side ** 2

    def __post_init__(self):
        # detectors at one third and two thirds of the channels, so they
        # scale automatically when m_side is swept
        if self.detectors is None:
            self.detectors = (self.N // 3, 2 * self.N // 3)


def _copy(params):
    """Deep copy of a weight list -- step() updates the arrays in place."""
    return [p.copy() for p in params]


class Trainer:
    """Adjoint-based gradient descent on an MZI mesh.

    Only two data sets are used, as specified for the project: the model is
    trained on E_train, and the accuracy measured there is referred to as
    the validation accuracy. The test set is held out and evaluated
    once, at the end, via evaluate().
    """

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
        self.history = {"loss": [], "acc": [], "grad_norm": []}
        self.train_time = None

    def _layers(self):
        """Transfer matrices for the current weights. Rebuilt after every
        update, since the weights change; this is the runtime bottleneck."""
        return self.mesh.layer_matrices_separate(
            self.thetas, self.phis, self.cfg.eta_bs, self.cfg.alpha_fiber)

    def step(self, E_batch, y_batch):
        """One gradient step on one minibatch. Returns (loss, gradient norm)."""
        cfg = self.cfg
        d1, d7 = cfg.detectors
        layers = self._layers()

        fh = forward_history(E_batch, layers)                     # forward fields
        Gam = adjoint_source(fh, y_batch, d1, d7, cfg.loss_kind)  # Gamma_L
        bh = backward_history(Gam, layers)                        # adjoint fields
        g_th, g_ph = gradient(fh, bh, self.mesh.plan)

        # Descent: minus the gradient, element-wise per layer. Never write
        # "self.thetas -= ...": that is a list, and += on a list extends it.
        for l in range(len(self.thetas)):
            self.thetas[l] -= cfg.learning_rate * g_th[l]
            self.phis[l]   -= cfg.learning_rate * g_ph[l]

        loss = np.mean(loss_function(fh, y_batch, d1, d7, cfg.loss_kind)) #computing mean value of loss functions for given batch
        grad_norm = np.sqrt(sum(np.sum(g**2) for g in g_th + g_ph))
        return loss, grad_norm

    def evaluate(self, E, y):
        """Loss and accuracy on any data set, without changing the weights."""
        d1, d7 = self.cfg.detectors
        E_out = forward(E, self._layers())
        loss = np.mean(loss_function(E_out, y, d1, d7, self.cfg.loss_kind))
        conf = confusion_matrix(y, predict(E_out, d1, d7), classes=(1, 7))
        return loss, accuracy(conf)

    def fit(self, E_train, y_train):
        """Train until the loss stops improving or max_epochs is reached.
        """
        cfg = self.cfg
        t0 = time.perf_counter()
        best_loss = np.inf
        best_params = (_copy(self.thetas), _copy(self.phis))
        wait = 0

        for epoch in range(cfg.max_epochs):
            idx = self.rng.permutation(E_train.shape[1])
            losses, norms = [], []
            for s in range(0, len(idx), cfg.batch_size):
                b = idx[s:s + cfg.batch_size]
                loss, gn = self.step(E_train[:, b], y_train[b])
                losses.append(loss); norms.append(gn)

            epoch_loss = float(np.mean(losses))
            _, acc = self.evaluate(E_train, y_train)
            self.history["loss"].append(epoch_loss)
            self.history["acc"].append(acc)
            self.history["grad_norm"].append(float(np.mean(norms)))

            if epoch_loss < best_loss - cfg.min_delta:
                best_loss, wait = epoch_loss, 0
                best_params = (_copy(self.thetas), _copy(self.phis))
            else:
                wait += 1
                if wait >= cfg.patience:
                    break

        # keep the best weights, not the last ones (guards a diverging step)
        self.thetas, self.phis = best_params
        self.train_time = time.perf_counter() - t0
        return self.history

    def inference_time(self, E, repeats=20):
        """Seconds per single classification.

        The transfer matrices are built outside the timed region: on real
        hardware the weights are physically present in the chip, so building
        them is simulation overhead and not part of the inference cost.
        """
        layers = self._layers()
        one = E[:, :1]
        t0 = time.perf_counter()
        for _ in range(repeats):
            forward(one, layers)
        return (time.perf_counter() - t0) / repeats