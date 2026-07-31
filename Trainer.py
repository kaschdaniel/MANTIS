from dataclasses import dataclass, asdict, fields
import numpy as np, time, json, pathlib
from tqdm.auto import tqdm

from mesh import MZIMesh
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
    detectors: tuple | None = None   # None -> Det 1 @  1/3*N, Det 2 @ 2/3*N

    # --- optimization ---
    loss_kind: str = "mse_norm"
    learning_rate: float = 1e-2
    batch_size: int = 32
    init: str = "haar"               # "haar" | "random"
    max_epochs: int = 300
    patience: int = 20               # epochs without loss improvement
    min_delta: float = 1e-4          # smallest loss drop counted as progress
    param_init_seed: int | None = 0

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

def _jsonable(o):
    """Convert NumPy scalars/arrays that json does not know about."""
    if isinstance(o, np.integer):  return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray):  return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")

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
        self.rng = np.random.default_rng(cfg.param_init_seed)
        init = mesh.init_haar if cfg.init == "haar" else mesh.init_random
        self.thetas, self.phis = init(self.rng)
        self.history = {"loss": [], "acc": [], "grad_norm": [],
                        "batch_loss": [], "epoch_end_step": []}
        self.train_time = None
        self.test_acc = None               # set by the caller after evaluate()

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
        # Time tracking using tqdm
        num_samples = E_train.shape[1]
        batches_per_epoch = np.ceil(num_samples / cfg.batch_size) #how many batches per sample
        total_steps = cfg.max_epochs * batches_per_epoch
        with tqdm(total=total_steps, desc="Starte Training...") as pbar:
            for epoch in range(cfg.max_epochs):
                idx = self.rng.permutation(E_train.shape[1])
                losses, norms = [], []
                # Saving loss for every gradient step to see effect of batchsize
                for s in range(0, len(idx), cfg.batch_size):
                    b = idx[s:s + cfg.batch_size]
                    loss, gn = self.step(E_train[:, b], y_train[b])
                    losses.append(loss); norms.append(gn)
                    self.history["batch_loss"].append(float(loss))
                    #Plotting time progress using tqdm
                    pbar.update(1)
                    progr = len(self.history["batch_loss"]) / batches_per_epoch
                    pbar.set_description(f"Epoch {progr:.2f}/{cfg.max_epochs} | Loss: {loss:.4f}")

                epoch_loss = float(np.mean(losses))
                _, acc = self.evaluate(E_train, y_train)
                self.history["loss"].append(epoch_loss)
                self.history["acc"].append(acc)
                self.history["grad_norm"].append(float(np.mean(norms)))
                # step index at which this epoch ended, for marking epoch
                # boundaries when plotting batch_loss
                self.history["epoch_end_step"].append(len(self.history["batch_loss"]))

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
    # ------------------------------------------------ persistence
    def save(self, path, **extra):
        """Write the run to disk as JSON.

        NumPy arrays are converted to lists.
        Extra keyword arguments (test_acc, notes, ...) come back as trainer.extra.
        """
        state = {
            "format": 1,
            "cfg": asdict(self.cfg),
            "N": self.mesh.N,
            "plan": [[kind, list(ks) if isinstance(ks, np.ndarray) else ks]
                     for kind, ks in self.mesh.plan],
            "thetas": [t.tolist() for t in self.thetas],
            "phis": [p.tolist() for p in self.phis],
            "history": {k: [float(x) for x in v]
                        for k, v in self.history.items()},
            "train_time": float(self.train_time),
            "test_acc": self.test_acc,
            "extra": extra
        }
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "x") as f: #if file exists already, throw exception
            json.dump(state, f, indent=2, default=_jsonable)
        return path

    @classmethod
    def load(cls, path):
        """Rebuild a Trainer from a JSON file written by save()."""
        with open(path, "r") as f:
            state = json.load(f)

        # keep only fields the current TrainConfig knows
        known = {f.name for f in fields(TrainConfig)}
        unknown = set(state["cfg"]) - known
        if unknown:
            print(f"ignoring unknown config fields: {sorted(unknown)}")
        cfg = TrainConfig(**{k: v for k, v in state["cfg"].items() if k in known})

        # reconstruct plan: ("mzi", [0, 2, ...]) or ("perm", array)
        plan = []
        for kind, ks in state["plan"]:
            if kind == "perm" and isinstance(ks, list):
                ks = np.array(ks)
            plan.append((kind, ks))

        obj = cls(MZIMesh(state["N"], plan), cfg)
        obj.thetas = [np.array(t) for t in state["thetas"]]
        obj.phis = [np.array(p) for p in state["phis"]]
        obj.history = {k: [float(x) for x in v]
                       for k, v in state["history"].items()}
        obj.train_time = state["train_time"]
        # test_acc: top level for new files, inside extra for old ones
        obj.extra = state.get("extra", {})
        obj.test_acc = state.get("test_acc", obj.extra.get("test_acc"))
        return obj