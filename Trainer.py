from dataclasses import dataclass, field
import numpy as np, time

@dataclass
class TrainConfig:
    """Everything that defines one run. Dumped verbatim into the results."""
    loss_kind: str = "mse_norm"
    learning_rate: float = 1e-2
    batch_size: int = 32
    detectors: tuple = (33, 66)
    init: str = "haar"          # "haar" | "random"
    max_epochs: int = 300
    patience: int = 20
    min_delta: float = 1e-4
    seed: int | None = 0


class Trainer:
    def __init__(self, mesh, cfg: TrainConfig):
        self.mesh, self.cfg = mesh, cfg
        rng = np.random.default_rng(cfg.seed)
        init = mesh.init_haar if cfg.init == "haar" else mesh.init_random
        self.thetas, self.phis = init(rng)
        self.history = {"train_loss": [], "val_loss": [], "val_acc": [],
                        "grad_norm": []}
        self.best = {"val_acc": -np.inf, "params": None, "epoch": -1}

    # one gradient step on one minibatch
    def step(self, E_batch, y_batch): ...

    # returns (loss, accuracy) without touching the parameters
    def evaluate(self, E, y): ...

    def fit(self, E_train, y_train, E_val, y_val):
        t0 = time.perf_counter()
        wait = 0
        for epoch in range(self.cfg.max_epochs):
            self._run_epoch(E_train, y_train)
            val_loss, val_acc = self.evaluate(E_val, y_val)
            # ... record history, early stopping, keep best params ...
            if wait >= self.cfg.patience:
                break
        self.thetas, self.phis = self.best["params"]
        self.train_time = time.perf_counter() - t0
        return self.history
