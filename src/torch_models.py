"""
PyTorch multilayer perceptron with a scikit-learn classifier API.

``TorchMLPClassifier`` exposes ``fit(X, y, sample_weight)``,
``predict_proba``, ``predict`` and ``classes_``, so it plugs into the
existing pipeline (training, evaluation, model bundle, CLI, API) exactly
like ``LogisticRegression`` — on top of either TF-IDF or sentence
embeddings.

Design notes:
    - Sparse TF-IDF input stays sparse; only the current mini-batch is
      densified, so memory use is bounded by ``batch_size``.
    - Confidence-aware training is preserved: the loss is the
      cross-entropy weighted by ``sample_weight`` x class weight
      (``class_weight="balanced"`` mirrors scikit-learn's formula).
    - Early stopping monitors the weighted loss on a stratified validation
      slice carved from the TRAINING data; the test set is never touched.
      The best epoch's weights are restored.
    - The fitted network is pickled as plain numpy arrays (not as a torch
      module), so the model bundle stays a regular joblib file. torch is
      imported lazily: the baseline pipeline runs without it.
"""
from collections.abc import Sequence

import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from .utils.logging_config import setup_logger

logger = setup_logger(__name__)


def _import_torch():
    """Import torch lazily with an actionable error message."""
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "Model type 'mlp' requires PyTorch. Install it with:\n"
            "  pip install torch --index-url "
            "https://download.pytorch.org/whl/cpu"
        ) from e
    return torch


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """
    Feed-forward network: [Linear → ReLU → Dropout] x len(hidden_dims)
    → Linear(n_classes), trained with AdamW on weighted cross-entropy.

    Args:
        hidden_dims: Sizes of hidden layers; ``()`` gives a softmax
            (multinomial logistic) regression trained in torch
        dropout: Dropout probability after each hidden layer
        learning_rate: AdamW learning rate
        weight_decay: AdamW decoupled weight decay (L2-like regularization)
        batch_size: Mini-batch size (training and inference)
        max_epochs: Upper bound on training epochs
        patience: Epochs without validation improvement before stopping
        validation_fraction: Share of training rows held out for early
            stopping; 0 disables early stopping (trains ``max_epochs``)
        class_weight: "balanced", None/"none", or {label: weight}
        random_state: Seed for weight init, shuffling and validation split
        device: Torch device, e.g. "cpu" or "cuda"

    Attributes (set by ``fit``):
        classes_, n_features_in_, n_epochs_, best_epoch_,
        best_val_loss_, history_ (per-epoch train/val loss)
    """

    # Word-level explanations need linear per-word coefficients (coef_)
    _word_explanations_supported = False

    def __init__(
        self,
        hidden_dims: Sequence[int] = (128,),
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 50,
        patience: int = 5,
        validation_fraction: float = 0.1,
        class_weight: str | dict | None = "balanced",
        random_state: int | None = 42,
        device: str = "cpu",
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.class_weight = class_weight
        self.random_state = random_state
        self.device = device

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_module(self, n_features: int, n_classes: int):
        torch = _import_torch()
        nn = torch.nn
        layers = []
        in_dim = n_features
        for width in tuple(self.hidden_dims):
            layers += [
                nn.Linear(in_dim, int(width)),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ]
            in_dim = int(width)
        layers.append(nn.Linear(in_dim, n_classes))
        return nn.Sequential(*layers)

    @staticmethod
    def _as_matrix(X):
        """CSR for sparse input (fast row slicing), float32 array otherwise."""
        if issparse(X):
            return X.tocsr()
        return np.asarray(X, dtype=np.float32)

    @staticmethod
    def _batch_tensor(torch, X, rows: np.ndarray, device):
        """Densify only the selected rows and move them to ``device``."""
        batch = X[rows]
        if issparse(batch):
            batch = batch.toarray()
        return torch.as_tensor(
            np.asarray(batch, dtype=np.float32), device=device
        )

    def _compute_class_weights(self, y_idx: np.ndarray) -> np.ndarray:
        """Per-class weights indexed like ``classes_``."""
        n_classes = len(self.classes_)
        cw = self.class_weight
        if cw is None or cw == "none":
            return np.ones(n_classes)
        if cw == "balanced":
            counts = np.bincount(y_idx, minlength=n_classes)
            return len(y_idx) / (n_classes * np.maximum(counts, 1))
        if isinstance(cw, dict):
            return np.array(
                [float(cw.get(label, 1.0)) for label in self.classes_]
            )
        raise ValueError(
            f"class_weight must be 'balanced', None or dict, got {cw!r}"
        )

    def _weighted_loss_sums(self, torch, module, X, rows, y_idx, weights,
                            loss_fn, device):
        """Return (sum of weighted losses, sum of weights) for ``rows``."""
        logits = module(self._batch_tensor(torch, X, rows, device))
        targets = torch.as_tensor(y_idx[rows], dtype=torch.long,
                                  device=device)
        w = torch.as_tensor(weights[rows], dtype=torch.float32,
                            device=device)
        return (loss_fn(logits, targets) * w).sum(), w.sum()

    def _get_module(self):
        """Return the network, rebuilding it after unpickling if needed."""
        module = getattr(self, "module_", None)
        if module is not None:
            return module
        torch = _import_torch()
        module = self._build_module(self.n_features_in_, len(self.classes_))
        module.load_state_dict({
            name: torch.as_tensor(array)
            for name, array in self._module_state.items()
        })
        module.to(torch.device(self.device))
        module.eval()
        self.module_ = module
        return module

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X, y, sample_weight=None) -> "TorchMLPClassifier":
        torch = _import_torch()
        X = self._as_matrix(X)
        y = np.asarray(y)
        if X.shape[0] != len(y):
            raise ValueError(
                f"X has {X.shape[0]} samples but y has {len(y)} labels"
            )

        self.classes_, y_idx = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("Need samples of at least 2 classes to fit")
        self.n_features_in_ = X.shape[1]

        if sample_weight is None:
            sample_weight = np.ones(len(y))
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if sample_weight.shape[0] != len(y):
            raise ValueError("sample_weight length does not match y")

        seed = 0 if self.random_state is None else int(self.random_state)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        device = torch.device(self.device)

        # Stratified validation slice of the training data
        all_rows = np.arange(len(y))
        if self.validation_fraction and self.validation_fraction > 0:
            try:
                train_rows, val_rows = train_test_split(
                    all_rows,
                    test_size=self.validation_fraction,
                    random_state=seed,
                    stratify=y_idx,
                )
            except ValueError:
                # Too few samples of some class to stratify (tiny datasets)
                train_rows, val_rows = train_test_split(
                    all_rows,
                    test_size=self.validation_fraction,
                    random_state=seed,
                )
        else:
            train_rows, val_rows = all_rows, None

        class_weights = self._compute_class_weights(y_idx[train_rows])
        weights = sample_weight * class_weights[y_idx]

        module = self._build_module(self.n_features_in_, len(self.classes_))
        module.to(device)
        optimizer = torch.optim.AdamW(
            module.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        def snapshot():
            return {
                k: v.detach().clone() for k, v in module.state_dict().items()
            }

        best_state = snapshot()
        best_loss = np.inf
        best_epoch = 0
        epochs_without_improvement = 0
        history = []

        for epoch in range(1, int(self.max_epochs) + 1):
            # --- train ---
            module.train()
            loss_sum, weight_sum = 0.0, 0.0
            order = rng.permutation(train_rows)
            for start in range(0, len(order), int(self.batch_size)):
                rows = order[start:start + int(self.batch_size)]
                optimizer.zero_grad()
                batch_loss, batch_weight = self._weighted_loss_sums(
                    torch, module, X, rows, y_idx, weights, loss_fn, device
                )
                (batch_loss / batch_weight.clamp_min(1e-12)).backward()
                optimizer.step()
                loss_sum += float(batch_loss.detach())
                weight_sum += float(batch_weight)
            train_loss = loss_sum / max(weight_sum, 1e-12)

            # --- validate ---
            val_loss = None
            if val_rows is not None:
                module.eval()
                v_loss, v_weight = 0.0, 0.0
                with torch.no_grad():
                    for start in range(0, len(val_rows),
                                       int(self.batch_size)):
                        rows = val_rows[start:start + int(self.batch_size)]
                        b_loss, b_weight = self._weighted_loss_sums(
                            torch, module, X, rows, y_idx, weights,
                            loss_fn, device,
                        )
                        v_loss += float(b_loss)
                        v_weight += float(b_weight)
                val_loss = v_loss / max(v_weight, 1e-12)

            history.append({
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss
            })

            monitored = val_loss if val_loss is not None else train_loss
            if monitored < best_loss - 1e-6:
                best_loss = monitored
                best_epoch = epoch
                best_state = snapshot()
                epochs_without_improvement = 0
            elif val_rows is not None:
                epochs_without_improvement += 1
                if epochs_without_improvement >= int(self.patience):
                    break

        if val_rows is None:
            # No validation: keep the final weights
            best_state = snapshot()
            best_epoch = len(history)

        module.load_state_dict(best_state)
        module.eval()
        self.module_ = module
        self.n_epochs_ = len(history)
        self.best_epoch_ = best_epoch
        self.best_val_loss_ = (
            float(best_loss) if val_rows is not None else None
        )
        self.history_ = history

        logger.info(
            f"MLP trained: {self.n_epochs_} epochs "
            f"(best epoch {self.best_epoch_}"
            + (
                f", val loss {self.best_val_loss_:.4f})"
                if self.best_val_loss_ is not None else ")"
            )
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, "classes_")
        torch = _import_torch()
        X = self._as_matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, model expects "
                f"{self.n_features_in_}"
            )
        n_classes = len(self.classes_)
        if X.shape[0] == 0:
            return np.empty((0, n_classes), dtype=np.float64)

        module = self._get_module()
        module.eval()
        device = torch.device(self.device)
        rows_all = np.arange(X.shape[0])
        chunks = []
        with torch.no_grad():
            for start in range(0, X.shape[0], int(self.batch_size)):
                rows = rows_all[start:start + int(self.batch_size)]
                logits = module(self._batch_tensor(torch, X, rows, device))
                chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(chunks).astype(np.float64)

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    # ------------------------------------------------------------------
    # Pickling: store weights as numpy arrays, not as a torch module
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        state = dict(super().__getstate__())
        module = state.pop("module_", None)
        if module is not None:
            state["_module_state"] = {
                name: tensor.detach().cpu().numpy()
                for name, tensor in module.state_dict().items()
            }
        return state
