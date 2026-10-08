"""
Sentence-embedding feature extractor with a scikit-learn vectorizer API.

``SentenceEmbeddingVectorizer`` turns raw texts into dense sentence
embeddings using a pretrained Sentence-Transformers model. It mimics the
contract the rest of the pipeline expects from a vectorizer
(``fit`` / ``transform`` / ``fit_transform``), so classifiers, evaluation,
the model bundle, the CLI and the API work with it unchanged.

Design notes:
    - The encoder is pretrained and frozen: ``fit`` only validates input
      and loads the model; nothing is learned from the data, so fitting on
      the full dataset would not leak — but the pipeline still fits on the
      training split only, for uniformity with TF-IDF.
    - The heavy encoder is NOT pickled into the model bundle: only its name
      and settings are. It is loaded lazily on first use (from the local
      Hugging Face cache, or downloaded once).
    - ``sentence-transformers`` (and torch) are imported lazily, so the
      baseline TF-IDF pipeline and the test suite run without them.
    - Deliberately no ``get_feature_names_out``: embedding dimensions are
      not words, and word-level explanations must be disabled for this
      vectorizer (see ``interpretability.supports_word_explanations``).
"""
from collections.abc import Callable, Iterable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .utils.logging_config import setup_logger

logger = setup_logger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _load_sentence_transformer(model_name: str, device: str):
    """
    Load a SentenceTransformer model (lazy import).

    Kept as a module-level function so tests can replace it with a
    lightweight fake encoder (no torch, no model download).

    Raises:
        ImportError: If sentence-transformers is not installed
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "Vectorizer type 'sentence_embedding' requires the optional "
            "deep-learning dependencies. Install them with:\n"
            "  pip install torch --index-url "
            "https://download.pytorch.org/whl/cpu\n"
            "  pip install -r requirements-dl.txt"
        ) from e

    logger.info(f"Loading sentence-transformers model '{model_name}' "
                f"on {device}")
    return SentenceTransformer(model_name, device=device)


class SentenceEmbeddingVectorizer(TransformerMixin, BaseEstimator):
    """
    Encode texts into dense sentence embeddings.

    Args:
        model_name: Sentence-Transformers model id
            (Hugging Face Hub name or local path)
        batch_size: Encoding batch size
        normalize_embeddings: L2-normalize embeddings (recommended for
            linear classifiers: puts all samples on the same scale)
        device: Torch device, e.g. "cpu" or "cuda"
        preprocessor: Optional callable applied to each raw text before
            encoding (light cleaning; see ``build_text_preprocessor``)
        show_progress_bar: Show the encoding progress bar

    Attributes (set by ``fit``):
        embedding_dim_: Dimensionality of produced embeddings
        n_features_in_: Same as ``embedding_dim_`` (sklearn convention)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        device: str = "cpu",
        preprocessor: Callable[[str], str] | None = None,
        show_progress_bar: bool = False,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.device = device
        self.preprocessor = preprocessor
        self.show_progress_bar = show_progress_bar

    # ------------------------------------------------------------------
    # Encoder lifecycle
    # ------------------------------------------------------------------

    def _get_encoder(self):
        """Return the loaded encoder, loading it on first use."""
        encoder = getattr(self, "_encoder", None)
        if encoder is None:
            encoder = _load_sentence_transformer(self.model_name, self.device)
            self._encoder = encoder
        return encoder

    def warmup(self) -> None:
        """
        Load the encoder and run one tiny encode.

        Call at service startup so the first real request does not pay the
        model-loading cost.
        """
        self.transform(["warmup"])

    def __getstate__(self) -> dict:
        """Exclude the loaded encoder from pickling (bundle stays small)."""
        state = super().__getstate__()  # keeps sklearn's version stamp
        state = dict(state)
        state.pop("_encoder", None)
        return state

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def _prepare(self, texts: Iterable[str]) -> list[str]:
        if isinstance(texts, str):
            raise TypeError(
                "Expected an iterable of texts, got a single string. "
                "Wrap it in a list: transform([text])."
            )
        texts = ["" if t is None else str(t) for t in texts]
        if self.preprocessor is not None:
            texts = [self.preprocessor(t) for t in texts]
        return texts

    def fit(
        self, texts: Iterable[str], y=None
    ) -> "SentenceEmbeddingVectorizer":
        """
        Load the encoder and record the embedding dimensionality.

        Nothing is learned from ``texts``: the encoder is pretrained.
        """
        self._prepare(texts)  # validates input type
        dim = self.transform(["dimension probe"]).shape[1]
        self.embedding_dim_ = dim
        self.n_features_in_ = dim
        return self

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        """
        Encode texts into a dense float32 matrix (n_samples, embedding_dim).
        """
        prepared = self._prepare(texts)
        if not prepared:
            dim = getattr(self, "embedding_dim_", 0)
            return np.empty((0, dim), dtype=np.float32)

        embeddings = self._get_encoder().encode(
            prepared,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)
