import os
import pathlib
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel, ConfigDict, Field

from ollama import Client
from langchain.embeddings import Embeddings, init_embeddings
from langchain_community.vectorstores import FAISS
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


# TODO: maybe embeddings should be a sub-class of indexing


class EmbeddingConfig(BaseModel):
    """Embedding model configuration (name, instantiated model and embedding dimension)."""
    model_name: str #: str : Model identifier used for both the embedding wrapper and the HF config.
    ollama: bool  #: bool : Whether to serve the embedding model via ollama
    model: Embeddings = None #: :langchain:`Embeddings <embeddings>` : Instantiated embedding model.
    kwargs: Dict[str, Any] = None #: dict : Provider-specific kwargs passed to the embedding model.
    dim: int #: int : Embedding dimensionality (taken from the HuggingFace model config).
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "EmbeddingConfig":
        """
        Build an :class:`~cardiology_gen_ai.models.EmbeddingConfig` from a configuration mapping.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Mapping that must contain ``"deployment"`` (model name) and optional ``"kwargs"``.
            Recognized keys under ``kwargs``: ``normalize_embeddings``, ``prompt_name`` (encode kwargs), and ``device`` (model kwargs; auto-detected GPU/CPU if omitted).

        Returns
        -------
        :class:`~cardiology_gen_ai.models.EmbeddingConfig`
            Populated configuration with ``model`` instantiated and ``dim`` read from :class:`~transformers.AutoConfig`

        Raises
        ------
        OSError
            If the Hugging Face model/config cannot be resolved locally or downloaded.
        """
        import torch
        kwargs = config_dict["kwargs"]
        if not isinstance(kwargs, dict):
            kwargs = dict()
        encode_kwargs = {k: v for k, v in kwargs.items() if k in ["normalize_embeddings", "prompt_name"]}
        model_kwargs = {k: v for k, v in kwargs.items() if k in ["device"]}
        if model_kwargs.get("device", None) is None or model_kwargs.get("device", None) == "cuda":
            model_kwargs["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = config_dict["deployment"]
        ollama_model = config_dict.get("ollama", False)
        if ollama_model:
            client = Client(os.getenv("OLLAMA_URL"))
            client.pull(model_name)
            model = init_embeddings(
                model=model_name,
                provider="ollama",
                base_url=os.getenv("OLLAMA_URL")
            )
        else:
            model = init_embeddings(
                model=model_name,
                provider="huggingface",
                cache_folder=os.environ.get("HUGGINGFACE_HUB_CACHE"),
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
        dim = len(model.embed_query("Computing embedding dimension."))
        return cls(model_name=model_name, ollama=ollama_model, model=model, dim=dim, kwargs=kwargs)

    def to_config(self) -> Dict[str, Any]:
        embedding_config = dict()
        embedding_config["deployment"] = self.model_name
        embedding_config["ollama"] = self.ollama
        embedding_config["kwargs"] = {k: v for k, v in self.kwargs.items()
                                      if k not in ["device"]}
        return embedding_config


class IndexTypeNames(Enum):
    """Backend type for the vectorstore index."""
    qdrant = "qdrant" #: :langchain:`Qdrant <qdrant/qdrant/langchain_qdrant.qdrant.QdrantVectorStore.html#langchain_qdrant.qdrant.QdrantVectorStore>`
    faiss = "faiss" #: :langchain:`FAISS <community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html>`


class DistanceTypeNames(Enum):
    """Distance metric to use in the vector index."""
    cosine = "cosine" #: Cosine similarity (inner product / normalized).
    euclidean = "euclidean" #: L2 (Euclidean) distance.


class RetrievalTypeNames(Enum):
    """Retrieval strategy for the vector store."""
    dense = "dense" #: Dense-only retrieval.
    sparse = "sparse" #: Sparse-only retrieval.
    hybrid = "hybrid" #: Hybrid dense + sparse retrieval.


class IndexingConfig(BaseModel):
    """Vector index configuration (name, backend type, distance, storage folder and retrieval mode)."""
    name: str #: str : Index (collection) name.
    description: str #: str : Human-readable description of the index.
    folder: pathlib.Path #: pathlib.Path : Root folder vectorstores are saved (defaults to ``os.getenv("INDEX_ROOT")``).
    type: IndexTypeNames #: :class:`~cardiology_gen_ai.models.IndexTypeNames` : Backend type (``qdrant`` or ``faiss``).
    distance: DistanceTypeNames #: :class:`~cardiology_gen_ai.models.DistanceTypeNames` : Similarity/distance metric.
    retrieval_mode: RetrievalTypeNames #: :class:`~cardiology_gen_ai.models.RetrievalTypeNames` : Retrieval strategy (default ``dense``).

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "IndexingConfig":
        """
        Build an :class:`~cardiology_gen_ai.models.IndexingConfig` from a configuration mapping.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Mapping with at least ``type`` and ``distance``; optional ``retrieval_mode``. Other keys are forwarded as-is (e.g. ``name``, ``description``, ``folder``).

        Returns
        -------
        :class:`~cardiology_gen_ai.models.IndexingConfig`
            Validated configuration object.
        """
        index_type = IndexTypeNames(config_dict["type"])
        distance = DistanceTypeNames(config_dict["distance"])
        retrieval = config_dict.get("retrieval_mode", RetrievalTypeNames.dense)
        retrieval_mode = RetrievalTypeNames(retrieval) if isinstance(retrieval, str) is not None else  RetrievalTypeNames.dense
        folder = pathlib.Path(os.getenv("INDEX_ROOT"))
        other_config_dict = {k: v for k, v in config_dict.items() if k not in ["type", "distance", "retrieval_mode"]}
        return cls(type=index_type, distance=distance, retrieval_mode=retrieval_mode, folder=folder, **other_config_dict)

    def to_config(self) -> Dict[str, Any]:
        index_config = dict()
        index_config["name"] = self.name
        index_config["description"] = self.description
        index_config["type"] = self.type.value
        index_config["distance"] = self.distance.value
        index_config["retrieval_mode"] = self.retrieval_mode.value
        return index_config


class Vectorstore(BaseModel, ABC):
    """
    Abstract base class for vector store adapters (:langchain:`Qdrant <qdrant/qdrant/langchain_qdrant.qdrant.QdrantVectorStore.html#langchain_qdrant.qdrant.QdrantVectorStore>`, :langchain:`FAISS <community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html>`).

    .. rubric:: Notes

    Subclasses must implement creation/loading and existence checks, and expose a way to count stored documents/chunks.
    """
    config: IndexingConfig #: IndexingConfig :  Index configuration.
    vectorstore: QdrantVectorStore | FAISS = None #: :langchain:`Qdrant <qdrant/qdrant/langchain_qdrant.qdrant.QdrantVectorStore.html#langchain_qdrant.qdrant.QdrantVectorStore>` | :langchain:`FAISS <community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html>` : Underlying vector store instance.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def vectorstore_exists(self) -> bool:
        """
        Check whether the underlying vector store already exists.

        Returns
        -------
        bool
            ``True`` if the index is present, ``False`` otherwise.
        """
        pass

    @abstractmethod
    def load_vectorstore(self, **kwargs) -> QdrantVectorStore | FAISS:
        """
        Load/open the underlying vector store.

        Returns
        -------
        :langchain:`QdrantVectorStore <qdrant/qdrant/langchain_qdrant.qdrant.QdrantVectorStore.html#langchain_qdrant.qdrant.QdrantVectorStore>` | :langchain:`FAISS <community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html>`
            Loaded vector store instance.
        """
        pass

    @abstractmethod
    def get_n_documents_in_vectorstore(self) -> int:
        """
        Return the number of stored items (documents/chunks) in the vector store.

        Returns
        -------
        int
            Number of stored items.
        """
        pass


class QdrantVectorstore(Vectorstore):
    """Vector store adapter backed by Qdrant."""
    # TODO: maybe should be changed
    url: Optional[str] = Field(default_factory=lambda: os.getenv("QDRANT_URL")) #: str : Qdrant vectorstore endpoint URL (defaults to ``os.getenv("QDRANT_URL")``).
    client: QdrantClient = None #: :qdrant:`QdrantClient <qdrant_client.qdrant_client>` : Low-level Qdrant client bound to ``url``.
    vectorstore: QdrantVectorStore = None #:  :langchain:`QdrantVectorStore <qdrant/qdrant/langchain_qdrant.qdrant.QdrantVectorStore.html#langchain_qdrant.qdrant.QdrantVectorStore>` : LangChain Qdrant vector store instance when loaded.

    def _get_client(self):
        if self.client is None:
            from qdrant_client import QdrantClient
            if not self.url:
                raise RuntimeError("QDRANT_URL is not set")
            self.client = QdrantClient(self.url)
        return self.client

    def vectorstore_exists(self) -> bool:
        """
        Check whether the configured collection exists in Qdrant.

        Returns
        -------
        bool
            ``True`` if the collection exists, ``False`` otherwise.
        """
        client = self._get_client()
        return any(collection.name == self.config.name for collection in client.get_collections().collections)

    def load_vectorstore(self, embeddings_model: EmbeddingConfig, retrieval_mode: str) -> QdrantVectorStore:
        """
        Open the existing Qdrant collection as a LangChain vector store.

        Parameters
        ----------
        embeddings_model : :class:`~cardiology_gen_ai.models.EmbeddingConfig`
            Embedding backend to bind (dense).
        retrieval_mode : str
            Retrieval mode (ignored in favor of ``self.config.retrieval_mode``; kept for compatibility).

        Returns
        -------
        :langchain:`QdrantVectorStore <qdrant/qdrant/langchain_qdrant.qdrant.QdrantVectorStore.html#langchain_qdrant.qdrant.QdrantVectorStore>`
            Loaded vector store instance configured with dense and sparse embeddings.
        """
        from langchain_qdrant import FastEmbedSparse, RetrievalMode
        retrieval_mode_dict = \
            {"dense": RetrievalMode.DENSE, "sparse": RetrievalMode.SPARSE, "hybrid": RetrievalMode.HYBRID}
        retrieval_mode = retrieval_mode_dict.get(self.config.retrieval_mode.value)
        cache_dir = os.environ.get("FASTEMBED_CACHE_PATH")
        local_files_only = True if cache_dir is not None else False
        qdrant_vectorstore = QdrantVectorStore.from_existing_collection(
            url=self.url,
            collection_name=self.config.name,
            embedding=embeddings_model.model,
            sparse_embedding=FastEmbedSparse(
                model_name="Qdrant/bm25", cache_dir=cache_dir, local_files_only=local_files_only
            ),
            vector_name="",
            sparse_vector_name="sparse",
            content_payload_key="page_content",
            metadata_payload_key="metadata",
            retrieval_mode=retrieval_mode,
            force_recreate=False,
        )
        self.vectorstore = qdrant_vectorstore
        return qdrant_vectorstore

    def get_n_documents_in_vectorstore(self) -> int:
        """
        Count stored points/documents in the current collection.

        Returns
        -------
        int
            Exact number of stored points.
        """
        return self.vectorstore.client.count(self.vectorstore.collection_name, exact=True).count


class FaissVectorstore(Vectorstore):
    """
    Vector store adapter backed by FAISS (local index files).

    .. rubric:: Notes

    If persisted, stores two files in ``config.folder``: ``<name>.faiss`` (index) and ``<name>.pkl`` (docstore).
    """
    vectorstore: FAISS = None #: :langchain:`FAISS <community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html>` : LangChain FAISS vector store instance when loaded.

    def vectorstore_exists(self) -> bool:
        """
        Check whether both FAISS artifacts (index and docstore) exist on disk.

        Returns
        -------
        bool
            ``True`` if both ``.faiss`` and ``.pkl`` files are present, ``False`` otherwise.
        """
        vectorstore_embedding_path = self.config.folder / (self.config.name + ".faiss")
        vectorstore_doc_path = self.config.folder / (self.config.name + ".pkl")
        return vectorstore_embedding_path.is_file() and vectorstore_doc_path.is_file()

    def load_vectorstore(self, embeddings_model: EmbeddingConfig, **kwargs) -> FAISS:
        """
        Load the FAISS index from local files as a LangChain vector store.

        Parameters
        ----------
        embeddings_model : :class:`cardiology_gen_ai.models.EmbeddingConfig`
            Embedding backend to bind.

        Returns
        -------
        :langchain:`FAISS <community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html>`
            Loaded vector store instance.
        """
        faiss_vectorstore = FAISS.load_local(
            folder_path=self.config.folder.as_posix(),
            index_name=self.config.name,
            embeddings=embeddings_model.model,
            allow_dangerous_deserialization=True,
        )
        self.vectorstore = faiss_vectorstore
        return faiss_vectorstore

    def get_n_documents_in_vectorstore(self) -> int:
        """
        Return the number of vectors stored in the index.

        Returns
        -------
        int
            Number of stored vectors.
        """
        return int(self.vectorstore.index.ntotal)
