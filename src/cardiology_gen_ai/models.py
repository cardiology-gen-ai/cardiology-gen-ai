import os
import pathlib
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any
from pydantic import BaseModel, ConfigDict

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from transformers import AutoConfig


class EmbeddingConfig(BaseModel):
    model_name: str
    model: HuggingFaceEmbeddings = None
    dim: int

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "EmbeddingConfig":
        kwargs = config_dict["kwargs"]
        if not isinstance(kwargs, dict):
            kwargs = dict()
        encode_kwargs = {k: v for k, v in kwargs.items() if k in ["normalize_embeddings", "prompt_name"]}
        model_kwargs = {k: v for k, v in kwargs.items() if k in ["device"]}
        if model_kwargs.get("device", None) is None:
            model_kwargs["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = config_dict["deployment"]
        model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        dim = AutoConfig.from_pretrained(model_name).hidden_size
        return cls(model_name=model_name, model=model, dim=dim)


class IndexTypeNames(Enum):
    qdrant = "qdrant"
    faiss = "faiss"


class DistanceTypeNames(Enum):
    cosine = "cosine"
    euclidean = "euclidean"


class RetrievalTypeNames(Enum):
    dense = "dense"
    sparse = "sparse"
    hybrid = "hybrid"


class IndexingConfig(BaseModel):
    name: str
    description: str
    # TODO: change afterwards [check how to handle .env variables in this repo]
    # folder: pathlib.Path = pathlib.Path(os.getenv("INDEX_ROOT", pathlib.Path(__file__).parent.parent.parent / "index"))
    folder: pathlib.Path = pathlib.Path(os.getenv("INDEX_ROOT"))
    type: IndexTypeNames
    distance: DistanceTypeNames
    retrieval_mode: RetrievalTypeNames = RetrievalTypeNames.dense

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "IndexingConfig":
        index_type = IndexTypeNames(config_dict["type"])
        distance = DistanceTypeNames(config_dict["distance"])
        retrieval = config_dict.get("retrieval_mode", RetrievalTypeNames.dense)
        retrieval_mode = RetrievalTypeNames(retrieval) if isinstance(retrieval, str) is not None else retrieval
        other_config_dict = {k: v for k, v in config_dict.items() if k not in ["type", "distance", "retrieval_mode"]}
        return cls(type=index_type, distance=distance, retrieval_mode=retrieval_mode, **other_config_dict)


class Vectorstore(BaseModel, ABC):
    config: IndexingConfig
    vectorstore: QdrantVectorStore | FAISS = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def vectorstore_exists(self) -> bool:
        pass

    @abstractmethod
    def load_vectorstore(self, **kwargs) -> QdrantVectorStore | FAISS:
        pass

    @abstractmethod
    def get_n_documents_in_vectorstore(self) -> int:
        pass


class QdrantVectorstore(Vectorstore):
    url: str = os.getenv("QDRANT_URL")  # TODO: maybe should be changed
    client: QdrantClient = QdrantClient(url)
    vectorstore: QdrantVectorStore = None

    def vectorstore_exists(self) -> bool:
        return any(collection.name == self.config.name for collection in self.client.get_collections().collections)

    def load_vectorstore(self, embeddings_model: EmbeddingConfig, retrieval_mode: str) -> QdrantVectorStore:
        retrieval_mode_dict = \
            {"dense": RetrievalMode.DENSE, "sparse": RetrievalMode.SPARSE, "hybrid": RetrievalMode.HYBRID}
        retrieval_mode = retrieval_mode_dict.get(self.config.retrieval_mode.value)
        qdrant_vectorstore = QdrantVectorStore.from_existing_collection(
            url=self.url,
            collection_name=self.config.name,
            embedding=embeddings_model.model,
            sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
            vector_name="dense",
            sparse_vector_name="sparse",
            content_payload_key="page_content",
            metadata_payload_key="metadata",
            retrieval_mode=retrieval_mode,
        )
        self.vectorstore = qdrant_vectorstore
        return qdrant_vectorstore

    def get_n_documents_in_vectorstore(self) -> int:
        return self.vectorstore.client.count(self.vectorstore.collection_name, exact=True).count


class FaissVectorstore(Vectorstore):
    vectorstore: FAISS = None

    def vectorstore_exists(self) -> bool:
        vectorstore_embedding_path = self.config.folder / (self.config.name + ".faiss")
        vectorstore_doc_path = self.config.folder / (self.config.name + ".pkl")
        return vectorstore_embedding_path.is_file() and vectorstore_doc_path.is_file()

    def load_vectorstore(self, embeddings_model: EmbeddingConfig, **kwargs) -> FAISS:
        faiss_vectorstore = FAISS.load_local(
            folder_path=self.config.folder.as_posix(),
            index_name=self.config.name,
            embeddings=embeddings_model.model,
            allow_dangerous_deserialization=True,
        )
        self.vectorstore = faiss_vectorstore
        return faiss_vectorstore

    def get_n_documents_in_vectorstore(self) -> int:
        return int(self.vectorstore.index.ntotal)
