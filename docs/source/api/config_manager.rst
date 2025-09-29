Configuration Manager
=====================

Configuration loader for the common abstractions of the application.

This module defines :pydantic:`Pydantic BaseModel <base_model>` used to configure:

- embedding strategy and model (:class:`~cardiology_gen_ai.models.EmbeddingConfig`);
- abstract vectorstore index (:class:`~cardiology_gen_ai.models.IndexingConfig`);
- concrete vectorsotres scheletons backed by :langchain:`Qdrant <qdrant/qdrant/langchain_qdrant.qdrant.QdrantVectorStore.html#langchain_qdrant.qdrant.QdrantVectorStore>` (:class:`~cardiology_gen_ai.models.QdrantVectorstore`) and :langchain:`FAISS <community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html>` (:class:`~cardiology_gen_ai.models.FaissVectorstore`).

.. automodule:: cardiology_gen_ai.config.manager
   :members:
   :undoc-members:
   :show-inheritance:
