import logging
import os
import re
from typing import List, Optional

import omegaconf
import torch
from langchain.docstore.document import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_milvus import Milvus


class TextEmbeddings:
    def __init__(
        self,
        cfg: omegaconf.DictConfig,
        documents: List[Document],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.cfg = cfg
        self.documents = documents
        self.logger = logger or logging.getLogger(__name__)
        self.uri_path: str = os.path.join(
            self.cfg.embeddings.text_embeddings.embeddings_path,
            self.cfg.embeddings.text_embeddings.index_name,
        )
        self.embeddings_model_name: str = re.sub(
            r'[<>:"/\\|?*]',
            "_",
            self.cfg.embeddings.text_embeddings.model_name.split("/")[-1],
        )
        self.embeddings_path: str = os.path.join(
            self.cfg.embeddings.text_embeddings.embeddings_path,
            self.cfg.text_splitter.name,
            self.embeddings_model_name,
        )

        os.makedirs(self.embeddings_path, exist_ok=True)
        os.makedirs(self.uri_path, exist_ok=True)

    def _load_embeddings_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Embedding model will be loaded to {device}.\n")

        model_config = {
            "model_name": self.cfg.embeddings.text_embeddings.model_name,
            "show_progress": self.cfg.embeddings.text_embeddings.show_progress,
            "model_kwargs": {"device": device},
        }
        embedding_model = HuggingFaceInstructEmbeddings(**model_config)
        self.logger.info(f"Embedding Model loaded to {device.upper()}.\n")

        return embedding_model

    def _split_text(
        self,
        embedding_model: HuggingFaceInstructEmbeddings,
    ) -> List[Document]:
        self.logger.info(f"Using {self.cfg.text_splitter.name.replace('_', ' ')}.\n")

        if self.cfg.text_splitter.name.lower() == "recursive_character_text_splitter":
            text_splitter = RecursiveCharacterTextSplitter(
                **self.cfg.text_splitter.text_splitter
            )

        elif (
            self.cfg.text_splitter.name.lower()
            == "sentence_transformers_token_text_splitter"
        ):
            text_splitter = SentenceTransformersTokenTextSplitter(
                model_name=self.cfg.embeddings.text_embeddings.model_name,
                chunk_overlap=self.cfg.text_splitter.text_splitter.chunk_overlap,
            )

        elif self.cfg.text_splitter.name.lower() == "semantic_chunker":
            text_splitter = SemanticChunker(
                embeddings=embedding_model,
                breakpoint_threshold_type=self.cfg.text_splitter.text_splitter.breakpoint_threshold_type,
            )

        else:
            raise ValueError(f"Unknown text splitter: {self.cfg.text_splitter.name}.")

        chunks = text_splitter.split_documents(self.documents)
        self.logger.info(f"Text split into {len(chunks)} parts.")

        return chunks

    def _embed_document(
        self,
        embedding_model: HuggingFaceInstructEmbeddings,
        chunks: List[Document],
    ):
        self.logger.info(
            f"Generating Vector Embeddings, it will be saved @ {self.embeddings_path}.\n"
        )

        Milvus.from_documents(
            documents=chunks,
            embedding=embedding_model,
            drop_old=True,
            connection_args={"uri": self.uri_path},
        )

        self.logger.info("Successfully generated and saved Vector Embeddings.\n")

    def generate_vectordb(self):
        self.logger.info("Embedding text.")
        embedding_model = self._load_embeddings_model()
        chunks = self._split_text(embedding_model=embedding_model)
        self._embed_document(embedding_model=embedding_model, chunks=chunks)
