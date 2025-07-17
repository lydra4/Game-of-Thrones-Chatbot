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
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


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
        self.embeddings_model_name: str = re.sub(
            r'[<>:"/\\|?*]',
            "_",
            self.cfg.embeddings.text_embeddings.model_name.split("/")[-1],
        )
        self.persist_directory: str = os.path.join(
            self.cfg.embeddings.text_embeddings.embeddings_path,
            self.cfg.text_splitter.name,
            self.embeddings_model_name,
        )

        os.makedirs(self.persist_directory, exist_ok=True)

    def _load_embeddings_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Embedding model will be loaded to {device}.")
        embedding_model = HuggingFaceEmbeddings(
            model_name=self.cfg.embeddings.text_embeddings.model_name,
            show_progress=self.cfg.embeddings.text_embeddings.show_progress,
            model_kwargs={"device": device},
        )
        self.logger.info(f"Embedding Model loaded to {device.upper()}.")

        return embedding_model

    def _split_text(
        self,
        embedding_model: HuggingFaceEmbeddings,
    ) -> List[Document]:
        self.logger.info(f"Using {self.cfg.text_splitter.name.replace('_', ' ')}.")

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

        documents = text_splitter.split_documents(self.documents)
        self.logger.info(f"Text split into {len(documents)} parts.")

        return documents

    def _embed_document(
        self,
        embedding_model: HuggingFaceEmbeddings,
        documents: List[Document],
    ):
        self.logger.info(
            f"Generating Vector Embeddings, it will be saved @ {self.persist_directory}."
        )

        Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=self.persist_directory,
        )

        self.logger.info("Successfully generated and saved Vector Embeddings.")

    def generate_vectordb(self):
        self.logger.info("Embedding text.")
        embedding_model = self._load_embeddings_model()
        documents = self._split_text(embedding_model=embedding_model)
        self._embed_document(embedding_model=embedding_model, documents=documents)
