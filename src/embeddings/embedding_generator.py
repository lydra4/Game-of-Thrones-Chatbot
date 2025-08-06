import logging
import os
import re
from typing import Dict, List, Optional

import torch
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from langchain.docstore.document import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from omegaconf import DictConfig


class GenerateEmbeddings:
    def __init__(
        self,
        cfg: DictConfig,
        documents: List[Document],
        image_paths: List[str],
        metadata_list: List[Dict[str, str]],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.cfg = cfg
        self.documents = documents
        self.logger: logging.Logger = logger or logging.getLogger(__name__)
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_function = OpenCLIPEmbeddingFunction(
            model_name=self.cfg.embeddings.model_name,
            checkpoint=self.cfg.embeddings.checkpoint,
            device=self.device,
        )
        self.cfg.embeddings.model_name_cleaned = re.sub(
            r'[<>:"/\\|?*]', "_", self.cfg.embeddings.model_name
        )
        self.persist_directory: str = os.path.join(
            self.cfg.text_embeddings.embeddings_path,
            self.cfg.text_splitter.name,
            self.cfg.embeddings.model_name_cleaned,
        )

        os.makedirs(self.persist_directory, exist_ok=True)

    def _split_text(self, embedding_model: HuggingFaceEmbeddings) -> List[Document]:
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
                model_name=self.cfg.embeddings.embeddings_model.model_name,
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

    def _embed_documents(
        self,
        embedding_model: HuggingFaceEmbeddings,
        documents: List[Document],
    ):
        self.logger.info(
            f"Generating Embeddings, it will be save at {self.persist_directory}."
        )

        Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=self.persist_directory,
        )

        self.logger.info("Successfully generated and saved Vector Embeddings.")

    def generate_vectordb(self):
        self.logger.info("Starting document processing and generating embeddings.\n")
        self._split_text()
        self._embed_documents()
