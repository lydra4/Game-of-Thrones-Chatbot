import logging
import os
import re
from typing import List, Optional

from langchain.docstore.document import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from omegaconf import DictConfig


class TextEmbeddings:
    def __init__(
        self,
        cfg: DictConfig,
        documents: List[Document],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.cfg = cfg
        self.documents = documents
        self.logger = logger or logging.getLogger(__name__)
        self.embeddings_model_name = re.sub(
            r'[<>:"/\\|?*]', "_", self.cfg.text_embeddings.model_name
        )
        self.persist_directory = os.path.join(
            self.cfg.text_embeddings.embeddings_path,
            self.cfg.text_splitter.name,
            self.embeddings_model_name,
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

    def _embed_documents(self) -> FAISS:
        """
        Embeds the documents using the loaded embedding model and saves them as a FAISS vector store.

        The function ensures the text is split before embedding. It constructs an output path,
        generates the embeddings, and saves them locally.

        Returns:
            FAISS: The FAISS vector store containing document embeddings.
        """
        if not self.texts:
            self._split_text()

        self.embeddings_model_name = re.sub(
            r'[<>:"/\\|?*]',
            "_",
            self.cfg.embeddings.embeddings_model.model_name.split("/")[-1],
        )

        self.embeddings_path = os.path.join(
            self.cfg.embeddings.embed_documents.embeddings_path,
            self.cfg.text_splitter.name,
            self.embeddings_model_name,
        )
        os.makedirs(self.embeddings_path, exist_ok=True)
        self.logger.info(f"Embeddings will be saved @ {self.embeddings_path}\n")

        self.logger.info("Generating Vector Embeddings.\n")

        vectordb = FAISS.from_documents(
            documents=self.texts, embedding=self.embedding_model
        )

        self.logger.info("Saving Vector Embeddings.\n")

        vectordb.save_local(
            folder_path=self.embeddings_path,
            index_name=self.cfg.embeddings.embed_documents.index_name,
        )

        self.logger.info("Successfully saved.\n")

    def generate_vectordb(self) -> FAISS:
        """
        Generates and saves the vector store from the provided documents.

        This is the public method that orchestrates the document splitting, embedding,
        and FAISS vector store creation and saving.

        Returns:
            FAISS: The saved FAISS vector store containing embeddings.
        """
        self.logger.info("Starting document processing and generating embeddings.\n")
        self._split_text()
        self._embed_documents()
