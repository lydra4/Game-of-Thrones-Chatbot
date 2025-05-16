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
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker


class PerformEmbeddings:
    """
    Handles document embedding and FAISS vector database operations.

    This class provides functionality for text splitting, embedding generation
    using HuggingFace models, and saving the embeddings in a FAISS vector store.

    Attributes:
        cfg (omegaconf.DictConfig): Configuration settings for text splitting
            and embedding generation.
        documents (List[Document]): A list of LangChain Document objects to process.
        logger (logging.Logger): Logger instance used to log messages.
        texts (List[Document]): The resulting list of split text documents.
        embedding_model (Optional[HuggingFaceInstructEmbeddings]):
            The loaded HuggingFace embedding model.
        embeddings_path (Optional[str]): The local path where embeddings will be stored.
        embeddings_model_name (Optional[str]): The cleaned name of the embedding model.
    """

    def __init__(
        self,
        cfg: omegaconf.DictConfig,
        documents: List[Document],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initializes the PerformEmbeddings instance with configuration, documents, and logger.

        Args:
            cfg (omegaconf.DictConfig): Configuration dictionary with settings
                for embedding and text splitting.
            documents (List[Document]): A list of LangChain Document objects to be embedded.
            logger (Optional[logging.Logger], optional): Optional logger instance for
                debugging or tracking. Defaults to None.
        """
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.documents = documents
        self.texts: List[Document] = []
        self.embedding_model: Optional[HuggingFaceInstructEmbeddings] = None
        self.embeddings_path: Optional[str] = None
        self.embeddings_model_name: Optional[str] = None

    def _split_text(self) -> List[Document]:
        """
        Splits the input documents into smaller chunks using the configured text splitter.

        The splitter is selected based on the configuration. Supported types include:
        - RecursiveCharacterTextSplitter
        - SentenceTransformersTokenTextSplitter
        - SemanticChunker

        Returns:
            List[Document]: A list of LangChain Document objects after splitting.
        """
        self.embedding_model = self._load_embeddings_model()

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
                embeddings=self.embedding_model,
                breakpoint_threshold_type=self.cfg.text_splitter.text_splitter.breakpoint_threshold_type,
            )

        self.texts = text_splitter.split_documents(self.documents)
        self.logger.info(f"Text split into {len(self.texts)} parts.")

        return self.texts

    def _load_embeddings_model(self) -> HuggingFaceInstructEmbeddings:
        """
        Loads the HuggingFace embedding model to either GPU or CPU based on availability.

        Returns:
            HuggingFaceInstructEmbeddings: The initialized embedding model ready for inference.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Embedding model will be loaded to {device}.\n")

        model_config = {
            "model_name": self.cfg.embeddings.embeddings_model.model_name,
            "show_progress": self.cfg.embeddings.embeddings_model.show_progress,
            "model_kwargs": {"device": device},
        }
        self.embedding_model = HuggingFaceInstructEmbeddings(**model_config)

        self.logger.info(f"Embedding Model loaded to {device.upper()}.\n")

        return self.embedding_model

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
