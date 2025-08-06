import logging
import os
import re
from typing import Dict, List, Optional

from langchain.docstore.document import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from langchain_chroma.vectorstores import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
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
        self.image_paths = image_paths
        self.metadata_list = metadata_list
        self.logger: logging.Logger = logger or logging.getLogger(__name__)
        self.clip = OpenCLIPEmbeddings(
            model_name=self.cfg.embeddings.model_name,
            checkpoint=self.cfg.embeddings.checkpoint,
            model=None,
            preprocess=None,
            tokenizer=None,
        )

        self.cfg.embeddings.model_name_cleaned = re.sub(
            r'[<>:"/\\|?*]', "_", self.cfg.embeddings.model_name
        )
        self.persist_directory: str = os.path.join(
            self.cfg.embeddings.embeddings_path,
            self.cfg.text_splitter.name,
            self.cfg.embeddings.model_name_cleaned,
        )
        os.makedirs(self.persist_directory, exist_ok=True)

        self.chroma = Chroma(
            collection_name="GOT",
            embedding_function=self.clip,
            persist_directory=self.persist_directory,
        )

    def _split_text(
        self,
    ) -> (
        RecursiveCharacterTextSplitter
        | SentenceTransformersTokenTextSplitter
        | SemanticChunker
    ):
        self.logger.info(f"Using {self.cfg.text_splitter.name.replace('_', ' ')}.\n")

        if self.cfg.text_splitter.name.lower() == "recursive_character_text_splitter":
            text_splitter = RecursiveCharacterTextSplitter(
                separators=self.cfg.text_splitter.separators,
                chunk_size=self.cfg.text_splitter.chunk_size,
                chunk_overlap=self.cfg.text_splitter.chunk_overlap,
            )

        elif (
            self.cfg.text_splitter.name.lower()
            == "sentence_transformers_token_text_splitter"
        ):
            text_splitter = SentenceTransformersTokenTextSplitter(
                model_name=self.cfg.text_splitter.model_name,
                chunk_overlap=self.cfg.text_splitter.chunk_overlap,
            )

        elif self.cfg.text_splitter.name.lower() == "semantic_chunker":
            text_splitter = SemanticChunker(
                embeddings=self.cfg.text_splitter.model_name,
                breakpoint_threshold_type=self.cfg.text_splitter.breakpoint_threshold_type,
            )
        else:
            raise ValueError(f"Unknown text splitter: {self.cfg.text_splitter.name}.")

        return text_splitter

    def generate_vectordb(self):
        self.logger.info("Generating Embeddings.")
        image_ids = [f"image_{i}" for i in range(len(self.image_paths))]
        self.chroma.add_images(
            ids=image_ids,
            uris=self.image_paths,
            metadatas=self.metadata_list,
        )
        text_splitter = self._split_text()
        texts = text_splitter.split_documents(documents=self.documents)
        text_ids = [f"text_{i}" for i in range(len(texts))]
        self.chroma.add_documents(ids=text_ids, documents=texts)
