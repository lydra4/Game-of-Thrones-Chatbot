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

        model_name_cleaned = re.sub(
            r'[<>:"/\\|?*]', "_", self.cfg.embeddings.model_name
        )
        self.persist_directory: str = os.path.join(
            self.cfg.embeddings.embeddings_path,
            self.cfg.text_splitter.name,
            model_name_cleaned,
        )
        os.makedirs(self.persist_directory, exist_ok=True)

        self.chroma: Chroma = Chroma(
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
            embeddings = HuggingFaceEmbeddings(
                model_name=self.cfg.text_splitter.model_name,
            )
            text_splitter = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type=self.cfg.text_splitter.breakpoint_threshold_type,
            )
        else:
            raise ValueError(f"Unknown text splitter: {self.cfg.text_splitter.name}.")

        return text_splitter

    def _embed_text(
        self,
        documents: List[Document],
        text_splitter,
        chroma_db: Chroma,
    ):
        self.logger.info("Embedding text")
        texts = text_splitter.split_documents(documents=documents)
        text_ids = [f"text_{i}" for i in range(len(texts))]
        chroma_db.add_documents(ids=text_ids, documents=texts)

    def _embed_images(
        self,
        images_path: List[str],
        metadata_list: List[Dict[str, str]],
        chroma_db: Chroma,
    ):
        self.logger.info("Embedding images")
        image_ids = [f"image_{i}" for i in range(len(images_path))]
        chroma_db.add_images(
            ids=image_ids,
            uris=images_path,
            metadatas=metadata_list,
        )

    def generate_vectordb(self):
        text_splitter = self._split_text()
        self._embed_text(
            documents=self.documents,
            text_splitter=text_splitter,
            chroma_db=self.chroma,
        )
        self._embed_images(
            images_path=self.image_paths,
            metadata_list=self.metadata_list,
            chroma_db=self.chroma,
        )
