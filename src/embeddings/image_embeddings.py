import logging
import os
import re
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from omegaconf import DictConfig


class ImageEmbeddings:
    """
    Handles document embedding and FAISS vector database operations.

    This class provides functionality for text splitting, embedding generation
    using HuggingFace models, and saving the embeddings in a FAISS vector store.
    """

    def __init__(
        self,
        cfg: DictConfig,
        saved_images: List[str],
        metadata_list: List[dict],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.saved_images = saved_images
        self.metadata_list = metadata_list

        if len(saved_images) != len(metadata_list):
            raise ValueError("Mismatch between number of saved images and metadata.")

        self.embedding_model = OpenCLIPEmbeddings(
            model=None,
            model_name=self.cfg.embeddings.image_embeddings.model_name,
            checkpoint=self.cfg.embeddings.image_embeddings.checkpoint,
            preprocess=None,
            tokenizer=None,
        )
        model_name_cleaned: str = re.sub(
            r'[<>:"/\\|?*]', "_", self.cfg.embeddings.image_embeddings.model_name
        )
        self.persist_directory: str = os.path.join(
            self.cfg.embeddings.image_embeddings.embeddings_path,
            model_name_cleaned,
        )

        os.makedirs(self.persist_directory, exist_ok=True)
        self.logger.info(f"Image embeddings will be saved @ {self.persist_directory}.")

    def generate_image_vectordb(self):
        if not self.saved_images:
            raise ValueError("No image paths provided.")

        self.logger.info(f"Embedding {len(self.saved_images)} images with CLIP.")

        vector_store = Chroma(
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory,
        )
        vector_store.add_images(uris=self.saved_images, metadatas=self.metadata_list)
        self.logger.info(
            f"Successfully generated and saved image embeddings @ {self.persist_directory}."
        )
