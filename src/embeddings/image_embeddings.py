import logging
import os
from typing import List, Optional

import ebooklib
import omegaconf
from ebooklib import epub
from langchain.docstore.document import Document as TextDocument
from langchain.schema import Document as LCDocument
from langchain.vectorstores import FAISS
from langchain_experimental.open_clip import OpenCLIPEmbeddings


class ImageEmbeddings:
    """
    Handles document embedding and FAISS vector database operations.

    This class provides functionality for text splitting, embedding generation
    using HuggingFace models, and saving the embeddings in a FAISS vector store.
    """

    def __init__(
        self,
        cfg: omegaconf.DictConfig,
        documents: List[TextDocument],
        logger: logging.Logger,
    ) -> None:
        self.cfg = cfg
        self.logger = logger
        self.documents = documents
        self.embeddings_path: Optional[str] = None
        self.embedding_model = OpenCLIPEmbeddings(
            model=None,
            model_name=self.cfg.image_embedding.model_name,
            checkpoint=self.cfg.image_embedding.checkpoint,
            preprocess=None,
            tokenizer=None,
        )

    def _extract_images_from_epub(self, epub_path: str) -> List[tuple]:
        book_name = os.path.splitext(os.path.basename(epub_path))[0]
        book = epub.read_epub(epub_path)
        images = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_IMAGE:
                images.append((item.get_content(), item.get_name(), book_name))
        return images

    def embed_images(self):
        all_bytes = []
        metadata_list = []
        for document in self.documents:
            book_name = document.metadata.get("source")
            epub_path = os.path.join(self.cfg.preprocessing.path, f"{book_name}.epub")
            if not os.path.isfile(epub_path):
                self.logger.warning(f"EPUB not found for '{book_name}': {epub_path}")
                continue

            extracted_images = self._extract_images_from_epub(epub_path)
            for image_bytes, image_name, _ in extracted_images:
                all_bytes.append(image_bytes)
                metadata_list.append({"source": book_name, "image_name": image_name})

        if not all_bytes:
            raise ValueError("No images found in provided EPUB documents.")

        self.logger.info(f"Embedding {len(all_bytes)} images with CLIP.")
        image_embeddings = self.embedding_model.embed_documents(all_bytes)

        lc_docs = [
            LCDocument(page_content="Game of Thrones Image", metadata=meta)
            for meta in metadata_list
        ]
        store = FAISS.from_documents(image_embeddings, self.embedding_model, lc_docs)
