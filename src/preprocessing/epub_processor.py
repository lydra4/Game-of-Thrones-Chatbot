import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub
from langchain.docstore.document import Document
from omegaconf import DictConfig


class EPUBProcessor:
    def __init__(
        self, cfg: DictConfig, logger: Optional[logging.Logger] = None
    ) -> None:
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.rules: List = [
            (re.compile(p), r)
            for p, r in [
                (
                    self.cfg.preprocessing.whitespace,
                    self.cfg.preprocessing.whitespace_replacement,
                ),
                (self.cfg.preprocessing.url, self.cfg.preprocessing.url_replacement),
                (
                    self.cfg.preprocessing.non_alphanumeric,
                    self.cfg.preprocessing.non_alphanumeric_replacement,
                ),
                (
                    self.cfg.preprocessing.consecutive_non_alphanumeric,
                    self.cfg.preprocessing.consecutive_non_alphanumeric_replacement,
                ),
                (
                    self.cfg.preprocessing.newlines_pattern,
                    self.cfg.preprocessing.two_lines,
                ),
            ]
        ]

    def _preprocess_text(self, text: str) -> str:
        for pattern, replacement in self.rules:
            text = pattern.sub(replacement, text)
        return text

    def _extract_content_from_epub(
        self,
        epub_path: str,
    ) -> Tuple[str, List[str], List[Dict[str, str]]]:
        book_name = os.path.splitext(os.path.basename(epub_path))[0]
        output_dir = os.path.join(self.cfg.image_embeddings.out_dir, book_name)
        os.makedirs(output_dir, exist_ok=True)

        all_text: List[str] = []
        saved_image_paths: List[str] = []
        metadata_list: List[Dict[str, str]] = []

        book = epub.read_epub(epub_path)

        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), "html.parser")
                all_text.append(soup.get_text(separator="\n"))

            elif item.get_type() == ebooklib.ITEM_IMAGE:
                image_name = item.get_name()
                clean_name = os.path.basename(image_name)
                full_path = os.path.join(output_dir, clean_name)

                with open(full_path, "wb") as f:
                    f.write(item.get_content())

                saved_image_paths.append(full_path)
                metadata_list.append(
                    {
                        "image_name": image_name,
                        "book_name": book_name,
                        "source_path": full_path,
                    }
                )

        raw_text = "\n".join(all_text).lower().strip()
        processed_text = self._preprocess_text(text=raw_text)

        return processed_text, saved_image_paths, metadata_list

    def load(self) -> Tuple[List[Document], List[str], List[Dict[str, str]]]:
        extracted_documents = []
        all_saved_image_paths: List[str] = []
        all_metadata_list: List[Dict[str, str]] = []

        if not os.path.isdir(self.cfg.preprocessing.path):
            raise FileNotFoundError(
                f"Directory not found: {self.cfg.preprocessing.path}"
            )

        epub_files = [
            os.path.join(self.cfg.preprocessing.path, file)
            for file in os.listdir(self.cfg.preprocessing.path)
            if file.endswith(self.cfg.preprocessing.file_extension)
        ]

        if not epub_files:
            raise FileNotFoundError(
                f"No epub files found in {self.cfg.preprocessing.path}"
            )
        for epub_file in epub_files:
            book_name = os.path.splitext(os.path.basename(epub_file))[0]
            self.logger.info(f"Processing {book_name}")

            try:
                processed_text, saved_image_paths, metadata_list = (
                    self._extract_content_from_epub(epub_path=epub_file)
                )

                if not processed_text:
                    self.logger.warning(f"No text extracted from {book_name}")
                    continue

                extracted_documents.append(
                    Document(
                        page_content=processed_text, metadata={"source": book_name}
                    )
                )
                all_saved_image_paths.extend(saved_image_paths)
                all_metadata_list.extend(metadata_list)
                self.logger.info("Successfull!\n")

            except Exception as e:
                self.logger.error(f"Error Processing {book_name}: {e}", exc_info=True)

        if not extracted_documents:
            raise ValueError(
                f"No valid text extracted from {self.cfg.preprocessing.path}"
            )

        return extracted_documents, all_saved_image_paths, all_metadata_list
