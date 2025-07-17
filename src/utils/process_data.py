"""
EPUB Processing module.

This module defines the EPUBProcessor class used to extract, clean, and preprocess text from EPUB files.
It uses Apache Tika for content extraction and supports custom preprocessing rules via configuration.
"""

import logging
import os
import re
from typing import List, Optional

import ebooklib
import ebooklib.epub
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from omegaconf import DictConfig
from tqdm import tqdm


class EPUBProcessor(BaseLoader):
    """Processes EPUB files by extracting and cleaning text.

    This class scans a directory for `.epub` files, extracts their text using
    BeautifulSoup, and applies configurable preprocessing steps such as
    whitespace normalization, URL removal, and filtering of non-alphanumeric
    characters.

    Attributes:
        cfg (omegaconf.DictConfig): Configuration dictionary containing preprocessing settings.
        logger (logging.Logger): Logger instance for logging messages.
    """

    def __init__(
        self, cfg: DictConfig, logger: Optional[logging.Logger] = None
    ) -> None:
        """Initializes the EPUBProcessor instance.

        Args:
            cfg (omegaconf.dictconfig): Configuration dictionary containing preprocessing settings.
            logger (Optional[logging.Logger], optional): Logger instance for logging messages.
                If not provided, a default logger will be created.
        """
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

    def _preprocess_text(self, text: str) -> str:
        """Cleans and preprocesses the extracted text using configurable regex rules.

        This includes:
            - Whitespace normalization
            - URL removal
            - Non-alphanumeric filtering
            - Consecutive non-alphanumeric character replacement
            - Newline normalization

        Args:
            text (str): Raw text extracted from an EPUB file.

        Returns:
            str: Cleaned and preprocessed text.
        """
        text = re.sub(
            self.cfg.preprocessing.whitespace,
            self.cfg.preprocessing.whitespace_replacement,
            text,
        )
        text = re.sub(
            self.cfg.preprocessing.url, self.cfg.preprocessing.url_replacement, text
        )
        text = re.sub(
            self.cfg.preprocessing.non_alphanumeric,
            self.cfg.preprocessing.non_alphanumeric_replacement,
            text,
        )
        text = re.sub(
            self.cfg.preprocessing.consecutive_non_alphanumeric,
            self.cfg.preprocessing.consecutive_non_alphanumeric_replacement,
            text,
        )
        text = re.sub(
            self.cfg.preprocessing.normalizes_newlines,
            self.cfg.preprocessing.two_lines,
            text,
        )

        return text

    def _extract_preprocess_text_from_epub(self, epub_path: str) -> str:
        """Extracts and preprocesses text from a single EPUB file.

        This method parses the EPUB using `ebooklib`, extracts text from HTML
        content using BeautifulSoup, and then preprocesses the combined text.

        Args:
            epub_path (str): Path to the EPUB file.

        Returns:
            str: Preprocessed text content extracted from the EPUB file.
        """
        book = ebooklib.epub.read_epub(name=epub_path)
        all_text = []

        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), "html.parser")
                all_text.append(soup.get_text(separator="\n"))

        return self._preprocess_text(text="\n".join(all_text).strip())

    def load(self) -> List[Document]:
        """Loads and processes all EPUB files in the configured directory.

        This method searches for `.epub` files in the specified path,
        extracts and preprocesses the text content from each file, and
        returns them as a list of LangChain `Document` objects.

        Returns:
            List[Document]: A list of documents with cleaned page content and metadata.

        Raises:
            FileNotFoundError: If the specified directory doesn't exist or contains no EPUB files.
            ValueError: If no valid content is extracted from the EPUB files.
        """

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

        extracted_documents = []

        for epub_file in tqdm(epub_files, desc="Processing EPUBs"):
            book_name = os.path.splitext(os.path.basename(epub_file))[0]
            self.logger.info(f"Processing {book_name}")

            try:
                preprocess_text = self._extract_preprocess_text_from_epub(
                    epub_path=epub_file
                )

                if not preprocess_text:
                    self.logger.warning(f"No text extracted from {book_name}")
                    continue

                extracted_documents.append(
                    Document(
                        page_content=preprocess_text, metadata={"source": book_name}
                    )
                )
                self.logger.info(f"Successfully processed {book_name}.")

            except Exception as e:
                self.logger.error(f"Error Processing {book_name}: {e}", exc_info=True)

        if not extracted_documents:
            raise ValueError(
                f"No valid text extracted from {self.cfg.preprocessing.path}"
            )

        return extracted_documents
