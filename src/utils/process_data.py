"""
EPUB Processing module.

This module defines the EPUBProcessor class used to extract, clean, and preprocess text from EPUB files.
It uses Apache Tika for content extraction and supports custom preprocessing rules via configuration.
"""

import logging
import os
import re
from typing import List, Optional

import omegaconf
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from tika import parser


class EPUBProcessor(BaseLoader):
    """Processes EPUB files by extracting and cleaning text.

    This class scans a directory for `.epub` files, extracts their text using Apache Tika,
    and applies configurable preprocessing steps such as whitespace normalization,
    URL removal, and filtering of non-alphanumeric characters.

    Attributes:
        cfg (omegaconf.DictConfig): Configuration dictionary containing preprocessing settings.
        logger (logging.Logger): Logger instance for logging messages.
    """

    def __init__(
        self, cfg: omegaconf.dictconfig, logger: Optional[logging.Logger] = None
    ) -> None:
        """Initializes the EPUBProcessor.

        Args:
            cfg (omegaconf.dictconfig): Configuration dictionary containing preprocessing settings.
            logger (Optional[logging.Logger], optional): Logger instance for logging messages.
                If not provided, defaults to a logger with the module name.
        """
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

    def _preprocess_text(self, text: str) -> str:
        """Cleans and preprocesses the extracted text.

        Applies a series of configurable preprocessing rules including:
        - Whitespace normalization
        - URL removal
        - Non-alphanumeric filtering
        - Consecutive character cleanup
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

    def load(self) -> List[Document]:
        """Loads and processes EPUB files from the configured directory.

        Extracts text content from all `.epub` files in the given directory using Tika,
        preprocesses the content, and returns a list of LangChain `Document` objects.

        Returns:
            List[Document]: A list of documents containing the cleaned text and metadata.

        Raises:
            FileNotFoundError: If the directory is not found or contains no EPUB files.
            ValueError: If no valid text is extracted from any EPUB file.
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

        for epub_file in epub_files:
            book_name = os.path.splitext(os.path.basename(epub_file))[0]
            self.logger.info(f"Processing {book_name}")

            try:
                raw_text = (
                    parser.from_file(epub_file).get("content", "").lower().strip()
                )

                if not raw_text:
                    self.logger.warning(f"No text extracted from {book_name}")
                    continue

                cleaned_text = self._preprocess_text(raw_text)
                extracted_documents.append(
                    Document(page_content=cleaned_text, metadata={"source": book_name})
                )
                self.logger.info("Successfull!\n")

            except Exception as e:
                self.logger.error(f"Error Processing {book_name}: {e}", exc_info=True)

        if not extracted_documents:
            raise ValueError(
                f"No valid text extracted from {self.cfg.preprocessing.path}"
            )

        return extracted_documents
