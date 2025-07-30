"""
Embedding pipeline module.

This module defines the main entry point for running the embedding pipeline,
which involves processing EPUB documents, generating embeddings, and storing
them in a vector database. Configuration is managed via Hydra.
"""

import logging
import os

import hydra
from embeddings.image_embeddings import ImageEmbeddings
from embeddings.text_embeddings import TextEmbeddings
from preprocessing.epub_processor import EPUBProcessor
from utils.general_utils import setup_logging


@hydra.main(version_base=None, config_path="../conf", config_name="embeddings.yaml")
def main(cfg):
    """Main function to execute the embedding pipeline.

    This function performs the following steps:
    1. Sets up logging configuration.
    2. Loads and preprocesses EPUB documents using `EPUBProcessor`.
    3. Generates embeddings and stores them in a vector database using `PerformEmbeddings`.

    Args:
        cfg (omegaconf.DictConfig): Hydra configuration object containing
            all necessary parameters for the embedding pipeline.
    """
    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.\n")
    setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "conf", "logging.yaml"
        )
    )

    epub_processor = EPUBProcessor(cfg=cfg, logger=logger)
    extracted_documents, saved_images, metadata_list = epub_processor.load()

    text_embeddings = TextEmbeddings(
        cfg=cfg, documents=extracted_documents, logger=logger
    )
    text_embeddings.generate_text_vectordb()

    image_embeddings = ImageEmbeddings(
        cfg=cfg,
        saved_images=saved_images,
        metadata_list=metadata_list,
        logger=logger,
    )
    image_embeddings.generate_image_vectordb()


if __name__ == "__main__":
    main()
