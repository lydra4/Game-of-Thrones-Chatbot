"""
Embedding pipeline module.

This module defines the main entry point for running the embedding pipeline,
which involves processing EPUB documents, generating embeddings, and storing
them in a vector database. Configuration is managed via Hydra.
"""

import logging
import os

import hydra
from embeddings.perform_embeddings import PerformEmbeddings
from preprocessing.epub_processor import EPUBProcessor
from utils.general_utils import setup_logging


@hydra.main(version_base=None, config_path="../conf", config_name="training.yaml")
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
    extracted_documents, all_saved_image_paths, all_metadata_list = (
        epub_processor.load()
    )

    perform_embeddings = PerformEmbeddings(
        cfg=cfg, logger=logger, documents=extracted_documents
    )
    perform_embeddings.generate_vectordb()


if __name__ == "__main__":
    main()
