"""
Embedding pipeline module.

This module defines the main entry point for running the embedding pipeline,
which involves processing EPUB documents, generating embeddings, and storing
them in a vector database. Configuration is managed via Hydra.
"""

import logging
import os

import hydra
from embeddings.embedding_generator import GenerateEmbeddings
from preprocessing.epub_processor import EPUBProcessor
from utils.general_utils import setup_logging


@hydra.main(version_base=None, config_path="../conf", config_name="embeddings.yaml")
def main(cfg):
    print(cfg)
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

    perform_embeddings = GenerateEmbeddings(
        cfg=cfg,
        logger=logger,
        documents=extracted_documents,
        image_paths=all_saved_image_paths,
        metadata_list=all_metadata_list,
    )
    perform_embeddings.generate_vectordb()


if __name__ == "__main__":
    main()
