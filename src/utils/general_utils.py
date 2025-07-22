import logging
import logging.config
import os

import yaml
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


def setup_logging(
    logging_config_path="../conf/logging.yaml", default_level=logging.INFO
):
    """
    Logging configuration module.

    This module provides functionality to set up logging using a YAML configuration file.
    If the configuration file is missing or invalid, it defaults to basic logging with a specified level.

    Attributes:
        logger (logging.Logger): Logger used to capture logs during setup.

    Functions:
        setup_logging(logging_config_path, default_level): Initializes logging from YAML or falls back to basic config.
    """
    try:
        os.makedirs("logs", exist_ok=True)
        with open(logging_config_path, encoding="utf-8") as file:
            log_config = yaml.safe_load(file.read())
        logging.config.dictConfig(log_config)

    except Exception as error:
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=default_level,
        )
        logger.info(error)
        logger.info("Logging config file is not found. Basic config is used.")


def load_embedding_model(model_name: str, show_progress: bool, **kwargs):
    logger.info("Loading embedding model.")
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name, show_progress=show_progress, model_kwargs={**kwargs}
        )
    except Exception as e:
        logger.error(f"Failed to load embedding mode: '{model_name}': {e}.")
        raise ValueError(e)
    logger.info("Embedding model loaded.")
    return embedding_model
