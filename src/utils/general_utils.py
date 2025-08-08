import logging
import logging.config
import os

import yaml
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


def setup_logging(
    logging_config_path="../conf/logging.yaml", default_level=logging.INFO
):
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


def load_embedding_model(model_name: str, show_progress: bool = True, **kwargs):
    logger.info("Loading embedding model.")
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            show_progress=show_progress,
            model_kwargs={**kwargs},
        )
    except Exception as e:
        logger.error(f"Failed to load embedding model:'{model_name}':{e}")
        raise ValueError(e)
    logger.info("Embedding model loaded.")
    return embedding_model


def get_api_key(env_var: str) -> str:
    api_key = os.getenv(key=env_var)
    if not api_key:
        raise ValueError(f"Missing enviroment variable:{env_var}")
    return api_key
