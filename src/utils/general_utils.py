import ast
import logging
import logging.config
import os
import re
from typing import List, Union

import yaml
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai.chat_models import ChatOpenAI

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


def load_embedding_model(model_name: str, show_progress: bool = True, **kwargs):
    logger.info("Loading embedding model.")
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            show_progress=show_progress,
            model_kwargs={**kwargs},
        )
    except Exception as e:
        logger.error(f"Failed to load embedding mode: '{model_name}': {e}.")
        raise ValueError(e)
    logger.info("Embedding model loaded.")
    return embedding_model


def initialize_llm(model_name: str, temperature: Union[float, int]):
    load_dotenv()
    model_name_cleaned = model_name.strip().lower()
    api_key_env_var = (
        "OPENAI_API_KEY" if "gpt-" in model_name_cleaned else "GEMINI_API_KEY"
    )
    api_key = os.getenv(api_key_env_var)

    if not api_key:
        raise ValueError(f"{api_key_env_var} not found in enviroment variables.")

    logger.info(f"Initializing LLM:{model_name_cleaned}")

    try:
        if model_name_cleaned.startswith("gpt-"):
            return ChatOpenAI(
                model=model_name_cleaned,
                temperature=temperature,
                api_key=api_key,
            )

        elif model_name_cleaned.startswith("gemini-"):
            return ChatGoogleGenerativeAI(
                model=model_name_cleaned,
                temperature=temperature,
                api_key=api_key,
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}.")

    except Exception as e:
        logger.error(f"Failed to initialize LLM: {model_name_cleaned}.")
        raise


def extract_list_from_string(text: str) -> List[str]:
    match = re.search(r"\[.*?\]", text)
    if match:
        parsed_list = ast.literal_eval(match.group())

        return parsed_list

    else:
        raise ValueError(f"No list found in text: {text}.")
