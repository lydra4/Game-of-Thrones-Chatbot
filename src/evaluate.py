"""
Evaluation module to run the evaluation pipeline.

This module defines the main entry point for executing the EvaluationPipeline
using configuration managed by Hydra. It sets up logging, initializes the pipeline,
and triggers the evaluation process.
"""

import logging
import os

import hydra
from evaluation.evaluation_pipeline import EvaluationPipeline
from omegaconf import DictConfig
from utils.general_utils import setup_logging


@hydra.main(version_base=None, config_path="../conf", config_name="evaluate.yaml")
def main(cfg: DictConfig):
    """Main function to set up logging and run the evaluation pipeline.

    Args:
        cfg (omegaconf.dictconfig): Configuration dictionary provided by Hydra
            containing all necessary settings for evaluation.

    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration\n")
    setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "conf", "logging.yaml"
        )
    )

    evaluation_pipeline = EvaluationPipeline(cfg=cfg, logger=logger)
    evaluation_pipeline.evaluation()


if __name__ == "__main__":
    main()
