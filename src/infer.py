"""
Inference module to run the inference pipeline.

This module defines the main entry point for executing the InferencePipeline
using configuration managed by Hydra. It handles logging setup, pipeline
initialization, and triggering the inference execution.
"""

import logging
import os

import hydra
from inference.inference_pipeline import InferencePipeline
from utils.general_utils import setup_logging


@hydra.main(version_base=None, config_path="../conf", config_name="inference.yaml")
def main(cfg):
    """Main function to set up logging and execute the inference pipeline.

    Args:
            cfg (omegaconf.DictConfig): Hydra configuration object containing
                        all necessary parameters for inference.
    """
    print(cfg)
    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration")
    setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "conf", "logging.yaml"
        )
    )

    infer_pipeline = InferencePipeline(cfg=cfg, logger=logger)
    infer_pipeline.run_inference()


if __name__ == "__main__":
    main()
