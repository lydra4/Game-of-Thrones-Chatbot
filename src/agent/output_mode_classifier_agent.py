import logging
from typing import Any, List, Optional, Union

from agent.base_agent import BaseAgent
from omegaconf import DictConfig
from utils.general_utils import extract_list_from_string


class OutputModeClassifierAgent(BaseAgent):
    def __init__(
        self,
        cfg: DictConfig,
        llm: Any,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(cfg=cfg, llm=llm, logger=logger)

    def run(self, input_data: Union[str, List[str]]):
        self.logger.info(f"Classifying query: '{input_data}'")
        input_mode = super().run(input_data=input_data)
        input_mode = input_mode.content
        input_mode = extract_list_from_string(text=input_mode)
        if not input_mode:
            raise ValueError("Output agent return an empty list.")

        self.logger.info(
            f"Query: {input_data}, classified as '{' and '.join(input_mode)}'."
        )
        return input_mode
