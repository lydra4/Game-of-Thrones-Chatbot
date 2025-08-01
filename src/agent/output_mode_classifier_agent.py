import ast
import logging
from typing import Any, List, Optional, Union

from agent.base_agent import BaseAgent
from omegaconf import DictConfig


class OutputModeClassifierAgent(BaseAgent):
    def __init__(
        self,
        cfg: DictConfig,
        llm: Any,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(cfg=cfg, llm=llm, logger=logger)

    def run(self, input_data: Union[str, List[str]]):
        self.logger.info(f"Classifying query: {input_data}")
        input_mode = super().run(input_data=input_data)
        input_mode = input_mode.content
        if not input_mode:
            raise ValueError("Output agent return an empty list.")

        input_mode = ast.literal_eval(input_mode)
        self.logger.info(f"Query: {input_data}, classified as {input_mode}.")
        return input_mode
