import logging
from typing import Any, List, Optional

from agent.base_agent import BaseAgent
from langchain.tools import BaseTool
from omegaconf import DictConfig


class InputModeClassifierAgent(BaseAgent):
    def __init__(
        self,
        cfg: DictConfig,
        llm: Any,
        tools: Optional[List[BaseTool]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(cfg=cfg, llm=llm, tools=tools, logger=logger)

    def run(self, query: str):
        return super().run(input_data=query)
