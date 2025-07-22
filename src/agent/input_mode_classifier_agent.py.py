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
        self.cfg = cfg
        self.llm = llm
        self.tools = tools
        self.logger = logger
        super().__init__(cfg=cfg, llm=llm, tools=tools, logger=logger)
