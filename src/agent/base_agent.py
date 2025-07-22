import logging
from typing import Any, List, Optional, Union

from langchain.agents import initialize_agent
from omegaconf import DictConfig
from langchain.tools import BaseTool


class BaseAgent:
    def __init__(
        self,
        cfg: DictConfig,
        llm: Any,
        tools: Optional[List[BaseTool]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.cfg = cfg
        self.llm = llm
        self.tools = tools or []
        self.logger = logger or logging.getLogger(__name__)
        self.agent = initialize_agent(tools=self.tools, llm=self.llm, verbose=True)

        self.logger.debug(f"{self.__class__.__name__} initialized with config: {cfg}")s

    def run(self, input_data: Union[str, List[str]]):
        if not hasattr(self, 'agent'):
            raise NotImplementedError("Each agent must implement its own run method.")
        
        return self.agent.run(input_data)
