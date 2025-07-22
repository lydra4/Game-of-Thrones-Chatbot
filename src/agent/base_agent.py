import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

from langchain.agents import initialize_agent
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from omegaconf import DictConfig


class BaseAgent(ABC):
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

        self.logger.debug(f"{self.__class__.__name__} initialized with config: {cfg}")

    def open_prompt_template(
        self,
        template: str,
        input_variables: List[str],
    ) -> PromptTemplate:
        return PromptTemplate(template=template, input_variables=input_variables)

    @abstractmethod
    def run(self, input_data: Union[str, List[str]]):
        pass
