import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

from langchain.agents import AgentExecutor
from langchain.agents.openai_tools.base import create_openai_tools_agent
from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
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
        self.agent = self._initialize_agent()

        self.logger.debug(f"{self.__class__.__name__} initialized with config: {cfg}")

    def _initialize_agent(self):
        prompt = ChatPromptTemplate.from_template(
            template=self.cfg.template,
            input_variables=self.cfg.input_variables,
        )
        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )

        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    @abstractmethod
    def run(self, input_data: Union[str, List[str]]):
        pass
