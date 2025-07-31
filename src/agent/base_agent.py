import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.chains.llm import LLMChain
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
        self.agent = self._initialize_agent()

        self.logger.debug(f"{self.__class__.__name__} initialized with config: {cfg}")

    def _initialize_agent(self):
        prompt = PromptTemplate(
            template=self.cfg.template,
            input_variables=self.cfg.input_variables,
        )
        llm_chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
        )
        agent = ZeroShotAgent(llm_chain=llm_chain)

        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    @abstractmethod
    def run(self, input_data: Union[str, List[str]]):
        pass
