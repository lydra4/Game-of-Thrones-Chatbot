import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from omegaconf import DictConfig


class BaseAgent(ABC):
    def __init__(
        self,
        cfg: DictConfig,
        llm: Any,
        logger: Optional[logging.Logger] = None,
    ):
        self.cfg = cfg
        self.llm = llm
        self.logger = logger or logging.getLogger(__name__)
        self.logger.debug(f"{self.__class__.__name__} initialized with config: {cfg}")

    def _initialize_chain(self) -> RunnableSequence:
        prompt = ChatPromptTemplate.from_template(
            template=self.cfg.template,
        )

        return prompt | self.llm

    @abstractmethod
    def run(self, input_data: Union[str, List[str]]):
        chain = self._initialize_chain()
        output = chain.invoke(input=input_data)
        return output
