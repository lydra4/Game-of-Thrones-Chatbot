import logging
from typing import Any, List, Optional, Union

from omegaconf import DictConfig


class BaseAgent:
    def __init__(
        self,
        cfg: DictConfig,
        llm: Any,
        input_data: Union[str, List[str]],
        tools: Optional[List[Any]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.cfg = cfg
        self.llm = llm
        self.input_data = input_data
        self.tools = tools
        self.logger = logger or logging.getLogger(__name__)

        self.logger.debug(f"{self.__class__.__name__} initialized with config: {cfg}")

    def run(self):
        raise NotImplementedError("Each agent must implement it's own run method.")
