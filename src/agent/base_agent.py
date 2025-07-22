import logging
from typing import Optional

from omegaconf import DictConfig


class BaseAgent:
    def __init__(
        self,
        cfg: DictConfig,
        llm,
        tools=None,
        logger: Optional[logging.Logger] = None,
    ):
        self.cfg = cfg
        self.llm = llm
        self.tools = tools
        self.logger = logger or logging.getLogger(__name__)
