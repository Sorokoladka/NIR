from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass
class BaseModelParams:
    pass

class BaseModel(ABC):
    def __init__(self, params: BaseModelParams):
        self.params = params

    @abstractmethod
    def simulate(self, *args, **kwargs) -> Any:
        pass

