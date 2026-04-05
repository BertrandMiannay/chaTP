from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class APIResponse:
    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: float

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class BaseAPI(ABC):
    @abstractmethod
    def send(self, messages: list[dict], system: str | None = None) -> APIResponse:
        pass
