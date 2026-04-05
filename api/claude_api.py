import time
import anthropic
from api.base import BaseAPI, APIResponse


class ClaudeAPI(BaseAPI):
    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.client = anthropic.Anthropic()

    def send(self, messages: list[dict], system: str | None = None) -> APIResponse:
        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        start = time.monotonic()
        response = self.client.messages.create(**kwargs)
        latency_ms = (time.monotonic() - start) * 1000

        return APIResponse(
            content=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
        )
