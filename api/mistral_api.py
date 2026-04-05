import os
import time
from mistralai.client import Mistral
from api.base import BaseAPI, APIResponse


class MistralAPI(BaseAPI):
    DEFAULT_MODEL = "mistral-small-latest"

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    def send(self, messages: list[dict], system: str | None = None) -> APIResponse:
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        start = time.monotonic()
        response = self.client.chat.complete(
            model=self.model,
            messages=all_messages,
        )
        latency_ms = (time.monotonic() - start) * 1000

        return APIResponse(
            content=response.choices[0].message.content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            latency_ms=latency_ms,
        )
