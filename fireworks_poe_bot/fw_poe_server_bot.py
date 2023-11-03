import copy
from typing import AsyncIterable, Dict, List, Union
from fastapi_poe import PoeBot
from sse_starlette.sse import ServerSentEvent
from fastapi_poe.types import (
    PartialResponse,
    QueryRequest,
    ReportErrorRequest,
    ReportFeedbackRequest,
    SettingsRequest,
    SettingsResponse,
    ErrorResponse,
)

from fireworks.client import ChatCompletion
from fireworks.client.api import ChatCompletionResponseStreamChoice, ChatMessage
from fireworks.client.error import InvalidRequestError

from typing import Callable
from itertools import groupby
import logging
import time
import os


class FireworksPoeServerBot(PoeBot):
    def __init__(
        self,
        model: str,
        environment: str,
        server_version: str,
        completion_async_method: Callable = ChatCompletion.acreate,
    ):
        super().__init__()
        self.model = model
        self.environment = environment
        self.server_version = server_version
        self.completion_async_method = completion_async_method

    def _log_warn(self, payload: Dict):
        payload = copy.copy(payload)
        payload.update(
            {
                "severity": "WARNING",
                "environment": self.environment,
                "model": self.model,
                "server_version": self.server_version,
            }
        )
        logging.warning(payload)

    def _log_info(self, payload: Dict):
        payload = copy.copy(payload)
        payload.update(
            {
                "severity": "INFO",
                "environment": self.environment,
                "model": self.model,
                "server_version": self.server_version,
            }
        )
        logging.info(payload)

    async def get_response(
        self, query: QueryRequest
    ) -> AsyncIterable[Union[PartialResponse, ServerSentEvent]]:
        if len(query.query) == 0:
            yield ErrorResponse(allow_retry=False, text="Empty query")
            return

        messages: List[ChatMessage] = []

        redacted_msgs = []
        for protocol_message in query.query:
            # Redacted message for logging
            log_msg = copy.copy(protocol_message.dict())
            log_msg.update({"content": f"Content(len={len(protocol_message.content)})"})
            redacted_msgs.append(log_msg)

            # OpenAI/Fireworks use the "assistant" role for the LLM, but Poe uses the
            # "bot" role. Replace that one. Otherwise, ignore the role
            if protocol_message.role not in {"system", "user", "bot"}:
                self._log_warn({"msg": "Unknown role", **log_msg})
                continue
            if protocol_message.content_type not in {"text/plain", "text/markdown"}:
                self._log_warn({"msg": "Unknown content type", **log_msg})
                continue
            # TODO: support protocol_message.feedback and protocol_message.attachments
            # if needed
            if protocol_message.role == "bot":
                role = "assistant"
            else:
                role = protocol_message.role
            messages.append({"role": role, "content": protocol_message.content})

            self._log_info(
                {
                    "msg": "Message received",
                    **log_msg,
                }
            )

        # The poe servers send us arbitrary lists of messages. We need to do a few things
        # to normalize for our chat completion API:
        # 1. Ensure that all assistant messages are preceded by a user message
        # 2. Merge adjacent messages from the same role
        # 3. Ensure that the last message is a user message

        # Ensure that all assistant messages are preceded by a user message
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "assistant" and (
                i == 0 or messages[i - 1]["role"] != "user"
            ):
                self._log_warn(
                    {
                        "msg": f"Assistant message {messages[i]} not preceded by user message"
                    }
                )
                messages.insert(i, {"role": "user", "content": ""})

        # Merge adjacent messages from the same role
        merged_messages = []

        for role, group in groupby(messages, key=lambda x: x["role"]):
            content = " ".join(message["content"] for message in group)
            merged_messages.append({"role": role, "content": content})

        messages = merged_messages

        # Ensure last message is a user message
        if messages[-1]["role"] != "user":
            self._log_warn({"msg": f"Last message {messages[-1]} not a user message"})
            messages.append({"role": "user", "content": ""})

        log_query = copy.copy(query.dict())
        log_query.update({"query": redacted_msgs})
        try:
            generated_len = 0
            start_t = time.time()
            async for response in self.completion_async_method(
                model=self.model,
                messages=messages,
                stream=True,
                request_timeout=600,
                temperature=query.temperature,
                stop=query.stop_sequences[:4],
                max_tokens=4096,  # TODO: make arg
                prompt_truncate_len=2048,  # TODO: make arg
                frequency_penalty=0.5,
            ):
                # Step 3: Transform the CompletionStreamResponse into PartialResponse format
                for choice in response.choices:
                    assert isinstance(choice, ChatCompletionResponseStreamChoice)
                    if choice.delta.content is None:
                        continue

                    generated_len += len(choice.delta.content)
                    yield PartialResponse(
                        text=choice.delta.content,
                        raw_response=response,
                        request_id=response.id,
                    )

            end_t = time.time()
            elapsed_sec = end_t - start_t
            self._log_info(
                {
                    "severity": "INFO",
                    "msg": "Request completed",
                    **log_query,
                    "generated_len": generated_len,
                    "elapsed_sec": elapsed_sec,
                }
            )
            yield ServerSentEvent(event="done")
            return
        except InvalidRequestError as e:
            end_t = time.time()
            logging.error(
                {
                    "severity": "ERROR",
                    "msg": "Invalid request",
                    "error": e,
                    "elapsed_sec": end_t - start_t,
                    **log_query,
                }
            )
            if "prompt is too long" in str(e):
                error_type = "user_message_too_long"
            else:
                error_type = None
            yield ErrorResponse(allow_retry=False, error_type=error_type, text=str(e))
            return

    async def get_settings(self, setting: SettingsRequest) -> SettingsResponse:
        """Override this to return non-standard settings."""
        return SettingsResponse()

    async def on_feedback(self, feedback_request: ReportFeedbackRequest) -> None:
        """Override this to record feedback from the user."""
        pass

    async def on_error(self, error_request: ReportErrorRequest) -> None:
        """Override this to record errors from the Poe server."""
        logging.error(
            {
                "severity": "ERROR",
                "msg": "Error reported",
                **error_request.dict(),
            }
        )
