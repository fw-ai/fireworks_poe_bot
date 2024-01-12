import copy
from typing import AsyncIterable, Dict, List, Optional, Union, Any
from .fastapi_poe import PoeBot
from sse_starlette.sse import ServerSentEvent
from .fastapi_poe.types import (
    PartialResponse,
    QueryRequest,
    ReportErrorRequest,
    ReportFeedbackRequest,
    SettingsRequest,
    SettingsResponse,
    ErrorResponse,
)

import fireworks.client
from fireworks.client import ChatCompletion
from fireworks.client.api import ChatCompletionResponseStreamChoice, ChatMessage
from fireworks.client.error import InvalidRequestError
from fireworks_poe_bot.plugin import log_error, log_info, log_warn, register_bot_plugin
from fireworks_poe_bot.config import ModelConfig

from typing import Callable
from itertools import groupby
import time
import io
import base64
import httpx
from PIL import Image
import traceback


class TextModelConfig(ModelConfig):
    allow_attachments: Optional[bool] = False
    input_image_size: Optional[int] = None
    prompt_truncate_len: int = 2048
    max_tokens: int = 4096
    system_prompt_override: Optional[str] = None
    additional_args: Optional[Dict[str, int | str | float | List[str]]] = None
    # "alpaca" or None
    chat_format: Optional[str] = None
    alpaca_instruction_msg: Optional[str] = None


@register_bot_plugin("text_models", TextModelConfig)
class FireworksPoeTextBot(PoeBot):
    def __init__(
        self,
        model: str,
        api_key: str,
        environment: str,
        deployment: str,
        server_version: str,
        allow_attachments: bool,
        input_image_size: int,
        prompt_truncate_len: int,
        max_tokens: int,
        system_prompt_override: Optional[str],
        additional_args: Optional[Dict[str, int | str]],
        chat_format: Optional[str],
        alpaca_instruction_msg: Optional[str],
        completion_async_method: Callable = ChatCompletion.acreate,
    ):
        super().__init__()
        self.model = model
        self.api_key = api_key
        self.environment = environment
        self.deployment = deployment
        self.server_version = server_version
        self.input_image_size = input_image_size
        self.completion_async_method = completion_async_method
        self.allow_attachments = allow_attachments
        self.prompt_truncate_len = prompt_truncate_len
        self.max_tokens = max_tokens
        self.chat_format = chat_format
        self.alpaca_instruction_msg = alpaca_instruction_msg
        self.system_prompt_override = system_prompt_override
        self.additional_args = additional_args or {}

    def _log_warn(self, payload: Dict):
        payload = copy.copy(payload)
        payload.update(
            {
                "severity": "WARNING",
                "environment": self.environment,
                "deployment": self.deployment,
                "model": self.model,
                "server_version": self.server_version,
            }
        )
        log_warn(payload)

    def _log_info(self, payload: Dict):
        payload = copy.copy(payload)
        payload.update(
            {
                "severity": "INFO",
                "environment": self.environment,
                "deployment": self.deployment,
                "model": self.model,
                "server_version": self.server_version,
            }
        )
        log_info(payload)

    async def download_image_and_encode_to_base64(
        self,
        url: str,
    ) -> str:
        async with httpx.AsyncClient() as client:
            image_download_start = time.perf_counter()
            r = await client.get(url)
            image_download_end = time.perf_counter()
            if r.status_code == 200:
                resize_encode_start = time.perf_counter()
                pil_img = Image.open(io.BytesIO(r.content))
                pil_img = pil_img.convert("RGB")
                width, height = pil_img.size
                if width >= height:
                    new_size = (
                        self.input_image_size,
                        int(height * self.input_image_size / width),
                    )
                else:
                    new_size = (
                        int(width * self.input_image_size / height),
                        self.input_image_size,
                    )
                pil_img_resized = pil_img.resize(new_size)
                buffered = io.BytesIO()
                pil_img_resized.save(buffered, format="JPEG")
                img_buffer = buffered.getvalue()
                img = "data:image/jpeg;base64,{}".format(
                    base64.b64encode(img_buffer).decode("utf-8")
                )
                resize_encode_end = time.perf_counter()
                self._log_info(
                    {
                        "download_image_ms": int(
                            (image_download_end - image_download_start) * 1000
                        ),
                        "encode_image_ms": int(
                            (resize_encode_end - resize_encode_start) * 1000
                        ),
                        "url": url,
                    }
                )
                return img
            raise Exception(f"Unable to download image, error code {r.status_code}")

    async def get_response(
        self, query: QueryRequest
    ) -> AsyncIterable[Union[PartialResponse, ServerSentEvent]]:
        if len(query.query) == 0:
            yield ErrorResponse(allow_retry=False, text="Empty query")
            return

        orig_api_key = fireworks.client.api_key
        fireworks.client.api_key = self.api_key
        try:
            start_t = time.time()
            messages: List[ChatMessage] = []

            cumulative_image_size_mb = 0
            for protocol_message in query.query:
                log_msg = protocol_message.dict()

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
                img_base64 = None
                if protocol_message.role == "bot":
                    role = "assistant"
                else:
                    role = protocol_message.role
                    if protocol_message.attachments and protocol_message.attachments[
                        0
                    ].content_type in ["image/png", "image/jpeg"]:
                        try:
                            img_base64 = await self.download_image_and_encode_to_base64(
                                protocol_message.attachments[0].url
                            )
                        except Exception as e:
                            yield ErrorResponse(allow_retry=False, text=str(e))
                            raise RuntimeError(str(e))

                if img_base64:
                    if cumulative_image_size_mb > 8:
                        # Apigee has a limit of 10MB for payload, we set image total limit to 8MB
                        yield ErrorResponse(
                            allow_retry=False, text="The total image size is too big"
                        )
                        raise RuntimeError("The total image size is too big")
                    messages.append(
                        {
                            "role": role,
                            "content": [
                                {"type": "text", "text": protocol_message.content},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": img_base64},
                                },
                            ],
                        }
                    )
                    cumulative_image_size_mb += len(img_base64) / 1024 / 1024
                else:
                    messages.append({"role": role, "content": protocol_message.content})

            if self.system_prompt_override is not None:
                system_prompt_msg = None
                for msg in messages:
                    if msg["role"] == "system":
                        system_prompt_msg = msg
                        break
                if system_prompt_msg is None:
                    system_prompt_msg = {
                        "role": "system",
                    }
                    messages.insert(0, system_prompt_msg)

                system_prompt_msg["content"] = [
                    {"type": "text", "text": self.system_prompt_override},
                ]

            if self.chat_format == "alpaca":
                # Discard all messages except "system" and the last "user"
                # message
                system_message = None
                user_message = None
                for msg in messages:
                    if msg["role"] == "system":
                        system_message = msg
                    elif msg["role"] == "user":
                        user_message = msg

                new_messages = []
                if system_message is not None:
                    new_messages.append(system_message)
                # Insert instruction message, if applicable
                if self.alpaca_instruction_msg is not None:
                    new_messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.alpaca_instruction_msg}
                            ],
                        }
                    )
                    if user_message is not None:
                        user_message["role"] = "input"
                        # HACKS: move the image to the instruction message
                        content_non_image = [x for x in user_message["content"] if x["type"] != "image_url"]
                        content_image = [x for x in user_message["content"] if x["type"] == "image_url"]
                        if content_image:
                            new_messages[-1]["content"].append(content_image[0])
                        user_message["content"] = content_non_image
                        new_messages.append(user_message)
                else:
                    if user_message is not None:
                        new_messages.append(user_message)
                messages = new_messages

            self._log_info(
                {
                    "msg": "Request received",
                    **query.dict(),
                }
            )

            if self.chat_format != "alpaca":
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

                # Now there could be images in the messages, in which case the message content is a list
                def merge_messages_groups(
                    message_group: List[Union[str, List[Dict[str, Any]]]]
                ) -> Union[str, List[Dict[str, Any]]]:
                    text = []
                    images = []
                    for msg in message_group:
                        if isinstance(msg, str):
                            text.append(msg)
                        elif isinstance(msg, list):
                            assert msg[0]["type"] == "text"
                            text.append(msg[0]["text"])
                            images.extend(msg[1:])
                    if images:
                        return [{"type": "text", "text": " ".join(text)}, *images]
                    return " ".join(text)

                for role, group in groupby(messages, key=lambda x: x["role"]):
                    content = merge_messages_groups([message["content"] for message in group])
                    merged_messages.append({"role": role, "content": content})

                messages = merged_messages

                # Ensure last message is a user message
                if messages[-1]["role"] != "user":
                    self._log_warn({"msg": f"Last message {messages[-1]} not a user message"})
                    messages.append({"role": "user", "content": ""})

            additional_args = copy.deepcopy(self.additional_args)
            if "stop" in additional_args:
                stop_seqs = additional_args["stop"]
                additional_args.pop("stop")
            else:
                stop_seqs = query.stop_sequences[:4]
            generated_len = 0
            complete_response = ""
            async for response in self.completion_async_method(
                model=self.model,
                messages=messages,
                stream=True,
                request_timeout=600,
                temperature=query.temperature,
                stop=stop_seqs,
                max_tokens=self.max_tokens,
                prompt_truncate_len=self.prompt_truncate_len,
                **additional_args,
            ):
                # Step 3: Transform the CompletionStreamResponse into PartialResponse format
                for choice in response.choices:
                    assert isinstance(choice, ChatCompletionResponseStreamChoice)
                    if choice.delta.content is None:
                        continue

                    generated_len += len(choice.delta.content)
                    complete_response += choice.delta.content
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
                    "query": query.dict(),
                    "response": complete_response,
                    "generated_len": generated_len,
                    "elapsed_sec": elapsed_sec,
                }
            )
            yield ServerSentEvent(event="done")
            return
        except Exception as e:
            end_t = time.time()
            log_error(
                {
                    "severity": "ERROR",
                    "msg": "Invalid request",
                    "error": "\n".join(traceback.format_exception(e)),
                    "elapsed_sec": end_t - start_t,
                    "query": query.dict(),
                }
            )
            if "prompt is too long" in str(e):
                error_type = "user_message_too_long"
            else:
                error_type = None
            yield ErrorResponse(allow_retry=False, error_type=error_type, text=str(e))
            return
        finally:
            fireworks.client.api_key = orig_api_key

    async def get_settings(self, setting: SettingsRequest) -> SettingsResponse:
        return SettingsResponse(allow_attachments=self.allow_attachments)

    async def on_feedback(self, feedback_request: ReportFeedbackRequest) -> None:
        """Override this to record feedback from the user."""
        pass

    async def on_error(self, error_request: ReportErrorRequest) -> None:
        """Override this to record errors from the Poe server."""
        log_error(
            {
                "severity": "ERROR",
                "msg": "Error reported",
                **error_request.dict(),
            }
        )
