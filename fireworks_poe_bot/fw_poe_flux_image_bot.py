import asyncio
import copy
import io
from typing import AsyncIterable, Dict, List, Optional, Union
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
    MetaResponse,
)

import httpx
from itertools import groupby
import time
from PIL import Image
import uuid
from google.cloud import storage
import traceback
from fireworks_poe_bot.plugin import register_bot_plugin, log_error, log_info, log_warn
from fireworks_poe_bot.config import ModelConfig
import random


class FluxImageModelConfig(ModelConfig):
    gcs_bucket_name: str
    num_steps: int = 30
    multi_turn: bool = False
    meta_response: Optional[MetaResponse] = None


VALID_ASPECT_RATIOS = [
    "1:1",
    "21:9",
    "16:9",
    "3:2",
    "5:4",
    "4:5",
    "2:3",
    "9:16",
    "9:21",
    "4:3",
    "3:4",
]

@register_bot_plugin("flux_image_models", FluxImageModelConfig)
class FireworksPoeFluxImageBot(PoeBot):
    def __init__(
        self,
        model: str,
        api_key: str,
        environment: str,
        deployment: str,
        server_version: str,
        gcs_bucket_name: str,
        num_steps: int,
        multi_turn: bool,
        meta_response: Optional[MetaResponse],
    ):
        super().__init__()
        self.model = model
        self.api_key = api_key
        self.environment = environment
        self.deployment = deployment
        self.server_version = server_version

        model_atoms = model.split("/")
        if len(model_atoms) != 4:
            raise ValueError(
                f"Expected model name to be in the form accounts/{{modelname}}/models/{{model}}, but got {model}"
            )

        if model_atoms[0] != "accounts" or model_atoms[2] != "models":
            raise ValueError(
                f"Expected model name to be in the form accounts/{{modelname}}/models/{{model}}, but got {model}"
            )

        self.account = model_atoms[1]
        self.model = model_atoms[3]

        self.num_steps = num_steps
        self.gcs_bucket_name = gcs_bucket_name
        self.multi_turn = multi_turn
        self.meta_response = MetaResponse(**meta_response) if meta_response else meta_response

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

    def _log_error(self, payload: Dict):
        payload = copy.copy(payload)
        payload.update(
            {
                "severity": "ERROR",
                "environment": self.environment,
                "deployment": self.deployment,
                "model": self.model,
                "server_version": self.server_version,
            }
        )
        log_error(payload)

    async def get_response(
        self, query: QueryRequest
    ) -> AsyncIterable[Union[PartialResponse, ServerSentEvent]]:
        if self.meta_response is not None:
            yield self.meta_response

        try:
            start_t = time.time()

            if len(query.query) == 0:
                yield ErrorResponse(allow_retry=False, text="Empty query")
                return

            messages: List[Dict[str, str]] = []

            for protocol_message in query.query:
                role = "assistant" if protocol_message.role == "bot" else protocol_message.role
                if role not in {"system", "user", "assistant"}:
                    self._log_warn({"msg": "Unknown role", **protocol_message})
                    continue
                if protocol_message.content_type not in {"text/plain", "text/markdown"}:
                    self._log_warn({"msg": "Unknown content type", **protocol_message})
                    continue

                messages.append({"role": role, "content": protocol_message.content})

            messages = self._normalize_messages(messages)

            log_query = copy.copy(query.dict())
            self._log_info({"msg": "Request received", **log_query, "processed_msgs": messages})

            prompt = messages[-1]["content"]
            prompt, *ar = prompt.split("--aspect", maxsplit=1)
            if len(ar):
                ar_str = ar[0].strip()
                if ar_str not in VALID_ASPECT_RATIOS:
                    yield self.replace_response_event(text=f"ERROR Invalid --aspect {ar_str}")
                    yield ServerSentEvent(event="done")
                    return
            else:
                ar_str = None



            # Call the async image generation function
            inference_task = asyncio.create_task(self._generate_image_async(prompt, ar_str))

            inference_task_timer = 0
            while not inference_task.done():
                yield self.replace_response_event(
                    text=f"Generating image... ({inference_task_timer} seconds)"
                )
                await asyncio.sleep(1)
                inference_task_timer += 1

            answer = await inference_task

            if answer is None:
                yield ErrorResponse(allow_retry=False, text="Image generation failed")
                return

            public_image_url = self._upload_image_to_gcs(answer, self.gcs_bucket_name)
            response_text = f"![image]({public_image_url})"

            elapsed_sec = time.time() - start_t
            self._log_info({"msg": "Request completed", "response": response_text, "elapsed_sec": elapsed_sec})
            yield self.replace_response_event(text=response_text)
            yield ServerSentEvent(event="done")
            return

        except Exception as e:
            end_t = time.time()
            self._log_error({
                "msg": "Invalid request",
                "error": "\n".join(traceback.format_exception(e)),
                "elapsed_sec": end_t - start_t,
                **log_query,
            })
            error_type = "user_message_too_long" if "prompt is too long" in str(e) else None
            yield ErrorResponse(allow_retry=False, error_type=error_type, text=str(e))
            return

    def _normalize_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        # Ensure assistant messages are preceded by user messages
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "assistant" and (i == 0 or messages[i - 1]["role"] != "user"):
                messages.insert(i, {"role": "user", "content": ""})

        if messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": ""})

        return messages

    async def _generate_image_async(self, prompt: str, aspect_ratio: Optional[str]) -> Optional[Image.Image]:
        async with httpx.AsyncClient(timeout=None) as client:  # Set timeout to None
            url = f"https://api.fireworks.ai/inference/v1/workflows/accounts/{self.account}/models/{self.model}/text_to_image"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "image/jpeg",
            }
            json_data = {
                "prompt": prompt,
                "steps": self.num_steps,
                "seed": random.randint(0, 2**32 - 1),  # Generate a random seed
            }

            if aspect_ratio is not None:
                json_data["aspect_ratio"] = aspect_ratio

            try:
                response = await client.post(url, headers=headers, json=json_data)
                response.raise_for_status()

                # Convert response bytes into an image
                image_bytes = io.BytesIO(response.content)
                return Image.open(image_bytes)

            except httpx.HTTPStatusError as e:
                self._log_error({"msg": "Image generation failed", "error": str(e)})
                return None

    def _upload_image_to_gcs(self, image: Image, bucket_name: str) -> str:
        """Uploads a given PIL.Image to a GCS bucket."""
        random_uuid = str(uuid.uuid4()).replace("-", "")
        filename = f"{random_uuid}.jpg"

        client = storage.Client()
        bucket = client.get_bucket(bucket_name)

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG")
        img_byte_arr = img_byte_arr.getvalue()

        blob = bucket.blob(filename)
        blob.upload_from_string(img_byte_arr, content_type="image/jpeg")

        return blob.public_url

    async def get_settings(self, setting: SettingsRequest) -> SettingsResponse:
        return SettingsResponse(enable_multi_bot_chat_prompting=True)

    async def on_feedback(self, feedback_request: ReportFeedbackRequest) -> None:
        pass

    async def on_error(self, error_request: ReportErrorRequest) -> None:
        self._log_error({"msg": "Error reported", **error_request.dict()})
