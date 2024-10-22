import base64
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
from fireworks_poe_bot.plugin import log_error, log_info, log_warn, register_bot_plugin


class FluxImageModelConfig:
    gcs_bucket_name: str
    num_steps: int = 30
    multi_turn: bool = False
    meta_response: Optional[MetaResponse] = None

@register_bot_plugin("image_models", FluxImageModelConfig)
class FireworksPoeImageBot(PoeBot):
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

    async def _log(self, severity: str, payload: Dict):
        payload = copy.copy(payload)
        payload.update(
            {
                "severity": severity,
                "environment": self.environment,
                "deployment": self.deployment,
                "model": self.model,
                "server_version": self.server_version,
            }
        )
        print(payload)  # Replace this with proper logging

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
                    await self._log("WARNING", {"msg": "Unknown role", **protocol_message})
                    continue
                if protocol_message.content_type not in {"text/plain", "text/markdown"}:
                    await self._log("WARNING", {"msg": "Unknown content type", **protocol_message})
                    continue

                messages.append({"role": role, "content": protocol_message.content})

            # Ensure the message format for chat API
            messages = self._normalize_messages(messages)

            log_query = copy.copy(query.dict())
            await self._log("INFO", {"msg": "Request received", **log_query, "processed_msgs": messages})

            prompt = messages[-1]["content"]

            # Call the async image generation function
            answer = await self._generate_image_async(prompt)

            if answer is None:
                yield ErrorResponse(allow_retry=False, text="Image generation failed")
                return

            public_image_url = self._upload_image_to_gcs(answer, self.gcs_bucket_name)
            response_text = f"![image]({public_image_url})"

            elapsed_sec = time.time() - start_t
            await self._log("INFO", {"msg": "Request completed", "response": response_text, "elapsed_sec": elapsed_sec})
            yield PartialResponse(text=response_text)
            yield ServerSentEvent(event="done")
            return

        except Exception as e:
            await self._log("ERROR", {"msg": "Invalid request", "error": traceback.format_exc()})
            yield ErrorResponse(allow_retry=False, text=str(e))

    def _normalize_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        # Ensure assistant messages are preceded by user messages and merge adjacent messages
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "assistant" and (i == 0 or messages[i - 1]["role"] != "user"):
                messages.insert(i, {"role": "user", "content": ""})

        merged_messages = []
        for role, group in groupby(messages, key=lambda x: x["role"]):
            content = " ".join(message["content"] for message in group)
            merged_messages.append({"role": role, "content": content})

        if merged_messages[-1]["role"] != "user":
            merged_messages.append({"role": "user", "content": ""})

        return merged_messages

    async def _generate_image_async(self, prompt: str) -> Optional[Image.Image]:
        async with httpx.AsyncClient() as client:
            url = f"https://api.fireworks.ai/inference/v1/workflows/accounts/{self.account}/models/{self.model}/text_to_image"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "image/jpeg",
            }
            json_data = {
                "prompt": prompt,
                "guidance_scale": 7,
                "steps": self.num_steps,
                "seed": 0,
            }

            try:
                response = await client.post(url, headers=headers, json=json_data)
                response.raise_for_status()

                # Convert response bytes into an image
                image_bytes = io.BytesIO(response.content)
                return Image.open(image_bytes)

            except httpx.HTTPStatusError as e:
                await self._log("ERROR", {"msg": "Image generation failed", "error": str(e)})
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
        blob.make_public()

        return blob.public_url

    async def get_settings(self, setting: SettingsRequest) -> SettingsResponse:
        return SettingsResponse(enable_multi_bot_chat_prompting=True)

    async def on_feedback(self, feedback_request: ReportFeedbackRequest) -> None:
        pass

    async def on_error(self, error_request: ReportErrorRequest) -> None:
        await self._log("ERROR", {"msg": "Error reported", **error_request.dict()})
