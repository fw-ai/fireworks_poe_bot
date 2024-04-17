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

import fireworks.client
from fireworks.client.api import ChatMessage
from fireworks_poe_bot.plugin import log_error, log_info, log_warn, register_bot_plugin
from fireworks_poe_bot.config import ModelConfig

from itertools import groupby
import time
from PIL import Image
import uuid
import requests
from google.cloud import storage
import traceback
import aiohttp

class StabilityImageModelConfig(ModelConfig):
    gcs_bucket_name: str
    stability_api_key: str

    meta_response: Optional[MetaResponse] = None

@register_bot_plugin("stability_image_models", StabilityImageModelConfig)
class FireworksPoeImageBot(PoeBot):
    def __init__(
        self,
        model: str,
        api_key: str,
        environment: str,
        deployment: str,
        server_version: str,
        gcs_bucket_name: str,
        stability_api_key: str,
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

        self.client = ImageInference(account=self.account, model=self.model)


        self.gcs_bucket_name = gcs_bucket_name
        self.stability_api_key = stability_api_key
        if meta_response:
            self.meta_response = MetaResponse(**meta_response)
        else:
            self.meta_response = meta_response

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

            messages: List[ChatMessage] = []

            for protocol_message in query.query:
                # OpenAI/Fireworks use the "assistant" role for the LLM, but Poe uses the
                # "bot" role. Replace that one. Otherwise, ignore the role
                if protocol_message.role not in {"system", "user", "bot"}:
                    self._log_warn({"msg": "Unknown role", **protocol_message})
                    continue
                if protocol_message.content_type not in {"text/plain", "text/markdown"}:
                    self._log_warn({"msg": "Unknown content type", **protocol_message})
                    continue
                # TODO: support protocol_message.feedback and protocol_message.attachments
                # if needed
                if protocol_message.role == "bot":
                    role = "assistant"
                else:
                    role = protocol_message.role
                messages.append({"role": role, "content": protocol_message.content})

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
            self._log_info(
                {
                    "msg": "Request received",
                    **log_query,
                    "processed_msgs": messages,

                }
            )

            # generated_len = 0

            assert messages[-1]["role"] == "user"
            prompt = messages[-1]["content"]

            url = "https://api.stability.ai/v2beta/stable-image/generate/sd3"
            headers = {
                "Authorization": f"Bearer {self.stability_api_key}",
                "Accept": "image/*"
            }
            data = {
                "prompt": prompt,
                "output_format": "jpeg",
                "model": self.model,
            }
            # Using FormData to handle files and data
            form_data = aiohttp.FormData()
            form_data.add_field('none', '')
            for key, value in data.items():
                form_data.add_field(key, value)

            # Create session and post data
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, data=form_data) as response:
                    if not response.ok:
                        raise Exception(f"Generation failed with error {response.reason}")

                    finish_reason = response.headers.get("finish_reason", "SUCCESS")
                    image_data = await response.read()
                    image = Image.open(io.BytesIO(image_data))

            end_t_inference = time.time()
            start_t_encode = time.time()

            if finish_reason == "CONTENT_FILTERED":
                yield self.text_event(text="Potentially sensitive content detected")

            public_image_url = self._upload_image_to_gcs(
                image, self.gcs_bucket_name
            )
            response_text = f"![image]({public_image_url})"

            end_t = time.time()
            elapsed_sec = end_t - start_t
            self._log_info(
                {
                    "severity": "INFO",
                    "msg": "Request completed",
                    **log_query,
                    "response": response_text,
                    "elapsed_sec": elapsed_sec,
                    "elapsed_sec_inference": end_t_inference - start_t,
                    "elapsed_sec_upload": end_t - start_t_encode,
                }
            )
            yield PartialResponse(text=response_text)
            yield ServerSentEvent(event="done")
            return
        except Exception as e:
            end_t = time.time()
            self._log_error(
                {
                    "msg": "Invalid request",
                    "error": "\n".join(traceback.format_exception(e)),
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
        finally:
            fireworks.client.api_key = orig_api_key

    # Function to upload a PIL Image to an S3 bucket with a presigned URL
    def _upload_image_to_s3_with_ttl(
        self, bucket_name, object_name, image: Image, expiration=600
    ):
        """
        Upload a PIL Image to an S3 bucket with TTL by generating a presigned URL.

        :param bucket_name: String name of the bucket to which the image is uploaded.
        :param object_name: S3 object name. If not specified then file_name is used.
        :param image: PIL Image object to be uploaded.
        :param expiration: Time in seconds for the presigned URL to remain valid.
        """
        # In-memory binary streams
        in_mem_file = io.BytesIO()

        # Save the PIL image to in-memory file as JPEG
        image.save(in_mem_file, format="JPEG")
        in_mem_file.seek(0)  # Reset file pointer to the beginning

        # Upload the image to S3
        # self.s3_client.upload_fileobj(in_mem_file, bucket_name, object_name)
        self.s3_client.put_object(
            Bucket=self.s3_bucket_name,
            Key=object_name,
            Body=in_mem_file,
            ContentType="image/jpeg",
        )

        # Generate a presigned URL for the S3 object
        url = self.s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": object_name},
            ExpiresIn=expiration,
        )

        return url

    def _upload_image_to_gcs(self, image: Image, bucket_name: str):
        """Uploads a given PIL.Image to a GCS bucket."""
        # Generate a (statistically) unique filename with a uuid4
        random_uuid = str(uuid.uuid4()).replace("-", "")
        filename = f"{random_uuid}.jpg"

        # Initialize the GCS client
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)

        # Convert the PIL.Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG")
        img_byte_arr = img_byte_arr.getvalue()

        # Create a new blob (i.e., object) in the bucket and upload the image bytes
        blob = bucket.blob(filename)
        blob.upload_from_string(img_byte_arr, content_type=f"image/jpeg")

        blob.make_public()

        # The public URL can be accessed with the `public_url` attribute
        public_url = blob.public_url

        return public_url

    def _download_image(self, image_url):
        # Send an HTTP GET request to the image URL
        response = requests.get(image_url)

        # Check if the request was successful
        if response.status_code == 200:
            # Read the image content into an in-memory bytes buffer
            image_bytes = io.BytesIO(response.content)

            # Use Pillow to open the image from the bytes buffer
            img = Image.open(image_bytes)

            return img
        else:
            # If the request failed, raise an HTTPError with the response
            response.raise_for_status()

    async def get_settings(self, setting: SettingsRequest) -> SettingsResponse:
        """Override this to return non-standard settings."""
        return SettingsResponse()

    async def on_feedback(self, feedback_request: ReportFeedbackRequest) -> None:
        """Override this to record feedback from the user."""
        pass

    async def on_error(self, error_request: ReportErrorRequest) -> None:
        """Override this to record errors from the Poe server."""
        self._log_error(
            {
                "msg": "Error reported",
                **error_request.dict(),
            }
        )
