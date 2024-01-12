import traceback
from .fastapi_poe import PoeBot
from .fastapi_poe.types import (
    PartialResponse,
    QueryRequest,
    ReportErrorRequest,
    ReportFeedbackRequest,
    SettingsRequest,
    SettingsResponse,
    ErrorResponse,
)

import copy
import httpx
import time
import io
from PIL.Image import Image
from sse_starlette.sse import ServerSentEvent
from typing import AsyncIterable, Dict, List, Optional, Union
from fireworks_poe_bot.plugin import log_error, log_info, log_warn, register_bot_plugin
from fireworks_poe_bot.config import ModelConfig
import fireworks.client
from fireworks.client.api import ChatMessage
from fireworks.client.image import ImageInference, AnswerVideo


class VideoModelConfig(ModelConfig):
    poe_bot_access_key: str


@register_bot_plugin("video_models", VideoModelConfig)
class FireworksPoeVideoBot(PoeBot):
    def __init__(self,
            model: str,
            api_key: str,
            environment: str,
            deployment: str,
            server_version: str,
            poe_bot_access_key: str,
    ):
        super().__init__()
        self.model = model
        self.api_key = api_key
        self.environment = environment
        self.deployment = deployment
        self.server_version = server_version
        self.poe_bot_access_key = poe_bot_access_key

        self.client = ImageInference(account=self.account, model=self.model)


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

    async def download_image_and_encode_to_pil(
        self,
        url: str,
    ) -> Image:
        async with httpx.AsyncClient() as client:
            r = await client.get(url)
            if r.status_code == 200:
                pil_img = Image.open(io.BytesIO(r.content))
                pil_img = pil_img.convert("RGB")
                return pil_img
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

                if protocol_message.role == "bot":
                    protocol_message.role = "assistant"

            last_user_message = None
            for protocol_message in query.query:
                if protocol_message.role == "user":
                    last_user_message = protocol_message.content

            if not last_user_message:
                yield ErrorResponse(allow_retry=False, text="No user message")
                return

            if len(protocol_message.attachments) != 1:
                yield ErrorResponse(allow_retry=False, text="Please upload a single image attachment to generate a video")
                return

            attachment = protocol_message.attachments[0]
            if attachment.content_type not in ["image/png", "image/jpeg"]:
                # FIXME: more image types?
                yield ErrorResponse(allow_retry=False, text=f"Invalid image type {attachment.content_type}, expected a PNG or JPEG image")
                return

            try:
                img_pil: Image = await self.download_image_and_encode_to_pil(
                    protocol_message.attachments[0].url
                )
            except Exception as e:
                yield ErrorResponse(allow_retry=False, text=str(e))
                raise RuntimeError(str(e))


            self._log_info(
                {
                    "msg": "Request received",
                    **query.dict(),
                }
            )

            answer: AnswerVideo = await self.client.image_to_video_async(
                input_image=img_pil,
                safety_check=True,
                # TODO: more params
            )

            # Upload file as attachment
            await self.post_message_attachment(
                self.poe_bot_access_key, query.message_id, file_data=answer.video, filename="video.mp4"
            )
            end_t_inference = time.time()

            if answer.finish_reason == "CONTENT_FILTERED":
                response_text = "Your video was generated, but it was filtered by the content filter. Please try again with a different image."
            else:
                response_text = "Your video was generated. Please download the attachment."

            end_t = time.time()
            elapsed_sec = end_t - start_t
            self._log_info(
                {
                    "severity": "INFO",
                    "msg": "Request completed",
                    **query.dict(),
                    "response": response_text,
                    "elapsed_sec": elapsed_sec,
                    "elapsed_sec_inference": end_t_inference - start_t,
                }
            )
            yield PartialResponse(text=response_text)
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
                    **query.dict(),
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
        """Override this to return non-standard settings."""
        return SettingsResponse()

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