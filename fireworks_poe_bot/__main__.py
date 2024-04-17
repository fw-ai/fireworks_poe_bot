# Imports needed to intialize plugins
from fireworks_poe_bot.fw_poe_text_bot import FireworksPoeTextBot
from fireworks_poe_bot.fw_poe_image_bot import FireworksPoeImageBot
from fireworks_poe_bot.fw_poe_qr_bot import FireworksPoeQRBot
from fireworks_poe_bot.fw_poe_video_bot import FireworksPoeVideoBot
from fireworks_poe_bot.fw_poe_stability_image_bot import FireworksPoeImageBot
from fireworks_poe_bot.logging import UVICORN_LOGGING_CONFIG
from fireworks_poe_bot.plugin import LoggingPlugin, register_logging_plugin, BOT_PLUGINS, log_info
from sse_starlette.sse import ServerSentEvent


import argparse
from dataclasses import dataclass
from typing import Any, AsyncIterable, Dict, Union
import fastapi
import fastapi_poe
from fastapi import HTTPException
from fastapi_poe import make_app
import uvicorn
import logging
import os
import json


@dataclass
class ServerArgs:
    host: str = "0.0.0.0"
    port: int = 80
    config_file_path: str = "config.json"
    environment: str = ""
    deployment: str = "poe-omnibot"


class PyLoggingPlugin(LoggingPlugin):
    def log_warn(self, payload: Dict[str, Any]):
        logging.warning(payload)

    def log_info(self, payload: Dict[str, Any]):
        logging.info(payload)

    def log_error(self, payload: Dict[str, Any]):
        logging.error(payload)

def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(
            prog="fireworks_poe_bot",
            description=f"""
        Fireworks LLM Poe Server Bot

        Copyright (c) 2023 Fireworks.ai, Inc. and affiliates.
        """,
        )

        # Server args.
        server_args = ServerArgs()
        server_group = parser.add_argument_group("server", "Server arguments")
        server_group.add_argument("--host", type=str, default=server_args.host)
        server_group.add_argument("-p", "--port", type=int, default=server_args.port)
        server_group.add_argument(
            "-c", "--config-file-path", type=str, default=server_args.config_file_path
        )
        server_group.add_argument(
            "-e", "--environment", type=str, default=server_args.environment
        )
        server_group.add_argument(
            "-d", "--deployment", type=str, default=server_args.deployment
        )

        args = parser.parse_args()

        # Parse arguments.
        for k, v in vars(args).items():
            for g in [server_args]:
                if hasattr(g, k):
                    setattr(g, k, v)
                    break
            else:
                assert k in ["print_supported_models"], f"Unknown argument {k}"

    # Register default logging plugin
    register_logging_plugin(PyLoggingPlugin())

    # Load bots from config
    with open(args.config_file_path) as f:
        config = json.load(f)

    remaining_config_keys = set(config.keys())

    bots = {}

    for plugin in BOT_PLUGINS:
        if plugin.config_key in config:
            remaining_config_keys.remove(plugin.config_key)
            for config_dict in config[plugin.config_key]:
                bot_config = plugin.BotConfigClass(**config_dict)
                model_fqn = bot_config.model_fqn
                ctor_dict = bot_config.dict()
                for k in list(ctor_dict.keys()):
                    if k.startswith("SERVER_"):
                        ctor_dict.pop(k)
                bots[model_fqn] = plugin.BotPluginClass(
                    environment=args.environment,
                    deployment=args.deployment,
                    server_version="0.0.1",  # FIXME: versioneer?
                    **ctor_dict
                )

    if len(remaining_config_keys) > 0:
        raise ValueError(
            f"Unknown config keys: {remaining_config_keys}, supported keys: {set([plugin.config_key for plugin in BOT_PLUGINS])}"
        )

    log_info({'message': f"Loaded bots: {bots}"})

    assert (
        len(bots) > 0
    ), "No bots specified, use --text-models or --image-models to specify models to serve"


    # Bot that proxies into the bots contained in the `bots` dictionary above
    class FWProxyBot(fastapi_poe.PoeBot):
        def __init__(self, bots):
            super().__init__()
            self.bots = bots

        def find_bot(self, account: str, model: str) -> fastapi_poe.PoeBot:
            bot_fqn = f"accounts/{account}/models/{model}"
            if bot_fqn not in self.bots:
                raise HTTPException(status_code=404, detail=f"Bot {bot_fqn} not found")
            return bots[bot_fqn]

        def find_bot_from_query_params(self, params) -> fastapi_poe.PoeBot:
            if "account" not in params:
                raise HTTPException(status_code=400, detail=f"Missing account query parameter")
            if "model" not in params:
                raise HTTPException(status_code=400, detail=f"Missing model query parameter")
            return self.find_bot(params["account"], params["model"])

        async def get_response_with_context(
            self, request: fastapi_poe.QueryRequest, context: fastapi_poe.RequestContext
        ) -> AsyncIterable[Union[fastapi_poe.PartialResponse, ServerSentEvent]]:
            bot = self.find_bot_from_query_params(context.http_request.query_params)
            async for event in bot.get_response(request):
                yield event

        async def get_settings_with_context(
            self,
            setting: fastapi_poe.SettingsRequest,
            context: fastapi_poe.RequestContext,
        ) -> fastapi_poe.SettingsResponse:
            bot = self.find_bot_from_query_params(context.http_request.query_params)
            return await bot.get_settings(setting)

        async def on_feedback_with_context(
            self,
            feedback_request: fastapi_poe.ReportFeedbackRequest,
            context: fastapi_poe.RequestContext,
        ) -> None:
            bot = self.find_bot_from_query_params(context.http_request.query_params)
            await bot.on_feedback(feedback_request)

        async def on_error_with_context(
            self,
            error_request: fastapi_poe.ReportErrorRequest,
            context: fastapi_poe.RequestContext,
        ) -> None:
            bot = self.find_bot_from_query_params(context.http_request.query_params)
            await bot.on_error(error_request)

    app = make_app(FWProxyBot(bots), allow_without_key=True)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        server_header=False,
        log_config=UVICORN_LOGGING_CONFIG,
    )


if __name__ == "__main__":
    main()
