from fireworks_poe_bot.fw_poe_text_bot import FireworksPoeTextBot
from fireworks_poe_bot.fw_poe_image_bot import FireworksPoeImageBot
from fireworks_poe_bot.fw_poe_qr_bot import FireworksPoeQRBot
from fireworks_poe_bot.logging import UVICORN_LOGGING_CONFIG
from fireworks_poe_bot.plugin import LoggingPlugin, register_logging_plugin, BOT_PLUGINS, log_info


import argparse
from dataclasses import dataclass
from typing import Any, Dict
from .fastapi_poe import make_app
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

    app = make_app(bots, allow_without_key=True)

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
