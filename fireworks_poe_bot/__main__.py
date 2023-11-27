from fireworks_poe_bot.fw_poe_text_bot import FireworksPoeTextBot
from fireworks_poe_bot.fw_poe_image_bot import FireworksPoeImageBot
from fireworks_poe_bot.logging import UVICORN_LOGGING_CONFIG
from fireworks_poe_bot.config import Config, load_config


import argparse
from dataclasses import dataclass
from .fastapi_poe import make_app
import uvicorn
import os


@dataclass
class ServerArgs:
    host: str = "0.0.0.0"
    port: int = 80
    config_file_path: str = "config.json"
    image_size: int = 336
    allow_attachments: bool = False
    environment: str = ""


def main():
    parser = argparse.ArgumentParser(
        prog="fireworks_poe_bot",
        description=f"""
    Fireworks LLM Poe Server Bot v0.0.1.

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
        "-s", "--image-size", type=int, default=server_args.image_size
    )
    server_group.add_argument("-a", "--allow-attachments", action="store_true")
    server_group.add_argument(
        "-e", "--environment", type=str, default=server_args.environment
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

    config = load_config(args.config_file_path)

    bots = {}

    for text_model_spec in config.text_models:
        model_fqn, api_key = text_model_spec.model, text_model_spec.API_KEY
        bots[model_fqn] = FireworksPoeTextBot(
            model=model_fqn,
            api_key=api_key,
            environment=args.environment,
            server_version="0.0.1",
            image_size=args.image_size,
            allow_attachments=args.allow_attachments,
        )

    for image_model_spec in config.image_models:
        model_fqn, api_key = image_model_spec.model, image_model_spec.API_KEY
        bots[model_fqn] = FireworksPoeImageBot(
            model=model_fqn,
            api_key=api_key,
            environment=args.environment,
            server_version="0.0.1",
            gcs_bucket_name=os.environ["GCS_BUCKET_NAME"],
        )

    assert (
        len(bots) > 0
    ), "No bots specified, use --text-models or --image-models to specify models to serve"

    app = make_app(bots, allow_without_key=True)

    uvicorn.run(
        app,
        host=server_args.host,
        port=server_args.port,
        log_level="info",
        server_header=False,
        log_config=UVICORN_LOGGING_CONFIG,
    )


if __name__ == "__main__":
    main()
