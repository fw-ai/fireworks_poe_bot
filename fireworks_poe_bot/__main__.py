from fireworks_poe_bot.fw_poe_server_bot import FireworksPoeServerBot

import fireworks.client

import argparse
from dataclasses import dataclass
from fastapi_poe import make_app
import uvicorn


@dataclass
class ServerArgs:
    fireworks_api_key: str = ""
    fireworks_api_base_url: str = ""
    host: str = "0.0.0.0"
    port: int = 80
    model: str = ""
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
    server_group.add_argument("-m", "--model", type=str, default=server_args.model)
    server_group.add_argument("-f", "--fireworks-api-key", type=str, default="")
    server_group.add_argument("-u", "--fireworks-api-base-url", type=str, default="")
    server_group.add_argument("-a", "--allow-attachments", action="store_true")
    server_group.add_argument(
        "-e", "--environment", type=str, default=server_args.environment
    )

    args = parser.parse_args()
    assert args.model, "Model must be specified"

    # Parse arguments.
    for k, v in vars(args).items():
        for g in [server_args]:
            if hasattr(g, k):
                setattr(g, k, v)
                break
        else:
            assert k in ["print_supported_models"], f"Unknown argument {k}"

    fireworks.client.api_key = server_args.fireworks_api_key
    if server_args.fireworks_api_base_url:
        fireworks.client.base_url = server_args.fireworks_api_base_url

    bot = FireworksPoeServerBot(
        model=args.model,
        environment=args.environment,
        server_version="0.0.1",
        allow_attachments=args.allow_attachments,
    )
    app = make_app(bot, allow_without_key=True)

    uvicorn.run(
        app,
        host=server_args.host,
        port=server_args.port,
        log_level="info",
        server_header=False,
    )


if __name__ == "__main__":
    main()
