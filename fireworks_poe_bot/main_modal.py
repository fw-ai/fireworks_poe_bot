from fireworks_poe_bot.fw_poe_server_bot import FireworksPoeServerBot

import fireworks.client

import argparse
from dataclasses import dataclass
from fastapi_poe import make_app
import os
from modal import Image, Stub, asgi_app


fireworks.client.api_key = os.environ["FW_ACCESS_KEY"]

bot = FireworksPoeServerBot(os.environ["MODEL"], "", "0.0.1")

image = (
    Image.debian_slim()
    .pip_install("fastapi-poe==0.0.23")
    .pip_install("fireworks-ai>=0.6.0")
    .env(
        {
            "FW_ACCESS_KEY": os.environ["FW_ACCESS_KEY"],
            "MODEL": os.environ["MODEL"],
        }
    )
)

stub = Stub("fw-poe-bot")


@stub.function(image=image)
@asgi_app()
def fastapi_app():
    app = make_app(bot, allow_without_key=True)
    return app
