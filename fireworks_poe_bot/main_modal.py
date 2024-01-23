from fireworks_poe_bot.fw_poe_text_bot import FireworksPoeTextBot

import fireworks.client

import argparse
from dataclasses import dataclass
from fastapi_poe import make_app
import os
from modal import Image, Stub, asgi_app


bot = FireworksPoeTextBot(os.environ["MODEL"], "", "0.0.1")

image = (
    Image.debian_slim()
    .pip_install("fastapi-poe==0.0.23")
    .pip_install("fireworks-ai>=0.6.0")
    .env(
        {
            "FIREWORKS_API_BASE": os.environ.get(
                "FIREWORKS_API_BASE", fireworks.client.base_url
            ),
            "FIREWORKS_API_KEY": os.environ.get(
                "FIREWORKS_API_KEY", fireworks.client.api_key
            ),
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
