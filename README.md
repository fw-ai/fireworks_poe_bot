# Fireworks Poe Bot

This codebase is the underlying implementation of the [Mistral-7B](https://poe.com/fw-mistral-7b) on Poe. This implementation calls into the `accounts/fireworks/models/mistral-7b-instruct-4k` model from [fireworks.ai](https://app.fireworks.ai/). However, this bot implementation is fully general and can be used to create bots that call into other Fireworks models.


## Server Arguments

The server has several important arguments:

* `--model` - This flag specifies the model to call into on Fireworks. For example: `accounts/fireworks/models/mistral-7b-instruct-4k`
* `--fireworks-api-key` - This flag specifies the API key to use to call into Fireworks. API keys can be retrieved from the [Fireworks console](https://app.fireworks.ai/users?tab=apps).

There are several optional arguments that can usually be left as default:

* `--host` specifies the hostname to bind to. You usually don't need to set this
* `--port` specifies the port for the bot to listen on. By default this is port 80.
* `--fireworks-api-base-url` specifies the base URL to call into Fireworks. This is `https://api.fireworks.ai` by default. You usually don't need to modify this.


## Running the bot

The bot can be run locally by installing the package and running the module:

```bash
$ pip install -e .
$ python -m fireworks_poe_bot --model accounts/fireworks/models/mistral-7b-instruct-4k --fireworks-api-key <your API key>
INFO:     Started server process [50060]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:80 (Press CTRL+C to quit)
```

The server will then be listening on port 80. You can then communicate with the bot using the [Poe simulator](https://github.com/poe-platform/poe-protocol/tree/main/simulator_poe) for testing


## Testing

Test cases can be found in `test/test_poebot.py`. That file can be invoked directly or with `pytest`. The test file mocks out the Fireworks service -- it is intended for quick iteration of development of the poebot logic itself.

## Deployment

As the bot listens on port 80, it can be deployed in many contexts, including through Modal and in other infrastructure like k8s.

### Deploying to Modal

From the root of this repository, run:

```bash
$ MODEL=accounts/fireworks/models/mistral-7b-instruct-4k FW_ACCESS_KEY=<your API key> modal deploy fireworks_poe_bot/main_modal.py
✓ Created objects.
├── 🔨 Created mount /root/fireworks/public-py/fireworks
├── 🔨 Created mount /root/fireworks/py/fireworks
├── 🔨 Created mount /root/fireworks_poe_bot/fireworks_poe_bot
└── 🔨 Created fastapi_app => https://XXXX--fw-poe-bot-fastapi-app.modal.run
✓ App deployed! 🎉
View Deployment: https://modal.com/apps/XXXX/fw-poe-bot
```

You should then be able to curl the `https` URL returned by Modal:

```bash
$ curl https://XXXX--fw-poe-bot-fastapi-app.modal.run
<html><body><h1>FastAPI Poe bot server</h1><p>Congratulations! Your server is running. To connect it to Poe, create a bot at <a href="https://poe.com/create_bot?server=1">https://poe.com/create_bot?server=1</a>.</p></body></html>
```

After this, follow the instructions on the [Poe documentation](https://developer.poe.com/server-bots/quick-start#integrating-with-poe) to create a Poe bot and point it to your running deployment.

## Contributing

Contributions and bug reports are greatly appreciated!
