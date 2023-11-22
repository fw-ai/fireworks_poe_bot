import copy
import logging
import sys
import uvicorn

# Force all logs at level < ERROR to go to stdout, and all logs at level >= ERROR to go to stderr
# This makes it so that unstructured log parsing by google cloud logging (formerly StackDriver)
# doesn't spuriously count random log lines as errors
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set the lowest level this logger will handle

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)

stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s %(levelname)s %(processName)s:%(process)d:%(thread)d %(filename)s:%(lineno)s] %(message)s"
)
stdout_handler.setFormatter(formatter)
stderr_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(stdout_handler)
logger.addHandler(stderr_handler)

UVICORN_LOGGING_CONFIG = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)
UVICORN_LOGGING_CONFIG["handlers"]["default"]["stream"] = "ext://sys.stdout"