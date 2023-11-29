from abc import ABC, abstractmethod
from typing import Any, Dict, List
from dataclasses import dataclass
from pydantic import BaseModel
from fireworks_poe_bot.config import ModelConfig

class LoggingPlugin(ABC):
    @abstractmethod
    def log_warn(self, payload: Dict[str, Any]):
        ...

    @abstractmethod
    def log_info(self, payload: Dict[str, Any]):
        ...

    @abstractmethod
    def log_error(self, payload: Dict[str, Any]):
        ...


_LOGGING_PLUGINS: List[LoggingPlugin] = []


def register_logging_plugin(plugin: LoggingPlugin):
    _LOGGING_PLUGINS.append(plugin)


def log_warn(payload: Dict[str, Any]):
    for plugin in _LOGGING_PLUGINS:
        plugin.log_warn(payload)


def log_info(payload: Dict[str, Any]):
    for plugin in _LOGGING_PLUGINS:
        plugin.log_info(payload)


def log_error(payload: Dict[str, Any]):
    for plugin in _LOGGING_PLUGINS:
        plugin.log_error(payload)


@dataclass
class _BotPlugin:
    BotPluginClass: type
    BotConfigClass: BaseModel
    config_key: str


BOT_PLUGINS: List[_BotPlugin] = []

def register_bot_plugin(config_key: str, BotConfigClass: type = ModelConfig):
    def decorator(BotPluginClass: type):
        BOT_PLUGINS.append(_BotPlugin(
            BotPluginClass=BotPluginClass,
            BotConfigClass=BotConfigClass,
            config_key=config_key,
        ))

    return decorator
