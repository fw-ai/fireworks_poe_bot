from pydantic import BaseModel, Field
from typing import List
import json


# Define the JSON schema using Pydantic
class ModelConfig(BaseModel):
    model: str
    API_KEY: str


class Config(BaseModel):
    text_models: List[ModelConfig] = []
    image_models: List[ModelConfig] = []


def load_config(file_path: str) -> Config:
    with open(file_path, "r") as config_file:
        data = json.load(config_file)

    config = Config(**data)

    return config
