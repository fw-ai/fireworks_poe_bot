from pydantic import BaseModel, Field
from typing import List, Optional
import json


# Define the JSON schema using Pydantic
class ModelConfig(BaseModel):
    model: str
    API_KEY: str

    account_override: Optional[str] = None
    model_override: Optional[str] = None

class TextModelConfig(ModelConfig):
    allow_attachments: Optional[bool] = False
    input_image_size: Optional[int] = None

class QRCodeConfig(ModelConfig):
    conditioning_scale: Optional[float] = None


class Config(BaseModel):
    text_models: List[TextModelConfig] = []
    image_models: List[ModelConfig] = []
    qr_models: List[QRCodeConfig] = []


def load_config(file_path: str) -> Config:
    with open(file_path, "r") as config_file:
        data = json.load(config_file)

    config = Config(**data)

    return config
