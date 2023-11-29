from pydantic import BaseModel, Field
from typing import List, Optional


# Define the JSON schema using Pydantic
class ModelConfig(BaseModel):
    model: str
    api_key: str

    SERVER_endpoint_account_override: Optional[str] = None
    SERVER_endpoint_model_override: Optional[str] = None

    @property
    def model_fqn(self):
        if (
            self.SERVER_endpoint_account_override is not None
            or self.SERVER_endpoint_model_override is not None
        ):
            _, account, _, model = self.model.split("/")
            account = self.SERVER_endpoint_account_override or account
            model = self.SERVER_endpoint_model_override or model
            return f"accounts/{account}/models/{model}"
        else:
            return self.model
