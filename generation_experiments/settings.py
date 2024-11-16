from typing import Literal
from enum import Enum

from langchain_mistralai.chat_models import ChatMistralAI
from langchain_groq import ChatGroq

from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class LLMSettings(BaseSettings):
    model_name: str
    model_source: Literal["groq", "mistral"]

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", protected_namespaces=("settings_",)
    )

class GroqSettings(LLMSettings):
    groq_api_key: str
    model_source: str = Field(exclude=True) 

class MistralSettings(LLMSettings):
    mistral_api_key: str
    model_source: str = Field(exclude=True)

class ModelSource(Enum):
    GROQ = ("groq", GroqSettings, ChatGroq)
    MISTRAL = ("mistral", MistralSettings, ChatMistralAI)

    def __init__(self, source: str, settings_class: type, chat_class: type):
        self.source = source
        self.settings_class = settings_class
        self.chat_class = chat_class

    @classmethod
    def get_by_source(cls, source: str):
        for item in cls:
            if item.source == source:
                return item
        return None


def get_settings(model_source: str | None = None) -> LLMSettings:
    if model_source:
        model_enum = ModelSource.get_by_source(model_source)
        if model_enum:
            return model_enum.settings_class()
    return LLMSettings()
