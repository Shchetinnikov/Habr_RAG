from settings import get_settings
from settings import ModelSource


def get_chat():
    base_settings = get_settings()
    model_enum = ModelSource.get_by_source(base_settings.model_source)

    if not model_enum:
        raise ValueError(f"Unsupported model source: {base_settings.model_source}")

    settings = model_enum.settings_class()
    return model_enum.chat_class(**settings.model_dump())
