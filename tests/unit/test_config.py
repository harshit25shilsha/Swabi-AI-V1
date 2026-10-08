from app.core.config import settings


def test_settings_loaded():
    assert settings.app_name
    assert settings.llm_provider == "groq"
    assert settings.llm_model


def test_llm_retries_default():
    assert settings.llm_max_retries >= 0
