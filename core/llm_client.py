"""
LLM router with automatic provider fallback.

Every other module calls:
    responses_create(...)
    responses_parse(...)

This router automatically tries Vercel -> Groq -> Gemini -> OpenRouter.
"""

from openai import OpenAI
from core.config import config

from openai import (
    APIError,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    AuthenticationError,
)
_clients = {}


def _get_client(provider: str) -> OpenAI:
    if provider not in _clients:

        if provider == "vercel":
            _clients[provider] = OpenAI(
                api_key=config.AI_GATEWAY_API_KEY,
                base_url=config.AI_GATEWAY_BASE_URL,
            )

        elif provider == "groq":
            _clients[provider] = OpenAI(
                api_key=config.GROQ_API_KEY,
                base_url=config.GROQ_BASE_URL,
            )

        elif provider == "gemini":
            _clients[provider] = OpenAI(
                api_key=config.GEMINI_API_KEY,
                base_url=config.GEMINI_BASE_URL,
            )

        elif provider == "openrouter":
            _clients[provider] = OpenAI(
                api_key=config.OPENROUTER_API_KEY,
                base_url=config.OPENROUTER_BASE_URL,
            )

        else:
            raise ValueError(f"Unknown provider: {provider}")

    return _clients[provider]


def _provider_and_model(model: str):
    """
    Example:
        openai/gpt-5.4-mini
        groq/llama-3.3-70b-versatile
        google/gemini-2.5-flash
        openrouter/meta-llama/llama-3.3-70b-instruct
    """

    if model.startswith("openai/"):
        return "vercel", model

    if model.startswith("groq/"):
        return "groq", model.split("/", 1)[1]

    if model.startswith("google/"):
        return "gemini", model.split("/", 1)[1]

    if model.startswith("openrouter/"):
        return "openrouter", model.split("/", 1)[1]

    raise ValueError(model)


def responses_create(models, **kwargs):

    last_error = None

    for model in models:

        provider, actual_model = _provider_and_model(model)

        try:

            client = _get_client(provider)

            kwargs["model"] = actual_model

            print(f"Trying {provider}: {actual_model}")

            return client.responses.create(**kwargs)

        except (RateLimitError, APIError, APITimeoutError,AuthenticationError) as e:

            print(f"{provider} failed: {e}")

            last_error = e

            continue

    raise last_error


def responses_parse(models, **kwargs):

    last_error = None

    for model in models:

        provider, actual_model = _provider_and_model(model)

        try:

            client = _get_client(provider)

            kwargs["model"] = actual_model

            return client.responses.parse(**kwargs)

        except (RateLimitError, APIError, APITimeoutError, AuthenticationError) as e:

            last_error = e

            continue

    raise last_error