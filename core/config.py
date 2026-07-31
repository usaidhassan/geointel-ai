"""
Central place for all environment-driven configuration.
Every other module imports from here instead of reading os.environ directly,
so there is exactly one place to look when something is misconfigured.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # LLM providers
    AI_GATEWAY_API_KEY = os.getenv("AI_GATEWAY_API_KEY", "")
    AI_GATEWAY_BASE_URL = os.getenv(
        "AI_GATEWAY_BASE_URL",
        "https://ai-gateway.vercel.sh/v1",
    )

    GROQ_BASE_URL = os.getenv(
        "GROQ_BASE_URL",
        "https://api.groq.com/openai/v1",
    )

    OPENROUTER_BASE_URL = os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1",
    )

    GEMINI_BASE_URL = os.getenv(
        "GEMINI_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

    # Default model if you still want one
    LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-5.4-mini")

    # Fallback order for each task
    RAG_MODELS = [
        m.strip() for m in os.getenv(
            "RAG_MODELS",
            "openai/gpt-5.4-mini,groq/llama-3.3-70b-versatile,gemini-2.5-flash,openrouter/meta-llama/llama-3.3-70b"
        ).split(",")
        if m.strip()
    ]

    AGENT_MODELS = [
        m.strip() for m in os.getenv(
            "AGENT_MODELS",
            "openai/gpt-5.4-mini,groq/llama-3.3-70b-versatile,gemini-2.5-flash,openrouter/meta-llama/llama-3.3-70b"
        ).split(",")
        if m.strip()
    ]

    JUDGE_MODELS = [
        m.strip() for m in os.getenv(
            "JUDGE_MODELS",
            "gemini-2.5-flash,groq/llama-3.3-70b-versatile,openai/gpt-5-mini"
        ).split(",")
        if m.strip()
    ]

    # Postgres
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "geointel")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "geointel")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "geointel")
    POSTGRES_SSLMODE = os.getenv("POSTGRES_SSLMODE", "require")

    # Embeddings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

    # App
    API_PORT = int(os.getenv("API_PORT", "8000"))

    @classmethod
    def pg_conninfo(cls) -> str:
        return (
            f"host={cls.POSTGRES_HOST} "
            f"port={cls.POSTGRES_PORT} "
            f"dbname={cls.POSTGRES_DB} "
            f"user={cls.POSTGRES_USER} "
            f"password={cls.POSTGRES_PASSWORD} "
            f"sslmode={cls.POSTGRES_SSLMODE} "
            f"connect_timeout=5"
        )


config = Config()
