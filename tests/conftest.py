"""
Shared pytest setup. Sets safe default env vars BEFORE any project module is
imported, so importing core.config never crashes during test collection just
because .env doesn't exist yet in a fresh checkout / CI environment.
"""
import os

os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "geointel")
os.environ.setdefault("POSTGRES_USER", "geointel")
os.environ.setdefault("POSTGRES_PASSWORD", "geointel")
os.environ.setdefault("AI_GATEWAY_API_KEY", "test-key-not-used-by-offline-tests")
