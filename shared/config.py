"""
Application settings — local development version.

Uses:
  - Mistral AI API  (chat + embeddings) instead of Azure OpenAI
  - ChromaDB        (local vector store) instead of Azure AI Search
  - SQLite          (local file DB)      instead of Azure Cosmos DB
  - No Azure services, no managed identity, no Azure SDK dependencies.

All values are loaded from .env (copy .env.local.example → .env to get started).
"""
from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEVELOPMENT = "development"
    STAGING     = "staging"
    PRODUCTION  = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Deployment environment ─────────────────────────────────────────────────
    ENVIRONMENT: Environment = Environment.DEVELOPMENT

    # ── Mistral AI (replaces Azure OpenAI) ────────────────────────────────────
    # Get your key from: https://console.mistral.ai/
    MISTRAL_API_KEY: SecretStr
    MISTRAL_BASE_URL: str        = "https://api.mistral.ai/v1"
    MISTRAL_CHAT_MODEL: str      = "mistral-small-latest"
    MISTRAL_EMBEDDING_MODEL: str = "mistral-embed"

    # Keep the same field names that agent code references for deployments so
    # we don't have to touch workflow / classify / synthesis code.
    # These alias to the Mistral model names internally.
    @property
    def AZURE_OPENAI_CHAT_DEPLOYMENT(self) -> str:
        return self.MISTRAL_CHAT_MODEL

    @property
    def AZURE_OPENAI_EMBEDDING_DEPLOYMENT(self) -> str:
        return self.MISTRAL_EMBEDDING_MODEL

    # Kept for compatibility — not used in local mode.
    AZURE_OPENAI_API_VERSION: str = "n/a"

    # ── Local vector store (ChromaDB, replaces Azure AI Search) ───────────────
    LOCAL_SEARCH_DB_PATH: str      = "./local_data/chroma"
    LOCAL_SEARCH_COLLECTION: str   = "rag-documents"
    # Azure Search field names kept for config references in tools.
    AZURE_SEARCH_INDEX: str        = "local"
    AZURE_SEARCH_SEMANTIC_CONFIG: str = "local"

    # ── SQLite (replaces Azure Cosmos DB) ─────────────────────────────────────
    SQLITE_DB_PATH: str             = "./local_data/rag.db"
    # Cosmos field names kept so model/memory code references still resolve.
    COSMOS_CONTAINER_CHAT: str      = "chat_history"
    COSMOS_CONTAINER_FEEDBACK: str  = "feedback"
    COSMOS_CONTAINER_SESSIONS: str  = "sessions"
    COSMOS_CONTAINER_LTM: str       = "long_term_memory"

    # ── Inter-agent URLs ───────────────────────────────────────────────────────
    MAIN_AGENT_URL:   str = "http://localhost:8000"
    ORCHESTRATOR_URL: str = "http://localhost:8001"
    RETRIEVAL_URL:    str = "http://localhost:8002"

    # ── Inter-agent auth ───────────────────────────────────────────────────────
    INTERNAL_API_SECRET: SecretStr | None = None

    # ── Teams Bot (optional for local testing) ─────────────────────────────────
    MICROSOFT_APP_ID:        str           = ""
    MICROSOFT_APP_PASSWORD:  SecretStr | None = None
    MICROSOFT_APP_TYPE:      str           = "MultiTenant"
    MICROSOFT_APP_TENANT_ID: str           = ""
    BOT_PORT:                int           = Field(default=3978, ge=1, le=65535)

    # ── Escalation (stubbed locally — logs to stdout) ─────────────────────────
    AZURE_SERVICE_BUS_NAMESPACE: str | None = None
    SB_QUEUE_ESCALATION: str               = "escalation-requests"
    ZENDESK_SUBDOMAIN:   str | None        = None
    ZENDESK_API_TOKEN:   SecretStr | None  = None
    ZENDESK_USER_EMAIL:  str | None        = None
    ZENDESK_GROUP_ID_TICKET: int | None    = None
    ZENDESK_GROUP_ID_SME:    int | None    = None

    # ── RAG tuning ─────────────────────────────────────────────────────────────
    CONFIDENCE_THRESHOLD: float          = Field(default=0.30, ge=0.0, le=1.0)
    CITATION_CONFIDENCE_THRESHOLD: float = Field(default=0.20, ge=0.0, le=1.0)
    MAX_RETRIEVAL_ATTEMPTS: int  = Field(default=2,     ge=1,  le=5)
    RETRIEVAL_TOP_K: int         = Field(default=5,     ge=1,  le=20)
    SYNTHESIS_TEMPERATURE: float = Field(default=0.0,   ge=0.0, le=1.0)
    MAX_QUERY_LENGTH: int        = Field(default=2000,  ge=50, le=8000)
    SYNTHESIS_MAX_CONTEXT_CHARS: int = Field(default=12000, ge=2000, le=40000)
    SYNTHESIS_MAX_SOURCES: int   = Field(default=5,     ge=1,  le=10)
    SYNTHESIS_MAX_ANSWER_CHARS: int = Field(default=10000, ge=500, le=20000)
    SYNTHESIS_MAX_TOKENS: int    = Field(default=4000,  ge=500, le=16000)

    # ── Memory ─────────────────────────────────────────────────────────────────
    SESSION_MAX_TURNS: int     = Field(default=10,     ge=1,   le=50)
    SESSION_TTL_SECONDS: int   = Field(default=604800, ge=3600)
    LTM_SUMMARY_EVERY_N: int   = Field(default=5,      ge=1,   le=20)
    LTM_MAX_SUMMARY_CHARS: int = Field(default=3000,   ge=500, le=10000)
    LTM_MAX_FACTS: int         = Field(default=10,     ge=3,   le=30)

    # ── Rate limiting (in-memory, Redis not needed locally) ───────────────────
    REDIS_URL: str | None     = None
    RATE_LIMIT_RPM:   int     = Field(default=60, ge=1, le=600)
    RATE_LIMIT_BURST: int     = Field(default=10, ge=1, le=50)

    # ── Domain classification ──────────────────────────────────────────────────
    DOMAIN_CONFIDENCE_THRESHOLD: float = Field(default=0.6, ge=0.0, le=1.0)

    # ── Escalation SLAs ────────────────────────────────────────────────────────
    ESCALATION_SLA_TICKET: str = "4 business hours"
    ESCALATION_SLA_SME:    str = "2 business hours"

    # ── Observability ──────────────────────────────────────────────────────────
    APPLICATIONINSIGHTS_CONNECTION_STRING: str | None = None
    LOG_LEVEL: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
