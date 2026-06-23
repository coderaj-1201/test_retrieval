"""
Application settings — loaded from environment variables only.

In production (Azure Container Apps) all authentication uses DefaultAzureCredential
(managed identity). No API keys or connection strings are used.

How config is injected in ACA:
  - Non-secret values (endpoints, deployment names, container names) are set
    directly as ACA environment variables in Bicep/Terraform.
  - Secret values (INTERNAL_API_SECRET, ZENDESK_API_TOKEN, MicrosoftAppPassword)
    are injected via Key Vault secret references on the container app.
  - Azure services (OpenAI, Search, Cosmos, Service Bus) are accessed via the
    managed identity — no credential env vars needed for those.

See infra/PERMISSIONS.md for the full list of RBAC roles the managed identity
must be assigned before deployment.
"""
from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEVELOPMENT = "development"
    STAGING     = "staging"
    PRODUCTION  = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Loads .env when present (local dev convenience).
        # In ACA, env vars are injected directly — no .env file needed.
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Deployment environment ─────────────────────────────────────────────────
    ENVIRONMENT: Environment = Environment.PRODUCTION

    # ── Azure AI Foundry ───────────────────────────────────────────────────────
    # Optional: teams-bot does not use these directly
    AZURE_FOUNDRY_PROJECT_ENDPOINT: AnyHttpUrl | None = None
    AZURE_OPENAI_ENDPOINT: AnyHttpUrl | None          = None
    AZURE_OPENAI_CHAT_DEPLOYMENT: str      = "gpt-41-mini"
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = "text-embedding-3-large"
    AZURE_OPENAI_API_VERSION: str          = "2025-01-01-preview"

    # ── Claude (complex query routing) ────────────────────────────────────────
    # Set CLAUDE_ENDPOINT to the Azure AI Foundry model inference endpoint for
    # Claude. If unset, all queries fall back to AZURE_OPENAI_CHAT_DEPLOYMENT.
    CLAUDE_ENDPOINT:          AnyHttpUrl | None = None
    CLAUDE_CHAT_DEPLOYMENT:   str               = "claude-sonnet-4-6"

    # ── Azure AI Search ────────────────────────────────────────────────────────
    AZURE_SEARCH_ENDPOINT: AnyHttpUrl | None = None
    AZURE_SEARCH_INDEX: str               = "idx-rag"
    AZURE_SEARCH_SEMANTIC_CONFIG: str     = "rag-semantic-config"

    # ── Cosmos DB ──────────────────────────────────────────────────────────────
    COSMOS_ENDPOINT: AnyHttpUrl | None = None
    COSMOS_DATABASE: str                   = "csmsdb-aishrdsvcs-eus-prod"
    COSMOS_CONTAINER_CHAT: str             = "chat-history"
    COSMOS_CONTAINER_FEEDBACK: str         = "feedback"
    COSMOS_CONTAINER_SESSIONS: str         = "sessions"
    COSMOS_CONTAINER_LTM: str             = "long-term-memory"

    # ── Inter-agent URLs ───────────────────────────────────────────────────────
    # Set these to the internal ACA URLs of each agent container.
    MAIN_AGENT_URL:    AnyHttpUrl          = "http://main-agent:8000"
    ORCHESTRATOR_URL:  AnyHttpUrl          = "http://orchestrator:8001"
    RETRIEVAL_URL:     AnyHttpUrl          = "http://retrieval:8002"

    # ── Inter-agent auth ───────────────────────────────────────────────────────
    # Shared HMAC secret sent as X-Internal-Secret header between agents.
    # Injected by ACA from Key Vault. Can be left unset in local dev only.
    INTERNAL_API_SECRET: SecretStr | None  = None

    # ── Teams Bot ──────────────────────────────────────────────────────────────
    # MicrosoftAppId and MicrosoftAppTenantId are non-secret (they appear in
    # Azure AD app registrations). MicrosoftAppPassword is a secret injected
    # from Key Vault via ACA secret reference.
    MICROSOFT_APP_ID:        str           = ""
    MICROSOFT_APP_PASSWORD:  SecretStr | None = None
    MICROSOFT_APP_TYPE:      str           = "MultiTenant"
    MICROSOFT_APP_TENANT_ID: str           = ""
    BOT_PORT:                int           = Field(default=3978, ge=1, le=65535)

    # ── Service Bus (escalation fallback) ─────────────────────────────────────
    # Set AZURE_SERVICE_BUS_NAMESPACE; managed identity provides access.
    # Connection strings are NOT supported — use managed identity only.
    AZURE_SERVICE_BUS_NAMESPACE: str | None = None
    SB_QUEUE_ESCALATION: str               = "escalation-requests"

    # ── Zendesk (primary escalation channel) ──────────────────────────────────
    # Zendesk does not support managed identity.
    # ZENDESK_API_TOKEN must be injected from Key Vault via ACA secret reference.
    ZENDESK_SUBDOMAIN:       str | None       = None
    ZENDESK_API_TOKEN:       SecretStr | None = None
    ZENDESK_USER_EMAIL:      str | None       = None
    ZENDESK_GROUP_ID_TICKET: int | None       = None
    ZENDESK_GROUP_ID_SME:    int | None       = None

    # ── RAG tuning ─────────────────────────────────────────────────────────────
    CONFIDENCE_THRESHOLD: float          = Field(default=0.65, ge=0.0, le=1.0)
    CITATION_CONFIDENCE_THRESHOLD: float = Field(default=0.40, ge=0.0, le=1.0)
    MAX_RETRIEVAL_ATTEMPTS: int  = Field(default=3,    ge=1,   le=5)
    RETRIEVAL_TOP_K: int         = Field(default=5,    ge=1,   le=20)
    SYNTHESIS_TEMPERATURE: float = Field(default=0.0,  ge=0.0, le=1.0)
    MAX_QUERY_LENGTH: int        = Field(default=2000, ge=50,  le=8000)
    SYNTHESIS_MAX_CONTEXT_CHARS: int = Field(default=12000, ge=2000, le=40000)
    SYNTHESIS_MAX_SOURCES: int   = Field(default=5,    ge=1,   le=10)
    SYNTHESIS_MAX_ANSWER_CHARS: int = Field(default=10000, ge=500, le=20000)
    SYNTHESIS_MAX_TOKENS: int = Field(default=6000, ge=500, le=16000)

    # ── Memory ─────────────────────────────────────────────────────────────────
    SESSION_MAX_TURNS: int       = Field(default=10,     ge=1,   le=50)
    SESSION_TTL_SECONDS: int     = Field(default=604800, ge=3600)
    LTM_SUMMARY_EVERY_N: int     = Field(default=5,      ge=1,   le=20)
    LTM_MAX_SUMMARY_CHARS: int   = Field(default=3000,   ge=500, le=10000)
    LTM_MAX_FACTS: int           = Field(default=10,     ge=3,   le=30)

    # ── Rate limiting ──────────────────────────────────────────────────────────
    REDIS_URL: str | None = None
    RATE_LIMIT_RPM:   int = Field(default=20, ge=1,  le=600)
    RATE_LIMIT_BURST: int = Field(default=5,  ge=1,  le=50)

    # ── Domain classification ──────────────────────────────────────────────────
    DOMAIN_CONFIDENCE_THRESHOLD: float = Field(default=0.6, ge=0.0, le=1.0)

    # ── Escalation SLAs ────────────────────────────────────────────────────────
    ESCALATION_SLA_TICKET: str = "4 business hours"
    ESCALATION_SLA_SME:    str = "2 business hours"

    # ── Observability ──────────────────────────────────────────────────────────
    # Application Insights connection string — non-privileged telemetry push.
    # Set via ACA environment variable; not sensitive enough for Key Vault.
    APPLICATIONINSIGHTS_CONNECTION_STRING: str | None = None
    LOG_LEVEL: str = "INFO"

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        import logging
        level = getattr(logging, v.upper(), None)
        if level is None:
            raise ValueError(f"Invalid LOG_LEVEL '{v}'. Must be DEBUG/INFO/WARNING/ERROR/CRITICAL.")
        return v.upper()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
