# Container Managed Identity — Required Permissions

Every container app (main-agent, orchestrator-agent, retrieval-agent, teams-bot)
shares the **same user-assigned managed identity** (or each has its own system-assigned
identity). The identity must be assigned every role listed below before first deployment.

> All commands below use Azure CLI. Replace `<PLACEHOLDER>` values with your actual
> resource names/IDs.

---

## 1. Azure OpenAI / AI Foundry

**Role:** `Cognitive Services OpenAI User`
**Scope:** The Azure OpenAI resource (or AI Foundry hub)

```bash
az role assignment create \
  --role "Cognitive Services OpenAI User" \
  --assignee <MANAGED_IDENTITY_PRINCIPAL_ID> \
  --scope /subscriptions/<SUB_ID>/resourceGroups/<RG>/providers/Microsoft.CognitiveServices/accounts/<OPENAI_RESOURCE>
```

> Grants: `chat/completions`, `embeddings`, and model listing.
> Required by: orchestrator-agent (classify), retrieval-agent (synthesis, HyDE, decompose, embeddings), main-agent (LTM summary).

---

## 2. Azure AI Foundry Project (AIProjectClient)

**Role:** `Azure AI Developer`
**Scope:** The AI Foundry project resource

```bash
az role assignment create \
  --role "Azure AI Developer" \
  --assignee <MANAGED_IDENTITY_PRINCIPAL_ID> \
  --scope /subscriptions/<SUB_ID>/resourceGroups/<RG>/providers/Microsoft.MachineLearningServices/workspaces/<FOUNDRY_PROJECT>
```

> Required by: `get_foundry_client()` — used for tracing and evaluation features.

---

## 3. Azure AI Search

### 3a. Read and query the index
**Role:** `Search Index Data Reader`
**Scope:** The Search service resource

```bash
az role assignment create \
  --role "Search Index Data Reader" \
  --assignee <MANAGED_IDENTITY_PRINCIPAL_ID> \
  --scope /subscriptions/<SUB_ID>/resourceGroups/<RG>/providers/Microsoft.Search/searchServices/<SEARCH_SERVICE>
```

> Required by: retrieval-agent (hybrid search queries).

### 3b. Create / update index (setup scripts only)
**Role:** `Search Index Data Contributor`
**Scope:** The Search service resource

```bash
az role assignment create \
  --role "Search Index Data Contributor" \
  --assignee <MANAGED_IDENTITY_PRINCIPAL_ID> \
  --scope /subscriptions/<SUB_ID>/resourceGroups/<RG>/providers/Microsoft.Search/searchServices/<SEARCH_SERVICE>
```

> Required by: `scripts/setup_search.py` (index provisioning). Not needed at runtime
> if the index already exists. Safe to assign permanently.

---

## 4. Cosmos DB — Data Plane RBAC

> **Important:** Cosmos DB has two separate permission systems:
> - **Azure IAM** (control plane) — manages the account itself. Not needed here.
> - **Cosmos DB native RBAC** (data plane) — controls read/write access to data.
>   This is what the application needs, and it is NOT visible in the Azure portal IAM blade.

**Role:** `Cosmos DB Built-in Data Contributor`
**Role ID:** `00000000-0000-0000-0000-000000000002`
**Scope:** The Cosmos DB account

```bash
az cosmosdb sql role assignment create \
  --account-name <COSMOS_ACCOUNT> \
  --resource-group <RG> \
  --role-definition-id "00000000-0000-0000-0000-000000000002" \
  --principal-id <MANAGED_IDENTITY_PRINCIPAL_ID> \
  --scope "/"
```

> Grants: read, write, upsert, delete, query on all containers.
> Required by: all four agents (session memory, chat history, feedback, LTM).

To verify the assignment was created:
```bash
az cosmosdb sql role assignment list \
  --account-name <COSMOS_ACCOUNT> \
  --resource-group <RG>
```

---

## 5. Azure Service Bus (escalation fallback)

**Role:** `Azure Service Bus Data Sender`
**Scope:** The Service Bus namespace

```bash
az role assignment create \
  --role "Azure Service Bus Data Sender" \
  --assignee <MANAGED_IDENTITY_PRINCIPAL_ID> \
  --scope /subscriptions/<SUB_ID>/resourceGroups/<RG>/providers/Microsoft.ServiceBus/namespaces/<SB_NAMESPACE>
```

> Required by: main-agent (escalation fallback when Zendesk is unavailable).
> Skip if Service Bus is not used in your deployment.

---

## 6. Key Vault (secret references)

**Role:** `Key Vault Secrets User`
**Scope:** The Key Vault resource

```bash
az role assignment create \
  --role "Key Vault Secrets User" \
  --assignee <MANAGED_IDENTITY_PRINCIPAL_ID> \
  --scope /subscriptions/<SUB_ID>/resourceGroups/<RG>/providers/Microsoft.KeyVault/vaults/<KV_NAME>
```

> Required for ACA Key Vault secret references (INTERNAL_API_SECRET, MicrosoftAppPassword,
> ZENDESK_API_TOKEN). The identity reads the secret value at container startup.

---

## 7. Container Registry (image pull)

**Role:** `AcrPull`
**Scope:** The Azure Container Registry

```bash
az role assignment create \
  --role "AcrPull" \
  --assignee <MANAGED_IDENTITY_PRINCIPAL_ID> \
  --scope /subscriptions/<SUB_ID>/resourceGroups/<RG>/providers/Microsoft.ContainerRegistry/registries/<ACR_NAME>
```

> Required for ACA to pull the container image from a private registry.
> Assign on the registry, not on an individual image.

---

## Summary Table

| Service | Role | Required? |
|---|---|---|
| Azure OpenAI / Foundry | Cognitive Services OpenAI User | **Yes** |
| Azure AI Foundry Project | Azure AI Developer | **Yes** |
| Azure AI Search | Search Index Data Reader | **Yes** |
| Azure AI Search | Search Index Data Contributor | Setup scripts only |
| Cosmos DB (data plane) | Cosmos DB Built-in Data Contributor | **Yes** |
| Azure Service Bus | Azure Service Bus Data Sender | If using SB escalation |
| Key Vault | Key Vault Secrets User | **Yes** (for secrets) |
| Container Registry | AcrPull | **Yes** (private ACR) |

---

## How to Find the Managed Identity Principal ID

If using a **user-assigned managed identity:**
```bash
az identity show \
  --name <IDENTITY_NAME> \
  --resource-group <RG> \
  --query principalId -o tsv
```

If using a **system-assigned managed identity** on the container app:
```bash
az containerapp show \
  --name <APP_NAME> \
  --resource-group <RG> \
  --query identity.principalId -o tsv
```

---

## ACA Environment Variables (non-secret)

These are set directly on the container app — no Key Vault needed:

| Variable | Description |
|---|---|
| `AZURE_FOUNDRY_PROJECT_ENDPOINT` | AI Foundry project endpoint URL |
| `AZURE_OPENAI_ENDPOINT` | OpenAI endpoint (includes `/openai/v1`) |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | Deployment name e.g. `gpt-41-mini` |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Embedding deployment name |
| `AZURE_SEARCH_ENDPOINT` | Search service URL |
| `AZURE_SEARCH_INDEX` | Search index name |
| `COSMOS_ENDPOINT` | Cosmos DB account URL |
| `COSMOS_DATABASE` | Database name |
| `ORCHESTRATOR_URL` | Internal ACA URL of orchestrator container |
| `RETRIEVAL_URL` | Internal ACA URL of retrieval container |
| `MAIN_AGENT_URL` | Internal ACA URL of main-agent container |
| `ENVIRONMENT` | `production` |
| `LOG_LEVEL` | `INFO` |

## ACA Secret References (Key Vault)

These must be stored in Key Vault and referenced as secrets on the container app:

| Variable | Description |
|---|---|
| `INTERNAL_API_SECRET` | Shared HMAC secret for inter-agent auth |
| `MICROSOFT_APP_PASSWORD` | Bot Framework app password |
| `ZENDESK_API_TOKEN` | Zendesk API token for ticket creation |
