# Kaori N2G Core Features

```cypher
CREATE VECTOR INDEX `rel_embedding_index`
FOR ()-[r:DYNAMIC_RELATIONSHIP]-()
ON (r.embedding)
OPTIONS {indexConfig: {
 `vector.dimensions`: 384,
 `vector.similarity_function`: 'cosine'
}}
```
Brief technical documentation of the Narrative-to-Graph architecture for the Kaori Journaling Companion.

---

## Table of Contents

1. [⚙️ How to Run](#️-how-to-run)  
2. [🧠 Core Features](#-core-features)  
   - [Temporal Context Awareness](#1-temporal-context-awareness)  
   - [Multimodality of Entities](#2-multimodality-of-entities)  
   - [Semantic Similarity Handler](#3-semantic-similarity-handler)  
   - [Persistence & Deduplication](#4-persistence--deduplication)  
   - [Relationship History](#5-relationship-history)  
   - [Persistent LLM Caching](#6-persistent-llm-caching)  

---

## ⚙️ How to Run

### 1. Environment Configuration

Create a `.env` file in the root directory:

```env
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
OPENROUTER_API_KEY=your_api_key
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet  # optional

```

### 2. Infrastructure Setup

Start the Neo4j database using Docker Compose:

```bash
docker compose up -d

```

The Neo4j Browser will be accessible at `http://localhost:7474`.

### 3. Application Execution

Install dependencies and run the application using `uv`:

```bash
uv sync
uv run main.py
```

or run the MCP server directly:

```bash
uv run server.py
```

---

## 📁 Project Layout (main files and folders)

- `prompts/phase1_detective.txt`: phase 1 prompt template (LLM detective)
- `prompts/phase2_architect.txt`: phase 2 prompt template (LLM architect)
- `config.py`: environment config (Neo4j, OpenRouter, server host/port)
- `docker-compose.yml`: Neo4j container with APOC plugin and local data volume
- `neo4j_client.py`: async Neo4j graph operations (nodes/relationships/query)
- `openrouter_client.py`: OpenRouter API wrapper + response caching
- `prompt_manager.py`: prompt templating and substitution engine
- `server.py`: MCP server implementation with `process_journal_entry`, `query_graph`, `get_person_summary`
- `test_client.py` (temporary): MCP client for integration testing
- `vectorizer.py`: sentence embedding (SentenceTransformer) helper

---

## 🧠 Core Features

This project currently implements the following behavior in the core modules:

- `process_journal_entry` tool in `server.py`: coordinates phase 1 and phase 2 prompts, handles LLM calls, vectorizes relationship context, and writes nodes/relationships into Neo4j.
- `prompt_manager.py`: templates phase 1 and phase 2 prompt files in `prompts/` and fills placeholders (`user_name`, `user_aliases`, `entry_text`, etc.).
- `openrouter_client.py`: async OpenRouter chat completion client with robust JSON extraction and local `temp/response_*.json` logging.
- `neo4j_client.py`: async graph operations (connection, create nodes/relationships, query, person timeline, person events, summary).
- `vectorizer.py`: sentence embeddings via `SentenceTransformer` (`all-MiniLM-L6-v2`) with CPU/GPU detection.
- `test_client.py`: integration test harness for MCP server operations (list tools, call tool, check timeline results).
- `docker-compose.yml`: local Neo4j container configuration including APOC plugin and data volume persistence.

### Workflows

1. Take user journal entry (`entry_text`) → phase 1 prompt (detective analysis).
2. If graph update is approved → phase 2 prompt (architect graph structure output).
3. Phase 2 nodes/relationships are vectorized and committed to Neo4j.
4. Graph querying from tool API: `query_graph` (timeline/person_events/custom) and `get_person_summary`.

### Current limitations

- No explicit entity deduplication / sophisticated canonicalization in current graph API beyond basic node insertion.
- No built-in JSON schema validation beyond minimal MCP input schema in `server.py`.
- LLM output is expected to be well-formed JSON (openrouter client has best-effort extraction).

---
**Tech Stack:** Neo4j, OpenRouter, Neo4j Python driver, MCP server/client, SentenceTransformer, Docker Compose.
