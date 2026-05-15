
Just a stateless FastAPI microservice for entity extraction and mapping stuff into Neo4j.

## Overview

Storing raw text as giant strings is kinda useless, so this just extracts your narratives into a clean, straightforward graph structure in Neo4j:

* **`Entry ───[CONTAINS]───> Event`** (What happened)
* **`Event ───[INVOLVES]───> Person`** (Who was there)
* **`Event ───[TRIGGERS]───> EmotionState`** (The emotional impact)
* **`EmotionState ───[INSTANCE_OF]───> EmotionType`** (Categorizing the mood)

The schema is intentionally kept simple to strip away the noise. By breaking down messy text into these basic nodes and relationships, it creates a structured knowledge graph that an LLM can easily pull from and utilize as precise context for RAG.

Hopefully, this actually makes your AI memory system a lot smarter and prevents the LLM from hallucinating nonsense.

## Requirements

*   Docker / Docker Compose
*   An API key (OpenRouter or Gemini)

## Setup

### Environment Variables

Copy the template:

```bash
cp .env.example .env

```

Fill the `.env` with your keys:

```ini
# Neo4j
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# OpenRouter
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet

# Google Gemini
GOOGLE_API_KEY=your_google_key
GEMINI_MODEL=gemini-1.5-pro

MODEL_TYPE=gemini_or_openrouter

```

### Run the Application

Standard compose command:

```bash
docker compose up -d

```

### Verify Installation

* **FastAPI Docs:** [http://localhost:8001/docs](http://localhost:8000/docs)
* **Neo4j Browser:** [http://localhost:7474](http://localhost:7474)

## Useful Commands

* **View Logs:** `docker compose logs -f`
* **Rebuild Containers:** `docker compose up -d --build`
* **Stop Services:** `docker compose down`