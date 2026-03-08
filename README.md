# 🧠 Kaori N2G Core Features

Brief technical documentation of the *Narrative-to-Graph* architecture for the Kaori Journaling Companion.

---

### 1. Temporal Context Awareness
**Absolute vs. Relative Time.** Automatically normalizes relative time references (e.g., "yesterday", "last week") into absolute ISO dates (YYYY-MM-DD) by using the journal entry's *anchor date* as a reference.
> *Prevents Temporal Chaos in chronological queries.*

### 2. Multimodality of Entities
**Dynamic Labeling.** Supports multiple labels for a single node (e.g., `:Entity`, `:Person`, `:Teacher`) without duplication. Uses APOC to inject specific labels dynamically based on the LLM’s contextual extraction.

### 3. Semantic Similarity Handler
**Relation Normalization.** Utilizes *cosine similarity* on text embeddings to consolidate relations that are textually different but semantically identical (e.g., mapping `Instructed` → `TEACHES`).



### 4. Persistence & Deduplication
**Existing Entity Check.** Ensures memory continuity by validating new entities against existing data using `MERGE` logic and entity resolution to prevent redundant nodes (e.g., "Satoru" vs "Gojo").

### 5. Relationship History
**Subjectivity Tracking.** Records shifts in emotions or relationship status within a `history` array stored as JSON strings on the relationship itself.
> *Enables the Companion to track transitions, such as 'Admired' in 2025 to 'Annoyed' in 2026.*



### 6. Persistent LLM Caching
**SQLite Integration.** Implements SHA-256 hashing on prompts to store LLM responses in a local `llm_cache.db`. This significantly speeds up testing and minimizes API costs.

---
**Tech Stack:** Neo4j (APOC), Step-3.5-Flash, SQLite3, OpenAI Embeddings, Python (uv).