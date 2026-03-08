import os
import re
import json
import hashlib
import sqlite3
import numpy as np
from openai import OpenAI
from neo4j import GraphDatabase
from scipy.spatial.distance import cosine
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
driver = GraphDatabase.driver(
    NEO4J_URI, 
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)
CACHE_DB = "llm_cache.db"
RELATION_VOCAB = {} 

# --- CACHE ENGINE (SQLite) ---
def init_cache():
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS prompt_cache (hash TEXT PRIMARY KEY, prompt TEXT, response TEXT)")
    conn.commit()
    conn.close()

def get_cached_response(prompt):
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT response FROM prompt_cache WHERE hash = ?", (prompt_hash,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

def set_cached_response(prompt, response):
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO prompt_cache (hash, prompt, response) VALUES (?, ?, ?)", (prompt_hash, prompt, response))
    conn.commit()
    conn.close()

# --- SEMANTIC ENGINE ---
def get_embedding(text):
    response = client.embeddings.create(model="openai/text-embedding-3-small", input=[text])
    return response.data[0].embedding

def get_semantic_relation(raw_rel):
    global RELATION_VOCAB
    clean_rel = raw_rel.strip().lower()
    if not RELATION_VOCAB:
        label = clean_rel.upper().replace(" ", "_")
        RELATION_VOCAB[label] = get_embedding(clean_rel)
        return label
    
    new_vec = get_embedding(clean_rel)
    best_match = None
    min_dist = 0.15 
    for existing_label, existing_vec in RELATION_VOCAB.items():
        dist = cosine(new_vec, existing_vec)
        if dist < min_dist:
            min_dist = dist
            best_match = existing_label
    
    if best_match: return best_match
    new_label = clean_rel.upper().replace(" ", "_")
    RELATION_VOCAB[new_label] = new_vec
    return new_label

# --- N2G PIPELINE ---
def process_journal_entry(entry_text):
    date_match = re.search(r"Tanggal:\s*(\d{4}-\d{2}-\d{2})", entry_text)
    anchor_date = date_match.group(1) if date_match else "2026-03-08"

    prompt = f"""
    You are a Journaling Intelligence. Extract entities and facts from this entry.
    Reference Date: {anchor_date}
    RULES:
    1. TEMPORAL: Convert relative times (yesterday, last week) to YYYY-MM-DD based on {anchor_date}.
    2. ENTITY TYPES: Use PERSON, ORGANIZATION, LOCATION, or CONCEPT.
    3. RESOLUTION: Map nicknames or titles (e.g., 'Six Eyes User') to the actual entity name if identifiable.
    4. Output ONLY JSON: [{{"sub": "Name", "type": "TYPE", "role": "Role", "rel": "Relation", "obj": "Target", "time": "YYYY-MM-DD"}}]
    
    Entry: {entry_text}
    """

    cached_data = get_cached_response(prompt)
    if cached_data:
        facts = json.loads(cached_data)
    else:
        response = client.chat.completions.create(model="stepfun/step-3.5-flash:free", messages=[{"role": "user", "content": prompt}])
        clean_json = re.sub(r'```json|```', '', response.choices[0].message.content).strip()
        facts = json.loads(clean_json)
        set_cached_response(prompt, clean_json)

    with driver.session() as session:
        for fact in tqdm(facts, desc=f"Syncing {anchor_date}", unit="fact"):
            rel_type = get_semantic_relation(fact['rel'])
            history_entry = json.dumps({"mood_rel": fact['rel'], "time": fact['time']})
            
            # --- APOC POWERED QUERY ---
            # 1. Merge core entity
            # 2. Dynamically add labels (Person, Location, etc)
            # 3. Handle roles as a unique set
            query = f"""
            MERGE (s:Entity {{name: $sub}})
            WITH s
            CALL apoc.create.addLabels(s, [$sub_type]) YIELD node AS snode
            SET snode.roles = apoc.coll.toSet(coalesce(snode.roles, []) + $sub_role)
            
            MERGE (o:Entity {{name: $obj}})
            
            MERGE (snode)-[r:{rel_type}]->(o)
            ON CREATE SET r.history = [$history_entry]
            ON MATCH SET r.history = apoc.coll.toSet(r.history + $history_entry)
            """
            session.run(query, 
                sub=fact['sub'], sub_type=fact['type'].capitalize(), 
                sub_role=fact['role'], obj=fact['obj'], 
                history_entry=history_entry
            )

init_cache()

if __name__ == "__main__":
    print("Satoru N2G Engine (APOC Edition) Ready.")