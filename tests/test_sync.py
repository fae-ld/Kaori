import asyncio
import json
import os
from dotenv import load_dotenv
from utils.graph_utils import validate_and_sanitize 
from neo4j_client import Neo4jClient
from pprintpp import pprint

async def run_test():
    load_dotenv()
    
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    client = Neo4jClient(uri, user, password)
    
    try:
        entry_1_path = "temp/logs/002_entry_karen_phase1_output_175702.json"
        entry_2_path = "temp/logs/003_entry_karen_phase1_output_103056.json" 
        entry_3_path = "temp/logs/004_entry_karen_phase1_output_173745.json"
        
        with open(entry_2_path, "r") as f:
            raw_llm_data = json.load(f)
        
        entry_id = "test-journal-001"
        print(f"--- Starting Sync for Entry: {entry_id} ---")

        # 4. Step 1: Post-processing (Clean & Validate)
        print("[1/3] Validating and sanitizing graph...")
        clean_data = validate_and_sanitize(raw_llm_data)

        # 5. Step 2: Sync to Neo4j
        print("[2/3] Syncing to Neo4j (Nodes & Edges with Embedding)...")
        id_map = await client.sync_graph_data(clean_data, entry_id)
        
        print(f"[3/3] Success! Processed {len(id_map)} nodes.")
        print("ID Mapping (Local -> Neo4j):", json.dumps(id_map, indent=2))

    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        await client.close()
        print("--- Connection Closed ---")

if __name__ == "__main__":
    asyncio.run(run_test())