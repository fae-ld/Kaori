from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase
from typing import Dict, List, Optional
import uuid
from datetime import datetime
import logging
import os

from utils.nlp_utils import get_embedding, calculate_similarity

logger = logging.getLogger(__name__)

_neo4j_client_instance = None

class Neo4jClient:
    def __init__(self, uri, user, password):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def close(self):
        await self.driver.close()

    async def sync_graph_data(self, clean_json: Dict, entry_id: str) -> Dict[str, str]:
        id_map = await self._upsert_nodes(clean_json.get("nodes", []))
        await self._upsert_edges(clean_json.get("edges", []), id_map, entry_id)
        return id_map

    async def _upsert_nodes(self, nodes: List[Dict]) -> Dict[str, str]:
        id_map = {}
        async with self.driver.session() as session:
            for node in nodes:
                local_id = node["id"]
                label = node["label"]
                props = node["properties"]
                
                # Metadata for the node itself (Global, not entry-specific)
                props["last_seen_at"] = datetime.now().isoformat()

                query = f"""
                MERGE (n:{label} {{ name: $name }})
                ON CREATE SET n = $props, n.id = $new_id, n.created_at = $now
                ON MATCH SET n += $props
                RETURN n.id as final_id
                """
                
                result = await session.run(
                    query, 
                    name=props["name"], 
                    props=props, 
                    new_id=str(uuid.uuid4()),
                    now=datetime.now().isoformat()
                )
                record = await result.single()
                id_map[local_id] = record["final_id"]
        return id_map

    async def _upsert_edges(self, edges: List[Dict], id_map: Dict[str, str], entry_id: str):
        async with self.driver.session() as session:
            for edge in edges:
                source_uuid = id_map.get(edge["source"])
                target_uuid = id_map.get(edge["target"])
                rel_type = edge["relationship"]
                props = edge.get("properties", {})
                
                if not source_uuid or not target_uuid:
                    continue

                # Generate embedding
                desc = props.get("context", "")
                edge_embedding = get_embedding(desc) if desc else None

                # Pastikan nama parameter ($embedding atau $new_emb) KONSISTEN
                query = f"""
                MATCH (a {{id: $source_uuid}})
                MATCH (b {{id: $target_uuid}})
                MERGE (a)-[r:{rel_type}]->(b)
                ON CREATE SET 
                    r = $props, 
                    r.embedding = $embedding,  // Sesuaikan di sini
                    r.source_entry_ids = [$entry_id], 
                    r.created_at = $now
                ON MATCH SET 
                    r += $props,
                    r.embedding = $embedding,  // Update embedding juga jika ada perubahan
                    r.source_entry_ids = CASE 
                        WHEN NOT $entry_id IN r.source_entry_ids 
                        THEN r.source_entry_ids + $entry_id 
                        ELSE r.source_entry_ids 
                    END,
                    r.updated_at = $now
                """
                
                await session.run(
                    query, 
                    source_uuid=source_uuid, 
                    target_uuid=target_uuid, 
                    props=props,
                    embedding=edge_embedding,  # Nama parameter di sini harus sama dengan di query ($embedding)
                    entry_id=entry_id,
                    now=datetime.now().isoformat()
                )

    async def find_node_by_name_or_alias(self, label: str, name: str) -> Optional[Dict]:
        """Generic search for nodes by name or alias."""
        query = f"""
        MATCH (n:{label})
        WHERE n.name = $name OR (n.aliases IS NOT NULL AND $name IN n.aliases)
        RETURN n.id as id, n.name as name, labels(n) as labels
        LIMIT 1
        """
        async with self.driver.session() as session:
            result = await session.run(query, name=name)
            record = await result.single()
            return dict(record) if record else None
        
    async def _upsert_edges(self, edges: List[Dict], id_map: Dict[str, str], entry_id: str):
        async with self.driver.session() as session:
            for edge in edges:
                source_uuid = id_map.get(edge["source"])
                target_uuid = id_map.get(edge["target"])
                rel_type = edge["relationship"]
                props = edge.get("properties", {})
                
                if not source_uuid or not target_uuid:
                    continue

                # 1. Generate embedding untuk konteks baru
                desc = props.get("context", "")
                new_emb = get_embedding(desc) if desc else None

                # 2. Cari semua relasi dengan tipe yang sama antara dua node tersebut
                existing_rels_query = f"""
                MATCH (a {{id: $source_uuid}})-[r:{rel_type}]->(b {{id: $target_uuid}})
                RETURN r.id as rel_id, r.embedding as emb, id(r) as native_id
                """
                result = await session.run(existing_rels_query, source_uuid=source_uuid, target_uuid=target_uuid)
                records = await result.data()

                best_match_id = None
                if new_emb:
                    highest_sim = 0
                    for rec in records:
                        if rec["emb"]:
                            # Hitung similarity di sisi Python (lebih fleksibel tanpa GDS)
                            sim = calculate_similarity(new_emb, rec["emb"])
                            if sim > 0.85 and sim > highest_sim:
                                highest_sim = sim
                                best_match_id = rec["native_id"] # Gunakan internal ID Neo4j untuk presisi

                # 3. Eksekusi: Update jika ada match, Create jika tidak ada
                if best_match_id is not None:
                    # LOGIKA MERGE (Update existing)
                    sync_query = f"""
                    MATCH ()-[r]->() WHERE id(r) = $rel_id
                    SET r += $props,
                        r.source_entry_ids = CASE 
                            WHEN NOT $entry_id IN r.source_entry_ids 
                            THEN r.source_entry_ids + $entry_id 
                            ELSE r.source_entry_ids 
                        END,
                        r.updated_at = $now
                    """
                    await session.run(sync_query, rel_id=best_match_id, props=props, entry_id=entry_id, now=datetime.now().isoformat())
                    print(f"✅ SEMANTIC MERGE: Match found (>0.85), updated edge {rel_type}")
                else:
                    # LOGIKA CREATE (Buat baru)
                    create_query = f"""
                    MATCH (a {{id: $source_uuid}})
                    MATCH (b {{id: $target_uuid}})
                    CREATE (a)-[r:{rel_type}]->(b)
                    SET r = $props,
                        r.id = $new_rel_id,
                        r.embedding = $embedding,
                        r.source_entry_ids = [$entry_id],
                        r.created_at = $now
                    """
                    await session.run(
                        create_query, 
                        source_uuid=source_uuid, 
                        target_uuid=target_uuid, 
                        props=props, 
                        embedding=new_emb,
                        new_rel_id=str(uuid.uuid4()),
                        entry_id=entry_id, 
                        now=datetime.now().isoformat()
                    )
                    print(f"🆕 CREATE NEW EDGE: No semantic match, created new {rel_type}")

    async def connect(self):
        """Verify connectivity for server initialization."""
        await self.driver.verify_connectivity()
        print("✅ Neo4j connection verified.")

    async def get_person_events(self, person_name: str) -> List[Dict]:
        """Query events related to a person for the MCP tool."""
        query = """
        MATCH (p:Person {name: $name})-[r]->(event)
        RETURN properties(p) as person, type(r) as relationship, properties(event) as event, r.source_entry_ids as sources
        """
        async with self.driver.session() as session:
            result = await session.run(query, name=person_name)
            return [dict(record) for record in await result.data()]

    async def find_existing_person(self, name: str) -> Optional[Dict]:
        """Find a person node by name."""
        async with self.driver.session() as session:
            result = await session.run("MATCH (p:Person {name: $name}) RETURN properties(p) as props", name=name)
            record = await result.single()
            return record["props"] if record else None

    async def get_timeline(self, limit: int = 50) -> List[Dict]:
        """Get recent graph updates."""
        query = """
        MATCH (a)-[r]->(b)
        RETURN a.name as source, type(r) as relationship, b.name as target, r.updated_at as time
        ORDER BY r.updated_at DESC LIMIT $limit
        """
        async with self.driver.session() as session:
            result = await session.run(query, limit=limit)
            return [dict(record) for record in await result.data()]

def get_neo4j_client():
    """
    Returns a singleton instance of Neo4jClient.
    Initializes it if it doesn't exist yet.
    """
    global _neo4j_client_instance
    
    if _neo4j_client_instance is None:
        load_dotenv()
        
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
    
        _neo4j_client_instance = Neo4jClient(uri, user, password)
        print("⚡ Neo4jClient Initialized (Singleton)")
        
    return _neo4j_client_instance

async def close_neo4j_client():
    """
    Closes the global Neo4jClient singleton instance.
    Call this when your application is shutting down.
    """
    global _neo4j_client_instance
    
    if _neo4j_client_instance is not None:
        try:
            await _neo4j_client_instance.close()
            print("💤 Neo4jClient Connection Closed.")
        except Exception as e:
            print(f"❌ Error while closing Neo4jClient: {e}")
        finally:
            _neo4j_client_instance = None
    else:
        print("ℹ️ Neo4jClient was not initialized, nothing to close.")