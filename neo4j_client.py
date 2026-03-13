# neo4j_client.py
from neo4j import AsyncGraphDatabase
from typing import Dict, List, Any, Optional
import logging
import uuid
from datetime import datetime
from config import config

logger = logging.getLogger(__name__)

class Neo4jClient:
    def __init__(self):
        self.driver = None
        self.uri = config.neo4j_uri
        self.user = config.neo4j_user
        self.password = config.neo4j_password
    
    async def connect(self):
        """Establish connection to Neo4j"""
        self.driver = AsyncGraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password)
        )
        # Test connection
        async with self.driver.session() as session:
            await session.run("RETURN 1")
        logger.info("Connected to Neo4j")
    
    async def close(self):
        """Close Neo4j connection"""
        if self.driver:
            await self.driver.close()
            logger.info("Disconnected from Neo4j")
    
    async def create_nodes(self, nodes: List[Dict], entry_id: str) -> Dict[str, str]:
        """Create multiple nodes and return ID mapping"""
        id_map = {}
        
        async with self.driver.session() as session:
            for node in nodes:
                # Generate Neo4j ID
                neo4j_id = str(uuid.uuid4())
                id_map[node["id"]] = neo4j_id
                
                # Prepare node properties
                properties = {
                    k: v for k, v in node.items() 
                    if k not in ['id', 'node_type']
                }
                
                # Add metadata
                properties['source_entry_id'] = entry_id
                properties['created_at'] = datetime.now().isoformat()
                
                # Create node with explicit ID
                await session.run(
                    f"""
                    CREATE (n:{node['node_type']} {{
                        id: $id,
                        {', '.join([f'{k}: ${k}' for k in properties.keys()])}
                    }})
                    RETURN n
                    """,
                    id=neo4j_id,
                    **properties
                )
                
                logger.debug(f"Created node: {node['node_type']} - {neo4j_id}")
        
        return id_map
    
    async def create_relationships(self, relationships: List[Dict], id_map: Dict[str, str], entry_id: str):
        """Create relationships between nodes"""
        async with self.driver.session() as session:
            for rel in relationships:
                from_id = id_map.get(rel["from"])
                to_id = id_map.get(rel["to"])
                
                if not from_id or not to_id:
                    logger.warning(f"Skipping relationship: missing node IDs for {rel}")
                    continue
                
                # Add metadata to properties
                properties = rel.get('properties', {}).copy()
                properties['source_entry_id'] = entry_id
                properties['created_at'] = datetime.now().isoformat()
                
                await session.run(
                    f"""
                    MATCH (a {{id: $from_id}})
                    MATCH (b {{id: $to_id}})
                    CREATE (a)-[r:{rel['type']} {{ 
                        {', '.join([f'{k}: ${k}' for k in properties.keys()])}
                    }}]->(b)
                    RETURN r
                    """,
                    from_id=from_id,
                    to_id=to_id,
                    **properties
                )
                
                logger.debug(f"Created relationship: {rel['type']}")
    
    async def find_existing_person(self, canonical_name: str) -> Optional[Dict]:
        """Find existing person by canonical name"""
        async with self.driver.session() as session:
            result = await session.run(
                "MATCH (p:Person {canonical_name: $name}) RETURN p",
                name=canonical_name
            )
            record = await result.single()
            return dict(record["p"]) if record else None
    
    async def query(self, cypher: str, params: Dict = None) -> List[Dict]:
        """Execute custom Cypher query"""
        async with self.driver.session() as session:
            result = await session.run(cypher, params or {})
            records = []
            async for record in result:
                records.append({k: dict(v) if hasattr(v, 'keys') else v for k, v in record.items()})
            return records
    
    async def get_person_events(self, person_name: str) -> List[Dict]:
        """Get all events involving a person"""
        cypher = """
        MATCH (p:Person {canonical_name: $name})<-[:INVOLVES]-(e:Event)
        OPTIONAL MATCH (e)-[:ABOUT]->(i:Interest)
        OPTIONAL MATCH (e)-[:LOCATED_AT]->(pl:Place)
        RETURN e, collect(DISTINCT i) as interests, collect(DISTINCT pl) as places
        ORDER BY e.created_at DESC
        """
        
        results = await self.query(cypher, {"name": person_name})
        return [
            {
                "event": r["e"],
                "interests": r["interests"],
                "places": r["places"]
            }
            for r in results
        ]
    
    async def get_timeline(self, limit: int = 50) -> List[Dict]:
        """Get recent events timeline"""
        cypher = """
        MATCH (e:Event)
        OPTIONAL MATCH (e)-[:INVOLVES]->(p:Person)
        OPTIONAL MATCH (e)-[:ABOUT]->(i:Interest)
        RETURN e, collect(DISTINCT p) as people, collect(DISTINCT i) as interests
        ORDER BY e.created_at DESC
        LIMIT $limit
        """
        
        results = await self.query(cypher, {"limit": limit})
        return [
            {
                "event": r["e"],
                "people": r["people"],
                "interests": r["interests"]
            }
            for r in results
        ]

# Singleton instance
_neo4j_client: Optional[Neo4jClient] = None

def get_neo4j_client() -> Neo4jClient:
    global _neo4j_client
    if _neo4j_client is None:
        _neo4j_client = Neo4jClient()
    return _neo4j_client

async def close_neo4j_client():
    global _neo4j_client
    if _neo4j_client:
        await _neo4j_client.close()
        _neo4j_client = None