# server.py
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types
import asyncio
import logging
from typing import List, Dict, Any
import json
import os
from datetime import datetime

from config import config
from openrouter_client import get_openrouter_client, close_openrouter_client
from neo4j_client import get_neo4j_client, close_neo4j_client
from prompt_manager import get_prompt_manager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("neo4j-memory-server")

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

class Neo4jMemoryServer:
    def __init__(self):
        self.server = Server("neo4j-memory-server")
        self.openrouter = get_openrouter_client()
        self.neo4j = get_neo4j_client()
        self.prompts = get_prompt_manager()
    
    async def initialize(self):
        """Initialize connections"""
        await self.neo4j.connect()
        await self._setup_handlers()
        logger.info("Server initialized")
    
    async def _setup_handlers(self):
        """Setup MCP handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[types.Tool]:
            return [
                types.Tool(
                    name="process_journal_entry",
                    description="Process a journal entry and update Neo4j graph",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entry_id": {"type": "string"},
                            "entry_text": {"type": "string"},
                            "user_name": {"type": "string"},
                            "user_aliases": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["entry_id", "entry_text", "user_name"]
                    }
                ),
                types.Tool(
                    name="query_graph",
                    description="Query the Neo4j graph",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query_type": {
                                "type": "string",
                                "enum": ["person_events", "timeline", "custom"]
                            },
                            "person_name": {"type": "string"},
                            "cypher_query": {"type": "string"},
                            "params": {"type": "object"},
                            "limit": {"type": "integer"}
                        },
                        "required": ["query_type"]
                    }
                ),
                types.Tool(
                    name="get_person_summary",
                    description="Get summary of a person",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "person_name": {"type": "string"},
                            "include_events": {"type": "boolean"}
                        },
                        "required": ["person_name"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            try:
                if name == "process_journal_entry":
                    result = await self._process_entry(arguments)
                elif name == "query_graph":
                    result = await self._query_graph(arguments)
                elif name == "get_person_summary":
                    result = await self._get_person_summary(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
                
                return [types.TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, default=str)
                )]
            except Exception as e:
                logger.error(f"Error in {name}: {str(e)}")
                return [types.TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)}, indent=2)
                )]
    
    async def _save_to_temp(self, entry_id: str, suffix: str, data: Any, is_json: bool = True):
        timestamp = datetime.now().strftime("%H%M%S")
        ext = "json" if is_json else "txt"
        filename = f"{entry_id}_{suffix}_{timestamp}.{ext}".replace("/", "_")
        filepath = os.path.join("temp", filename)
        
        try:
            os.makedirs("temp", exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                if is_json:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                else:
                    f.write(data)
            logger.debug(f"File saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save {filepath}: {str(e)}")
    
    async def _process_entry(self, args: Dict) -> Dict:
        """Process journal entry through both phases"""
        entry_id = args["entry_id"]
        entry_text = args["entry_text"]
        user_name = args["user_name"]
        user_aliases = args.get("user_aliases", [])
        
        logger.info(f"Processing entry {entry_id}")
        
        # Phase 1: Life Graph Detective
        phase1_prompt = self.prompts.get_phase1_prompt(
            user_name, user_aliases, entry_text
        )

        # DEBUGGING PURPOSES
        await self._save_to_temp(entry_id, "phase1_input", phase1_prompt, is_json=False)

        phase1_result = await self.openrouter.chat_completion(
            phase1_prompt, 
            temperature=0.3
        )

        # DEBUGGING PURPOSES
        await self._save_to_temp(entry_id, "phase1_output", phase1_result)
        
        # Check verdict
        if phase1_result["final_verdict"]["decision"] == "skip_to_postgresql":
            return {
                "graph_update": False,
                "reason": phase1_result["final_verdict"]["reason"],
                "phase1_analysis": phase1_result
            }
        
        # Phase 2: Life Graph Architect
        phase2_prompt = self.prompts.get_phase2_prompt(
            phase1_result, entry_id, user_name
        )

        # DEBUGGING PURPOSES
        await self._save_to_temp(entry_id, "phase2_input", phase2_prompt, is_json=False)

        phase2_result = await self.openrouter.chat_completion(
            phase2_prompt,
            temperature=0.2
        )

        # DEBUGGING PURPOSES
        await self._save_to_temp(entry_id, "phase2_output", phase2_result)
        
        if not phase2_result.get("graph_update", False):
            return phase2_result
        
        # Apply to Neo4j
        neo4j_result = await self._apply_to_neo4j(phase2_result, entry_id)
        
        return {
            "entry_id": entry_id,
            "processed": True,
            "phase1_analysis": phase1_result,
            "phase2_graph": phase2_result,
            "neo4j_update": neo4j_result
        }
    
    async def _apply_to_neo4j(self, graph_update: Dict, entry_id: str) -> Dict:
        """Apply graph updates to Neo4j"""
        try:
            # Create nodes
            id_map = await self.neo4j.create_nodes(
                graph_update.get("nodes", []),
                entry_id
            )
            
            # Create relationships
            await self.neo4j.create_relationships(
                graph_update.get("relationships", []),
                id_map,
                entry_id
            )
            
            return {
                "applied": True,
                "node_count": len(graph_update.get("nodes", [])),
                "relationship_count": len(graph_update.get("relationships", []))
            }
            
        except Exception as e:
            logger.error(f"Neo4j update failed: {str(e)}")
            return {
                "applied": False,
                "error": str(e)
            }
    
    async def _query_graph(self, args: Dict) -> Dict:
        """Query the graph"""
        query_type = args["query_type"]
        
        if query_type == "person_events":
            events = await self.neo4j.get_person_events(args["person_name"])
            return {"person": args["person_name"], "events": events}
        
        elif query_type == "timeline":
            timeline = await self.neo4j.get_timeline(args.get("limit", 50))
            return {"timeline": timeline}
        
        elif query_type == "custom":
            results = await self.neo4j.query(
                args["cypher_query"],
                args.get("params", {})
            )
            return {"results": results}
    
    async def _get_person_summary(self, args: Dict) -> Dict:
        """Get person summary"""
        person_name = args["person_name"]
        
        # Get person node
        person = await self.neo4j.find_existing_person(person_name)
        if not person:
            return {"error": f"Person {person_name} not found"}
        
        # Get events
        events = await self.neo4j.get_person_events(person_name)
        
        # Calculate stats
        stats = {
            "total_events": len(events),
            "event_types": list(set(e["event"].get("event_type") for e in events if e["event"].get("event_type"))),
            "interests": list(set(
                interest["name"] 
                for e in events 
                for interest in e["interests"] 
                if interest.get("name")
            ))
        }
        
        result = {
            "person": person,
            "stats": stats
        }
        
        if args.get("include_events", False):
            result["events"] = events[:20]  # Last 20 events
        
        return result
    
    async def run(self):
        """Run the server"""
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="neo4j-memory-server",
                    server_version="0.1.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )

async def main():
    """Main entry point"""
    server = Neo4jMemoryServer()
    
    try:
        await server.initialize()
        await server.run()
    finally:
        # Cleanup
        await close_openrouter_client()
        await close_neo4j_client()

if __name__ == "__main__":
    asyncio.run(main())