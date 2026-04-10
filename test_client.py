# test_client.py
import asyncio
import json
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_server():
    """Test the MCP server"""
    
    # Setup server parameters
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["server.py"]
    )
    
    # Connect ke server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            
            # Initialize
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            print("📋 Available tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")
            
            # Test process journal entry
            print("\n📝 Testing process_journal_entry...")
            
            file_name = "003_entry_karen.txt"
            with open(f"temp/entries/{file_name}", "r", encoding="utf-8") as f:
                entry_text = f.read()
            
            result = await session.call_tool(
                "process_journal_entry",
                arguments={
                    "entry_id": file_name.split(".")[0],
                    "entry_text": entry_text,
                    "user_name": "Karen Miura",
                    "user_aliases": ["Karen", "Ren"]
                }
            )

            print(entry_text)
            
            # # Parse result
            # if result.content and result.content[0].text:
            #     data = json.loads(result.content[0].text)
            #     print(f"\n✅ Result:")
            #     print(f"Graph Update: {data.get('graph_update', False)}")
                
            #     if data.get('graph_update'):
            #         neo4j_update = data.get('neo4j_update', {})
            #         print(f"Nodes created: {neo4j_update.get('node_count', 0)}")
            #         print(f"Relationships created: {neo4j_update.get('relationship_count', 0)}")
                
            #     # Show phase 1 analysis summary
            #     phase1 = data.get('phase1_analysis', {})
            #     verdict = phase1.get('final_verdict', {})
            #     print(f"\n📊 Phase 1 Verdict: {verdict.get('decision')}")
            #     print(f"Confidence: {verdict.get('confidence')}")
            #     print(f"Reason: {verdict.get('reason')}")
            
            # # Test query graph
            # print("\n🔍 Testing query_graph (timeline)...")
            # timeline = await session.call_tool(
            #     "query_graph",
            #     arguments={
            #         "query_type": "timeline",
            #         "limit": 5
            #     }
            # )
            
            # if timeline.content and timeline.content[0].text:
            #     timeline_data = json.loads(timeline.content[0].text)
            #     events = timeline_data.get('timeline', [])
            #     print(f"Found {len(events)} events")
                
            #     for i, event in enumerate(events[:3]):  # Show first 3
            #         print(f"  {i+1}. {event.get('event', {}).get('name', 'Unknown')}")

if __name__ == "__main__":
    asyncio.run(test_server())