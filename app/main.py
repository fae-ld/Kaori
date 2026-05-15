# app/main.py
from neomodel import config

import os
import json
import logging
import shutil
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse

from app.schemas import JournalRequest, ChatRequest
from app.models.graph_models import Person, Entry
from app.utils import process_and_log, process_and_persist_graph, get_content_from_chunk, load_llm_and_agent, parse_cypher_shell
from app.exceptions.llm_exceptions import LLMSchemaValidationError
import app.tools as tools

from langchain_core.messages import HumanMessage, AIMessage

from neomodel import db

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

BACKUP_DIR = os.getenv("NEO4J_BACKUP_DIR")
MODEL_TYPE = os.getenv("MODEL_TYPE")
MODEL_NAME = os.getenv("OPENROUTER_MODEL") if MODEL_TYPE == "openrouter" else os.getenv("GEMINI_MODEL")

llm, agent_executor = load_llm_and_agent(tools)

config.DATABASE_URL = f"bolt://{os.getenv('NEO4J_USER')}:{os.getenv('NEO4J_PASSWORD')}@{os.getenv('NEO4J_URI').split('//')[1]}"

@app.get("/")
async def hello_world():
    return {"message": "Kaori Service is Online"}

@app.get("/hello")
async def say_hello():
    try:
        prompt = "Katakan 'Kaori is ready!' dengan sangat singkat."
        response = llm.invoke(prompt)
        
        return {
            "status": "online",
            "provider": MODEL_TYPE,
            "model": MODEL_NAME,
            "message": response.content.strip()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini Error: {str(e)}")

@app.post("/test-graph")
async def test_graph(name: str, entry_text: str):
    person = Person.get_or_create({"name": name})[0]
    entry = Entry(content=entry_text).save()
    
    entry.people.connect(person)
    
    return {"status": "success", "detail": f"Entry linked to {name}"}

@app.get("/api/graph/tree/{input_id}")
async def get_entry_graph_tree(input_id: str):
    query = """
    MATCH (e:Entry)
    WHERE e.uid = $identifier OR e.label = $identifier
    MATCH (e)-[:CONTAINS]->(ev:Event)
    OPTIONAL MATCH (ev)-[:INVOLVES]->(p:Person)
    OPTIONAL MATCH (ev)-[:TRIGGERS]->(es:EmotionState)-[:INSTANCE_OF]->(et:EmotionType)
    RETURN e, ev, p, es, et
    """
    results, _ = db.cypher_query(query, {"identifier": input_id})
    
    if not results:
        raise HTTPException(status_code=404, detail="Entry not found or has no events")

    def format_node(node):
        if not node: return None
        
        # Determine primary label and fallback display name
        node_type = list(node.labels)[0] if node.labels else "Unknown"
        node_label = node.get('label') or node.get('name') or f"{node_type}_{node.get('uid')[:4]}"
        
        return {"type": node_type, "label": node_label}

    # Initialize tree with the root entry
    entry_node = results[0][0]
    tree = {**format_node(entry_node), "events": []}
    
    event_map = {}

    for row in results:
        _, ev, p, es, et = row
        ev_uid = ev.get('uid')

        # Initialize event entry if not already processed
        if ev_uid not in event_map:
            event_data = {**format_node(ev), "people": [], "emotions": []}
            event_map[ev_uid] = event_data
            tree["events"].append(event_data)
        
        current_event = event_map[ev_uid]

        # Attach involved people
        if p:
            p_formatted = format_node(p)
            if p_formatted not in current_event["people"]:
                current_event["people"].append(p_formatted)
        
        # Attach emotion states and their types
        if es:
            emotion_state = format_node(es)
            if et:
                emotion_state["emotion_type"] = format_node(et)
            
            if emotion_state not in current_event["emotions"]:
                current_event["emotions"].append(emotion_state)

    return tree

@app.get("/api/graph/backup")
async def export_graph():
    """
    Export database ke file .cypher dan kirim sebagai download
    """
    file_name = "memora_backup.cypher"
    file_path = os.path.join(BACKUP_DIR, file_name)
    
    try:
        query = f"""    
        CALL apoc.export.cypher.all('{file_name}', {{
            format: 'cypher-shell',
            useOptimizations: {{type: 'UNWIND_BATCH', unwindBatchSize: 20}},
            ifNotExists: true
        }})
        """
        db.cypher_query(query)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=500, detail="Backup file was not created by Neo4j")
            
        return FileResponse(
            path=file_path,
            filename=file_name,
            media_type='text/plain'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.post("/api/graph/restore")
async def import_graph(file: UploadFile = File(...)):
    """
    Import/restore database dari file .cypher hasil export cypher-shell
    """
    if not file.filename.endswith(".cypher"):
        raise HTTPException(status_code=400, detail="File harus berekstensi .cypher")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".cypher") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        with open(tmp_path, "r", encoding="utf-8") as f:
            content = f.read()

        statements = parse_cypher_shell(content)

        if not statements:
            raise HTTPException(status_code=400, detail="Tidak ada statement valid yang ditemukan di file")

        # satu statement satu transaksi
        executed = 0
        errors = []
        for stmt in statements:
            try:
                db.cypher_query(stmt)
                executed += 1
            except Exception as e:
                errors.append({"statement_preview": stmt[:120], "error": str(e)})

        result = {
            "status": "success" if not errors else "partial",
            "executed": executed,
            "failed": len(errors),
        }
        if errors:
            result["errors"] = errors[:10]  # batasin biar response ga membludak

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Restore failed: {str(e)}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
    
@app.post('/api/extract')
async def extract_endpoint(data: JournalRequest):
    try:
        result = process_and_log(
            llm=llm,
            data_input=data.model_dump(),
            filename='extraction.txt',
            variables={
                "username": data.username,
                "aliases": ", ".join(data.aliases),
                "entry_id": data.entry_id,
                "entry_text": data.entry_text
            }
        )
        
        process_and_persist_graph(result)
        
        return {"status": "success", "message": "Graph data persisted successfully"}

    except LLMSchemaValidationError as e:
        raise HTTPException(status_code=422, detail={"errors": e.errors})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")

@app.post("/api/chat/stream")
async def chat_stream_endpoint(req: ChatRequest):
    print(f"\n[DEBUG] Request from: {req.user_name}")

    formatted_messages = [
        HumanMessage(content=msg.content) if msg.role == "user" else AIMessage(content=msg.content)
        for msg in req.history
    ]

    async def event_generator():
        try:
            async for event in agent_executor.astream_events({"messages": formatted_messages}, version="v2"):
                kind = event["event"]
                
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    content = get_content_from_chunk(chunk, MODEL_TYPE)
                    
                    if content:
                        yield f"data: {json.dumps({'type': 'text', 'content': content})}\n\n"

                elif kind == "on_tool_start":
                    yield f"data: {json.dumps({'type': 'tool_start', 'tool': event['name']})}\n\n"
                
                elif kind == "on_tool_end":
                    yield f"data: {json.dumps({'type': 'tool_end', 'tool': event['name']})}\n\n"
        
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
        
        finally:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )