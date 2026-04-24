# app/main.py
from neomodel import config

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from app.schemas import JournalRequest
from app.models.graph_models import Person, Entry
from langchain_openai import ChatOpenAI
from neomodel import db

from app.utils import process_and_log, process_and_persist_graph
from app.exceptions.llm_exceptions import LLMSchemaValidationError

load_dotenv()

app = FastAPI()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model=os.getenv("OPENROUTER_MODEL")
)

load_dotenv()

config.DATABASE_URL = f"bolt://{os.getenv('NEO4J_USER')}:{os.getenv('NEO4J_PASSWORD')}@{os.getenv('NEO4J_URI').split('//')[1]}"

app = FastAPI(title="Kaori")

@app.get("/")
async def hello_world():
    return {"message": "Kaori Service is Online"}

@app.post("/test-graph")
async def test_graph(name: str, entry_text: str):
    person = Person.get_or_create({"name": name})[0]
    entry = Entry(content=entry_text).save()
    
    entry.people.connect(person)
    
    return {"status": "success", "detail": f"Entry linked to {name}"}

@app.get("/graph/tree/{entry_uid}")
async def get_entry_graph_tree(entry_uid: str):
    query = """
    MATCH (e:Entry {uid: $uid})-[:CONTAINS]->(ev:Event)
    OPTIONAL MATCH (ev)-[:INVOLVES]->(p:Person)
    OPTIONAL MATCH (ev)-[:TRIGGERS]->(es:EmotionState)-[:INSTANCE_OF]->(et:EmotionType)
    RETURN e, ev, p, es, et
    """
    results, _ = db.cypher_query(query, {"uid": entry_uid})
    
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

@app.post('/extract')
async def extract_endpoint(data: JournalRequest):
    try:
        result = process_and_log(
            llm=llm,
            data_input=data.model_dump(),
            variables={
                "username": data.username,
                "aliases": ", ".join(data.aliases),
                "entry_text": data.entry_text
            }
        )

        # with open('temp/outputs/006.json', 'r') as response:
        #     result = json.load(response)

        process_and_persist_graph(result)
        
        return {"status": "success", "message": "Graph data persisted successfully"}

    except LLMSchemaValidationError as e:
        raise HTTPException(status_code=422, detail={"errors": e.errors})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")