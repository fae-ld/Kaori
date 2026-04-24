# app/main.py
from neomodel import config

import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.schemas import JournalRequest
from app.models.graph_models import Person, Entry
from app.utils import process_and_log, process_and_persist_graph, parse_json_from_llm, load_prompt
from app.tools import *
from app.exceptions.llm_exceptions import LLMSchemaValidationError

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from neomodel import db


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
        # result = process_and_log(
        #     llm=llm,
        #     data_input=data.model_dump(),
        #     filename='extraction.txt',
        #     variables={
        #         "username": data.username,
        #         "aliases": ", ".join(data.aliases),
        #         "entry_text": data.entry_text
        #     }
        # )

        for i in range(2, 7):
            print(f'start {i}')
            with open(f'temp/outputs/00{i}.json', 'r') as response:
                result = json.load(response)

            process_and_persist_graph(result)
            print(f'done {i}')
        
        return {"status": "success", "message": "Graph data persisted successfully"}

    except LLMSchemaValidationError as e:
        raise HTTPException(status_code=422, detail={"errors": e.errors})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")
    
@app.post("/api/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    user_input = data.get("message")
    username = data.get("username")

    # Step 1: LLM planning and tool decision
    plan_variables = {"username": username, "user_input": user_input}
    raw_plan = process_and_log(llm, data, 'retrieval_1.txt', plan_variables)
    plan_json = parse_json_from_llm(raw_plan)
    
    # Step 2: Knowledge Graph retrieval
    graph_context = ""
    if plan_json.get("needs_tools"):
        context_parts = []
        for req in plan_json.get("requested_tools", []):
            tool = req.get('tool')
            query = req.get('search_query')
            
            if tool == 'get_event_patterns':
                res = get_event_patterns(query)
                context_parts.append(f"Events: {res}")
            elif tool == 'get_person_dynamics':
                res = get_person_dynamics(query)
                context_parts.append(f"People: {res}")
            elif tool == 'get_emotional_history':
                res = get_emotional_history(query)
                context_parts.append(f"Emotions: {res}")
            elif tool == 'check_sensitivity':
                res = check_sensitivity(query)
                context_parts.append(f"Sensitivity: {res}")
        
        graph_context = "\n".join(context_parts)

    # Step 3: Final streaming response
    final_variables = {
        "username": username,
        "user_input": user_input,
        "current_vibe": plan_json.get("current_vibe", "Emotional"),
        "graph_context": graph_context or "No memories.",
        "recent_history": ""
    }

    prompt = load_prompt(filename='retrieval_2.txt', **final_variables)

    async def generate_response():
        async for chunk in llm.astream([HumanMessage(content=prompt)]):
            yield chunk.content

    return StreamingResponse(generate_response(), media_type="text/plain")