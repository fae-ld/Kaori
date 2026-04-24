import os
import json
import uuid
from string import Template
from fastapi import HTTPException
from langchain_core.messages import HumanMessage
from datetime import datetime

from app.exceptions.llm_exceptions import LLMSchemaValidationError

from neomodel import db
from app.models.graph_models import Person, EmotionType, EmotionState, Event, Entry

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text: str):
    if not text:
        return None
    embedding = model.encode(text)
    return embedding.tolist()

def load_prompt(filename, **kwargs) -> str:
    """Loads a text file and substitutes placeholders using $ syntax."""
    try:
        base_dir = os.path.dirname(__file__)
        file_path = os.path.join(base_dir, "prompts", filename)
        
        with open(file_path, "r") as f:
            content = f.read()
            
        return Template(content).substitute(**kwargs)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Prompt file {filename} not found")
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing variable in prompt: {e}")
    
def validate_schema(data):
    errors = []
    
    # Schema rules (Source -> Relationship -> Target)
    VALID_TRIPLES = {
        ('Entry', 'CONTAINS', 'Event'),
        ('Event', 'INVOLVES', 'Person'),
        ('Event', 'TRIGGERS', 'EmotionState'),
        ('EmotionState', 'INSTANCE_OF', 'EmotionType'),
    }
    
    ALLOWED_NODE_TYPES = {'Entry', 'Event', 'Person', 'EmotionState', 'EmotionType'}

    # Node validation
    node_map = {}
    for node in data.get('nodes', []):
        node_id = node.get('id')
        n_type = node.get('node_type')
        
        if not n_type or n_type not in ALLOWED_NODE_TYPES:
            errors.append(f"Node {node_id} has an invalid node_type: {n_type}")
        
        if not node.get('label'):
            errors.append(f"Node {node_id} doesn't have field 'label'")
            
        node_map[node_id] = n_type

    # Edge Validation
    for edge in data.get('edges', []):
        source_id = edge.get('source')
        target_id = edge.get('target')
        rel = edge.get('relationship')

        if source_id not in node_map or target_id not in node_map:
            errors.append(f"Referential Integrity Error: ID {source_id} or {target_id} not found")
            continue

        # Check the rules (Source Type -> Relation -> Target Type)
        source_type = node_map[source_id]
        target_type = node_map[target_id]
        
        current_triple = (source_type, rel, target_type)
        
        if current_triple not in VALID_TRIPLES:
            errors.append(
                f"Illegal relationship: ({source_type}) -[{rel}]-> ({target_type}) is forbidden"
            )

    if errors:
        raise LLMSchemaValidationError("LLM Schema Violation", errors=errors)

def process_and_persist_graph(raw_content):
    created_nodes = {}

    with db.transaction:
        validate_schema(raw_content)

        for node_data in raw_content['nodes']:
            node_type = node_data['node_type']
            llm_id = node_data['id']
            props = node_data['properties']

            # Class mapping according to node_type
            if node_type == 'Person':
                node = Person.find_or_create(
                    extracted_name=props.get('name'),
                    extracted_aliases=props.get('aliases', [])
                )
    
            elif node_type == 'EmotionType':
                name_val = props.get('name').lower()
                node = EmotionType.nodes.get_or_none(name=name_val)
                if not node:
                    node = EmotionType(name=name_val).save()
            elif node_type == 'Entry':
                node = Entry(
                    summary=props.get('summary'),
                    embedding=get_embedding(props.get('summary'))
                ).save()
            elif node_type == 'Event':
                combined_text = f"{node_data.get('label')}: {props.get('description')}"
                node = Event(
                    label=node_data.get('label'), # Use label as event title
                    description=props.get('description'),
                    confidence=props.get('confidence', 1.0),
                    embedding=get_embedding(combined_text)
                ).save()
            elif node_type == 'EmotionState':
                node = EmotionState(
                    description=props.get('description'),
                    intensity=props.get('intensity'),
                    valence=props.get('valence'),
                    embedding=get_embedding(props.get('description'))
                ).save()
            
            created_nodes[llm_id] = node

        # --- Relationship Linking ---
        for edge in raw_content['edges']:
            source_obj = created_nodes.get(edge['source'])
            target_obj = created_nodes.get(edge['target'])
            rel = edge['relationship']

            if not source_obj or not target_obj:
                raise Exception(f"Failed to create relation: Node {edge['source']} or {edge['target']} not found")

            if rel == 'CONTAINS':
                source_obj.events.connect(target_obj)
            elif rel == 'INVOLVES':
                source_obj.people.connect(target_obj)
            elif rel == 'EXPRESSES':
                source_obj.emotions.connect(target_obj)
            elif rel == 'TRIGGERS':
                source_obj.emotions.connect(target_obj)
            elif rel == 'INSTANCE_OF':
                source_obj.emotion_type.connect(target_obj)

    return True

def process_and_log(llm, data_input, filename, variables):
    """Handles LLM execution and saves logs to temp folder."""
    rendered_prompt = load_prompt(filename, **variables)
    
    response = llm.invoke([HumanMessage(content=rendered_prompt)])
    raw_content = response.content

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_str = str(uuid.uuid4())[:4]
    session_id = f"{timestamp}_{random_str}"
    
    temp_dir = os.path.join("temp", session_id)
    os.makedirs(temp_dir, exist_ok=True)

    with open(os.path.join(temp_dir, "input.json"), "w") as f:
        json.dump(data_input, f, indent=4)
    
    with open(os.path.join(temp_dir, "rendered_prompt.txt"), "w") as f:
        f.write(rendered_prompt)
    
    with open(os.path.join(temp_dir, "output.json"), "w") as f:
        json.dump({"raw_output": raw_content}, f, indent=4)

    try:
        clean_content = raw_content.strip()
        if clean_content.startswith("```json"):
            clean_content = clean_content.removeprefix("```json").removesuffix("```").strip()
        elif clean_content.startswith("```"):
            clean_content = clean_content.removeprefix("```").removesuffix("```").strip()

        parsed_json = json.loads(clean_content)
        
        with open(os.path.join(temp_dir, "output_data.json"), "w") as f:
            json.dump(parsed_json, f, indent=4)
    except (json.JSONDecodeError, Exception):
        pass

    return raw_content

def parse_json_from_llm(content: str):
    """Utility to clean and parse JSON from LLM content."""
    try:
        clean = content.strip()
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0]
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0]
        return json.loads(clean.strip())
    except:
        return {"needs_tools": False, "current_vibe": "Sedang bercerita"}