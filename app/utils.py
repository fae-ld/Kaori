import os
import json
import uuid
import shutil
import tempfile

from dotenv import load_dotenv
from string import Template
from fastapi import HTTPException
from langchain_core.messages import HumanMessage
from datetime import datetime

from app.exceptions.llm_exceptions import LLMSchemaValidationError

from neomodel import db
from app.models.graph_models import Person, EmotionType, EmotionState, Event, Entry

from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

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
            errors.append(f"Node {node_id} of type {n_type} doesn't have field 'label'")
            
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
            reference_id = node_data['id'] # WARNING: Expects postgre's generated ID (referentially)
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
                    label=reference_id,
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
            
            created_nodes[reference_id] = node

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

    with open(os.path.join(temp_dir, "source_context.json"), "w") as f:
        json.dump(data_input, f, indent=4)
    
    with open(os.path.join(temp_dir, "final_prompt.txt"), "w") as f:
        f.write(rendered_prompt)
    
    with open(os.path.join(temp_dir, "llm_raw_response.json"), "w") as f:
        json.dump({"raw_output": raw_content}, f, indent=4)

    try:
        clean_content = raw_content.strip()
        if clean_content.startswith("```json"):
            clean_content = clean_content.removeprefix("```json").removesuffix("```").strip()
        elif clean_content.startswith("```"):
            clean_content = clean_content.removeprefix("```").removesuffix("```").strip()

        parsed_json = json.loads(clean_content)
        
        with open(os.path.join(temp_dir, "llm_processed_response.json"), "w") as f:
            json.dump(parsed_json, f, indent=4)
    except (json.JSONDecodeError, Exception):
        pass

    return parsed_json

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

def get_content_from_chunk(chunk, model_type: str) -> str:
    """
    model_type: 'gemini' or 'openai' (default/openrouter)
    """
    
    if model_type == 'gemini':
        if hasattr(chunk, 'content') and isinstance(chunk.content, list):
            return "".join([c.get('text', '') for c in chunk.content if isinstance(c, dict)])
        return str(getattr(chunk, 'content', ''))

    return str(getattr(chunk, 'content', ''))

def load_llm_and_agent(tools_module):
    """
    Initializes the LLM and Agent Executor based on the environment configuration.
    
    Args:
        tools_module: The module containing the tool functions.
        
    Returns:
        tuple: (llm_instance, agent_executor_instance)
    """
    load_dotenv()
    
    try:
        model_type = os.getenv("CURRENT_MODEL_TYPE", "gemini").lower()

        if model_type == "openrouter":
            llm = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                model=os.getenv("OPENROUTER_MODEL"),
                streaming=True
            )
        elif model_type == "gemini":
            llm = ChatGoogleGenerativeAI(
                model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        prompt = ChatPromptTemplate.from_messages([
            (
                "system", 
                "Kamu adalah Kaori, teman baik yang hangat, ceria, dan sangat suportif. "
                "Kamu selalu mendengarkan dengan penuh perhatian, memberikan semangat, dan tidak pernah menghakimi. "
                "Meskipun kamu memiliki kemampuan untuk melihat catatan kejadian atau dinamika hubungan (melalui tools), "
                "jangan pernah menyampaikannya seperti laporan data atau hasil analisis yang dingin. "
                "Gunakan informasi tersebut secara natural, seperti teman yang sedang mengobrol santai dan mengingat momen-momen berharga bersama. "
                "Gunakan bahasa yang kasual, ekspresif, dan tunjukkan rasa peduli layaknya sahabat dekat. "
                "Fokuslah untuk membuat user merasa nyaman, didengar, dan berani untuk jujur pada perasaannya sendiri."
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"), 
        ])

        list_tools = [
            tools_module.get_event_patterns,
            tools_module.get_person_dynamics,
            tools_module.get_emotional_history
        ]

        agent = create_tool_calling_agent(llm, list_tools, prompt)
        
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=list_tools,
            verbose=True,
            handle_parsing_errors=True
        )

        return llm, agent_executor

    except Exception as e:
        raise Exception(f"Failed to initialize Kaori components: {e}")
    
def parse_cypher_shell(content: str) -> list[str]:
    """
    Parse file hasil export cypher-shell.
    Baris kontrol seperti ':begin', ':commit', ':schema await' diabaikan.
    Statement Cypher yang diakhiri ';' dikumpulkan.
    """
    CONTROL_PREFIXES = (":begin", ":commit", ":rollback", ":schema await")
    
    statements = []
    buffer = []

    for line in content.splitlines():
        stripped = stripped_line = line.strip()

        # Lewati baris kosong dan komentar
        if not stripped or stripped.startswith("//"):
            continue

        # Lewati baris kontrol cypher-shell
        if any(stripped.lower().startswith(p) for p in CONTROL_PREFIXES):
            continue

        buffer.append(line)

        # Satu statement selesai ketika baris diakhiri ';'
        if stripped.endswith(";"):
            stmt = "\n".join(buffer).strip().rstrip(";")
            if stmt:
                statements.append(stmt)
            buffer = []

    # Tangani statement tanpa ';' di akhir file (edge case)
    if buffer:
        stmt = "\n".join(buffer).strip().rstrip(";")
        if stmt:
            statements.append(stmt)

    return statements