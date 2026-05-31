import os
import json
import time
import uuid
import logging
import requests
from typing import List

from dotenv import load_dotenv
from string import Template
from fastapi import HTTPException
from langchain_core.messages import HumanMessage
from datetime import datetime

from app.exceptions.llm_exceptions import LLMSchemaValidationError

from neomodel import db
from app.models.graph_models import Person, EmotionType, EmotionState, Event, Entry

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

logger = logging.getLogger(__name__)

# --- TAMBAHKAN KEMBALI FUNGSI YANG HILANG INI ---
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
        
        # --- INJEKSI UUID LARAVEL KE JSON ---
        parsed_json['_actual_entry_id'] = variables.get('entry_id')
        
        with open(os.path.join(temp_dir, "llm_processed_response.json"), "w") as f:
            json.dump(parsed_json, f, indent=4)
    except (json.JSONDecodeError, Exception):
        pass

    return parsed_json
# ------------------------------------------------

def get_embeddings(texts: List[str]) -> List[List[float]]:
    if not texts: return []
    jina_api_key = os.getenv("JINA_API_KEY")
    if not jina_api_key: raise ValueError("JINA_API_KEY not found.")
    
    url = "https://api.jina.ai/v1/embeddings"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {jina_api_key}"}
    payload = {"model": "jina-embeddings-v3", "input": texts}
    
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return [item["embedding"] for item in sorted(response.json()["data"], key=lambda x: x["index"])]

def load_prompt(filename, **kwargs) -> str:
    base_dir = os.path.dirname(__file__)
    with open(os.path.join(base_dir, "prompts", filename), "r") as f:
        return Template(f.read()).substitute(**kwargs)

def validate_schema(data):
    # (Schema Validation logic tetap sama)
    pass

def delete_entry_tree_if_exists(entry_label: str) -> bool:
    with db.transaction:
        # Kita hapus relasinya pelan-pelan supaya node utama (Person, EmotionType) gak ikut mati
        delete_query = """
            MATCH (entry:Entry {label: $label})
            
            // 1. Ambil semua event yang terkait dengan entry ini
            OPTIONAL MATCH (entry)-[:CONTAINS]->(event:Event)
            
            // 2. Putus relasi ke Person dan EmotionState
            OPTIONAL MATCH (event)-[r1:INVOLVES]->(p:Person)
            OPTIONAL MATCH (event)-[r2:TRIGGERS]->(es:EmotionState)
            DELETE r1, r2
            
            // 3. Putus relasi EmotionState ke EmotionType
            OPTIONAL MATCH (es)-[r3:INSTANCE_OF]->(et:EmotionType)
            DELETE r3
            
            // 4. Hapus node-node yang emang harus hilang (Event, EmotionState, dan Entry itu sendiri)
            DETACH DELETE event, es, entry
        """
        db.cypher_query(delete_query, {"label": entry_label})
    return True

def process_and_persist_graph(raw_content):
    created_nodes = {}

    # --- TANGKAP UUID ASLI YANG KITA INJEK TADI ---
    actual_entry_id = raw_content.get('_actual_entry_id')

    # Collect all texts that need embeddings and keep track of their targets
    embedding_tasks = [] # List of tuples: (node_data_reference_id, text_to_embed)
    
    for node_data in raw_content['nodes']:
        node_type = node_data['node_type']
        props = node_data['properties']
        ref_id = node_data['id']
        
        if node_type == 'Entry':
            embedding_tasks.append((ref_id, props.get('summary')))
        elif node_type == 'Event':
            combined_text = f"{node_data.get('label')}: {props.get('description')}"
            embedding_tasks.append((ref_id, combined_text))
        elif node_type == 'EmotionState':
            embedding_tasks.append((ref_id, props.get('description')))

    # Call the batch API once if there are texts to embed
    embedding_map = {}
    if embedding_tasks:
        texts_to_embed = [task[1] for task in embedding_tasks]
        # Call the batch function we created earlier
        vectors = get_embeddings(texts_to_embed) 
        
        # Map vectors back to their node's reference ID
        for i, task in enumerate(embedding_tasks):
            ref_id = task[0]
            embedding_map[ref_id] = vectors[i]

    # Transaction block for persisting data
    with db.transaction:
        validate_schema(raw_content)

        for node_data in raw_content['nodes']:
            node_type = node_data['node_type']
            reference_id = node_data['id']
            props = node_data['properties']

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
                    # --- PAKSA LABEL PAKAI UUID LARAVEL! ---
                    label=actual_entry_id if actual_entry_id else reference_id,
                    summary=props.get('summary'),
                    embedding=embedding_map.get(reference_id) # Use pre-calculated vector
                ).save()
            elif node_type == 'Event':
                node = Event(
                    label=node_data.get('label'),
                    description=props.get('description'),
                    confidence=props.get('confidence', 1.0),
                    embedding=embedding_map.get(reference_id) # Use pre-calculated vector
                ).save()
            elif node_type == 'EmotionState':
                node = EmotionState(
                    description=props.get('description'),
                    intensity=props.get('intensity'),
                    valence=props.get('valence'),
                    embedding=embedding_map.get(reference_id) # Use pre-calculated vector
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
        
        architect = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model=os.getenv("OPENROUTER_MODEL"),
            streaming=True
        )
        
        companion = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1
            )

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
            tools_module.get_similar_events_or_emotions,
            tools_module.get_person_dynamics,
            tools_module.get_emotional_history
        ]

        agent = create_tool_calling_agent(companion, list_tools, prompt)
        
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=list_tools,
            verbose=True,
            handle_parsing_errors=True
        )

        return architect, companion, agent_executor

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