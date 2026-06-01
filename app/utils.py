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

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate vector embeddings using Jina AI API (jina-embeddings-v3)."""
    if not texts:
        return []
        
    jina_api_key = os.getenv("JINA_API_KEY")
    if not jina_api_key:
        logger.error("Environment variable JINA_API_KEY is missing.")
        raise ValueError("JINA_API_KEY not found in environment variables.")

    item_count = len(texts)
    words_per_item = [len(text.split()) for text in texts]
    total_words = sum(words_per_item)
    
    logger.info(
        f"Requesting Jina AI embeddings | Total Items: {item_count} | "
        f"Total Words: {total_words} | Words per Item: {words_per_item}"
    )

    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {jina_api_key}"
    }
    payload = {
        "model": "jina-embeddings-v3",
        "input": texts
    }

    try:
        # Start timer right before the network request
        start_time = time.time()
        response = requests.post(url, json=payload, headers=headers)
        
        if not response.ok:
            logger.error(
                f"Jina AI API failure | Status Code: {response.status_code} | "
                f"Response Body: {response.text}"
            )
        response.raise_for_status()
        
        # Calculate elapsed time in seconds
        elapsed_time = time.time() - start_time
        logger.info(f"Jina AI API success | Time Took: {elapsed_time:.3f} seconds")
        
        response_data = response.json()
        
        # Sort by original index to maintain order consistency
        sorted_data = sorted(response_data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]

    except requests.exceptions.RequestException as e:
        logger.exception("An error occurred during Jina AI API request execution.")
        raise e

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
    
def delete_entry_tree_if_exists(entry_label: str) -> bool:
    """
    Checks if an Entry with the given label exists. If found, performs a conditional 
    cascade delete from the leaves upward (Person, EmotionState, EmotionType, Event, Entry).
    Leaves are only deleted if they are exclusively connected to this specific Entry tree.
    
    All operations are wrapped in a transaction. If any error occurs, the changes are rolled back.
    
    :param entry_label: The Postgres reference ID stored in the 'label' property.
    :return: True if the tree was found and successfully deleted, False otherwise.
    """
    logger.info(f"Initiating conditional cascade delete for Entry label: '{entry_label}'")
    
    try:
        # Step 1: Check if the target Entry exists before opening a write transaction
        check_query = "MATCH (e:Entry {label: $label}) RETURN e.uid LIMIT 1"
        results, _ = db.cypher_query(check_query, {"label": entry_label})
        
        if not results:
            logger.warning(f"Aborting delete operation: No Entry found with label '{entry_label}'")
            return False

        entry_uid = results[0][0]
        logger.debug(f"Target Entry found with internal UID: {entry_uid}. Starting database transaction.")

        # Step 2: Execute deletion inside an atomic transaction block
        with db.transaction:
            delete_query = """
            MATCH (entry:Entry {label: $label})
            
            // Collect all downstream Events related to this Entry
            OPTIONAL MATCH (entry)-[:CONTAINS]->(event:Event)
            
            // Collect all EmotionStates and People attached to those Events
            OPTIONAL MATCH (event)-[:TRIGGERS]->(emotionState:EmotionState)
            OPTIONAL MATCH (event)-[:INVOLVES]->(person:Person)
            
            // Collect EmotionTypes attached to those EmotionStates
            OPTIONAL MATCH (emotionState)-[:INSTANCE_OF]->(emotionType:EmotionType)
            
            // -------------------------------------------------------------
            // PHASE 1: Conditional Leaf Deletion (Degree-based check)
            // -------------------------------------------------------------
            
            // 1. Handle EmotionType: Delete only if its total relationship count is exactly 1
            // (meaning it's only connected to the EmotionState we are about to delete)
            FOREACH (et IN CASE WHEN emotionType IS NOT NULL AND size((et)--()) = 1 THEN [emotionType] ELSE [] END |
                DETACH DELETE et
            )
            
            // 2. Handle EmotionState: Delete only if its total relationship count is <= 2
            // (1 incoming from :Event and up to 1 outgoing to :EmotionType)
            FOREACH (es IN CASE WHEN emotionState IS NOT NULL AND size((es)--()) <= 2 THEN [emotionState] ELSE [] END |
                DETACH DELETE es
            )
            
            // 3. Handle Person: Delete only if they are not involved in any other Event/Entry outside this tree
            // (meaning they only have 1 relationship total)
            FOREACH (p IN CASE WHEN person IS NOT NULL AND size((p)--()) = 1 THEN [person] ELSE [] END |
                DETACH DELETE p
            )
            
            // -------------------------------------------------------------
            // PHASE 2: Branch & Root Deletion
            // -------------------------------------------------------------
            
            // 4. Safely detach and delete all collected Events
            FOREACH (ev IN CASE WHEN event IS NOT NULL THEN [event] ELSE [] END |
                DETACH DELETE ev
            )
            
            // 5. Finally, delete the root Entry node itself
            DETACH DELETE entry
            """
            
            logger.debug(f"Executing cascade delete Cypher query for Entry '{entry_label}'...")
            db.cypher_query(delete_query, {"label": entry_label})
            
        logger.info(f"Successfully deleted Entry tree for label '{entry_label}' and all its isolated downstream nodes.")
        return True
    
    except Exception as e:
        logger.error(
            f"Unexpected exception occurred while executing delete tree for Entry '{entry_label}': {str(e)}", 
            exc_info=True
        )
        raise e

def persist_graph_and_get_emotion_types(raw_content, entry_id):
    validate_schema(raw_content)

    created_nodes = {}

    # Collect all texts that need embeddings and keep track of their targets
    embedding_tasks = [] # List of tuples: (node_data_reference_id, text_to_embed)
    
    emotion_types_set = set()

    for node_data in raw_content['nodes']:
        node_type = node_data['node_type']
        props = node_data['properties']
        ref_id = node_data['id']
        
        if node_type == 'EmotionType':
            name_val = props.get('name', '').lower().strip()
            if name_val:
                emotion_types_set.add(name_val)

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
        delete_entry_tree_if_exists(entry_label=entry_id)
        
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
                    label=reference_id,
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

    return list(emotion_types_set)

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

def get_llm_provider(llm_instance):
    if isinstance(llm_instance, ChatGoogleGenerativeAI):
        return "gemini"
    elif isinstance(llm_instance, ChatOpenAI):
        return "openrouter"
    else:
        raise TypeError(f"Unknown provider type: {type(llm_instance).__name__}")

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