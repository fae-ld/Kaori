from neomodel import db
from app.utils import get_embeddings
from langchain_core.tools import tool

@tool
def get_similar_events_or_emotions(query_text: str, limit: int = 2):
    """
    Find 1-2 entities (either Events OR Emotion States) that are semantically similar to the query.
    It traverses the strict graph schema: [Entry]-(CONTAINS)->[Event]-(TRIGGERS)->[EmotionState] 
    and returns the connected structure.
    
    CRITICAL WARNING: This function triggers a 'get_embedding' calculation before querying, 
    introducing a severe response latency (worst case up to 2 minutes). 
    Use ONLY when semantic search for events OR emotions is strictly necessary.
    """
    vector = get_embeddings([query_text])[0]
    
    cypher = """
    MATCH (node)
    SEARCH node IN (VECTOR INDEX entity_description_index FOR $vector LIMIT $limit)
    SCORE AS score
    WHERE score > 0.7
    
    CALL (node) {
        // Kasus 1: Jika yang cocok adalah node Event
        WITH node WHERE node:Event
        MATCH (en:Entry)-[:CONTAINS]->(node)-[:TRIGGERS]->(emo:EmotionState)
        RETURN en, node as ev, emo
        
        UNION
        
        // Kasus 2: Jika yang cocok adalah node EmotionState
        WITH node WHERE node:EmotionState
        MATCH (en:Entry)-[:CONTAINS]->(ev:Event)-[:TRIGGERS]->(node)
        RETURN en, ev, node as emo
    }
    
    RETURN DISTINCT
        en.summary as entry_summary,
        ev.description as event_detail,
        emo.type as emotion_type,
        emo.intensity as emotion_intensity,
        score
    ORDER BY score DESC
    """
    
    results, _ = db.cypher_query(cypher, {"vector": vector, "limit": limit})
    
    formatted_results = []
    for r in results:
        formatted_results.append({
            "entry_context": r[0],
            "graph_structure": {
                "event": {
                    "detail": r[1]
                },
                "relationship": "TRIGGERS",
                "emotion_state": {
                    "type": r[2],
                    "intensity": r[3]
                }
            },
            "similarity_score": round(r[4], 2)
        })
        
    return formatted_results

@tool
def get_person_dynamics(name: str):
    """Who is this person and what were the significant previous dynamics."""
    cypher = """
    MATCH (p:Person)
    WHERE p.name CONTAINS $name
    
    MATCH (e:Entry)-[:HAS_EVENT]->(ev:Event)-[inv:INVOLVES]->(p)
    
    OPTIONAL MATCH (ev)-[tr:TRIGGERS]->(es:EmotionState)-[inst:INSTANCE_OF]->(et:EmotionType)

    RETURN 
        {name: p.name, aliases: p.aliases} AS person, 
        inv, 
        {description: ev.description} AS event, 
        tr, 
        {description: es.description} AS emotionState, 
        inst, 
        {name: et.name} AS emotionType
    ORDER BY e.created_at DESC
    LIMIT 5
    """
    results, _ = db.cypher_query(cypher, {"name": name})
    return results

@tool
def get_emotional_history(emotion_name: str):
    """Similar emotions in the past and whether there are changes in patterns."""
    cypher = """
    MATCH (et:EmotionType {name: $emotion_name})<-[:INSTANCE_OF]-(es:EmotionState)
    MATCH (en:Entry)-[:EXPRESSES]->(es)
    RETURN es.description as detail, en.created_at as date
    ORDER BY en.created_at DESC LIMIT 2
    """
    results, _ = db.cypher_query(cypher, {"emotion_name": emotion_name.lower()})
    return results