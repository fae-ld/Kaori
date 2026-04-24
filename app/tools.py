from neomodel import db
from app.utils import get_embedding

def get_event_patterns(query_text: str, limit: int = 2):
    """Find 1-2 most similar events: what happened + why they are similar."""
    vector = get_embedding(query_text)
    cypher = """
    CALL db.index.vector.queryNodes('entity_description_index', $limit, $vector)
    YIELD node, score
    WHERE score > 0.7
    MATCH (en:Entry)-[:CONTAINS]->(node)
    RETURN node.description as detail, en.summary as context, score
    """
    results, _ = db.cypher_query(cypher, {"vector": vector, "limit": limit})
    return [{"summary": r[0], "reason": r[1]} for r in results]

def get_person_dynamics(name: str):
    """Who is this person and what were the significant previous dynamics."""
    cypher = """
    MATCH (p:Person) WHERE p.name CONTAINS $name
    MATCH (p)<-[:INVOLVES]-(ev:Event)<-[:CONTAINS]-(en:Entry)
    RETURN p.name as name, ev.description as last_interaction, en.created_at as date
    ORDER BY en.created_at DESC LIMIT 2
    """
    results, _ = db.cypher_query(cypher, {"name": name})
    return results

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

def check_sensitivity(topic_keywords: str):
    """Check if this is a heavy or recurring topic (trauma/conflict)."""
    # Simple logic: If there are many negative emotions associated with similar topics
    vector = get_embedding(topic_keywords)
    cypher = """
    CALL db.index.vector.queryNodes('entity_description_index', 5, $vector)
    YIELD node, score
    MATCH (node)<-[:TRIGGERS]-(es:EmotionState)
    WHERE es.valence < 0.4
    RETURN count(es) as negative_hits
    """
    results, _ = db.cypher_query(cypher, {"vector": vector})
    return {"is_sensitive": results[0][0] > 3}