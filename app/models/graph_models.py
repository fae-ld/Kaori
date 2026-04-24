# app/models/graph_models.py
import uuid
from neomodel import (
    StructuredNode,
    StringProperty,
    DateTimeProperty,
    FloatProperty,
    ArrayProperty,
    RelationshipTo,
    db
)

generate_uuid = lambda: uuid.uuid4().hex

class EmotionType(StructuredNode):
    """Categorical emotion (e.g., 'fear', 'joy')"""
    uid = StringProperty(unique_index=True, default=generate_uuid)
    name = StringProperty(required=True)

class EmotionState(StructuredNode):
    """Instance of an emotion felt at a specific time"""
    uid = StringProperty(unique_index=True, default=generate_uuid)
    description = StringProperty()
    intensity = FloatProperty()
    valence = FloatProperty()
    timestamp = DateTimeProperty()
    confidence = FloatProperty()
    embedding = ArrayProperty(FloatProperty())
    
    # (:EmotionState)-[:INSTANCE_OF]->(:EmotionType)
    emotion_type = RelationshipTo('EmotionType', 'INSTANCE_OF')

class Person(StructuredNode):
    """Individual entity mentioned in entries"""
    uid = StringProperty(unique_index=True, default=generate_uuid)
    name = StringProperty(required=True)
    aliases = ArrayProperty(StringProperty(), default=[])

    @classmethod
    def find_or_create(cls, extracted_name, extracted_aliases):
        all_potential_names = list(set([extracted_name] + extracted_aliases))
        
        query = """
        MATCH (p:Person)
        WHERE p.name IN $names OR any(alias IN p.aliases WHERE alias IN $names)
        RETURN p
        LIMIT 1
        """
        results, _ = db.cypher_query(query, {'names': all_potential_names})

        if results:
            existing_node = Person.inflate(results[0][0])
            updated_aliases = list(set(existing_node.aliases + all_potential_names))
            existing_node.aliases = updated_aliases
            existing_node.save()
            return existing_node
        else:
            new_node = cls(name=extracted_name, aliases=all_potential_names).save()
            return new_node

class Event(StructuredNode):
    """Specific occurrence or activity"""
    uid = StringProperty(unique_index=True, default=generate_uuid)
    label = StringProperty(required=True)
    description = StringProperty()
    timestamp = DateTimeProperty()
    confidence = FloatProperty()
    embedding = ArrayProperty(FloatProperty())
    
    # (:Event)-[:INVOLVES]->(:Person)
    people = RelationshipTo('Person', 'INVOLVES')
    # (:Event)-[:TRIGGERS]->(:EmotionState)
    emotions = RelationshipTo('EmotionState', 'TRIGGERS')
    # TODO: (:Event)-[:LEADS_TO]->(:Event)
    next_events = RelationshipTo('Event', 'LEADS_TO')

class Entry(StructuredNode):
    """Daily journal record; UID synced from PostgreSQL"""
    uid = StringProperty(unique_index=True, default=generate_uuid)
    summary = StringProperty()
    created_at = DateTimeProperty()
    embedding = ArrayProperty(FloatProperty())
    
    # (:Entry)-[:CONTAINS]->(:Event)
    events = RelationshipTo('Event', 'CONTAINS')
    # (:Entry)-[:EXPRESSES]->(:EmotionState)
    emotions = RelationshipTo('EmotionState', 'EXPRESSES')
    # (:Entry)-[:INVOLVES]->(:Person)
    people = RelationshipTo('Person', 'INVOLVES')