# app/models/graph_models.py
import uuid
from neomodel import (
    StructuredNode,
    StringProperty,
    DateTimeProperty,
    FloatProperty,
    JSONProperty,
    RelationshipTo
)

class EmotionType(StructuredNode):
    """Categorical emotion (e.g., 'fear', 'joy')"""
    uid = StringProperty(unique_index=True, required=True)
    name = StringProperty(required=True)

class EmotionState(StructuredNode):
    """Instance of an emotion felt at a specific time"""
    uid = StringProperty(unique_index=True, default=lambda: uuid.uuid4().hex)
    description = StringProperty()
    intensity = FloatProperty()
    valence = FloatProperty()
    timestamp = DateTimeProperty()
    confidence = FloatProperty()
    
    # (:EmotionState)-[:INSTANCE_OF]->(:EmotionType)
    emotion_type = RelationshipTo('EmotionType', 'INSTANCE_OF')

class Person(StructuredNode):
    """Individual entity mentioned in entries"""
    uid = StringProperty(unique_index=True, default=lambda: uuid.uuid4().hex)
    name = StringProperty(required=True)
    aliases = JSONProperty() # Array of strings

class Event(StructuredNode):
    """Specific occurrence or activity"""
    uid = StringProperty(unique_index=True, default=lambda: uuid.uuid4().hex)
    label = StringProperty(required=True)
    description = StringProperty()
    timestamp = DateTimeProperty()
    confidence = FloatProperty()
    
    # (:Event)-[:INVOLVES]->(:Person)
    people = RelationshipTo('Person', 'INVOLVES')
    # (:Event)-[:TRIGGERS]->(:EmotionState)
    emotions = RelationshipTo('EmotionState', 'TRIGGERS')
    # (:Event)-[:LEADS_TO]->(:Event)
    next_events = RelationshipTo('Event', 'LEADS_TO')

class Entry(StructuredNode):
    """Daily journal record; UID synced from PostgreSQL"""
    uid = StringProperty(unique_index=True, required=True)
    raw_text = StringProperty(required=True)
    summary = StringProperty()
    created_at = DateTimeProperty()
    
    # (:Entry)-[:CONTAINS]->(:Event)
    events = RelationshipTo('Event', 'CONTAINS')
    # (:Entry)-[:EXPRESSES]->(:EmotionState)
    emotions = RelationshipTo('EmotionState', 'EXPRESSES')
    # (:Entry)-[:INVOLVES]->(:Person)
    people = RelationshipTo('Person', 'INVOLVES')