# app/schemas.py
from pydantic import BaseModel
from typing import List, Any, Dict

class JournalRequest(BaseModel):
    username: str
    aliases: List[str]
    entry_id: str
    entry_text: str

class ExtractionResponse(BaseModel):
    status: str
    data: Dict[str, Any]

class ChatRequest(BaseModel):
    user_name: str
    message: str
    history: List[Dict[str, str]]

class MessageItem(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    user_name: str
    history: List[MessageItem]
    # TODO: Optional field session_id or etc