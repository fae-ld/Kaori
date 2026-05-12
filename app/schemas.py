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