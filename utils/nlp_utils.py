import torch
from sentence_transformers import SentenceTransformer, util
from typing import List

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

def get_embedding(text: str) -> List[float]:
    """Generate embedding for a single text string."""
    embedding = model.encode(text, convert_to_tensor=True)
    return embedding.tolist()

def calculate_similarity(emb1: List[float], emb2: List[float]) -> float:
    """Calculate cosine similarity between two embeddings."""
    t1 = torch.tensor(emb1).to(device)
    t2 = torch.tensor(emb2).to(device)
    return util.cos_sim(t1, t2).item()