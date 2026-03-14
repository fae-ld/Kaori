import torch
import logging
from sentence_transformers import SentenceTransformer
from typing import List

logger = logging.getLogger(__name__)

class Vectorizer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        if torch.cuda.is_available():
            self.device = "cuda"
            device_name = torch.cuda.get_device_name(0)
        elif torch.backends.mps.is_available():
            self.device = "mps"
            device_name = "Apple Metal Performance Shaders (MPS)"
        else:
            self.device = "cpu"
            device_name = "System CPU"
        
        logger.info(f"Initializing Vectorizer with model: {model_name}")
        logger.info(f"Using device: [{self.device.upper()}] - {device_name}")
        
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            logger.info("SentenceTransformer model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def encode(self, text: str) -> List[float]:
        if not text:
            return []
        return self.model.encode(text).tolist()