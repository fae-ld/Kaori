# openrouter_client.py
import httpx
import json
import logging
from typing import Dict, Any, Optional
from config import config

logger = logging.getLogger(__name__)

class OpenRouterClient:
    def __init__(self):
        self.api_key = config.openrouter_key
        self.model = config.openrouter_model
        self.client = httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "Neo4j Memory MCP"
            },
            timeout=60.0
        )

        logger.info(f"OpenRouterClient initialized using model: {self.model}")
    
    async def close(self):
        await self.client.aclose()
    
    async def chat_completion(self, prompt: str, temperature: float = 0.3, max_tokens: int = 4000) -> Dict[str, Any]:
        """Send chat completion request to OpenRouter"""
        try:
            response = await self.client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )
            response.raise_for_status()
            result = response.json()

            if not result.get("choices"):
                logger.error(f"OpenRouter returned no choices: {result}")
                raise ValueError("No choices in OpenRouter response")
            
            message = result["choices"][0].get("message", {})
        
            content = message.get("content")
            
            if content is None:
                refusal = message.get("refusal")
                logger.error(f"LLM returned None content. Refusal: {refusal}")
                raise ValueError(f"LLM content is None. Refusal: {refusal}")

            logger.debug(f"Raw content received (first 100 chars): {content[:100]}...")

            cleaned_content = self._extract_json(content)
            
            return json.loads(cleaned_content)
            
        except Exception as e:
            logger.error(f"OpenRouter API error: {str(e)}")
            raise
    
    def _extract_json(self, content: str) -> str:
        """Extract JSON from markdown code blocks"""
        content = content.strip()
        
        # Remove ```json at start
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        
        # Remove ``` at end
        if content.endswith("```"):
            content = content[:-3]
        
        # Remove any leading/trailing whitespace
        content = content.strip()
        
        # If still wrapped in quotes, try to parse
        if content.startswith('"') and content.endswith('"'):
            try:
                content = json.loads(content)
            except:
                pass
        
        return content

# Singleton instance
_openrouter_client: Optional[OpenRouterClient] = None

def get_openrouter_client() -> OpenRouterClient:
    global _openrouter_client
    if _openrouter_client is None:
        _openrouter_client = OpenRouterClient()
    return _openrouter_client

async def close_openrouter_client():
    global _openrouter_client
    if _openrouter_client:
        await _openrouter_client.close()
        _openrouter_client = None