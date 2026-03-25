# prompt_manager.py
import os
from typing import Dict, Any, List
import json
from string import Template

class PromptManager:
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = prompts_dir
        self._cache = {}
    
    def _load_prompt(self, filename: str) -> str:
        """Load prompt from file"""
        if filename in self._cache:
            return self._cache[filename]
        
        filepath = os.path.join(self.prompts_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            self._cache[filename] = content
            return content
    
    def get_phase1_prompt(self, user_name: str, user_aliases: list, entry_text: str) -> str:
        """Get formatted phase 1 prompt"""
        template = self._load_prompt("phase1_detective.txt")
        t = Template(template)
        
        return t.safe_substitute(
            user_name=user_name,
            user_aliases=json.dumps(user_aliases),
            entry_text=entry_text
        )
    
    def get_phase2_prompt(self, phase1_result: Dict, entry_id: str, user_name: str, graph_context: List = None) -> str:
        """Get formatted phase 2 prompt with manual graph context"""
        template = self._load_prompt("phase2_architect.txt")
        t = Template(template)
        
        graph_context_json = json.dumps(graph_context, indent=2) if graph_context else "[]"
        
        return t.safe_substitute(
            user_name=user_name,
            entry_id=entry_id,
            phase1_json=json.dumps(phase1_result, indent=2),
            graph_context=graph_context_json 
        )

# Singleton instance
_prompt_manager: PromptManager = None

def get_prompt_manager() -> PromptManager:
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager

if __name__ == "__main__":
    # For testing purposes
    with open("temp/entry.txt", "r") as f:
        entry_text = f.read()

    pm = get_prompt_manager()
    phase1 = pm.get_phase1_prompt("Alice", ["A", "Ally"], entry_text)

    print("Phase 1 Prompt:")
    print(phase1)
    
    phase2 = pm.get_phase2_prompt({"final_verdict": {"decision": "skip_to_postgresql"}}, "entry123", "Alice")
    print("\nPhase 2 Prompt:")
    print(phase2)