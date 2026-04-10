import os
from neo4j_client import Neo4jClient 

class KaoriChatSim:
    def __init__(self):
        self.client = Neo4jClient()

    def get_narrative_context(self, person_a, person_b):
        """
        Mengambil jalur cerita terpendek antara dua orang untuk 
        mendapatkan rentetan kejadian (context) di antara mereka.
        """
        query = """
        MATCH path = shortestPath(
            (p1:Person {name: $name_a})-[*..3]-(p2:Person {name: $name_b})
        )
        RETURN [rel in relationships(path) | rel.context] AS narrative_chunks,
               [node in nodes(path) | node.sentiment] AS emotional_flow
        """
        params = {"name_a": person_a, "name_b": person_b}
        result = self.client.execute_query(query, params)
        
        if not result:
            return "Tidak ada konteks memori ditemukan."
        
        
        context_list = result[0]['narrative_chunks']
        return " ".join([c for c in context_list if c is not None])

    def chat_response(self, user_name, target_name, message):
        
        memory = self.get_narrative_context(user_name, target_name)
        
        
        
        prompt = f"""
        Kamu adalah {user_name}. Kamu sedang mengobrol dengan {target_name}.
        
        INGATAN KAMU (DARI GRAPH):
        {memory}
        
        PESAN DARI {target_name}: "{message}"
        
        Respons sebagai {user_name} dengan gaya bahasa jurnal kamu (kasual, jujur, agak overthinking):
        """
        
        
        
        print(f"--- DEBUG CONTEXT ---\n{memory}\n--------------------")
        print(f"\n[PROMPT KE LLM]:\n{prompt}")


if __name__ == "__main__":
    chat = KaoriChatSim()
    
    
    chat.chat_response(
        user_name="Karen Miura", 
        target_name="Kei", 
        message="Kenapa kemarin kamu diem banget pas aku ke rumah?"
    )