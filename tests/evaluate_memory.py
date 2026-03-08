from neo4j import GraphDatabase
import os
from difflib import SequenceMatcher
from dotenv import load_dotenv

load_dotenv()

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "admin12345"))

def similarity(a, b):
    # Standard Python fuzzy matching
    return SequenceMatcher(None, a, b).ratio()

def check_integrity():
    with driver.session() as session:
        print("=== INTEGRITY CHECK REPORT ===\n")

        # 1. Fetch all entity names to check duplicates in Python
        nodes = session.run("MATCH (e:Entity) RETURN e.name as name")
        names = [record["name"] for record in nodes]
        
        print("[1] Potential Duplicate Nodes:")
        found_dup = False
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                score = similarity(names[i], names[j])
                if score > 0.7: # Threshold 0.7 for fuzzy match
                    print(f"⚠️ Possible duplicate: '{names[i]}' and '{names[j]}' (Score: {score:.2f})")
                    found_dup = True
        if not found_dup: print("✅ No obvious duplicate nodes found.")

        # 2. Check Relationship Type Distribution
        rel_counts = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) as type, count(r) as count
        """)
        
        print("\n[2] Relationship Type Distribution:")
        for record in rel_counts:
            print(f"- {record['type']}: {record['count']} instances")

        # 3. Check Multi-Temporal Relations
        temporal = session.run("""
            MATCH ()-[r]->()
            WHERE r.time_contexts IS NOT NULL AND size(r.time_contexts) > 1
            RETURN type(r) as type, r.time_contexts as contexts
        """)
        
        print("\n[3] Multi-Temporal Relations (Successful Merges):")
        found_merge = False
        for record in temporal:
            print(f"✅ Relation '{record['type']}' has {len(record['contexts'])} time points.")
            found_merge = True
        if not found_merge: print("ℹ️ No multi-temporal relations found yet.")

if __name__ == "__main__":
    check_integrity()
    driver.close()