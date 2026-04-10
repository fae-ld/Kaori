from pprintpp import pprint

def validate_and_sanitize(llm_output: dict):
    nodes = llm_output.get("nodes", [])
    edges = llm_output.get("edges", [])

    referenced_ids = set()

    for e in edges:
        if "source" in e:
            referenced_ids.add(e["source"])
        
        if "target" in e:
            referenced_ids.add(e["target"])

    valid_nodes = [n for n in nodes if n["id"] in referenced_ids]

    for node in valid_nodes:
        raw_name = node["properties"].get("name", "")
        node["properties"]["name"] = raw_name.strip().title()

    return {
        "nodes": valid_nodes,
        "edges": edges
    }