"""
Build a graph of related concepts from a string using Wikidata.
"""
from typing import List, Optional, Dict, Tuple
from sparql_util import search_entity_by_label, get_entity_info, get_neighbor_triples, get_property_label
from extract_subgraph import get_id_from_uri, init_triples_dict, get_entity_triplets
import json


def get_entity_neighbors_with_ids(entity_id: str, entity_label: str, max_neighbors: int = 50) -> List[Dict]:
    """
    Get neighbors of an entity with both IDs and labels.
    
    Args:
        entity_id: Wikidata entity ID
        entity_label: Entity label
        max_neighbors: Maximum number of neighbors to return
    
    Returns:
        List of dictionaries with 'source_id', 'source_label', 'relation', 'target_id', 'target_label'
    """
    results = get_neighbor_triples(entity_id)
    neighbors = []
    
    if "results" not in results or "bindings" not in results["results"]:
        return neighbors
    
    for result in results["results"]["bindings"]:
        obj_type = result.get('object', {}).get('type', '')
        if obj_type != "uri":
            continue
        
        # Get IDs
        subject_id = get_id_from_uri(result['subject']['value'])
        predicate_id = get_id_from_uri(result['predicate']['value'])
        obj_id = get_id_from_uri(result['object']['value'])
        
        # Get labels
        subject_label = result.get('subjectLabel', {}).get('value', subject_id)
        predicate_label = result.get('predicateLabel', {}).get('value', predicate_id)
        obj_label = result.get('objectLabel', {}).get('value', obj_id)
        
        # Use provided label for subject if it matches
        if subject_id == entity_id:
            subject_label = entity_label
        
        # Get property label if missing
        if predicate_label == predicate_id:
            predicate_label = get_property_label(predicate_id)
        
        # Get object label if missing (it's an ID)
        if obj_label == obj_id:
            obj_info = get_entity_info(obj_id)
            if obj_info:
                obj_label = obj_info['label']
        
        neighbors.append({
            "source_id": subject_id,
            "source_label": subject_label,
            "relation": predicate_label,
            "target_id": obj_id,
            "target_label": obj_label
        })
        
        if len(neighbors) >= max_neighbors:
            break
    
    return neighbors


def build_graph_from_string(input_string: str, max_depth: int = 5, max_neighbors: int = 50) -> dict:
    """
    Build a graph of concepts related to the input string through Wikidata.
    Recursively expands the graph up to max_depth levels, with max_neighbors per node.
    
    Args:
        input_string: String to search for in Wikidata
        max_depth: Maximum depth of the graph (how many hops from the starting entity)
        max_neighbors: Maximum number of neighbor entities to include per node
    
    Returns:
        Dictionary containing the graph structure with nodes and edges
    """
    # Step 1: Search for the entity in Wikidata
    print(f"Searching for '{input_string}' in Wikidata...")
    entities = search_entity_by_label(input_string, limit=5)
    
    if not entities:
        print(f"No entities found for '{input_string}'")
        return {"nodes": [], "edges": [], "root_entity": None}
    
    # Use the first result (most relevant)
    root_entity = entities[0]
    print(f"Found entity: {root_entity['label']} ({root_entity['id']})")
    
    if len(entities) > 1:
        print(f"Other candidates:")
        for e in entities[1:]:
            print(f"  - {e['label']} ({e['id']})")
    
    # Step 2: Build graph recursively using BFS
    print(f"\nBuilding graph for {root_entity['label']} (max_depth={max_depth}, max_neighbors={max_neighbors})...")
    
    # Data structures for graph building
    nodes_dict = {}  # entity_id -> {id, label}
    edges = []  # List of {source, relation, target, source_id, target_id}
    visited = set()  # Track visited entity IDs to avoid cycles
    queue = []  # BFS queue: [(entity_id, entity_label, depth), ...]
    
    # Initialize with root entity
    root_id = root_entity['id']
    root_label = root_entity['label']
    nodes_dict[root_id] = {
        "id": root_id,
        "label": root_label
    }
    queue.append((root_id, root_label, 0))
    visited.add(root_id)
    
    # BFS traversal
    level_counts = {}  # Track nodes per level for progress reporting
    
    while queue:
        current_id, current_label, current_depth = queue.pop(0)
        
        # Track level counts
        if current_depth not in level_counts:
            level_counts[current_depth] = 0
        level_counts[current_depth] += 1
        
        # Print progress
        if current_depth == 0:
            print(f"  Level {current_depth}: {current_label} ({current_id})")
        elif level_counts[current_depth] <= 5 or level_counts[current_depth] % 10 == 0:
            print(f"  Level {current_depth}: Processing {current_label} ({current_id}) [node {level_counts[current_depth]}]")
        
        # Skip if we've reached max depth (don't process neighbors)
        if current_depth >= max_depth:
            continue
        
        # Get neighbors for current entity
        try:
            neighbors = get_entity_neighbors_with_ids(current_id, current_label, max_neighbors)
            
            neighbor_count = 0
            for neighbor in neighbors:
                target_id = neighbor['target_id']
                target_label = neighbor['target_label']
                relation = neighbor['relation']
                
                # Add target node if not already present
                if target_id not in nodes_dict:
                    nodes_dict[target_id] = {
                        "id": target_id,
                        "label": target_label
                    }
                
                # Add edge
                edges.append({
                    "source": current_label,
                    "source_id": current_id,
                    "relation": relation,
                    "target": target_label,
                    "target_id": target_id
                })
                
                # Add to queue if not visited and depth allows
                # Process up to max_depth levels (0 to max_depth-1)
                if target_id not in visited and current_depth + 1 < max_depth:
                    queue.append((target_id, target_label, current_depth + 1))
                    visited.add(target_id)
                    neighbor_count += 1
            
            if neighbor_count > 0 and (current_depth == 0 or level_counts[current_depth] <= 3):
                print(f"    Added {neighbor_count} neighbors at level {current_depth + 1}")
        
        except Exception as e:
            print(f"    Error processing {current_label}: {e}")
            continue
    
    # Print summary by level
    print(f"\n  Summary by level:")
    for level in sorted(level_counts.keys()):
        print(f"    Level {level}: {level_counts[level]} nodes")
    
    # Convert nodes dict to list
    nodes_list = list(nodes_dict.values())
    
    result = {
        "root_entity": {
            "id": root_entity['id'],
            "label": root_entity['label']
        },
        "nodes": nodes_list,
        "edges": edges,
        "total_nodes": len(nodes_list),
        "total_edges": len(edges),
        "max_depth": max_depth,
        "max_neighbors": max_neighbors,
        "actual_depth": max_depth  # Could calculate actual depth if needed
    }
    
    print(f"\nGraph built: {len(nodes_list)} nodes, {len(edges)} edges")
    
    return result


def save_graph_to_json(graph: dict, output_path: str):
    """Save the graph to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    print(f"Graph saved to {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python build_graph_from_string.py <input_string> [output_file.json]")
        sys.exit(1)
    
    input_string = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else f"graph_{input_string.replace(' ', '_')}.json"
    
    graph = build_graph_from_string(input_string)
    save_graph_to_json(graph, output_file)










