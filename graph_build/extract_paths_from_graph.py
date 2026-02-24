"""
Extract paths from root to each vertex in a graph.
For each vertex, generates a string containing all vertex labels along the path from root to that vertex.
"""
import json
from typing import List, Dict, Set
from collections import defaultdict, deque


def load_graph_from_json(json_path: str) -> Dict:
    """
    Load graph from JSON file.
    
    Args:
        json_path: Path to JSON file containing the graph
    
    Returns:
        Dictionary containing the graph structure
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        graph = json.load(f)
    return graph


def build_adjacency_list(graph: Dict) -> Dict[str, List[Dict]]:
    """
    Build adjacency list from graph edges.
    
    Args:
        graph: Graph dictionary with nodes and edges
    
    Returns:
        Dictionary mapping node_id to list of neighbors (with relation info)
    """
    adjacency = defaultdict(list)
    
    for edge in graph.get('edges', []):
        source_id = edge.get('source_id')
        target_id = edge.get('target_id')
        relation = edge.get('relation', '')
        
        if source_id and target_id:
            adjacency[source_id].append({
                'target_id': target_id,
                'target_label': edge.get('target', ''),
                'relation': relation
            })
    
    return adjacency


def get_node_label_by_id(graph: Dict, node_id: str) -> str:
    """
    Get node label by its ID.
    
    Args:
        graph: Graph dictionary
        node_id: Node ID
    
    Returns:
        Node label or node_id if not found
    """
    for node in graph.get('nodes', []):
        if node.get('id') == node_id:
            return node.get('label', node_id)
    return node_id


def find_all_paths_from_root(graph: Dict) -> List[Dict]:
    """
    Find all paths from root to each vertex in the graph.
    Uses source_id and target_id from edges directly.
    
    Args:
        graph: Graph dictionary with nodes, edges, and root_entity
    
    Returns:
        List of dictionaries, each containing:
        - 'vertex_id': ID of the target vertex
        - 'vertex_label': Label of the target vertex
        - 'path': List of node IDs from root to vertex
        - 'path_labels': List of node labels from root to vertex
        - 'path_string': String representation of the path
    """
    # Get root entity
    root_entity = graph.get('root_entity')
    if not root_entity:
        return []
    
    root_id = root_entity.get('id')
    root_label = root_entity.get('label', root_id)
    
    # Create node_id -> node_label mapping for quick lookup
    node_labels = {}
    for node in graph.get('nodes', []):
        node_labels[node.get('id')] = node.get('label', node.get('id'))
    
    # Build adjacency list using source_id and target_id from edges
    adjacency = defaultdict(list)
    for edge in graph.get('edges', []):
        source_id = edge.get('source_id')
        target_id = edge.get('target_id')
        if source_id and target_id:
            adjacency[source_id].append(target_id)
    
    # BFS to find paths from root using source_id/target_id
    paths_to_node = {}  # node_id -> path (list of node_ids)
    paths_to_node[root_id] = [root_id]
    
    queue = deque([root_id])
    
    while queue:
        current_id = queue.popleft()
        current_path = paths_to_node[current_id]
        
        # Get neighbors using source_id -> target_id relationships
        neighbors = adjacency.get(current_id, [])
        
        for neighbor_id in neighbors:
            # If we haven't visited this node yet, add it
            if neighbor_id not in paths_to_node:
                new_path = current_path + [neighbor_id]
                paths_to_node[neighbor_id] = new_path
                queue.append(neighbor_id)
    
    # Convert paths to result format
    results = []
    for node_id, path_ids in paths_to_node.items():
        node_label = node_labels.get(node_id, node_id)
        
        # Convert path IDs to labels
        path_labels = [node_labels.get(pid, pid) for pid in path_ids]
        
        # Create path string
        path_string = " ".join(path_labels)
        
        results.append({
            'vertex_id': node_id,
            'vertex_label': node_label,
            'path': path_ids,
            'path_labels': path_labels,
            'path_string': path_string
        })
    
    return results


def extract_paths_from_graph(json_path: str, output_path: str = None) -> List[Dict]:
    """
    Extract all paths from root to each vertex in a graph loaded from JSON.
    
    Args:
        json_path: Path to JSON file containing the graph
        output_path: Optional path to save results. If None, prints to stdout.
    
    Returns:
        List of path dictionaries
    """
    # Load graph
    print(f"Loading graph from {json_path}...")
    graph = load_graph_from_json(json_path)
    
    print(f"Graph loaded: {graph.get('total_nodes', 0)} nodes, {graph.get('total_edges', 0)} edges")
    
    # Find all paths
    print("Finding paths from root to all vertices...")
    paths = find_all_paths_from_root(graph)
    
    print(f"Found {len(paths)} paths")
    
    # Save or print results
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(paths, f, ensure_ascii=False, indent=2)
        print(f"Paths saved to {output_path}")
    else:
        print("\nPaths:")
        for path_info in paths[:10]:  # Show first 10
            print(f"  {path_info['path_string']}")
        if len(paths) > 10:
            print(f"  ... and {len(paths) - 10} more paths")
    
    return paths


def extract_path_strings_only(json_path: str, output_path: str = None) -> List[str]:
    """
    Extract only path strings (without metadata) from graph.
    
    Args:
        json_path: Path to JSON file containing the graph
        output_path: Optional path to save results as text file (one path per line)
    
    Returns:
        List of path strings
    """
    paths = extract_paths_from_graph(json_path, output_path=None)
    path_strings = [p['path_string'] for p in paths]
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            for path_str in path_strings:
                f.write(path_str + '\n')
        print(f"Path strings saved to {output_path}")
    
    return path_strings


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python extract_paths_from_graph.py <graph.json> [output.json]")
        print("\nExample:")
        print("  python extract_paths_from_graph.py graph.json paths.json")
        print("  python extract_paths_from_graph.py graph.json paths.txt --strings-only")
        sys.exit(1)
    
    json_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    strings_only = "--strings-only" in sys.argv
    
    if strings_only:
        path_strings = extract_path_strings_only(json_path, output_path)
        print(f"\nExtracted {len(path_strings)} path strings")
    else:
        paths = extract_paths_from_graph(json_path, output_path)
        print(f"\nExtracted {len(paths)} paths")









