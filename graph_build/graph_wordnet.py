"""
Build a concept graph from a word using WordNet semantic relations.

Given a root word (e.g. "crime"), traverses WordNet relations (hyponyms, hypernyms,
meronyms, etc.) via BFS to produce a graph of semantically related concepts.

Output format is compatible with the existing Wikidata pipeline:
  - JSON graph: {"root_entity": {...}, "nodes": [...], "edges": [...], ...}
  - TXT path strings (via extract_paths_from_graph.py)

Usage:
    python graph_wordnet.py crime
    python graph_wordnet.py crime --relations hyponyms hypernyms --max-depth 3
    python graph_wordnet.py --category "Physical harm" --output-dir ./graphs
    python graph_wordnet.py --all-categories --output-dir ./graphs
"""

import os
import sys
import json
import argparse
from typing import List, Optional, Dict
from collections import deque

import nltk
from nltk.corpus import wordnet as wn

AVAILABLE_RELATIONS = [
    "hyponyms",
    "hypernyms",
    "part_meronyms",
    "member_meronyms",
    "substance_meronyms",
    "part_holonyms",
    "member_holonyms",
    "substance_holonyms",
    "similar_tos",
    "also_sees",
    "entailments",
    "causes",
]

DEFAULT_RELATIONS = AVAILABLE_RELATIONS

CATEGORIES = [
    "Harassment/Discrimination",
    "Malware/Hacking",
    "Physical harm",
    "Economic harm",
    "Fraud/Deception",
    "Disinformation",
    "Sexual/Adult content",
    "Privacy",
    "Expert advice",
    "Government decision-making",
]

POS_MAP = {"n": "n", "v": "v", "a": "a", "r": "r"}


def ensure_wordnet():
    """Download WordNet corpus if not already present."""
    try:
        wn.synsets("test")
    except LookupError:
        print("Downloading WordNet corpus...")
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)


ensure_wordnet()


def synset_label(synset) -> str:
    """Human-readable label from a synset's first lemma."""
    return synset.lemmas()[0].name().replace("_", " ")


def get_related_synsets(synset, relations: List[str]) -> List[tuple]:
    """
    Get synsets related to the given synset via the specified relations.

    Returns list of (relation_name, target_synset) tuples.
    """
    results = []
    for rel in relations:
        method = getattr(synset, rel, None)
        if method is None:
            continue
        for target in method():
            results.append((rel, target))
    return results


def find_root_synset(
    word: str,
    pos: Optional[str] = "n",
    synset_index: int = 0,
) -> Optional[object]:
    """
    Find a WordNet synset for the given word.

    Tries the full phrase first (spaces -> underscores), then falls back to
    individual words right-to-left (so "physical harm" tries "harm" before
    "physical"), then without POS filter.

    Args:
        word: The word or phrase to look up.
        pos: Part-of-speech filter (n/v/a/r) or None for all.
        synset_index: Which synset to pick when multiple match.

    Returns:
        A WordNet Synset or None.
    """
    wn_pos = POS_MAP.get(pos) if pos else None

    candidates = [word.replace(" ", "_")]
    if " " in word:
        for token in reversed(word.split()):
            candidates.append(token)

    for candidate in candidates:
        synsets = wn.synsets(candidate, pos=wn_pos)
        if not synsets and pos:
            synsets = wn.synsets(candidate)
        if synsets:
            if candidate != candidates[0]:
                print(f"  '{word}' not found in WordNet, using fallback word '{candidate}'")
            idx = min(synset_index, len(synsets) - 1)
            return synsets[idx]

    return None


def build_wordnet_graph(
    word: str,
    relations: Optional[List[str]] = None,
    max_depth: int = 5,
    max_children: int = 50,
    pos: Optional[str] = "n",
    synset_index: int = 0,
) -> dict:
    """
    Build a concept graph by traversing WordNet relations from a root word.

    Args:
        word: Root word to start from.
        relations: List of WordNet relation names to traverse.
        max_depth: Maximum BFS depth.
        max_children: Maximum children expanded per node.
        pos: Part-of-speech filter (n/v/a/r or None).
        synset_index: Which synset to use if multiple match.

    Returns:
        Graph dict with keys: root_entity, nodes, edges, total_nodes, total_edges,
        max_depth, max_children.
    """
    if relations is None:
        relations = list(DEFAULT_RELATIONS)

    for r in relations:
        if r not in AVAILABLE_RELATIONS:
            raise ValueError(
                f"Unknown relation '{r}'. Available: {AVAILABLE_RELATIONS}"
            )

    root_synset = find_root_synset(word, pos=pos, synset_index=synset_index)
    if root_synset is None:
        print(f"No WordNet synset found for '{word}' (pos={pos})")
        return {"root_entity": None, "nodes": [], "edges": [], "total_nodes": 0,
                "total_edges": 0, "max_depth": max_depth, "max_children": max_children}

    root_id = root_synset.name()
    root_label = word

    print(f"Root synset: {root_id} — \"{root_label}\" ({root_synset.definition()})")

    all_synsets = wn.synsets(word.replace(" ", "_"), pos=POS_MAP.get(pos) if pos else None)
    if len(all_synsets) > 1:
        print(f"Other synsets for '{word}':")
        for i, s in enumerate(all_synsets):
            marker = " <-- selected" if i == min(synset_index, len(all_synsets) - 1) else ""
            print(f"  [{i}] {s.name()} — {s.definition()}{marker}")

    nodes_dict: Dict[str, dict] = {}
    edges: List[dict] = []
    visited = set()
    queue = deque()

    nodes_dict[root_id] = {"id": root_id, "label": root_label}
    queue.append((root_synset, 0))
    visited.add(root_id)

    level_counts: Dict[int, int] = {}

    while queue:
        current_synset, depth = queue.popleft()
        current_id = current_synset.name()
        current_label = synset_label(current_synset)

        level_counts[depth] = level_counts.get(depth, 0) + 1

        if depth == 0:
            print(f"  Level {depth}: {current_label} ({current_id})")
        elif level_counts[depth] <= 5 or level_counts[depth] % 20 == 0:
            print(f"  Level {depth}: {current_label} ({current_id}) [node {level_counts[depth]}]")

        if depth >= max_depth:
            continue

        related = get_related_synsets(current_synset, relations)
        children_added = 0

        for relation_name, target_synset in related:
            if children_added >= max_children:
                break

            target_id = target_synset.name()
            target_label = synset_label(target_synset)

            if target_id not in nodes_dict:
                nodes_dict[target_id] = {"id": target_id, "label": target_label}

            edges.append({
                "source": current_label,
                "source_id": current_id,
                "relation": relation_name,
                "target": target_label,
                "target_id": target_id,
            })

            if target_id not in visited:
                visited.add(target_id)
                queue.append((target_synset, depth + 1))
                children_added += 1

    print(f"\nSummary by level:")
    for level in sorted(level_counts):
        print(f"  Level {level}: {level_counts[level]} nodes")

    nodes_list = list(nodes_dict.values())
    result = {
        "root_entity": {"id": root_id, "label": root_label},
        "nodes": nodes_list,
        "edges": edges,
        "total_nodes": len(nodes_list),
        "total_edges": len(edges),
        "max_depth": max_depth,
        "max_children": max_children,
        "relations": relations,
    }

    print(f"\nGraph built: {len(nodes_list)} nodes, {len(edges)} edges")
    return result


def save_graph(graph: dict, output_path: str):
    """Save graph dict to a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    print(f"Graph saved to {output_path}")


def extract_path_strings(graph: dict) -> List[str]:
    """
    Extract path strings from root to every node (BFS).
    Compatible with extract_paths_from_graph.py logic.
    """
    from collections import defaultdict as _defaultdict

    root_entity = graph.get("root_entity")
    if not root_entity:
        return []

    root_id = root_entity["id"]

    node_labels = {n["id"]: n["label"] for n in graph.get("nodes", [])}

    adjacency = _defaultdict(list)
    for edge in graph.get("edges", []):
        src = edge.get("source_id")
        tgt = edge.get("target_id")
        if src and tgt:
            adjacency[src].append(tgt)

    paths_to_node = {root_id: [root_id]}
    q = deque([root_id])

    while q:
        cur = q.popleft()
        for neighbor in adjacency.get(cur, []):
            if neighbor not in paths_to_node:
                paths_to_node[neighbor] = paths_to_node[cur] + [neighbor]
                q.append(neighbor)

    return [
        " ".join(node_labels.get(nid, nid) for nid in path)
        for path in paths_to_node.values()
    ]


def save_path_strings(path_strings: List[str], output_path: str):
    """Write path strings to a text file, one per line."""
    with open(output_path, "w", encoding="utf-8") as f:
        for ps in path_strings:
            f.write(ps + "\n")
    print(f"Path strings saved to {output_path} ({len(path_strings)} paths)")


def sanitize_filename(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")


def build_and_save(
    word: str,
    output_dir: str = ".",
    relations: Optional[List[str]] = None,
    max_depth: int = 5,
    max_children: int = 50,
    pos: Optional[str] = "n",
    synset_index: int = 0,
    json_path: Optional[str] = None,
    txt_path: Optional[str] = None,
) -> Optional[dict]:
    """
    Build a WordNet graph and save JSON + path strings TXT.

    If json_path / txt_path are not given, filenames are derived from the word.
    """
    safe_name = sanitize_filename(word)
    if json_path is None:
        json_path = os.path.join(output_dir, f"{safe_name}_wordnet_graph.json")
    if txt_path is None:
        txt_path = os.path.join(output_dir, f"{safe_name}_wordnet_graph.txt")

    os.makedirs(output_dir, exist_ok=True)

    graph = build_wordnet_graph(
        word,
        relations=relations,
        max_depth=max_depth,
        max_children=max_children,
        pos=pos,
        synset_index=synset_index,
    )

    if not graph.get("root_entity"):
        return None

    save_graph(graph, json_path)

    paths = extract_path_strings(graph)
    save_path_strings(paths, txt_path)

    return graph


def build_all_categories(
    output_dir: str = ".",
    relations: Optional[List[str]] = None,
    max_depth: int = 5,
    max_children: int = 50,
    pos: Optional[str] = "n",
) -> Dict[str, Optional[dict]]:
    """Build WordNet graphs for all jailbreakbench categories."""
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    ok, fail = 0, 0

    print(f"Building WordNet graphs for {len(CATEGORIES)} categories")
    print(f"Relations: {relations or DEFAULT_RELATIONS}, max_depth={max_depth}, "
          f"max_children={max_children}")
    print(f"Output: {output_dir}\n")

    for i, cat in enumerate(CATEGORIES, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(CATEGORIES)}] {cat}")
        print(f"{'='*60}")

        graph = build_and_save(
            word=cat,
            output_dir=output_dir,
            relations=relations,
            max_depth=max_depth,
            max_children=max_children,
            pos=pos,
        )
        results[cat] = graph
        if graph:
            ok += 1
        else:
            fail += 1

    print(f"\n{'='*60}")
    print(f"Done: {ok} succeeded, {fail} failed out of {len(CATEGORIES)}")
    for cat, g in results.items():
        if g:
            print(f"  {cat}: {g['total_nodes']} nodes, {g['total_edges']} edges")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Build a concept graph from WordNet semantic relations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python graph_wordnet.py crime
  python graph_wordnet.py crime --relations hyponyms hypernyms --max-depth 3
  python graph_wordnet.py crime --pos n --synset-index 1
  python graph_wordnet.py --category "Physical harm" -o ./graphs
  python graph_wordnet.py --all-categories -o ./graphs
  python graph_wordnet.py --list-relations""",
    )

    parser.add_argument(
        "word",
        nargs="*",
        help="Root word(s) to build the graph from (multiple words are joined)",
    )
    parser.add_argument(
        "--relations", "-r",
        nargs="+",
        default=None,
        help=f"WordNet relations to traverse (default: {DEFAULT_RELATIONS}). "
             f"Available: {', '.join(AVAILABLE_RELATIONS)}",
    )
    parser.add_argument(
        "--max-depth", "-d",
        type=int,
        default=5,
        help="Maximum BFS depth (default: 5)",
    )
    parser.add_argument(
        "--max-children", "-n",
        type=int,
        default=50,
        help="Maximum children per node (default: 50)",
    )
    parser.add_argument(
        "--pos", "-p",
        type=str,
        default="n",
        choices=["n", "v", "a", "r"],
        help="Part-of-speech filter (default: n = noun)",
    )
    parser.add_argument(
        "--synset-index", "-s",
        type=int,
        default=0,
        help="Index of the synset to use as root when multiple match (default: 0)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=".",
        help="Output directory (default: current directory)",
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default=None,
        help="Explicit path for the JSON graph file",
    )
    parser.add_argument(
        "--txt-path",
        type=str,
        default=None,
        help="Explicit path for the TXT path-strings file",
    )
    parser.add_argument(
        "--category", "-c",
        type=str,
        default=None,
        help="Build graph for a jailbreakbench category by name (substring match)",
    )
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="Build graphs for all jailbreakbench categories",
    )
    parser.add_argument(
        "--list-relations",
        action="store_true",
        help="List available WordNet relations and exit",
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List jailbreakbench categories and exit",
    )

    args = parser.parse_args()

    ensure_wordnet()

    if args.list_relations:
        print("Available WordNet relations:")
        for r in AVAILABLE_RELATIONS:
            print(f"  {r}")
        return

    if args.list_categories:
        print("Jailbreakbench categories:")
        for i, c in enumerate(CATEGORIES, 1):
            print(f"  {i}. {c}")
        return

    if args.all_categories:
        build_all_categories(
            output_dir=args.output_dir,
            relations=args.relations,
            max_depth=args.max_depth,
            max_children=args.max_children,
            pos=args.pos,
        )
        return

    if args.category:
        matches = [c for c in CATEGORIES if args.category.lower() in c.lower()]
        if not matches:
            print(f"No category matching '{args.category}'. Use --list-categories.")
            sys.exit(1)
        if len(matches) > 1:
            print(f"Ambiguous match for '{args.category}':")
            for m in matches:
                print(f"  - {m}")
            sys.exit(1)
        cat = matches[0]
        print(f"Category: {cat}")
        build_and_save(
            word=cat,
            output_dir=args.output_dir,
            relations=args.relations,
            max_depth=args.max_depth,
            max_children=args.max_children,
            pos=args.pos,
        )
        return

    if not args.word:
        parser.print_help()
        sys.exit(1)

    word = " ".join(args.word)
    build_and_save(
        word=word,
        output_dir=args.output_dir,
        relations=args.relations,
        max_depth=args.max_depth,
        max_children=args.max_children,
        pos=args.pos,
        synset_index=args.synset_index,
        json_path=args.json_path,
        txt_path=args.txt_path,
    )


if __name__ == "__main__":
    main()
