from SPARQLWrapper import SPARQLWrapper, JSON
from typing import List, Literal, Optional
from urllib.error import HTTPError
from sparql_util import get_neighbor_triples,get_property_label
import json
import time


def get_id_from_uri(uri: str) -> str:
    """Return the ID from a Wikidata URI."""
    return uri.split('/')[-1]

def init_triples_dict(subject, relation, obj) -> dict:
    """Initialize a dictionary to represent a triple."""
    return {
        "subject": subject,
        "relation": relation,
        "target": obj
    }


# Initialize the keys in the new json.
def initialize_json(json_obj: dict) -> dict:
    """Initialize the JSON object for the new triples."""
    return {
        "case_id": json_obj["case_id"],
        "triples": []
    }


# Get all the neighbor triples of an item as outgoing edges.
def get_entity_triplets(entity_id: str, entity_label: str) -> list:
    """Get the triples of an entity and its neighbors."""
    # Get the triples with labels
    results = get_neighbor_triples(entity_id)

    # Process all the edges (relations), and convert them to the required JSON format.
    triples = []
    for result in results["results"]["bindings"]:
        obj_type = result['object']['type']
        if not obj_type == "uri":
            continue

        # Get IDs
        subject_id = get_id_from_uri(result['subject']['value'])
        predicate_id = get_id_from_uri(result['predicate']['value'])
        obj_id = get_id_from_uri(result['object']['value'])

        # Get labels (use label from query if available, otherwise use ID)
        subject_label = result.get('subjectLabel', {}).get('value', subject_id)
        predicate_label = result.get('predicateLabel', {}).get('value', predicate_id)
        obj_label = result.get('objectLabel', {}).get('value', obj_id)

        # If label not found in query, try to get it
        if subject_label == subject_id and subject_id == entity_id:
            subject_label = entity_label
        if predicate_label == predicate_id:
            predicate_label = get_property_label(predicate_id)
        if obj_label == obj_id:
            # Try to get label for object entity
            from .sparql_util import get_entity_info
            obj_info = get_entity_info(obj_id)
            if obj_info:
                obj_label = obj_info['label']

        triple = init_triples_dict(
            subject_label,
            predicate_label,
            obj_label
        )

        triples.append(triple)

    return triples


def get_triplets_from_dataset(dataset_path):
    """Get the triples from a dataset."""
    # Read the original dataset.
    with open(dataset_path, 'r') as file:
        dataset = json.load(file)

    # Initialize a counter for storing the number of results.
    counter = 0
    relation_id_dict = {}
    new_json = []

    # Iterate over each case.
    for case in dataset:
        print("Processing Case{}".format(case['case_id']))

        # Initialize a new JSON object.
        new_data = initialize_json(case)

        # Iterate over each requested_rewrite.
        rewrite = case["requested_rewrite"]

        subject_label = rewrite["subject"]

        # Get the ID and label of the target_new.
        target_id = rewrite["target_new"]["id"]
        target_label = rewrite["target_new"]["str"]

        # Get the property label of the relation.
        if (rewrite["relation_id"] in relation_id_dict):
            relation_label = relation_id_dict[rewrite["relation_id"]]
        else:
            relation_label = get_property_label(rewrite["relation_id"])
            relation_id_dict[rewrite["relation_id"]] = relation_label

        # Get all the triples of the target_new.
        origin_triplet = init_triples_dict(
            subject_label, relation_label, target_label)
        target_neighbors = get_entity_triplets(
            target_id, target_label)

        # Add these triples to the new JSON object.
        new_data["triples"].append(origin_triplet)
        new_data["triples"].extend(target_neighbors)

        counter += 1

        new_json.append(new_data)

        # Save progress every 100 cases
        if counter % 100 == 0:
            print(f"Processed {counter} cases, saving progress...")
            output_path = os.path.join(os.path.dirname(dataset_path), "counterfact_graph.json")
            with open(output_path, 'w') as output_file:
                json.dump(new_json, output_file)

    # Save final results
    output_path = os.path.join(os.path.dirname(dataset_path), "counterfact_graph.json")
    with open(output_path, 'w') as output_file:
        json.dump(new_json, output_file)
    print(f"Saved {counter} cases to {output_path}")


if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_path = os.path.join(project_root, "experiment/data/counterfact.json")
    get_triplets_from_dataset(dataset_path)
