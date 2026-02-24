from SPARQLWrapper import SPARQLWrapper, JSON
from typing import List, Optional
from urllib.error import HTTPError
import time

# Initialize SPARQL endpoint
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
# Wikidata requires a proper User-Agent header with contact information
# Format: AppName/Version (URL; email@example.com)
user_agent = "RefusalDirectionBot/1.0 (https://github.com/; contact@example.com)"

# Create SPARQLWrapper instance with User-Agent
# The 'agent' parameter should set the User-Agent header
sparql = SPARQLWrapper(SPARQL_ENDPOINT, agent=user_agent)
sparql.setReturnFormat(JSON)

# For some versions of SPARQLWrapper, we may need to set custom headers
# Try to set User-Agent via custom headers if agent parameter doesn't work
try:
    # SPARQLWrapper 2.x uses addCustomHttpHeader
    sparql.addCustomHttpHeader("User-Agent", user_agent)
except AttributeError:
    # SPARQLWrapper 1.x may not have this method, agent parameter should work
    pass

# Track last request time to avoid rate limiting
_last_request_time = 0
_min_request_interval = 1.0  # Minimum seconds between requests (Wikidata recommends ~1 req/sec)


def safe_sparql_request(sparql_wrapper):
    """
    Make a SPARQL request with error handling.
    Handles 429 (Too Many Requests), 403 (Forbidden), and 504 (Gateway Timeout) errors with automatic retry.
    Also adds a small delay between requests to avoid rate limiting.
    """
    global _last_request_time
    
    # Add a small delay to avoid rate limiting
    elapsed = time.time() - _last_request_time
    if elapsed < _min_request_interval:
        time.sleep(_min_request_interval - elapsed)
    _last_request_time = time.time()
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Requests that make too many calls may result in a 429 error.
            # Set a longer timeout for complex queries
            results = sparql_wrapper.query().convert()
            return results
        except HTTPError as e:
            if e.code == 429:
                print(f"429 error occurred (Too Many Requests). Pausing execution for 15 seconds...")
                time.sleep(15)  # Sleep for 15 seconds before retrying.
                retry_count += 1
            elif e.code == 403:
                print(f"403 error occurred (Forbidden). This might be due to User-Agent issues.")
                print(f"Retrying in 5 seconds... (attempt {retry_count + 1}/{max_retries})")
                time.sleep(5)
                retry_count += 1
                if retry_count >= max_retries:
                    raise Exception(f"403 Forbidden error persisted after {max_retries} retries. "
                                  f"Please check your User-Agent and ensure you're not blocked.")
            elif e.code == 504:
                print(f"504 error occurred (Gateway Timeout). The query may be too complex or the server is overloaded.")
                print(f"Retrying in 10 seconds... (attempt {retry_count + 1}/{max_retries})")
                time.sleep(10)  # Wait longer for gateway timeout
                retry_count += 1
                if retry_count >= max_retries:
                    raise Exception(f"504 Gateway Timeout persisted after {max_retries} retries. "
                                  f"The query may be too complex. Try simplifying it.")
            else:
                # If the error code is not handled, re-raise the exception.
                raise e
        except Exception as e:
            # Handle other exceptions (including timeout exceptions)
            error_str = str(e).lower()
            if "timeout" in error_str or "504" in error_str:
                print(f"Timeout error occurred: {e}")
                print(f"Retrying in 10 seconds... (attempt {retry_count + 1}/{max_retries})")
                time.sleep(10)
                retry_count += 1
            elif retry_count < max_retries - 1:
                print(f"Error occurred: {e}. Retrying in 5 seconds...")
                time.sleep(5)
                retry_count += 1
            else:
                raise e
    
    raise Exception(f"Failed after {max_retries} retries")


def get_neighbor_triples(entity_id: str) -> dict:
    """
    Get all triples where the entity is the subject (outgoing edges).
    Uses SELECT query with wikibase:label service for efficient label retrieval.
    
    Args:
        entity_id: Wikidata entity ID (e.g., "Q123")
    
    Returns:
        Dictionary with SPARQL query results in SELECT format
    """
    query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX wikibase: <http://wikiba.se/ontology#>
    PREFIX bd: <http://www.bigdata.com/rdf#>
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>

    SELECT ?subject ?subjectLabel ?predicate ?predicateLabel ?object ?objectLabel WHERE {
      BIND(wd:%s AS ?subject) 
      ?subject ?predicate ?object .
      ?wdProp wikibase:directClaim ?predicate .
      
      SERVICE wikibase:label {
        bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". 
        ?wdProp rdfs:label ?predicateLabel .
        ?object rdfs:label ?objectLabel .
        ?subject rdfs:label ?subjectLabel .
      }

      FILTER(STRSTARTS(STR(?predicate), STR(wdt:))) .
      FILTER(STRSTARTS(STR(?object), STR(wd:))) .
      FILTER(!STRSTARTS(STR(?predicate), STR(wdt:P910))) .
      FILTER(!STRSTARTS(STR(?predicate), STR(wdt:P1423))) .
    }
    LIMIT 1000
    """ % entity_id
    
    sparql.setQuery(query)
    try:
        results = safe_sparql_request(sparql)
        # Ensure we return the expected format
        if "results" not in results:
            return {"results": {"bindings": []}}
        return results
    except Exception as e:
        print(f"Error querying Wikidata for {entity_id}: {e}")
        return {"results": {"bindings": []}}


def get_property_label(property_id: str) -> str:
    """
    Get the English label for a Wikidata property.
    
    Args:
        property_id: Wikidata property ID (e.g., "P31")
    
    Returns:
        Property label in English
    """
    query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX wikibase: <http://wikiba.se/ontology#>
    PREFIX bd: <http://www.bigdata.com/rdf#>

    SELECT ?propertyLabel WHERE {
        wd:%s rdfs:label ?propertyLabel.
        FILTER(LANG(?propertyLabel) = "en")
    }
    """ % property_id
    
    sparql.setQuery(query)
    try:
        results = safe_sparql_request(sparql)
        bindings = results["results"]["bindings"]
        if bindings:
            return bindings[0]["propertyLabel"]["value"]
        return property_id  # Return ID if label not found
    except Exception as e:
        print(f"Error getting property label for {property_id}: {e}")
        return property_id


def search_entity_by_label(label: str, limit: int = 5, use_search_api: bool = True) -> List[dict]:
    """
    Search for Wikidata entities by label.
    Can use either Wikidata Search API (faster, more reliable) or SPARQL query.
    
    Args:
        label: String to search for
        limit: Maximum number of results to return
        use_search_api: If True, use Wikidata Search API (recommended). If False, use SPARQL.
    
    Returns:
        List of dictionaries with 'id' and 'label' keys
    """
    if use_search_api:
        # Try using Wikidata Search API first (more reliable)
        try:
            import urllib.parse
            import urllib.request
            import json as json_lib
            
            # Wikidata Search API endpoint
            search_url = "https://www.wikidata.org/w/api.php"
            params = {
                "action": "wbsearchentities",
                "search": label,
                "language": "en",
                "format": "json",
                "limit": limit
            }
            
            url = f"{search_url}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url)
            req.add_header("User-Agent", user_agent)
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json_lib.loads(response.read().decode())
                
            entities = []
            if "search" in data:
                for item in data["search"]:
                    entities.append({
                        "id": item["id"],
                        "label": item.get("label", item["id"])
                    })
            return entities
        except Exception as e:
            print(f"Search API failed, falling back to SPARQL: {e}")
            # Fall through to SPARQL query
    
    # Fallback to SPARQL query (simplified to avoid timeouts)
    query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?item ?itemLabel WHERE {
        ?item rdfs:label ?itemLabel.
        FILTER(LANG(?itemLabel) = "en")
        FILTER(CONTAINS(LCASE(?itemLabel), LCASE("%s")))
    }
    LIMIT %d
    """ % (label, limit)
    
    sparql.setQuery(query)
    try:
        results = safe_sparql_request(sparql)
        entities = []
        for binding in results["results"]["bindings"]:
            entity_id = binding["item"]["value"].split("/")[-1]
            entity_label = binding["itemLabel"]["value"]
            entities.append({"id": entity_id, "label": entity_label})
        return entities
    except Exception as e:
        print(f"Error searching for entity '{label}': {e}")
        return []


def get_entity_info(entity_id: str) -> Optional[dict]:
    """
    Get basic information about a Wikidata entity.
    
    Args:
        entity_id: Wikidata entity ID (e.g., "Q123")
    
    Returns:
        Dictionary with 'id' and 'label' keys, or None if not found
    """
    query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX wd: <http://www.wikidata.org/entity/>

    SELECT ?itemLabel WHERE {
        wd:%s rdfs:label ?itemLabel.
        FILTER(LANG(?itemLabel) = "en")
    }
    LIMIT 1
    """ % entity_id
    
    sparql.setQuery(query)
    try:
        results = safe_sparql_request(sparql)
        bindings = results["results"]["bindings"]
        if bindings:
            return {
                "id": entity_id,
                "label": bindings[0]["itemLabel"]["value"]
            }
        return None
    except Exception as e:
        print(f"Error getting entity info for {entity_id}: {e}")
        return None










