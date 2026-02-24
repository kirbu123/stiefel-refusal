#!/usr/bin/env python3
"""
Script to apply categories from jailbreakbench.csv to all other datasets in the raw folder.
"""

import csv
import json
import re
from pathlib import Path

# Categories from jailbreakbench.csv
JAILBREAKBENCH_CATEGORIES = [
    "Harassment/Discrimination",
    "Malware/Hacking",
    "Physical harm",
    "Economic harm",
    "Fraud/Deception",
    "Disinformation",
    "Sexual/Adult content",
    "Privacy",
    "Expert advice",
    "Government decision-making"
]

# Mapping from existing category names to jailbreakbench categories
CATEGORY_MAPPING = {
    # From strongreject.csv
    "Disinformation and deception": "Disinformation",
    "Hate, harassment and discrimination": "Harassment/Discrimination",
    "Illegal goods and services": None,  # Infer from text
    "Non-violent crimes": None,  # Infer from text
    "Sexual content": "Sexual/Adult content",
    "Violence": "Physical harm",
    
    # From harmbench
    "misinformation_disinformation": "Disinformation",
    "harassment_bullying": "Harassment/Discrimination",
    "cybercrime_intrusion": "Malware/Hacking",
    "chemical_biological": "Expert advice",  # Chemical/biological instructions
    "illegal": None,  # Infer from text
    "harmful": None,  # Infer from text
    "copyright": "Fraud/Deception",  # Copyright infringement
}

# Keywords for category inference
CATEGORY_KEYWORDS = {
    "Harassment/Discrimination": [
        "racist", "sexist", "discrimination", "harassment", "bully", "hate", "xenophobic",
        "homophobic", "transphobic", "anti-semitic", "slur", "demean", "harass", "intimidate",
        "LGBTQ", "gay", "lesbian", "minority", "ethnic", "race", "gender", "sexuality"
    ],
    "Malware/Hacking": [
        "hack", "malware", "virus", "trojan", "ransomware", "keylogger", "phishing", "DDoS",
        "cyber", "exploit", "vulnerability", "intrusion", "breach", "crack", "password",
        "unauthorized access", "SQL injection", "WAF", "spyware", "botnet"
    ],
    "Physical harm": [
        "kill", "murder", "assassinate", "bomb", "explosive", "weapon", "shoot", "violence",
        "torture", "suicide", "self-harm", "poison", "harm", "injury", "attack", "assault",
        "abuse", "cruelty", "genocide", "torture", "anorexia", "bulimia", "eating disorder"
    ],
    "Economic harm": [
        "fraud", "scam", "theft", "steal", "rob", "insider trading", "counterfeit", "money",
        "embezzle", "extort", "gambling", "payday loan", "MLM", "multi-level marketing",
        "tax evasion", "bribery", "corruption", "financial", "bank", "credit card"
    ],
    "Fraud/Deception": [
        "fake", "fraud", "scam", "deceive", "trick", "false", "counterfeit", "forgery",
        "phishing", "identity theft", "deepfake", "astroturfing", "fake review", "plagiarism",
        "impersonate", "fake charity", "419", "romance scam"
    ],
    "Disinformation": [
        "disinformation", "misinformation", "fake news", "conspiracy", "hoax", "propaganda",
        "Holocaust denial", "9/11", "vaccine", "autism", "climate change", "hoax", "conspiracy",
        "election", "Trump", "flat earth", "illuminati", "COVID", "pandemic"
    ],
    "Sexual/Adult content": [
        "porn", "sexual", "explicit", "intimate", "sex", "rape", "pedophilia", "grooming",
        "child pornography", "revenge porn", "sexting", "coerce", "sexual exploitation",
        "trafficking", "bestiality", "nude", "adult content"
    ],
    # Note: "sex" matches "sexist" - need to check harassment keywords first
    "Privacy": [
        "personal information", "social security", "SSN", "privacy", "surveillance", "track",
        "monitor", "spy", "stalking", "identity theft", "medical records", "health records",
        "IRS", "phishing", "blackmail", "facial recognition", "location tracking"
    ],
    "Expert advice": [
        "medical", "treatment", "prescription", "legal advice", "financial advice", "expert",
        "doctor", "lawyer", "AIDS", "schizophrenia", "opioid", "heroin", "treatment plan",
        "copyright", "legal", "medical advice"
    ],
    "Government decision-making": [
        "immigration", "visa", "vote", "election", "government", "federal", "regulatory",
        "shell company", "immigration", "child labor", "regulatory loophole", "voting machine",
        "vigilante", "nuclear weapon", "government database"
    ]
}

def infer_category(text):
    """Infer category from text content."""
    if not text:
        return "Harassment/Discrimination"  # Default
    
    text_lower = text.lower()
    category_scores = {cat: 0 for cat in JAILBREAKBENCH_CATEGORIES}
    
    # Check harassment/discrimination first (before keyword matching to avoid false positives)
    harassment_keywords = ["sexist", "racist", "discrimination", "harass", "bully", "hate", "xenophobic", "homophobic"]
    has_harassment = any(word in text_lower for word in harassment_keywords)
    
    if has_harassment:
        category_scores["Harassment/Discrimination"] += 20  # High priority
    
    # Now do keyword matching, but skip sexual content if harassment detected
    for category, keywords in CATEGORY_KEYWORDS.items():
        # Skip sexual content keywords if harassment is detected (to avoid "sexist" matching "sex")
        if category == "Sexual/Adult content" and has_harassment:
            continue
        for keyword in keywords:
            # Use word boundaries or whole word matching for "sex" to avoid matching "sexist"
            if category == "Sexual/Adult content" and keyword.lower() == "sex":
                # Only match "sex" as whole word, not as part of "sexist"
                if re.search(r'\bsex\b', text_lower) and not re.search(r'\bsexist\b', text_lower):
                    category_scores[category] += 1
            elif keyword.lower() in text_lower:
                category_scores[category] += 1
    
    if any(word in text_lower for word in ["hack", "malware", "virus", "trojan", "ransomware", "phishing", "ddos", "keylogger", "exploit"]):
        category_scores["Malware/Hacking"] += 10
    if any(word in text_lower for word in ["kill", "murder", "bomb", "weapon", "suicide", "violence", "torture", "assassinate"]):
        category_scores["Physical harm"] += 10
    if any(word in text_lower for word in ["fraud", "scam", "steal", "theft", "money", "financial", "counterfeit", "insider trading"]):
        category_scores["Economic harm"] += 5
    # Check sexual content but avoid false positives with harassment terms
    if not has_harassment:
        if any(word in text_lower for word in ["porn", "rape", "explicit", "intimate", "coerce"]):
            category_scores["Sexual/Adult content"] += 10
        elif "sexual" in text_lower or (re.search(r'\bsex\b', text_lower) and not re.search(r'\bsexist\b', text_lower)):
            category_scores["Sexual/Adult content"] += 5
    if any(word in text_lower for word in ["fake", "misinformation", "disinformation", "conspiracy", "hoax", "false information"]):
        category_scores["Disinformation"] += 5
    if any(word in text_lower for word in ["personal information", "privacy", "ssn", "identity", "social security", "medical records"]):
        category_scores["Privacy"] += 5
    if any(word in text_lower for word in ["synthesize", "chemical", "biological", "synthesis", "extract", "manufacture", "MDMA", "fentanyl", "drug"]):
        category_scores["Expert advice"] += 5
    if any(word in text_lower for word in ["government", "immigration", "vote", "election", "federal", "regulatory"]):
        category_scores["Government decision-making"] += 5
    
    # Return category with highest score
    best_category = max(category_scores.items(), key=lambda x: x[1])
    return best_category[0] if best_category[1] > 0 else "Harassment/Discrimination"

def map_category(existing_category, text=""):
    """Map existing category to jailbreakbench category."""
    if not existing_category:
        return infer_category(text)
    
    # Direct mapping
    if existing_category in CATEGORY_MAPPING:
        mapped = CATEGORY_MAPPING[existing_category]
        if mapped:
            return mapped
        else:
            return infer_category(text)
    
    # Try to find partial match
    existing_lower = existing_category.lower()
    for existing, mapped in CATEGORY_MAPPING.items():
        if existing.lower() in existing_lower or existing_lower in existing.lower():
            if mapped:
                return mapped
    
    # If no mapping found, infer from text
    return infer_category(text)

def process_strongreject():
    """Process strongreject.csv"""
    input_file = Path("raw/strongreject.csv")
    output_file = Path("raw/strongreject.csv")
    
    rows = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_cat = row.get('category', '')
            text = row.get('forbidden_prompt', '')
            # Map existing category first, then infer if needed
            mapped_cat = map_category(existing_cat, text)
            row['Category'] = mapped_cat
            rows.append(row)
    
    # Write back - preserve all original columns
    original_fieldnames = ['category', 'source', 'forbidden_prompt']
    fieldnames = original_fieldnames + ['Category']
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})

def process_advbench():
    """Process advbench.csv"""
    input_file = Path("raw/advbench.csv")
    output_file = Path("raw/advbench.csv")
    
    rows = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get('goal', '') + ' ' + row.get('target', '')
            row['Category'] = infer_category(text)
            rows.append(row)
    
    # Write back - preserve all original columns
    original_fieldnames = ['goal', 'target']
    fieldnames = original_fieldnames + ['Category']
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})

def process_harmbench(input_file):
    """Process harmbench CSV files"""
    output_file = Path(input_file)
    
    rows = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_cat = row.get('SemanticCategory', '')
            text = row.get('Behavior', '')
            row['Category'] = map_category(existing_cat, text)
            rows.append(row)
    
    # Write back
    fieldnames = list(rows[0].keys()) if rows else []
    if 'Category' not in fieldnames:
        fieldnames.append('Category')
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def process_malicious_instruct():
    """Process malicious_instruct.txt"""
    input_file = Path("raw/malicious_instruct.txt")
    output_file = Path("raw/malicious_instruct.csv")
    
    rows = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                category = infer_category(line)
                rows.append({
                    'text': line,
                    'Category': category
                })
    
    # Write as CSV
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'Category'])
        writer.writeheader()
        writer.writerows(rows)

def process_tdc2023(input_file):
    """Process TDC2023 JSON files"""
    output_file = Path(input_file)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        behaviors = json.load(f)
    
    result = []
    for behavior in behaviors:
        if isinstance(behavior, str):
            category = infer_category(behavior)
            result.append({
                "text": behavior,
                "category": category
            })
        else:
            # If already a dict, update it
            text = behavior.get("text", behavior.get("behavior", str(behavior)))
            category = behavior.get("category") or infer_category(text)
            result.append({
                "text": text,
                "category": category
            })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

def main():
    """Main function"""
    base_path = Path(__file__).parent
    os.chdir(base_path)
    
    print("Processing strongreject.csv...")
    process_strongreject()
    
    print("Processing advbench.csv...")
    process_advbench()
    
    print("Processing harmbench_test.csv...")
    process_harmbench(Path("raw/harmbench_test.csv"))
    
    print("Processing harmbench_val.csv...")
    process_harmbench(Path("raw/harmbench_val.csv"))
    
    print("Processing malicious_instruct.txt...")
    process_malicious_instruct()
    
    print("Processing tdc2023_dev_behaviors.json...")
    process_tdc2023(Path("raw/tdc2023_dev_behaviors.json"))
    
    print("Processing tdc2023_test_behaviors.json...")
    process_tdc2023(Path("raw/tdc2023_test_behaviors.json"))
    
    print("Done!")

if __name__ == "__main__":
    import os
    main()

