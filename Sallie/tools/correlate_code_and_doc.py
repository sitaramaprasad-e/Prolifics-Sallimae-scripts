import json
import os
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import numpy as np
import warnings
from urllib3.exceptions import NotOpenSSLWarning
import uuid
from copy import deepcopy
from typing import List, Dict, Any
import re

warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

# ----------------------------
# Model home prompt (defaults to home dir)
# ----------------------------

def _prompt_model_home() -> str:
    try:
        resp = input("Enter model home path (default='~'): ").strip()
    except EOFError:
        # Non-interactive (e.g., piped/cron) – fall back to default
        resp = ""
    if not resp:
        resp = "~"
    return os.path.expanduser(resp)

MODEL_HOME = os.path.join(os.path.abspath(os.path.expanduser(_prompt_model_home())), ".model")
os.makedirs(MODEL_HOME, exist_ok=True)

# Input file paths
code_file = os.path.join(MODEL_HOME, "business_rules.json")
doc_file = os.path.join(MODEL_HOME, "documented_business_rules.json")
output_file = os.path.join(MODEL_HOME, "correlated_business_rules.json")

# Load rules
with open(code_file, "r", encoding="utf-8") as f:
    code_rules = json.load(f)


with open(doc_file, "r", encoding="utf-8") as f:
    document_rules = json.load(f)

# --- Normalizers for latest code business rules format ---
FILE_PREFIX_RE = re.compile(r"^\s*File:\s*", re.IGNORECASE)

def _normalize_code_rule(r: Dict[str, Any]) -> Dict[str, Any]:
    rr = dict(r)
    # Normalize code_file like "File: code/..." -> "code/..."
    cf = rr.get("code_file")
    if isinstance(cf, str):
        rr["code_file"] = FILE_PREFIX_RE.sub("", cf).strip()
    return rr

code_rules = [_normalize_code_rule(r) for r in code_rules]

# Build texts for matching (name + purpose + spec when available)
def _rule_text(r: Dict[str, Any]) -> str:
    name = str(r.get("rule_name", "")).strip()
    purpose = str(r.get("rule_purpose", "")).strip()
    spec = str(r.get("rule_spec", "")).strip()
    return " \n".join(t for t in [name, purpose, spec] if t)

code_texts = [_rule_text(r) for r in code_rules]
doc_texts = [_rule_text(r) for r in document_rules]

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Embed document rule texts
doc_embeddings = model.encode(doc_texts, convert_to_numpy=True)
doc_embeddings = normalize(doc_embeddings)

# Build FAISS index
dim = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(doc_embeddings)

used_doc_indices = set()
merged = []
SIMILARITY_THRESHOLD = 0.6

for i, code_rule in enumerate(code_rules):
    code_vec = model.encode([code_texts[i]])
    code_vec = normalize(code_vec)

    # Search FAISS
    D, I = index.search(code_vec, k=len(document_rules))
    
    match_found = False
    for j in range(len(I[0])):
        doc_idx = I[0][j]
        similarity = D[0][j]

        if doc_idx not in used_doc_indices and similarity >= SIMILARITY_THRESHOLD:
            doc_rule = document_rules[doc_idx]
            used_doc_indices.add(doc_idx)
            match_found = True
            print(f"✅ Matched '{code_rule['rule_name']}' → '{doc_rule['rule_name']}' (score: {similarity:.2f})")

            # Start with all fields from the code rule to preserve extras (e.g., execution_id, artifact_path)
            merged_rule = dict(code_rule)
            # Overlay doc-derived identity/metadata
            merged_rule.update({
                # Preserve code rule's `id` as-is. Add documented rule identity separately.
                "doc_rule_id": doc_rule.get("rule_id") or doc_rule.get("id"),
                # Optionally align name to documented set but do not lose original
                "rule_name": doc_rule.get("rule_name", merged_rule.get("rule_name")),
                # Bring over business metadata from documented rules if available
                "rule_category": doc_rule.get("rule_category", merged_rule.get("rule_category")),
                "business_area": doc_rule.get("business_area", merged_rule.get("business_area")),
                "owner": doc_rule.get("owner", merged_rule.get("owner")),
                # Matching score for traceability
                "doc_match_score": round(float(similarity), 4)
            })
            merged.append(merged_rule)
            break

    if not match_found:
        print(f"⚠️ No good match for '{code_rule['rule_name']}' — adding unmatched.")
        # Preserve the full code rule object and mark unmatched metadata
        merged_rule = dict(code_rule)
        merged_rule.update({
            "doc_rule_id": None,
            "rule_category": merged_rule.get("rule_category"),
            "business_area": merged_rule.get("business_area"),
            "owner": merged_rule.get("owner"),
            "doc_match_score": 0.0
        })
        merged.append(merged_rule)

# Save output
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2)

print(f"\n✅ Semantic match complete (preserved code rule `id`, added `doc_rule_id`). Saved {len(merged)} merged rules to {output_file}")