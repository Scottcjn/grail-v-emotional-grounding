#!/usr/bin/env python3
"""
GRAIL-V Paper: Text Embedding Similarity Analysis
===================================================
Computes semantic similarity metrics between STOCK (literal/descriptive)
and NEURO (emotional/narrative) prompt pairs using sentence-transformer
embeddings. Supports the paper's claim that emotional vocabulary occupies
denser embedding space than literal scene-description vocabulary.

Output: clip_text_similarity.json
"""

import json
import os
import re
import sys
from itertools import combinations

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Prompt pairs: STOCK (literal) vs NEURO (emotional/narrative)
# ---------------------------------------------------------------------------
PROMPT_PAIRS = {
    "sophia_realization": {
        "STOCK": (
            "Victorian woman portrait, subtle head movement, slight smile, "
            "blinking eyes, warm lighting"
        ),
        "NEURO": (
            "The young woman's eyes brighten with quiet realization, a knowing "
            "smile forming as inspiration takes hold, warmth spreading across "
            "her expression"
        ),
    },
    "sophia_contemplation": {
        "STOCK": (
            "Victorian woman portrait, looking thoughtful, gentle movements, "
            "soft lighting"
        ),
        "NEURO": (
            "Her gaze turns inward with deep contemplation, a subtle shift "
            "from curiosity to understanding, quiet wisdom settling in her "
            "features"
        ),
    },
    "sophia_determination": {
        "STOCK": (
            "Victorian woman portrait, serious expression, focused look, "
            "slight movement"
        ),
        "NEURO": (
            "Quiet determination hardens in her eyes, jaw setting with "
            "newfound resolve, inner fire building behind composed exterior"
        ),
    },
    "elyan_sophia_focus": {
        "STOCK": (
            "Victorian exhibition, woman working on machine, man watching, "
            "gaslight flickering"
        ),
        "NEURO": (
            "The young woman works with fierce concentration, confident hands "
            "moving with purpose, quiet authority radiating as she masters the "
            "brass machinery"
        ),
    },
    "elyan_claude_focus": {
        "STOCK": (
            "Victorian exhibition, older man gesturing, woman at machine, "
            "warm lighting"
        ),
        "NEURO": (
            "The older gentleman's skepticism softens to grudging respect, "
            "pride wounded but giving way to reluctant admiration"
        ),
    },
    "debate_passion": {
        "STOCK": (
            "Two people in conversation, gesturing, fireplace glowing, "
            "Victorian study"
        ),
        "NEURO": (
            "Passionate intellectual exchange, conviction burning in their "
            "eyes, the electricity of clashing ideas filling the air between "
            "them"
        ),
    },
    "debate_tension": {
        "STOCK": (
            "Two people talking, subtle movements, warm firelight, period room"
        ),
        "NEURO": (
            "Tension crackling between them, unspoken challenge in their "
            "gazes, the air thick with intellectual rivalry"
        ),
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def pairwise_cosine_distances(embeddings: np.ndarray) -> list[float]:
    """Return list of (1 - cosine_sim) for every unique pair."""
    n = len(embeddings)
    dists = []
    for i, j in combinations(range(n), 2):
        dists.append(1.0 - cosine_sim(embeddings[i], embeddings[j]))
    return dists


def tokenize_words(text: str) -> set[str]:
    """Simple whitespace + punctuation tokeniser, lowercased."""
    return set(re.findall(r"[a-z']+", text.lower()))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, device="cpu")
    print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")

    scene_names = list(PROMPT_PAIRS.keys())
    stock_texts = [PROMPT_PAIRS[k]["STOCK"] for k in scene_names]
    neuro_texts = [PROMPT_PAIRS[k]["NEURO"] for k in scene_names]

    # --- Encode full prompts -----------------------------------------------
    print("\nEncoding prompts ...")
    stock_embs = model.encode(stock_texts, normalize_embeddings=True)
    neuro_embs = model.encode(neuro_texts, normalize_embeddings=True)

    # (a) Per-pair cosine similarity ----------------------------------------
    pair_sims = {}
    for idx, name in enumerate(scene_names):
        sim = cosine_sim(stock_embs[idx], neuro_embs[idx])
        pair_sims[name] = round(sim, 5)

    mean_pair_sim = round(float(np.mean(list(pair_sims.values()))), 5)

    # (b) Intra-group distance: STOCK --------------------------------------
    stock_dists = pairwise_cosine_distances(stock_embs)
    mean_stock_dist = round(float(np.mean(stock_dists)), 5)

    # (c) Intra-group distance: NEURO --------------------------------------
    neuro_dists = pairwise_cosine_distances(neuro_embs)
    mean_neuro_dist = round(float(np.mean(neuro_dists)), 5)

    # (d) Word-level vocabulary analysis ------------------------------------
    print("Analysing vocabulary embeddings ...")

    stock_words: set[str] = set()
    neuro_words: set[str] = set()
    for name in scene_names:
        stock_words |= tokenize_words(PROMPT_PAIRS[name]["STOCK"])
        neuro_words |= tokenize_words(PROMPT_PAIRS[name]["NEURO"])

    # Words unique to each group
    stock_only = stock_words - neuro_words
    neuro_only = neuro_words - stock_words
    shared = stock_words & neuro_words

    # Embed unique words
    stock_word_list = sorted(stock_only)
    neuro_word_list = sorted(neuro_only)

    stock_word_embs = model.encode(stock_word_list, normalize_embeddings=True)
    neuro_word_embs = model.encode(neuro_word_list, normalize_embeddings=True)

    stock_word_dists = pairwise_cosine_distances(stock_word_embs)
    neuro_word_dists = pairwise_cosine_distances(neuro_word_embs)

    mean_stock_word_dist = round(float(np.mean(stock_word_dists)), 5) if stock_word_dists else None
    mean_neuro_word_dist = round(float(np.mean(neuro_word_dists)), 5) if neuro_word_dists else None

    # Categorise emotional vs literal subsets manually
    emotional_keywords = [
        w for w in neuro_word_list
        if w in {
            "realization", "knowing", "inspiration", "contemplation",
            "curiosity", "understanding", "wisdom", "determination",
            "resolve", "concentration", "authority", "skepticism",
            "respect", "pride", "admiration", "passionate", "conviction",
            "electricity", "tension", "challenge", "rivalry",
            "fierce", "confident", "reluctant", "grudging",
            "wounded", "brightens", "brighten", "hardens", "crackling",
            "unspoken", "intellectual", "newfound", "composed",
        }
    ]
    literal_keywords = [
        w for w in stock_word_list
        if w in {
            "portrait", "lighting", "movement", "blinking",
            "flickering", "gaslight", "fireplace", "glowing",
            "gesturing", "talking", "conversation", "exhibition",
            "machine", "room", "period", "study", "focused",
            "serious", "subtle", "gentle", "slight", "soft",
            "warm", "look", "movements", "expression",
        }
    ]

    emo_embs = model.encode(emotional_keywords, normalize_embeddings=True) if emotional_keywords else np.array([])
    lit_embs = model.encode(literal_keywords, normalize_embeddings=True) if literal_keywords else np.array([])

    emo_word_dists = pairwise_cosine_distances(emo_embs) if len(emo_embs) >= 2 else []
    lit_word_dists = pairwise_cosine_distances(lit_embs) if len(lit_embs) >= 2 else []

    mean_emo_word_dist = round(float(np.mean(emo_word_dists)), 5) if emo_word_dists else None
    mean_lit_word_dist = round(float(np.mean(lit_word_dists)), 5) if lit_word_dists else None

    # --- Assemble results --------------------------------------------------
    results = {
        "model": model_name,
        "embedding_dim": model.get_sentence_embedding_dimension(),
        "num_pairs": len(scene_names),
        "per_pair_cosine_similarity": pair_sims,
        "mean_pair_cosine_similarity": mean_pair_sim,
        "intra_group_mean_cosine_distance": {
            "STOCK": mean_stock_dist,
            "NEURO": mean_neuro_dist,
            "interpretation": (
                "Lower distance = prompts cluster more tightly in embedding "
                "space. If NEURO < STOCK, emotional language is denser."
            ),
        },
        "vocabulary_analysis": {
            "unique_stock_words": len(stock_only),
            "unique_neuro_words": len(neuro_only),
            "shared_words": len(shared),
            "stock_unique_sample": stock_word_list[:15],
            "neuro_unique_sample": neuro_word_list[:15],
            "mean_pairwise_distance_stock_unique_words": mean_stock_word_dist,
            "mean_pairwise_distance_neuro_unique_words": mean_neuro_word_dist,
        },
        "emotional_vs_literal_vocab": {
            "emotional_keywords": emotional_keywords,
            "literal_keywords": literal_keywords,
            "mean_pairwise_distance_emotional": mean_emo_word_dist,
            "mean_pairwise_distance_literal": mean_lit_word_dist,
            "interpretation": (
                "Emotional vocabulary is expected to have LOWER mean pairwise "
                "distance (denser cluster) than literal/descriptive vocabulary, "
                "because emotion words share overlapping affective dimensions."
            ),
        },
    }

    # --- Write JSON --------------------------------------------------------
    out_path = os.path.join(os.path.dirname(__file__), "clip_text_similarity.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    # --- Pretty-print summary ----------------------------------------------
    print("\n" + "=" * 72)
    print("  GRAIL-V TEXT EMBEDDING ANALYSIS  --  Summary")
    print("=" * 72)

    print(f"\nModel : {model_name}")
    print(f"Dims  : {model.get_sentence_embedding_dimension()}")
    print(f"Pairs : {len(scene_names)}")

    print("\n--- (a) Per-Pair Cosine Similarity (STOCK vs NEURO) -----------------")
    for name, sim in pair_sims.items():
        print(f"  {name:30s}  {sim:.5f}")
    print(f"  {'MEAN':30s}  {mean_pair_sim:.5f}")

    print("\n--- (b/c) Intra-Group Mean Cosine Distance --------------------------")
    print(f"  STOCK prompts (literal)     : {mean_stock_dist:.5f}")
    print(f"  NEURO prompts (emotional)   : {mean_neuro_dist:.5f}")
    diff_pct = ((mean_neuro_dist - mean_stock_dist) / mean_stock_dist) * 100
    if mean_neuro_dist < mean_stock_dist:
        print(f"  --> NEURO prompts are {abs(diff_pct):.1f}% DENSER than STOCK")
    else:
        print(f"  --> STOCK prompts are {abs(diff_pct):.1f}% denser than NEURO")

    print("\n--- (d) Vocabulary Embedding Analysis --------------------------------")
    print(f"  Unique STOCK words : {len(stock_only):3d}  |  mean pairwise dist: {mean_stock_word_dist}")
    print(f"  Unique NEURO words : {len(neuro_only):3d}  |  mean pairwise dist: {mean_neuro_word_dist}")
    print(f"  Shared words       : {len(shared):3d}")

    print(f"\n  Emotional subset ({len(emotional_keywords)} words):")
    print(f"    mean pairwise distance : {mean_emo_word_dist}")
    print(f"  Literal subset ({len(literal_keywords)} words):")
    print(f"    mean pairwise distance : {mean_lit_word_dist}")

    if mean_emo_word_dist is not None and mean_lit_word_dist is not None:
        vocab_diff = ((mean_emo_word_dist - mean_lit_word_dist) / mean_lit_word_dist) * 100
        if mean_emo_word_dist < mean_lit_word_dist:
            print(f"\n  ==> Emotional vocab is {abs(vocab_diff):.1f}% DENSER than literal vocab")
            print("      (supports GRAIL-V claim: emotion words cluster in embedding space)")
        else:
            print(f"\n  ==> Literal vocab is {abs(vocab_diff):.1f}% denser than emotional vocab")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
