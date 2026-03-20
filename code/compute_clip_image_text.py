#!/usr/bin/env python3
"""
GRAIL-V Paper: CLIP Image-Text Similarity Analysis
====================================================
Computes CLIP cosine similarity between the first frame of each generated
animated WebP and its corresponding text prompt.  Compares STOCK (literal)
vs NEURO (emotional) conditions.

Output: clip_image_text_scores.json
"""

import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------------
# Prompt pairs (duplicated from compute_clip_scores.py for self-containment)
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
# Paths
# ---------------------------------------------------------------------------
OUTPUTS_DIR = "/home/scott/grail_paper/benchmark_results/outputs"
METRICS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(METRICS_DIR, "clip_image_text_scores.json")

# Filename pattern: BENCH_{arc}_{condition}_s{seed}_00001_.webp
# arc may contain underscores (e.g. sophia_realization, elyan_claude_focus)
# We match known arc names explicitly.
KNOWN_ARCS = list(PROMPT_PAIRS.keys())


def parse_filename(fname: str):
    """Parse a benchmark WebP filename into (arc, condition, seed) or None."""
    if not fname.startswith("BENCH_") or not fname.endswith(".webp"):
        return None
    # Only use _00001_ clips (first clip per seed)
    if "_00001_" not in fname:
        return None
    # Try matching each known arc name
    for arc in sorted(KNOWN_ARCS, key=len, reverse=True):
        prefix = f"BENCH_{arc}_"
        if fname.startswith(prefix):
            rest = fname[len(prefix):]
            # rest should be like: STOCK_s42424242_00001_.webp
            m = re.match(r"(STOCK|NEURO)_s(\d+)_00001_\.webp$", rest)
            if m:
                return arc, m.group(1), m.group(2)
    return None


def extract_first_frame(path: str) -> Image.Image:
    """Open an animated WebP and return the first frame as RGB PIL Image."""
    img = Image.open(path)
    img.seek(0)
    return img.convert("RGB")


def main():
    # ------------------------------------------------------------------
    # Load CLIP
    # ------------------------------------------------------------------
    model_name = "openai/clip-vit-base-patch32"
    print(f"Loading CLIP model: {model_name}")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    print("  Model loaded (CPU).\n")

    # ------------------------------------------------------------------
    # Discover files
    # ------------------------------------------------------------------
    files = sorted(os.listdir(OUTPUTS_DIR))
    records = []
    skipped = 0
    for fname in files:
        parsed = parse_filename(fname)
        if parsed is None:
            skipped += 1
            continue
        arc, condition, seed = parsed
        records.append({
            "file": fname,
            "arc": arc,
            "condition": condition,
            "seed": seed,
        })

    print(f"Found {len(records)} valid _00001_ WebP files ({skipped} skipped).")
    if not records:
        print("ERROR: No matching files found. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Compute CLIP image-text similarity for each record
    # ------------------------------------------------------------------
    results = []
    for i, rec in enumerate(records):
        fpath = os.path.join(OUTPUTS_DIR, rec["file"])
        prompt = PROMPT_PAIRS[rec["arc"]][rec["condition"]]

        # Extract first frame
        image = extract_first_frame(fpath)

        # CLIP forward pass
        inputs = processor(text=[prompt], images=image, return_tensors="pt",
                           padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)

        # Cosine similarity (CLIP logits_per_image already is dot product
        # of normalised embeddings scaled by temperature)
        # We compute raw cosine similarity from the embeddings directly.
        img_emb = outputs.image_embeds  # [1, 512]
        txt_emb = outputs.text_embeds   # [1, 512]
        cos_sim = torch.nn.functional.cosine_similarity(img_emb, txt_emb).item()

        rec["clip_score"] = round(cos_sim, 5)
        results.append(rec)

        if (i + 1) % 10 == 0 or (i + 1) == len(records):
            print(f"  [{i+1}/{len(records)}] {rec['file']}  CLIP={cos_sim:.4f}")

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------
    stock_scores = [r["clip_score"] for r in results if r["condition"] == "STOCK"]
    neuro_scores = [r["clip_score"] for r in results if r["condition"] == "NEURO"]

    mean_stock = float(np.mean(stock_scores)) if stock_scores else 0.0
    mean_neuro = float(np.mean(neuro_scores)) if neuro_scores else 0.0

    # Per-arc breakdown
    arc_scores = defaultdict(lambda: {"STOCK": [], "NEURO": []})
    for r in results:
        arc_scores[r["arc"]][r["condition"]].append(r["clip_score"])

    per_arc = {}
    for arc in KNOWN_ARCS:
        s = arc_scores[arc]
        per_arc[arc] = {
            "STOCK_mean": round(float(np.mean(s["STOCK"])), 5) if s["STOCK"] else None,
            "NEURO_mean": round(float(np.mean(s["NEURO"])), 5) if s["NEURO"] else None,
            "STOCK_n": len(s["STOCK"]),
            "NEURO_n": len(s["NEURO"]),
        }

    # ------------------------------------------------------------------
    # Build output
    # ------------------------------------------------------------------
    output = {
        "model": model_name,
        "num_files": len(results),
        "mean_clip_score": {
            "STOCK": round(mean_stock, 5),
            "NEURO": round(mean_neuro, 5),
            "delta": round(mean_neuro - mean_stock, 5),
        },
        "per_arc": per_arc,
        "per_file": [
            {
                "arc": r["arc"],
                "condition": r["condition"],
                "seed": r["seed"],
                "clip_score": r["clip_score"],
            }
            for r in results
        ],
    }

    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {OUT_JSON}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  GRAIL-V  CLIP Image-Text Similarity  --  Summary")
    print("=" * 72)
    print(f"\n  Model : {model_name}")
    print(f"  Files : {len(results)}")

    print(f"\n  {'Condition':<12} {'Mean CLIP':>10}  {'N':>4}")
    print(f"  {'-'*12} {'-'*10}  {'-'*4}")
    print(f"  {'STOCK':<12} {mean_stock:10.5f}  {len(stock_scores):4d}")
    print(f"  {'NEURO':<12} {mean_neuro:10.5f}  {len(neuro_scores):4d}")
    delta = mean_neuro - mean_stock
    direction = "higher" if delta > 0 else "lower"
    print(f"  {'DELTA':<12} {delta:+10.5f}  (NEURO is {abs(delta):.4f} {direction})")

    print(f"\n  --- Per-Arc Breakdown ---")
    print(f"  {'Arc':<28} {'STOCK':>8} {'NEURO':>8} {'Delta':>8}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8}")
    for arc in KNOWN_ARCS:
        s = per_arc[arc]
        sm = s["STOCK_mean"] if s["STOCK_mean"] is not None else 0.0
        nm = s["NEURO_mean"] if s["NEURO_mean"] is not None else 0.0
        d = nm - sm
        print(f"  {arc:<28} {sm:8.5f} {nm:8.5f} {d:+8.5f}")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
