#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
LPIPS Metrics for Neuromorphic Benchmark
========================================
Computes frame-level LPIPS between the STOCK and NEURO outputs of each
matched pair.  FVD is not computed here (the paper reports LPIPS and file
size only); the module name is kept because README.md and CONTRIBUTING.md
reference it as an entry point.

Paths come from grail_config (GRAIL_BENCHMARK_DIR / GRAIL_DATA_DIR).
"""

import os
import glob
import json
from collections import defaultdict

import numpy as np

import grail_config

BENCHMARK_DIR = grail_config.benchmark_dir()
RESULTS_FILE = grail_config.data_file("lpips_results.json")

# Renders are saved by ComfyUI as <prefix>_<clip index>_.webp.  Only the first
# clip of each seed is measured, matching compute_clip_image_text.py.
CLIP_INDEX = os.environ.get("GRAIL_CLIP_INDEX", "00001")


def load_lpips_model():
    """Load the LPIPS network.

    Imported lazily so the module can be imported (and its pair-matching
    logic tested) without torch/lpips installed and without downloading
    AlexNet weights.
    """
    import torch
    import lpips

    print("Loading LPIPS model (AlexNet)...")
    model = lpips.LPIPS(net='alex')
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using CUDA")
    else:
        print("Using CPU")
    return model


def load_webp_frames(webp_path, max_frames=24):
    """Load frames from animated WEBP"""
    from PIL import Image

    frames = []
    try:
        # Use webp library for animated webp
        import webp
        webp_data = webp.load_images(webp_path)
        for i, frame in enumerate(webp_data):
            if i >= max_frames:
                break
            # Convert to RGB if needed
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')
            frames.append(frame)
    except Exception as e:
        print(f"  Error loading {webp_path}: {e}")
        # Fallback: try PIL
        try:
            img = Image.open(webp_path)
            for i in range(max_frames):
                try:
                    img.seek(i)
                    frame = img.copy()
                    if frame.mode != 'RGB':
                        frame = frame.convert('RGB')
                    frames.append(frame)
                except EOFError:
                    break
        except Exception as e2:
            print(f"  PIL fallback failed: {e2}")

    return frames


def frames_to_tensor(frames):
    """Convert list of PIL images to tensor for LPIPS"""
    from PIL import Image
    import torch

    tensors = []
    for frame in frames:
        # Resize to 256x256 for LPIPS
        frame = frame.resize((256, 256), Image.LANCZOS)
        # Convert to tensor: [C, H, W] in range [-1, 1]
        arr = np.array(frame).astype(np.float32) / 255.0
        arr = arr * 2 - 1  # Scale to [-1, 1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        tensors.append(tensor)

    return torch.stack(tensors)  # [N, C, H, W]


def compute_lpips_pair(stock_path, neuro_path, loss_fn=None):
    """Compute LPIPS between STOCK and NEURO video pair"""
    import torch

    if loss_fn is None:
        loss_fn = load_lpips_model()

    stock_frames = load_webp_frames(stock_path)
    neuro_frames = load_webp_frames(neuro_path)

    if not stock_frames or not neuro_frames:
        return None

    # Use minimum frame count
    n_frames = min(len(stock_frames), len(neuro_frames))
    stock_frames = stock_frames[:n_frames]
    neuro_frames = neuro_frames[:n_frames]

    stock_tensor = frames_to_tensor(stock_frames)
    neuro_tensor = frames_to_tensor(neuro_frames)

    if torch.cuda.is_available():
        stock_tensor = stock_tensor.cuda()
        neuro_tensor = neuro_tensor.cuda()

    # Compute LPIPS for each frame pair
    lpips_scores = []
    with torch.no_grad():
        for i in range(n_frames):
            d = loss_fn(stock_tensor[i:i+1], neuro_tensor[i:i+1])
            lpips_scores.append(d.item())

    return {
        "mean": np.mean(lpips_scores),
        "std": np.std(lpips_scores),
        "min": np.min(lpips_scores),
        "max": np.max(lpips_scores),
        "n_frames": n_frames
    }


def find_pairs(benchmark_dir=None, clip_index=None):
    """Find STOCK/NEURO pairs in benchmark outputs.

    Only the clip whose index matches ``clip_index`` is considered.  A seed
    rendered as several clips would otherwise collapse onto one dict key and
    silently keep whichever file glob() happened to return last, so the two
    analyses in this repo could measure different renders.
    """
    benchmark_dir = benchmark_dir or BENCHMARK_DIR
    clip_index = clip_index or CLIP_INDEX
    pairs = defaultdict(dict)
    skipped_other_clips = 0

    for f in sorted(glob.glob(os.path.join(benchmark_dir, "BENCH_*.webp"))):
        basename = os.path.basename(f)
        # Parse: BENCH_<test_name>_<TYPE>_s<seed>_<clip index>_.webp
        parts = basename.replace(".webp", "").split("_")

        if f"_{clip_index}_" not in basename:
            skipped_other_clips += 1
            continue

        # Find STOCK or NEURO
        for condition, slot in (("STOCK", "stock"), ("NEURO", "neuro")):
            if condition not in parts:
                continue
            idx = parts.index(condition)
            test_name = "_".join(parts[1:idx])
            seed = parts[idx + 1]
            key = f"{test_name}_{seed}"
            if slot in pairs[key]:
                print(f"  WARN: duplicate {condition} render for {key}: "
                      f"keeping {os.path.basename(pairs[key][slot])}, "
                      f"ignoring {basename}")
                break
            pairs[key][slot] = f
            break

    if skipped_other_clips:
        print(f"  Skipped {skipped_other_clips} file(s) that are not clip "
              f"_{clip_index}_ (set GRAIL_CLIP_INDEX to change)")

    # Filter to complete pairs
    complete = {k: v for k, v in pairs.items() if "stock" in v and "neuro" in v}
    return complete


def main():
    grail_config.require_dir(BENCHMARK_DIR, "GRAIL_BENCHMARK_DIR")

    print("Finding STOCK/NEURO pairs...")
    pairs = find_pairs()
    print(f"Found {len(pairs)} complete pairs")
    if not pairs:
        raise SystemExit(
            f"ERROR: no complete STOCK/NEURO pairs in {BENCHMARK_DIR}"
        )

    loss_fn = load_lpips_model()

    results = {}
    by_arc = defaultdict(list)

    for i, (key, paths) in enumerate(sorted(pairs.items())):
        print(f"\n[{i+1}/{len(pairs)}] Processing {key}...")

        lpips_result = compute_lpips_pair(paths["stock"], paths["neuro"], loss_fn)

        if lpips_result:
            results[key] = lpips_result
            print(f"  LPIPS: {lpips_result['mean']:.4f} (+/- {lpips_result['std']:.4f})")

            # Extract arc from key
            arc = key.rsplit("_", 1)[0]  # Remove seed
            by_arc[arc].append(lpips_result["mean"])

    # Summary
    print("\n" + "=" * 60)
    print("LPIPS SUMMARY")
    print("=" * 60)

    all_means = [r["mean"] for r in results.values()]
    print(f"\nOverall LPIPS: {np.mean(all_means):.4f} (+/- {np.std(all_means):.4f})")
    print(f"  Range: {np.min(all_means):.4f} - {np.max(all_means):.4f}")

    print("\nBy Test Case:")
    for arc, scores in sorted(by_arc.items()):
        print(f"  {arc}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

    # Save results
    output = {
        "summary": {
            "overall_mean": float(np.mean(all_means)),
            "overall_std": float(np.std(all_means)),
            "n_pairs": len(results)
        },
        "by_test": {arc: {"mean": float(np.mean(scores)), "std": float(np.std(scores))}
                    for arc, scores in by_arc.items()},
        "per_pair": results
    }

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {RESULTS_FILE}")

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("""
LPIPS measures perceptual distance (lower = more similar).
- LPIPS < 0.1: Very similar (nearly identical)
- LPIPS 0.1-0.3: Similar (minor differences)
- LPIPS 0.3-0.5: Moderately different
- LPIPS > 0.5: Very different

For STOCK vs NEURO comparison:
- Low LPIPS = NEURO produces similar perceptual quality with fewer steps
- High LPIPS = NEURO produces different (potentially more expressive) output
    """)


if __name__ == "__main__":
    main()
