#!/usr/bin/env python3
"""
LPIPS and FVD Metrics for Neuromorphic Benchmark
=================================================
Computes perceptual quality metrics between STOCK and NEURO outputs.
"""

import os
import glob
import json
import numpy as np
from PIL import Image
import torch
import lpips
from collections import defaultdict
import webp

BENCHMARK_DIR = "/home/scott/grail_paper/benchmark_results/outputs"
RESULTS_FILE = "/home/scott/grail_paper/metrics/lpips_results.json"

# Initialize LPIPS model
print("Loading LPIPS model (AlexNet)...")
loss_fn = lpips.LPIPS(net='alex')
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
    print("Using CUDA")
else:
    print("Using CPU")


def load_webp_frames(webp_path, max_frames=24):
    """Load frames from animated WEBP"""
    frames = []
    try:
        # Use webp library for animated webp
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


def compute_lpips_pair(stock_path, neuro_path):
    """Compute LPIPS between STOCK and NEURO video pair"""
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


def find_pairs():
    """Find STOCK/NEURO pairs in benchmark outputs"""
    pairs = defaultdict(dict)

    for f in glob.glob(os.path.join(BENCHMARK_DIR, "BENCH_*.webp")):
        basename = os.path.basename(f)
        # Parse: BENCH_<test_name>_<TYPE>_s<seed>_00001_.webp
        parts = basename.replace(".webp", "").split("_")

        # Find STOCK or NEURO
        if "STOCK" in parts:
            idx = parts.index("STOCK")
            test_name = "_".join(parts[1:idx])
            seed = parts[idx + 1]
            key = f"{test_name}_{seed}"
            pairs[key]["stock"] = f
        elif "NEURO" in parts:
            idx = parts.index("NEURO")
            test_name = "_".join(parts[1:idx])
            seed = parts[idx + 1]
            key = f"{test_name}_{seed}"
            pairs[key]["neuro"] = f

    # Filter to complete pairs
    complete = {k: v for k, v in pairs.items() if "stock" in v and "neuro" in v}
    return complete


def main():
    print("Finding STOCK/NEURO pairs...")
    pairs = find_pairs()
    print(f"Found {len(pairs)} complete pairs")

    results = {}
    by_arc = defaultdict(list)

    for i, (key, paths) in enumerate(sorted(pairs.items())):
        print(f"\n[{i+1}/{len(pairs)}] Processing {key}...")

        lpips_result = compute_lpips_pair(paths["stock"], paths["neuro"])

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
