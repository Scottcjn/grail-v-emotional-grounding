#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Steps-vs-LPIPS Convergence Sweep for GRAIL-V Camera-Ready
Renders solo portraits at multiple step counts, computes LPIPS vs 60-step reference.
"""

import json
import os
import sys
import time
import uuid

import numpy as np

import grail_config

COMFYUI_SERVER = grail_config.comfyui_server()
OUTPUT_DIR = grail_config.sweep_dir()

# Solo portrait arcs only (reviewer asked about convergence for the efficiency claim)
ARCS = {
    "sophia_realization": {
        "STOCK": "Victorian woman portrait, subtle head movement, slight smile, blinking eyes, warm lighting",
        "NEURO": "The young woman's eyes brighten with quiet realization, a knowing smile forming as inspiration takes hold, warmth spreading across her expression",
    },
    "sophia_contemplation": {
        "STOCK": "Victorian woman portrait, looking thoughtful, gentle movements, soft lighting",
        "NEURO": "Her gaze turns inward with deep contemplation, a subtle shift from curiosity to understanding, quiet wisdom settling in her features",
    },
    "sophia_determination": {
        "STOCK": "Victorian woman portrait, serious expression, focused look, slight movement",
        "NEURO": "Quiet determination hardens in her eyes, jaw setting with newfound resolve, inner fire building behind composed exterior",
    },
}

NEGATIVE = "worst quality, blurry, distorted, frozen, static, still, motionless"
SEED = 42424242
# Controlled: same guidance and max_shift for both conditions
GUIDANCE = 7.5
MAX_SHIFT = 2.05
BASE_SHIFT = 0.95

STEP_COUNTS = [10, 15, 20, 24, 30, 40, 50, 60]

# LPIPS model, loaded on first use by main() (see load_lpips_model)
loss_fn = None


def load_lpips_model():
    """Load LPIPS lazily: importing this module must not pull in torch."""
    import lpips

    print("Loading LPIPS model...")
    return lpips.LPIPS(net='alex')


def build_workflow(prompt, steps, seed, prefix):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "ltx-2-19b-dev-fp8.safetensors"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "sophia_victorian_portrait.png"}},
        "3": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": "gemma_3_12B_it_fp4_mixed.safetensors", "ckpt_name": "ltx-2-19b-dev-fp8.safetensors", "device": "default"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["3", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["3", 0]}},
        "6": {"class_type": "LTXVConditioning", "inputs": {"positive": ["4", 0], "negative": ["5", 0], "frame_rate": 24.0}},
        "7": {"class_type": "LTXVImgToVideo", "inputs": {"positive": ["6", 0], "negative": ["6", 1], "vae": ["1", 2], "image": ["2", 0], "width": 512, "height": 320, "length": 49, "batch_size": 1, "strength": 1.0}},
        "8": {"class_type": "ModelSamplingLTXV", "inputs": {"model": ["1", 0], "max_shift": MAX_SHIFT, "base_shift": BASE_SHIFT, "latent": ["7", 2]}},
        "9": {"class_type": "LTXVScheduler", "inputs": {"steps": steps, "max_shift": MAX_SHIFT, "base_shift": BASE_SHIFT, "stretch": True, "terminal": 0.1, "latent": ["7", 2]}},
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "11": {"class_type": "CFGGuider", "inputs": {"model": ["8", 0], "positive": ["7", 0], "negative": ["7", 1], "cfg": GUIDANCE}},
        "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "13": {"class_type": "LTXVBaseSampler", "inputs": {"model": ["8", 0], "vae": ["1", 2], "width": 512, "height": 320, "num_frames": 49, "guider": ["11", 0], "sampler": ["12", 0], "sigmas": ["9", 0], "noise": ["10", 0], "optional_cond_images": ["2", 0], "optional_cond_indices": "0", "strength": 0.9}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["1", 2]}},
        "15": {"class_type": "SaveAnimatedWEBP", "inputs": {"images": ["14", 0], "filename_prefix": prefix, "fps": 24.0, "lossless": False, "quality": 90, "method": "default"}}
    }


def queue_and_wait(workflow, timeout=600):
    import requests

    client_id = str(uuid.uuid4())
    resp = requests.post(f"{COMFYUI_SERVER}/prompt",
                         json={"prompt": workflow, "client_id": client_id}, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Queue failed: {resp.status_code} {resp.text}")
    prompt_id = resp.json().get("prompt_id")
    print(f"    Queued: {prompt_id}")

    start = time.time()
    while time.time() - start < timeout:
        hist = requests.get(f"{COMFYUI_SERVER}/history/{prompt_id}", timeout=10).json()
        if prompt_id in hist:
            outputs = hist[prompt_id].get("outputs", {})
            for node_id, node_out in outputs.items():
                for img_info in node_out.get("images", []):
                    return img_info["filename"], img_info.get("subfolder", "")
            return None, None
        time.sleep(3)
    raise TimeoutError(f"Render timed out after {timeout}s")


def download_output(filename, subfolder=""):
    """Fetch a rendered clip from ComfyUI.

    ``/view`` answers a missing or expired file with an HTTP error whose body
    is a short text page.  Writing that body to ``<name>.webp`` unchanged
    produced a file that looks like a successful download (it even has a
    plausible byte count in the log) and only fails hours later, in the LPIPS
    stage, as an ``UnidentifiedImageError``.  Fail here instead.
    """
    import requests

    url = f"{COMFYUI_SERVER}/view?filename={filename}"
    if subfolder:
        url += f"&subfolder={subfolder}"
    url += "&type=output"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    content = resp.content
    # RIFF....WEBP -- cheap magic-number check; a proxy or an error page that
    # returns HTTP 200 would otherwise sail straight through.
    if len(content) < 12 or content[:4] != b"RIFF" or content[8:12] != b"WEBP":
        raise RuntimeError(
            f"{filename}: server returned {len(content)} bytes that are not a "
            f"WebP file (starts with {content[:16]!r})"
        )

    local_path = os.path.join(OUTPUT_DIR, filename)
    with open(local_path, 'wb') as f:
        f.write(content)
    return local_path


def load_frames(webp_path, max_frames=24):
    """Load frames from animated WebP; empty list if the file is unreadable.

    Both decode paths are guarded.  Previously only the ``webp`` module call
    was, so on a machine without that optional dependency -- the documented
    default -- any unreadable file raised straight out of Phase 2 and took
    the whole sweep down with it.
    """
    from PIL import Image

    frames = []
    try:
        import webp as webplib
        imgs = webplib.load_images(webp_path)
        for i, frame in enumerate(imgs[:max_frames]):
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')
            frames.append(frame)
        return frames
    except Exception:
        frames = []

    try:
        img = Image.open(webp_path)
        for i in range(max_frames):
            try:
                img.seek(i)
                frame = img.copy().convert('RGB')
                frames.append(frame)
            except EOFError:
                break
    except Exception as e:
        print(f"    WARN: cannot read {os.path.basename(webp_path)}: {e}")
        return []
    return frames


def compute_lpips(path_a, path_b):
    """Mean frame-level LPIPS between two animated WebPs."""
    import torch

    frames_a = load_frames(path_a)
    frames_b = load_frames(path_b)
    n = min(len(frames_a), len(frames_b))
    if n == 0:
        return float('nan')

    scores = []
    for i in range(n):
        fa = frames_a[i].resize((512, 320))
        fb = frames_b[i].resize((512, 320))
        ta = torch.from_numpy(np.array(fa)).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1
        tb = torch.from_numpy(np.array(fb)).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1
        with torch.no_grad():
            score = loss_fn(ta, tb).item()
        scores.append(score)
    return float(np.mean(scores))


def main():
    global loss_fn

    import requests

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    loss_fn = load_lpips_model()

    # Check server
    try:
        r = requests.get(f"{COMFYUI_SERVER}/system_stats", timeout=5)
        print(f"ComfyUI OK: {r.json()['system']['comfyui_version']}")
    except Exception as e:
        print(f"ERROR: ComfyUI not available: {e}")
        sys.exit(1)

    # Phase 1: Render all step counts
    render_paths = {}  # {(condition, arc, steps): local_path}
    render_failures = []
    total_renders = len(ARCS) * len(STEP_COUNTS) * 2
    done = 0

    for condition in ["STOCK", "NEURO"]:
        for arc_name, prompts in ARCS.items():
            prompt = prompts[condition]
            for steps in STEP_COUNTS:
                done += 1
                prefix = f"SWEEP_{arc_name}_{condition}_s{steps}"
                print(f"[{done}/{total_renders}] {condition} | {arc_name} | {steps} steps")

                try:
                    wf = build_workflow(prompt, steps, SEED, prefix)
                    filename, subfolder = queue_and_wait(wf)
                    if filename:
                        local = download_output(filename, subfolder)
                        render_paths[(condition, arc_name, steps)] = local
                        print(f"    OK: {filename} ({os.path.getsize(local)} bytes)")
                    else:
                        print(f"    WARN: No output file found")
                        render_failures.append((prefix, "no output file in history"))
                except Exception as e:
                    print(f"    FAILED: {e}")
                    render_failures.append((prefix, f"{type(e).__name__}: {e}"))

    print(f"\nRendered {len(render_paths)}/{total_renders} clips.")
    if render_failures:
        print(f"{len(render_failures)} render(s) failed:")
        for prefix, reason in render_failures[:10]:
            print(f"  - {prefix}: {reason}")
        if len(render_failures) > 10:
            print(f"  ... and {len(render_failures) - 10} more")
    if not render_paths:
        raise SystemExit(
            "ERROR: no clips were rendered -- nothing to measure. "
            "Check that ComfyUI has the LTX-2 checkpoint and the source image."
        )

    # Phase 2: Compute LPIPS vs 60-step reference
    print("\n" + "=" * 60)
    print("Computing LPIPS vs 60-step reference...")
    print("=" * 60)

    results = {"STOCK": {}, "NEURO": {}}

    for condition in ["STOCK", "NEURO"]:
        for steps in STEP_COUNTS:
            lpips_scores = []
            for arc_name in ARCS:
                ref_key = (condition, arc_name, 60)
                test_key = (condition, arc_name, steps)

                if ref_key not in render_paths or test_key not in render_paths:
                    print(f"  SKIP: {condition} {arc_name} {steps}→60 (missing render)")
                    continue

                if steps == 60:
                    lpips_scores.append(0.0)
                    continue

                score = compute_lpips(render_paths[test_key], render_paths[ref_key])
                if np.isnan(score):
                    print(f"  SKIP: {condition} {arc_name} {steps}→60 "
                          f"(no readable frames)")
                    continue
                lpips_scores.append(score)
                print(f"  {condition} {arc_name} {steps}→60: LPIPS={score:.4f}")

            if lpips_scores:
                results[condition][steps] = {
                    "mean": float(np.mean(lpips_scores)),
                    # Sample std: these are len(ARCS) arcs drawn as a sample,
                    # and every dispersion figure in the paper is a sample
                    # std.  ddof=0 here reported a systematically smaller
                    # spread than the published numbers.
                    "std": float(np.std(lpips_scores, ddof=1)) if len(lpips_scores) > 1 else 0.0,
                    "n": len(lpips_scores),
                    "scores": lpips_scores,
                }

    # Save results
    results_path = os.path.join(OUTPUT_DIR, "steps_vs_lpips_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Phase 3: Generate figure
    print("\nGenerating figure...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    for condition, color, marker, label in [
        ("STOCK", "#2196F3", "o", "STOCK (literal)"),
        ("NEURO", "#E91E63", "s", "NEURO (emotional)"),
    ]:
        steps_list = sorted(results[condition].keys())
        means = [results[condition][s]["mean"] for s in steps_list]
        stds = [results[condition][s]["std"] for s in steps_list]
        ax.errorbar(steps_list, means, yerr=stds, marker=marker, label=label,
                     color=color, capsize=3, linewidth=2, markersize=6)

    ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='Equivalence threshold')
    ax.set_xlabel('Diffusion Steps', fontsize=12)
    ax.set_ylabel('LPIPS vs. 60-step Reference', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xticks(STEP_COUNTS)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    figures_dir = grail_config.figures_dir()
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = os.path.join(figures_dir, "fig_steps_vs_lpips.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"Figure saved: {fig_path}")

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Steps':>6} | {'STOCK LPIPS':>12} | {'NEURO LPIPS':>12} | {'Delta':>8}")
    print("-" * 60)
    for steps in STEP_COUNTS:
        s = results.get("STOCK", {}).get(steps, {})
        n = results.get("NEURO", {}).get(steps, {})
        sm = s.get("mean", float('nan'))
        nm = n.get("mean", float('nan'))
        delta = nm - sm if not (np.isnan(sm) or np.isnan(nm)) else float('nan')
        print(f"{steps:>6} | {sm:>12.4f} | {nm:>12.4f} | {delta:>+8.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
