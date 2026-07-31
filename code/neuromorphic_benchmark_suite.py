#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Neuromorphic Benchmark Suite
============================
Systematic A/B testing for GRAIL-V paper statistical validation.

Test Matrix (per Grok recommendation):
- 3-4 different characters/subjects
- 4-5 emotional arcs
- 3-5 source images
- 5 seeds per test (for variance measurement)

Goal: Demonstrate 20% efficiency gain with statistical significance.
"""

import os
import json
import time
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Tuple
from datetime import datetime

import grail_config
from neuromorphic_prompt_translator import CognitiveFunction

try:
    # Author's private helper; not part of this repository.
    from seed_scaffolding import CognitiveSeedScaffolder
    HAVE_SEED_SCAFFOLDING = True
except ModuleNotFoundError:
    CognitiveSeedScaffolder = None
    HAVE_SEED_SCAFFOLDING = False

COMFYUI_SERVER = grail_config.comfyui_server()
# The manifest belongs next to the renders it describes, i.e. inside
# GRAIL_BENCHMARK_DIR -- which is also where run_lpips_fvd.py and
# compute_clip_image_text.py look for BENCH_*.webp.
OUTPUT_DIR = grail_config.benchmark_dir()

# Seed sequence of the published runs.  Every filename in
# data/clip_image_text_scores.json carries its seed, and the five seeds used
# there are BASE_SEED + 1000 * i, i = 0..4.  Used when seed_scaffolding is
# unavailable so the released suite reproduces the published seeds.
BASE_SEED = 42424242
SEED_STRIDE = 1000


def published_seeds(num_seeds: int, base_seed: int = BASE_SEED) -> List[int]:
    """Deterministic seeds matching the renders behind the paper's numbers."""
    return [base_seed + SEED_STRIDE * i for i in range(num_seeds)]


def source_image(filename: str) -> str:
    """Absolute path of a benchmark source image (GRAIL_SOURCE_IMAGE_DIR)."""
    return os.path.join(grail_config.source_image_dir(), filename)


def benchmark_seeds(num_seeds: int, base_seed: int = BASE_SEED) -> List[int]:
    """Scaffolded seeds when available, otherwise the published sequence."""
    if HAVE_SEED_SCAFFOLDING:
        scaffolder = CognitiveSeedScaffolder(base_seed=base_seed)
        scaffold = scaffolder.generate_scaffold(
            CognitiveFunction.LANGUAGE, num_frames=num_seeds * 10
        )
        return list(scaffold.frame_seeds[:num_seeds])
    return published_seeds(num_seeds, base_seed)


@dataclass
class TestCase:
    """Single test case configuration"""
    name: str
    image_path: str
    stock_prompt: str
    neuro_prompt: str
    subject_type: str  # woman, man, child, etc.
    emotional_arc: str  # realization, defiance, warmth, etc.


# =============================================================================
# TEST CASES
# =============================================================================

TEST_CASES = [
    # --- SOPHIA VICTORIAN (Woman, Portrait) ---
    TestCase(
        name="sophia_realization",
        image_path=source_image("sophia_victorian_frame.png"),
        stock_prompt="Victorian woman portrait, subtle head movement, slight smile, blinking eyes, warm lighting",
        neuro_prompt="The young woman's eyes brighten with quiet realization, a knowing smile forming as inspiration takes hold, warmth spreading across her expression",
        subject_type="woman",
        emotional_arc="realization"
    ),
    TestCase(
        name="sophia_contemplation",
        image_path=source_image("sophia_victorian_frame.png"),
        stock_prompt="Victorian woman portrait, looking thoughtful, gentle movements, soft lighting",
        neuro_prompt="Her gaze turns inward with deep contemplation, a subtle shift from curiosity to understanding, quiet wisdom settling in her features",
        subject_type="woman",
        emotional_arc="contemplation"
    ),
    TestCase(
        name="sophia_determination",
        image_path=source_image("sophia_victorian_frame.png"),
        stock_prompt="Victorian woman portrait, serious expression, focused look, slight movement",
        neuro_prompt="Quiet determination hardens in her eyes, jaw setting with newfound resolve, inner fire building behind composed exterior",
        subject_type="woman",
        emotional_arc="determination"
    ),

    # --- ELYAN LABS (Two characters - test sequential arcs) ---
    TestCase(
        name="elyan_sophia_focus",
        image_path=source_image("sophia_elyan_labs.png"),
        stock_prompt="Victorian exhibition, woman working on machine, man watching, gaslight flickering",
        neuro_prompt="The young woman works with fierce concentration, confident hands moving with purpose, quiet authority radiating as she masters the brass machinery",
        subject_type="woman_focus",
        emotional_arc="confidence"
    ),
    TestCase(
        name="elyan_claude_focus",
        image_path=source_image("sophia_elyan_labs.png"),
        stock_prompt="Victorian exhibition, older man gesturing, woman at machine, warm lighting",
        neuro_prompt="The older gentleman's skepticism softens to grudging respect, pride wounded but giving way to reluctant admiration",
        subject_type="man_focus",
        emotional_arc="respect"
    ),

    # --- DEBATE SCENE (Dynamic interaction) ---
    TestCase(
        name="debate_passion",
        image_path=source_image("sophia_claude_i2v_debate_preview.png"),
        stock_prompt="Two people in conversation, gesturing, fireplace glowing, Victorian study",
        neuro_prompt="Passionate intellectual exchange, conviction burning in their eyes, the electricity of clashing ideas filling the air between them",
        subject_type="interaction",
        emotional_arc="passion"
    ),
    TestCase(
        name="debate_tension",
        image_path=source_image("sophia_claude_i2v_debate_preview.png"),
        stock_prompt="Two people talking, subtle movements, warm firelight, period room",
        neuro_prompt="Tension crackling between them, unspoken challenge in their gazes, the air thick with intellectual rivalry",
        subject_type="interaction",
        emotional_arc="tension"
    ),
]

# Emotional arc vocabulary for analysis
EMOTIONAL_ARCS = {
    "realization": ["brighten", "dawning", "understanding", "clarity", "inspiration"],
    "contemplation": ["inward", "thoughtful", "wisdom", "quiet", "settling"],
    "determination": ["resolve", "fire", "hardening", "strength", "purpose"],
    "confidence": ["authority", "mastery", "assured", "commanding", "power"],
    "respect": ["softening", "admiration", "reluctant", "acknowledging", "yielding"],
    "passion": ["burning", "conviction", "electricity", "intensity", "fire"],
    "tension": ["crackling", "challenge", "rivalry", "charged", "unspoken"],
}


def build_workflow(prompt: str, negative: str, params: dict, image_path: str, prefix: str) -> dict:
    """Build LTX-2 I2V workflow"""
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "ltx-2-19b-dev-fp8.safetensors"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": os.path.basename(image_path)}},
        "3": {"class_type": "LTXAVTextEncoderLoader", "inputs": {"text_encoder": "gemma_3_12B_it_fp4_mixed.safetensors", "ckpt_name": "ltx-2-19b-dev-fp8.safetensors", "device": "default"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["3", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["3", 0]}},
        "6": {"class_type": "LTXVConditioning", "inputs": {"positive": ["4", 0], "negative": ["5", 0], "frame_rate": 24.0}},
        "7": {"class_type": "LTXVImgToVideo", "inputs": {"positive": ["6", 0], "negative": ["6", 1], "vae": ["1", 2], "image": ["2", 0], "width": params["width"], "height": params["height"], "length": params["frames"], "batch_size": 1, "strength": params.get("img_strength", 1.0)}},
        "8": {"class_type": "ModelSamplingLTXV", "inputs": {"model": ["1", 0], "max_shift": params["max_shift"], "base_shift": params["base_shift"], "latent": ["7", 2]}},
        "9": {"class_type": "LTXVScheduler", "inputs": {"steps": params["steps"], "max_shift": params["max_shift"], "base_shift": params["base_shift"], "stretch": True, "terminal": params.get("terminal", 0.1), "latent": ["7", 2]}},
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": params["seed"]}},
        "11": {"class_type": "CFGGuider", "inputs": {"model": ["8", 0], "positive": ["7", 0], "negative": ["7", 1], "cfg": params["guidance_scale"]}},
        "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "13": {"class_type": "LTXVBaseSampler", "inputs": {"model": ["8", 0], "vae": ["1", 2], "width": params["width"], "height": params["height"], "num_frames": params["frames"], "guider": ["11", 0], "sampler": ["12", 0], "sigmas": ["9", 0], "noise": ["10", 0], "optional_cond_images": ["2", 0], "optional_cond_indices": "0", "strength": params.get("denoise_strength", 0.9)}},
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["1", 2]}},
        "15": {"class_type": "SaveAnimatedWEBP", "inputs": {"images": ["14", 0], "filename_prefix": prefix, "fps": 24.0, "lossless": False, "quality": 90, "method": "default"}}
    }


def upload_image(image_path: str) -> Tuple[bool, str]:
    """Upload image to ComfyUI.  Returns (ok, detail)."""
    import requests

    try:
        with open(image_path, 'rb') as f:
            files = {'image': (os.path.basename(image_path), f, 'image/png')}
            resp = requests.post(f"{COMFYUI_SERVER}/upload/image", files=files, timeout=30)
    except OSError as e:
        return False, f"cannot read {image_path}: {e}"
    except Exception as e:                      # requests.RequestException et al.
        return False, f"{type(e).__name__}: {e}"

    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
    return True, ""


def describe_queue_error(status_code: int, payload) -> str:
    """Turn a ComfyUI /prompt rejection into one actionable line.

    ComfyUI answers a rejected job with HTTP 400 and a body carrying both a
    summary (``error``) and the offending inputs (``node_errors``); the node
    detail is what tells the user *which* checkpoint or image is missing, so
    it is worth surfacing rather than discarding.
    """
    if not isinstance(payload, dict):
        return f"HTTP {status_code}: {str(payload)[:200]}"

    error = payload.get("error")
    if isinstance(error, dict):
        head = error.get("message") or error.get("type") or "rejected"
        if error.get("details"):
            head = f"{head} ({error['details']})"
    elif error:
        head = str(error)
    else:
        head = f"HTTP {status_code}"

    details = []
    node_errors = payload.get("node_errors")
    if isinstance(node_errors, dict):
        for node_id, node in node_errors.items():
            for err in (node or {}).get("errors", []) if isinstance(node, dict) else []:
                details.append(
                    f"node {node_id}: {err.get('message', '')} {err.get('details', '')}".strip()
                )
    if details:
        head = f"{head}; " + "; ".join(details[:3])
    return head


def queue_prompt(workflow: dict) -> Tuple[str, float, str]:
    """Queue a prompt.

    Returns ``(prompt_id, queue_time, error)``.  ``prompt_id`` is non-empty
    only when ComfyUI actually accepted the job; on any failure ``error``
    carries the reason and ``prompt_id`` stays empty.  The error text is kept
    out of the prompt_id slot on purpose: it used to be returned *as* the
    prompt id, so a manifest of failures was indistinguishable from a
    manifest of queued renders.
    """
    import requests

    start = time.time()
    try:
        resp = requests.post(f"{COMFYUI_SERVER}/prompt", json={"prompt": workflow}, timeout=30)
    except Exception as e:
        return "", time.time() - start, f"{type(e).__name__}: {e}"

    elapsed = time.time() - start
    try:
        data = resp.json()
    except ValueError:
        return "", elapsed, f"HTTP {resp.status_code}: non-JSON response"

    if resp.status_code != 200:
        return "", elapsed, describe_queue_error(resp.status_code, data)

    prompt_id = (data or {}).get("prompt_id") or ""
    if not prompt_id:
        # 200 with no prompt_id still means nothing was scheduled.
        return "", elapsed, describe_queue_error(resp.status_code, data)
    return prompt_id, elapsed, ""


def run_single_test(test: TestCase, seed: int, test_type: str) -> dict:
    """Run a single STOCK or NEURO test"""

    negative = "worst quality, blurry, distorted, frozen, static, motionless, deformed"

    if test_type == "stock":
        prompt = test.stock_prompt
        params = {
            "guidance_scale": 7.5,
            "steps": 30,
            "width": 512, "height": 320, "frames": 49,  # ~2 seconds
            "max_shift": 2.05,
            "base_shift": 0.95,
            "terminal": 0.1,
            "img_strength": 1.0,
            "denoise_strength": 0.9,
            "seed": seed
        }
        prefix = f"BENCH_{test.name}_STOCK_s{seed}"
    else:
        prompt = test.neuro_prompt
        params = {
            "guidance_scale": 8.0,
            "steps": 24,  # 20% fewer
            "width": 512, "height": 320, "frames": 49,
            "max_shift": 2.10,
            "base_shift": 0.98,
            "terminal": 0.1,
            "img_strength": 1.0,
            "denoise_strength": 0.9,
            "seed": seed
        }
        prefix = f"BENCH_{test.name}_NEURO_s{seed}"

    workflow = build_workflow(prompt, negative, params, test.image_path, prefix)
    prompt_id, queue_time, error = queue_prompt(workflow)

    return {
        "test_name": test.name,
        "test_type": test_type,
        "seed": seed,
        "steps": params["steps"],
        "prompt_id": prompt_id,
        "queued": bool(prompt_id),
        "error": error,
        "queue_time": queue_time,
        "prefix": prefix,
        "subject_type": test.subject_type,
        "emotional_arc": test.emotional_arc
    }


def run_benchmark_suite(num_seeds: int = 3, tests_to_run: List[str] = None):
    """Run the full benchmark suite"""

    print("=" * 70)
    print("NEUROMORPHIC BENCHMARK SUITE")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Seeds per test: {num_seeds}")
    print(f"Test cases: {len(TEST_CASES)}")
    print(f"Total runs: {len(TEST_CASES) * num_seeds * 2} (stock + neuro)")
    print("=" * 70)

    # Generate seeds (cognitive scaffolding if the helper is installed,
    # otherwise the published seed sequence)
    test_seeds = benchmark_seeds(num_seeds)
    print(f"Seed source: "
          f"{'seed_scaffolding' if HAVE_SEED_SCAFFOLDING else 'published sequence'}")
    print(f"Seeds: {test_seeds}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"ComfyUI: {COMFYUI_SERVER}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    images_uploaded = set()
    skipped_tests = []

    for i, test in enumerate(TEST_CASES):
        if tests_to_run and test.name not in tests_to_run:
            continue

        print(f"\n[{i+1}/{len(TEST_CASES)}] {test.name}")
        print(f"    Subject: {test.subject_type} | Arc: {test.emotional_arc}")

        # Upload image if needed
        if test.image_path not in images_uploaded:
            print(f"    Uploading: {os.path.basename(test.image_path)}...")
            ok, detail = upload_image(test.image_path)
            if ok:
                images_uploaded.add(test.image_path)
                print("    Done")
            else:
                print(f"    UPLOAD FAILED - skipping test: {detail}")
                skipped_tests.append((test.name, detail))
                continue

        # Run tests for each seed
        for seed_idx, seed in enumerate(test_seeds):
            for test_type, steps in (("stock", 30), ("neuro", 24)):
                print(f"    [{seed_idx+1}/{num_seeds}] {test_type.upper()} "
                      f"({steps} steps, seed={seed})...", end=" ")
                result = run_single_test(test, seed, test_type)
                results.append(result)
                if result["queued"]:
                    print(f"queued: {result['prompt_id'][:8]}...")
                else:
                    print(f"NOT QUEUED: {result['error']}")

    queued = [r for r in results if r["queued"]]
    failed = [r for r in results if not r["queued"]]

    # Save results manifest
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "num_seeds": num_seeds,
        "test_cases": len(TEST_CASES),
        "total_runs": len(results),
        "queued_runs": len(queued),
        "failed_runs": len(failed),
        "skipped_tests": [{"test_name": n, "error": d} for n, d in skipped_tests],
        "seeds_used": test_seeds,
        "results": results
    }

    manifest_path = os.path.join(OUTPUT_DIR, f"benchmark_manifest_{int(time.time())}.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 70)
    print("BENCHMARK QUEUED" if queued and not failed else "BENCHMARK INCOMPLETE")
    print("=" * 70)
    print(f"Queued : {len(queued)}/{len(results)} "
          f"(stock {len([r for r in queued if r['test_type'] == 'stock'])}, "
          f"neuro {len([r for r in queued if r['test_type'] == 'neuro'])})")
    if skipped_tests:
        print(f"Skipped: {len(skipped_tests)} test case(s) whose source image "
              f"could not be uploaded")
        for name, detail in skipped_tests:
            print(f"         - {name}: {detail}")
    if failed:
        print(f"REJECTED: {len(failed)} run(s) -- ComfyUI did not schedule them")
        # One line per distinct reason: 70 identical "checkpoint missing"
        # lines say no more than one does.
        reasons = Counter(r["error"] for r in failed)
        for reason, count in reasons.most_common():
            print(f"         x{count}  {reason}")
    print(f"Manifest: {manifest_path}")
    print("\nStep comparison:")
    print("  STOCK: 30 steps")
    print("  NEURO: 24 steps (20% fewer)")

    if not queued:
        print("\nNOTHING WAS QUEUED -- no renders will appear and the analysis "
              "scripts will find no pairs. Fix the errors above and re-run.")
    elif failed or skipped_tests:
        print("\nPartial run: the analysis scripts will see fewer than the "
              "published 35 matched pairs.")
    else:
        print("\nWait for ComfyUI to complete, then run analysis.")

    return manifest


def quick_test(num_tests: int = 2, num_seeds: int = 2):
    """Quick test with subset for validation"""
    print("QUICK TEST MODE")
    print(f"Running {num_tests} tests with {num_seeds} seeds each\n")

    test_names = [t.name for t in TEST_CASES[:num_tests]]
    return run_benchmark_suite(num_seeds=num_seeds, tests_to_run=test_names)


def exit_code(manifest: dict) -> int:
    """0 only if every run was accepted; 1 otherwise.

    The suite queues renders that a human then waits hours for, so a partial
    or empty run has to be visible to whatever ran it -- not just to whoever
    happens to read the scrollback.
    """
    if not manifest.get("queued_runs"):
        return 1
    if manifest.get("failed_runs") or manifest.get("skipped_tests"):
        return 1
    return 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        manifest = quick_test(num_tests=2, num_seeds=2)
    elif len(sys.argv) > 1 and sys.argv[1] == "full":
        manifest = run_benchmark_suite(num_seeds=5)
    else:
        print("Usage:")
        print("  python neuromorphic_benchmark_suite.py quick  # 2 tests, 2 seeds each")
        print("  python neuromorphic_benchmark_suite.py full   # All tests, 5 seeds each")
        print("\nRunning quick test by default...")
        manifest = quick_test(num_tests=2, num_seeds=2)

    sys.exit(exit_code(manifest))
