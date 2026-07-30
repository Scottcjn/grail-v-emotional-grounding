#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Shared paths and endpoints for the GRAIL-V analysis scripts.
===========================================================
Every location the analysis scripts read or write is resolved here, with a
repo-relative default and an environment-variable override, so the documented
reproduction steps in README.md run outside the original author's machine.

Environment variables
---------------------
GRAIL_BENCHMARK_DIR   Directory holding the rendered BENCH_*.webp clips.
                      Default: <repo>/benchmark_results/outputs
GRAIL_DATA_DIR        Where computed metric JSON is written.
                      Default: <repo>/data
GRAIL_FIGURES_DIR     Where generated figures are written.
                      Default: <repo>/figures
GRAIL_SWEEP_DIR       Scratch directory for the steps-vs-LPIPS sweep renders.
                      Default: <repo>/steps_sweep_renders
GRAIL_COMFYUI_SERVER  ComfyUI endpoint used by the render pipelines.
                      Default: http://127.0.0.1:8188
GRAIL_SOURCE_IMAGE_DIR  Directory holding the benchmark source images.
                      Default: <repo>/source_images
"""

import os

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CODE_DIR)

DEFAULTS = {
    "GRAIL_BENCHMARK_DIR": os.path.join(REPO_ROOT, "benchmark_results", "outputs"),
    "GRAIL_DATA_DIR": os.path.join(REPO_ROOT, "data"),
    "GRAIL_FIGURES_DIR": os.path.join(REPO_ROOT, "figures"),
    "GRAIL_SWEEP_DIR": os.path.join(REPO_ROOT, "steps_sweep_renders"),
    "GRAIL_COMFYUI_SERVER": "http://127.0.0.1:8188",
    "GRAIL_SOURCE_IMAGE_DIR": os.path.join(REPO_ROOT, "source_images"),
}


def setting(name: str) -> str:
    """Return an environment override, or the documented default."""
    if name not in DEFAULTS:
        raise KeyError(f"Unknown GRAIL setting: {name}")
    return os.environ.get(name) or DEFAULTS[name]


def benchmark_dir() -> str:
    return setting("GRAIL_BENCHMARK_DIR")


def data_dir() -> str:
    return setting("GRAIL_DATA_DIR")


def figures_dir() -> str:
    return setting("GRAIL_FIGURES_DIR")


def sweep_dir() -> str:
    return setting("GRAIL_SWEEP_DIR")


def comfyui_server() -> str:
    return setting("GRAIL_COMFYUI_SERVER").rstrip("/")


def source_image_dir() -> str:
    return setting("GRAIL_SOURCE_IMAGE_DIR")


def data_file(filename: str) -> str:
    """Absolute path of a metrics artifact inside the data directory."""
    return os.path.join(data_dir(), filename)


def require_dir(path: str, env_var: str) -> str:
    """Fail with an actionable message instead of a bare traceback."""
    if not os.path.isdir(path):
        raise SystemExit(
            f"ERROR: directory not found: {path}\n"
            f"       Set {env_var} to the location of your renders, e.g.\n"
            f"       {env_var}=/path/to/outputs python {os.path.basename(__file__)}"
        )
    return path
