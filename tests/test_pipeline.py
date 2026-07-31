# SPDX-License-Identifier: MIT
"""
Tests for the analysis pipeline: configuration, filename handling and the
benchmark suite.

The oracles here are the committed artifacts in data/ -- the filenames,
seeds and conditions that produced the paper's numbers -- so the scripts and
the published results cannot drift apart silently.
"""

import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(REPO_ROOT, "code")
DATA_DIR = os.path.join(REPO_ROOT, "data")

import grail_config  # noqa: E402  (conftest puts code/ on sys.path)


# ---------------------------------------------------------------------------
# Import hygiene: the documented entry points must import anywhere
# ---------------------------------------------------------------------------

ENTRY_POINTS = [
    "grail_config",
    "neuromorphic_prompt_translator",
    "neuromorphic_benchmark_suite",
    "run_lpips_fvd",
    "compute_clip_image_text",
    "compute_clip_scores",
    "steps_vs_lpips_sweep",
]


@pytest.mark.parametrize("module", ENTRY_POINTS)
def test_entry_point_imports(module, tmp_path):
    """No missing modules, no model downloads, no side effects on import."""
    env = dict(os.environ)
    env["PYTHONPATH"] = CODE_DIR
    env["GRAIL_BENCHMARK_DIR"] = str(tmp_path / "outputs")
    env["GRAIL_SWEEP_DIR"] = str(tmp_path / "sweep")
    env["GRAIL_DATA_DIR"] = str(tmp_path / "data")
    proc = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True, text=True, env=env, cwd=str(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr
    # importing must not create directories or write artifacts
    assert list(tmp_path.iterdir()) == []


def test_benchmark_suite_imports_without_seed_scaffolding():
    import neuromorphic_benchmark_suite as suite

    assert suite.HAVE_SEED_SCAFFOLDING is False or suite.CognitiveSeedScaffolder
    assert suite.benchmark_seeds(3)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_defaults_are_repo_relative():
    assert grail_config.data_dir() == os.path.join(REPO_ROOT, "data")
    assert grail_config.benchmark_dir().startswith(REPO_ROOT)
    assert "/home/scott" not in grail_config.benchmark_dir()


def test_environment_overrides_win(monkeypatch, tmp_path):
    monkeypatch.setenv("GRAIL_BENCHMARK_DIR", str(tmp_path))
    monkeypatch.setenv("GRAIL_COMFYUI_SERVER", "http://render-box:8188/")
    assert grail_config.benchmark_dir() == str(tmp_path)
    assert grail_config.comfyui_server() == "http://render-box:8188"


def test_unknown_setting_is_rejected():
    with pytest.raises(KeyError):
        grail_config.setting("GRAIL_NOT_A_SETTING")


def test_require_dir_reports_the_env_var(tmp_path):
    missing = str(tmp_path / "nope")
    with pytest.raises(SystemExit) as excinfo:
        grail_config.require_dir(missing, "GRAIL_BENCHMARK_DIR")
    assert "GRAIL_BENCHMARK_DIR" in str(excinfo.value)
    assert missing in str(excinfo.value)


def test_no_source_file_hardcodes_the_author_home():
    for name in os.listdir(CODE_DIR):
        if not name.endswith(".py"):
            continue
        text = open(os.path.join(CODE_DIR, name), encoding="utf-8").read()
        assert "/home/scott" not in text, name


def test_metrics_are_written_into_the_data_directory():
    import compute_clip_image_text as clip_image
    import run_lpips_fvd as lpips_metrics

    assert os.path.dirname(clip_image.OUT_JSON) == grail_config.data_dir()
    assert os.path.dirname(lpips_metrics.RESULTS_FILE) == grail_config.data_dir()


# ---------------------------------------------------------------------------
# Filenames: suite prefix -> render -> both analyses
# ---------------------------------------------------------------------------

def published_records():
    with open(os.path.join(DATA_DIR, "clip_image_text_scores.json")) as f:
        return json.load(f)["per_file"]


def test_published_seeds_match_the_committed_results():
    import neuromorphic_benchmark_suite as suite

    published = sorted({int(r["seed"]) for r in published_records()})
    assert suite.published_seeds(len(published)) == published


def test_suite_prefix_round_trips_through_both_parsers():
    import compute_clip_image_text as clip_image
    import neuromorphic_benchmark_suite as suite
    import run_lpips_fvd as lpips_metrics

    seeds = suite.published_seeds(5)
    for case in suite.TEST_CASES:
        for condition in ("STOCK", "NEURO"):
            prefix = f"BENCH_{case.name}_{condition}_s{seeds[0]}"
            fname = f"{prefix}_00001_.webp"

            parsed = clip_image.parse_filename(fname)
            assert parsed == (case.name, condition, str(seeds[0])), fname

            parts = fname.replace(".webp", "").split("_")
            idx = parts.index(condition)
            assert "_".join(parts[1:idx]) == case.name
            assert parts[idx + 1] == f"s{seeds[0]}"
            assert lpips_metrics.CLIP_INDEX in fname


def test_every_published_filename_parses():
    import compute_clip_image_text as clip_image

    for record in published_records():
        fname = (f"BENCH_{record['arc']}_{record['condition']}"
                 f"_s{record['seed']}_00001_.webp")
        assert clip_image.parse_filename(fname) == (
            record["arc"], record["condition"], record["seed"])


@pytest.mark.parametrize("fname", [
    "BENCH_sophia_realization_STOCK_s42424242_00002_.webp",   # other clip
    "BENCH_unknown_arc_STOCK_s42424242_00001_.webp",          # unknown arc
    "sophia_realization_STOCK_s42424242_00001_.webp",         # no prefix
    "BENCH_sophia_realization_OTHER_s42424242_00001_.webp",   # bad condition
    "BENCH_sophia_realization_STOCK_sXYZ_00001_.webp",        # bad seed
])
def test_parse_filename_rejects(fname):
    import compute_clip_image_text as clip_image

    assert clip_image.parse_filename(fname) is None


def _touch(directory, name):
    path = os.path.join(directory, name)
    with open(path, "wb") as f:
        f.write(b"")
    return path


def test_find_pairs_matches_stock_and_neuro(tmp_path):
    import run_lpips_fvd as lpips_metrics

    outputs = tmp_path / "outputs"
    outputs.mkdir()
    for condition in ("STOCK", "NEURO"):
        _touch(str(outputs),
               f"BENCH_sophia_realization_{condition}_s42424242_00001_.webp")

    pairs = lpips_metrics.find_pairs(str(outputs))
    assert list(pairs) == ["sophia_realization_s42424242"]
    assert set(pairs["sophia_realization_s42424242"]) == {"stock", "neuro"}


def test_find_pairs_drops_incomplete_pairs(tmp_path):
    import run_lpips_fvd as lpips_metrics

    outputs = tmp_path / "outputs"
    outputs.mkdir()
    _touch(str(outputs), "BENCH_debate_tension_STOCK_s42424242_00001_.webp")
    assert lpips_metrics.find_pairs(str(outputs)) == {}


def test_find_pairs_ignores_other_clips_of_the_same_seed(tmp_path, capsys):
    """A second clip must not silently replace the measured render."""
    import run_lpips_fvd as lpips_metrics

    outputs = tmp_path / "outputs"
    outputs.mkdir()
    for condition in ("STOCK", "NEURO"):
        _touch(str(outputs),
               f"BENCH_debate_passion_{condition}_s42424242_00001_.webp")
        _touch(str(outputs),
               f"BENCH_debate_passion_{condition}_s42424242_00002_.webp")

    pairs = lpips_metrics.find_pairs(str(outputs))
    assert len(pairs) == 1
    measured = pairs["debate_passion_s42424242"]
    assert measured["stock"].endswith("_00001_.webp")
    assert measured["neuro"].endswith("_00001_.webp")
    assert "not clip" in capsys.readouterr().out


def test_find_pairs_reports_a_duplicate_instead_of_overwriting(tmp_path, capsys):
    import run_lpips_fvd as lpips_metrics

    outputs = tmp_path / "outputs"
    outputs.mkdir()
    # Same arc, condition and seed, rendered twice under different sub-runs
    _touch(str(outputs), "BENCH_debate_tension_STOCK_s42424242_00001_.webp")
    _touch(str(outputs), "BENCH_debate_tension_STOCK_s42424242_00001_2_.webp")
    _touch(str(outputs), "BENCH_debate_tension_NEURO_s42424242_00001_.webp")

    pairs = lpips_metrics.find_pairs(str(outputs))
    assert pairs["debate_tension_s42424242"]["stock"].endswith("_00001_.webp")
    assert "duplicate" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Benchmark suite parameters (supplementary: experimental parameters)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("condition,steps,guidance,max_shift,base_shift", [
    # supplementary.tex, "Experimental parameters"
    ("stock", 30, 7.5, 2.05, 0.95),
    ("neuro", 24, 8.0, 2.10, 0.98),
])
def test_condition_parameters_match_the_paper(monkeypatch, condition, steps,
                                              guidance, max_shift, base_shift):
    """Drive the real run_single_test path and inspect the queued workflow."""
    import neuromorphic_benchmark_suite as suite

    queued = {}

    def fake_queue(workflow):
        queued["workflow"] = workflow
        return "prompt-id", 0.0, ""

    monkeypatch.setattr(suite, "queue_prompt", fake_queue)
    case = suite.TEST_CASES[0]
    result = suite.run_single_test(case, 42424242, condition)

    wf = queued["workflow"]
    assert result["steps"] == steps
    assert result["prefix"] == f"BENCH_{case.name}_{condition.upper()}_s42424242"
    assert wf["9"]["inputs"]["steps"] == steps
    assert wf["11"]["inputs"]["cfg"] == guidance
    assert wf["8"]["inputs"]["max_shift"] == max_shift
    assert wf["8"]["inputs"]["base_shift"] == base_shift
    assert wf["13"]["inputs"]["num_frames"] == 49
    assert wf["4"]["inputs"]["text"] == (
        case.stock_prompt if condition == "stock" else case.neuro_prompt)


def test_workflow_uses_the_image_basename_only():
    import neuromorphic_benchmark_suite as suite

    params = {"guidance_scale": 7.5, "steps": 30, "width": 512, "height": 320,
              "frames": 49, "max_shift": 2.05, "base_shift": 0.95, "seed": 1}
    wf = suite.build_workflow("p", "n", params, "/tmp/some/dir/frame.png", "BENCH_x")
    assert wf["2"]["inputs"]["image"] == "frame.png"


def test_test_cases_cover_the_seven_published_arcs():
    import neuromorphic_benchmark_suite as suite

    published_arcs = {r["arc"] for r in published_records()}
    assert {c.name for c in suite.TEST_CASES} == published_arcs


def test_source_images_are_configurable(monkeypatch, tmp_path):
    import importlib

    import neuromorphic_benchmark_suite as suite

    monkeypatch.setenv("GRAIL_SOURCE_IMAGE_DIR", str(tmp_path))
    importlib.reload(suite)
    try:
        assert suite.TEST_CASES[0].image_path == str(tmp_path / "sophia_victorian_frame.png")
    finally:
        monkeypatch.delenv("GRAIL_SOURCE_IMAGE_DIR")
        importlib.reload(suite)
