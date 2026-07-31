# SPDX-License-Identifier: MIT
"""
Tests for pipeline integrity: a run that produced nothing must not report
success, and the released aggregates must reproduce the paper.

The ComfyUI cases run against a real local HTTP server returning the real
response shapes (ComfyUI answers a rejected job with HTTP 400 plus a
``node_errors`` map), so the assertions are about behaviour over the wire
rather than about a mock's call log.

The statistics cases use two oracles that are independent of the code under
test: the committed per-pair LPIPS means in ``data/lpips_results.json``, and
Student-t tail values taken from SciPy (hard-coded, so the suite itself
needs no SciPy).
"""

import json
import math
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")

import compute_clip_scores  # noqa: E402  (conftest puts code/ on sys.path)
import neuromorphic_benchmark_suite as suite  # noqa: E402
import run_lpips_fvd  # noqa: E402
import steps_vs_lpips_sweep as sweep  # noqa: E402
import verify_paper_stats as vps  # noqa: E402


# ---------------------------------------------------------------------------
# A local stand-in for ComfyUI
# ---------------------------------------------------------------------------

# Verbatim shape of a ComfyUI rejection when the checkpoint is not installed.
REJECTION_BODY = {
    "error": {
        "type": "prompt_outputs_failed_validation",
        "message": "Prompt outputs failed validation",
    },
    "node_errors": {
        "1": {
            "errors": [
                {
                    "type": "value_not_in_list",
                    "message": "Value not in list",
                    "details": "ckpt_name: 'ltx-2-19b-dev-fp8.safetensors' not in []",
                }
            ]
        }
    },
}

WEBP_HEADER = b"RIFF\x00\x00\x00\x00WEBPVP8 "


class _Handler(BaseHTTPRequestHandler):
    behaviour = "reject"

    def log_message(self, *args):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _raw(self, code, body, ctype="application/octet-stream"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path == "/upload/image":
            if self.behaviour == "upload_fails":
                return self._json(500, {"error": "disk full"})
            return self._json(200, {"name": "image.png"})
        if self.path == "/prompt":
            if self.behaviour == "reject":
                return self._json(400, REJECTION_BODY)
            if self.behaviour == "empty_ok":
                return self._json(200, {})
            if self.behaviour == "garbage":
                return self._raw(200, b"<html>proxy error</html>", "text/html")
            return self._json(200, {"prompt_id": "1234abcd-0000-0000-0000-000000000000"})
        self._json(404, {})

    def do_GET(self):
        if self.path.startswith("/view"):
            if self.behaviour == "view_404":
                return self._raw(404, b"404: Not Found", "text/plain")
            if self.behaviour == "view_html":
                return self._raw(200, b"<html>gateway timeout</html>", "text/html")
            return self._raw(200, WEBP_HEADER + b"payload")
        self._json(200, {"system": {"comfyui_version": "test"}})


@pytest.fixture
def comfy():
    """Start a local ComfyUI stand-in; yields a setter for its behaviour."""
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"

    class Control:
        base_url = url

        @staticmethod
        def behave(mode):
            _Handler.behaviour = mode

    Control.behave("reject")
    try:
        yield Control
    finally:
        server.shutdown()
        server.server_close()
        _Handler.behaviour = "reject"


@pytest.fixture
def suite_against(comfy, monkeypatch, tmp_path):
    """Point the benchmark suite at the stand-in and at a temp output dir."""
    monkeypatch.setattr(suite, "COMFYUI_SERVER", comfy.base_url)
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setattr(suite, "OUTPUT_DIR", str(out))
    image = tmp_path / "source.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    for case in suite.TEST_CASES:
        monkeypatch.setattr(case, "image_path", str(image), raising=False)
    return comfy, out


# ---------------------------------------------------------------------------
# queue_prompt: a failure must never look like a queued render
# ---------------------------------------------------------------------------

def test_rejected_job_yields_no_prompt_id(suite_against):
    comfy, _ = suite_against
    comfy.behave("reject")

    prompt_id, _elapsed, error = suite.queue_prompt({"1": {}})

    assert prompt_id == ""
    # The error must not masquerade as an id: it used to be *returned* as one.
    assert error
    assert "ckpt_name" in error, error


def test_http_200_without_prompt_id_is_a_failure(suite_against):
    comfy, _ = suite_against
    comfy.behave("empty_ok")

    prompt_id, _elapsed, error = suite.queue_prompt({"1": {}})

    assert prompt_id == ""
    assert error


def test_non_json_response_is_a_failure(suite_against):
    comfy, _ = suite_against
    comfy.behave("garbage")

    prompt_id, _elapsed, error = suite.queue_prompt({"1": {}})

    assert prompt_id == ""
    assert "non-JSON" in error


def test_unreachable_server_is_reported_not_raised(monkeypatch):
    # Port 1 is reserved and never listening.
    monkeypatch.setattr(suite, "COMFYUI_SERVER", "http://127.0.0.1:1")

    prompt_id, _elapsed, error = suite.queue_prompt({"1": {}})

    assert prompt_id == ""
    assert error


def test_accepted_job_yields_the_prompt_id(suite_against):
    comfy, _ = suite_against
    comfy.behave("accept")

    prompt_id, _elapsed, error = suite.queue_prompt({"1": {}})

    assert prompt_id == "1234abcd-0000-0000-0000-000000000000"
    assert error == ""


def test_describe_queue_error_keeps_the_node_detail():
    message = suite.describe_queue_error(400, REJECTION_BODY)

    assert "Prompt outputs failed validation" in message
    assert "node 1" in message
    assert "ckpt_name" in message


def test_describe_queue_error_survives_an_unexpected_body():
    assert "418" in suite.describe_queue_error(418, "not a dict")
    assert suite.describe_queue_error(500, {})


# ---------------------------------------------------------------------------
# The suite as a whole
# ---------------------------------------------------------------------------

def test_a_run_that_queued_nothing_is_not_reported_as_success(suite_against, capsys):
    comfy, _out = suite_against
    comfy.behave("reject")

    manifest = suite.run_benchmark_suite(num_seeds=2, tests_to_run=["sophia_realization"])

    assert manifest["total_runs"] == 4
    assert manifest["queued_runs"] == 0
    assert manifest["failed_runs"] == 4
    assert suite.exit_code(manifest) == 1
    assert all(r["prompt_id"] == "" and not r["queued"] for r in manifest["results"])

    out = capsys.readouterr().out
    assert "BENCHMARK QUEUED" not in out
    assert "NOTHING WAS QUEUED" in out
    assert "REJECTED: 4 run(s)" in out
    # The summary collapses identical reasons instead of repeating them.
    assert out.count("x4  Prompt outputs failed validation") == 1


def test_a_fully_accepted_run_reports_success(suite_against):
    comfy, _out = suite_against
    comfy.behave("accept")

    manifest = suite.run_benchmark_suite(num_seeds=2, tests_to_run=["sophia_realization"])

    assert manifest["queued_runs"] == manifest["total_runs"] == 4
    assert manifest["failed_runs"] == 0
    assert suite.exit_code(manifest) == 0


def test_upload_failure_is_recorded_not_just_printed(suite_against):
    comfy, _out = suite_against
    comfy.behave("upload_fails")

    manifest = suite.run_benchmark_suite(num_seeds=2, tests_to_run=["sophia_realization"])

    assert manifest["queued_runs"] == 0
    assert manifest["skipped_tests"]
    assert manifest["skipped_tests"][0]["test_name"] == "sophia_realization"
    assert suite.exit_code(manifest) == 1


def test_upload_image_reports_why_it_failed(suite_against, tmp_path):
    comfy, _out = suite_against

    comfy.behave("upload_fails")
    ok, detail = suite.upload_image(str(tmp_path / "source.png"))
    assert ok is False and "500" in detail

    comfy.behave("accept")
    ok, detail = suite.upload_image(str(tmp_path / "missing.png"))
    assert ok is False and "cannot read" in detail

    ok, detail = suite.upload_image(str(tmp_path / "source.png"))
    assert ok is True and detail == ""


def test_manifest_lands_inside_the_benchmark_dir(suite_against):
    """It used to be written to the *parent* of GRAIL_BENCHMARK_DIR."""
    comfy, out = suite_against
    comfy.behave("accept")

    suite.run_benchmark_suite(num_seeds=1, tests_to_run=["sophia_realization"])

    manifests = list(out.glob("benchmark_manifest_*.json"))
    assert len(manifests) == 1
    assert not list(out.parent.glob("benchmark_manifest_*.json"))


def test_output_dir_is_the_benchmark_dir_itself(monkeypatch):
    import importlib

    monkeypatch.setenv("GRAIL_BENCHMARK_DIR", "/tmp/grail-check/outputs")
    reloaded = importlib.reload(suite)
    try:
        assert reloaded.OUTPUT_DIR == "/tmp/grail-check/outputs"
    finally:
        monkeypatch.delenv("GRAIL_BENCHMARK_DIR")
        importlib.reload(suite)


@pytest.mark.parametrize(
    "manifest,expected",
    [
        ({"queued_runs": 4, "failed_runs": 0, "skipped_tests": []}, 0),
        ({"queued_runs": 0, "failed_runs": 4, "skipped_tests": []}, 1),
        ({"queued_runs": 3, "failed_runs": 1, "skipped_tests": []}, 1),
        ({"queued_runs": 4, "failed_runs": 0, "skipped_tests": [{"test_name": "x"}]}, 1),
        ({}, 1),
    ],
)
def test_exit_code(manifest, expected):
    assert suite.exit_code(manifest) == expected


# ---------------------------------------------------------------------------
# The sweep: a bad download must not become a .webp
# ---------------------------------------------------------------------------

@pytest.fixture
def sweep_against(comfy, monkeypatch, tmp_path):
    monkeypatch.setattr(sweep, "COMFYUI_SERVER", comfy.base_url)
    monkeypatch.setattr(sweep, "OUTPUT_DIR", str(tmp_path))
    return comfy, tmp_path


def test_download_rejects_an_http_error_instead_of_saving_it(sweep_against):
    comfy, out = sweep_against
    comfy.behave("view_404")

    with pytest.raises(Exception):
        sweep.download_output("clip_00001_.webp")

    # Nothing that later stages could mistake for a render.
    assert list(out.iterdir()) == []


def test_download_rejects_a_200_that_is_not_a_webp(sweep_against):
    comfy, out = sweep_against
    comfy.behave("view_html")

    with pytest.raises(RuntimeError, match="not a WebP"):
        sweep.download_output("clip_00001_.webp")

    assert list(out.iterdir()) == []


def test_download_writes_a_real_webp(sweep_against):
    comfy, out = sweep_against
    comfy.behave("accept")

    path = sweep.download_output("clip_00001_.webp")

    assert os.path.basename(path) == "clip_00001_.webp"
    assert open(path, "rb").read().startswith(b"RIFF")


def test_unreadable_clip_does_not_abort_the_whole_sweep(tmp_path, capsys):
    """One corrupt file used to raise straight out of Phase 2, discarding
    every LPIPS score computed so far -- results are only written afterwards."""
    bad = tmp_path / "bad_00001_.webp"
    bad.write_bytes(b"404: Not Found")

    assert sweep.load_frames(str(bad)) == []
    assert "cannot read" in capsys.readouterr().out

    assert math.isnan(sweep.compute_lpips(str(bad), str(bad)))


def test_load_frames_on_a_missing_file_returns_empty(tmp_path):
    assert sweep.load_frames(str(tmp_path / "nope.webp")) == []


# ---------------------------------------------------------------------------
# Aggregation: sample vs population standard deviation
# ---------------------------------------------------------------------------

def committed_pair_means():
    with open(os.path.join(DATA_DIR, "lpips_results.json")) as f:
        return {k: v["mean"] for k, v in json.load(f)["per_pair"].items()}


def solo_means():
    return [v for k, v in sorted(committed_pair_means().items())
            if k.rsplit("_", 1)[0] in vps.SOLO_ARCS]


def test_sample_std_is_bessel_corrected():
    values = [1.0, 2.0, 3.0, 4.0]
    # mean 2.5; sum sq dev 5.0 -> ddof=1: sqrt(5/3), ddof=0: sqrt(5/4)
    assert run_lpips_fvd.sample_std(values) == pytest.approx(math.sqrt(5 / 3))
    assert run_lpips_fvd.sample_std(values) != pytest.approx(math.sqrt(5 / 4))


def test_sample_std_degenerate_inputs():
    assert run_lpips_fvd.sample_std([]) == 0.0
    assert run_lpips_fvd.sample_std([0.5]) == 0.0


def test_aggregate_std_reproduces_the_published_dispersion():
    """The paper reports s = 0.005 for the solo pairs; ddof=0 gives 0.00478."""
    solo = solo_means()

    assert run_lpips_fvd.sample_std(solo) == pytest.approx(0.004944, abs=5e-6)
    assert vps.population_std(solo) == pytest.approx(0.004776, abs=5e-6)
    assert round(run_lpips_fvd.sample_std(solo), 3) == 0.005


def test_population_std_would_not_reproduce_the_published_t():
    solo = solo_means()
    n = len(solo)

    t_sample = (vps.mean(solo) - 0.1) / (run_lpips_fvd.sample_std(solo) / math.sqrt(n))
    t_population = (vps.mean(solo) - 0.1) / (vps.population_std(solo) / math.sqrt(n))

    assert t_sample == pytest.approx(-69.59, abs=0.01)
    assert t_population == pytest.approx(-72.03, abs=0.01)


# ---------------------------------------------------------------------------
# "N% denser" needs the compared-against group in the denominator
# ---------------------------------------------------------------------------

def test_density_gap_uses_the_looser_group_as_the_baseline():
    # A distance 40% below B's is "40% denser than B".
    assert compute_clip_scores.density_gap(0.6, 1.0) == pytest.approx(40.0)
    assert compute_clip_scores.density_gap(1.0, 1.0) == pytest.approx(0.0)


def test_density_gap_is_invertible():
    """tight * (1 - gap/100) round-trips, which the old formula did not."""
    tight, loose = 0.43099, 0.60534
    gap = compute_clip_scores.density_gap(tight, loose)

    assert loose * (1 - gap / 100) == pytest.approx(tight, abs=1e-9)
    assert gap == pytest.approx(28.80, abs=0.01)
    # The value the old, stock-denominated formula printed for this branch.
    assert gap != pytest.approx(40.45, abs=0.01)


def test_density_gap_matches_the_committed_vocabulary_numbers():
    with open(os.path.join(DATA_DIR, "clip_text_similarity.json")) as f:
        data = json.load(f)
    vocab = data["emotional_vs_literal_vocab"]

    gap = compute_clip_scores.density_gap(
        vocab["mean_pairwise_distance_emotional"],
        vocab["mean_pairwise_distance_literal"],
    )
    # This branch was already correct; it must stay unchanged.
    assert gap == pytest.approx(1.24, abs=0.01)


def test_density_gap_guards_a_zero_baseline():
    assert math.isnan(compute_clip_scores.density_gap(0.0, 0.0))


# ---------------------------------------------------------------------------
# The paper verifier
# ---------------------------------------------------------------------------

# Reference values from scipy.stats.t.cdf, so this suite needs no SciPy.
T_CDF_REFERENCE = [
    (-69.5913, 14, 1.7337926e-19),
    (-3.14, 8, 6.9026156e-03),
    (-1.0, 1, 2.5000000e-01),
    (0.0, 5, 5.0000000e-01),
    (2.7, 30, 9.9435777e-01),
    (12.0, 100, 1.0000000e+00),
]


@pytest.mark.parametrize("t,df,expected", T_CDF_REFERENCE)
def test_t_cdf_matches_scipy(t, df, expected):
    assert vps.t_cdf(t, df) == pytest.approx(expected, rel=1e-6)


def test_betainc_known_values():
    assert vps.betainc(1.0, 1.0, 0.25) == pytest.approx(0.25)
    assert vps.betainc(2.0, 3.0, 0.5) == pytest.approx(0.6875)
    assert vps.betainc(3.0, 1.0, 0.0) == 0.0
    assert vps.betainc(3.0, 1.0, 1.0) == 1.0


def test_t_cdf_is_symmetric():
    for df in (1, 4, 14, 60):
        for t in (0.3, 1.7, 4.2):
            assert vps.t_cdf(-t, df) == pytest.approx(1.0 - vps.t_cdf(t, df), rel=1e-9)


def test_published_claims_reproduce_from_the_committed_data(capsys):
    assert vps.main([]) == 0

    out = capsys.readouterr().out
    assert "ALL CLAIMS REPRODUCE" in out
    assert "[OFF]" not in out


def test_verifier_reports_the_exact_one_sided_p():
    """The paper writes p < 1e-19; the data give 1.73e-19, i.e. p < 1e-18."""
    _t, p, df = vps.one_sample_t(solo_means(), 0.1)

    assert df == 14
    assert p == pytest.approx(1.7338e-19, rel=1e-4)
    assert p > 1e-19
    assert p < 1e-18


def test_verifier_fails_loudly_on_tampered_data(tmp_path):
    with open(os.path.join(DATA_DIR, "lpips_results.json")) as f:
        data = json.load(f)
    for key in data["per_pair"]:
        if key.rsplit("_", 1)[0] in vps.SOLO_ARCS:
            data["per_pair"][key]["mean"] = 0.09  # just under the threshold
    tampered = tmp_path / "lpips_results.json"
    tampered.write_text(json.dumps(data))

    assert vps.main(["--results", str(tampered)]) == 1


def test_arc_grouping_covers_every_committed_pair():
    pair_means = committed_pair_means()
    solo = vps.collect(pair_means, vps.SOLO_ARCS)
    complex_ = vps.collect(pair_means, vps.COMPLEX_ARCS)

    assert len(pair_means) == 35
    assert len(solo) == 15
    assert len(complex_) == 15
    # The remaining five are the tension arc, which the paper reports
    # separately as the extended n=22 analysis.
    assert len(pair_means) - len(solo) - len(complex_) == 5
