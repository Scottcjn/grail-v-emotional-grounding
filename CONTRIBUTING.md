# Contributing to grail-v-emotional-grounding

Thanks for contributing to `grail-v-emotional-grounding`, the CVPR 2026 GRAIL-V workshop project on emotional vocabulary and diffusion efficiency in image-to-video generation.

This repository combines three kinds of artifacts:

- reproducible analysis code in `code/`
- evaluation materials in `human_eval/`
- paper sources in `paper/`

Please keep PRs focused on one of those areas so review stays fast and evidence remains easy to verify.

## Where to Contribute

### `code/`

Use this area for:

- analysis fixes
- reproduction workflow improvements
- metric computation changes
- prompt translation or benchmark pipeline updates

Current entry points include:

- `code/run_lpips_fvd.py`
- `code/compute_clip_image_text.py`
- `code/compute_clip_scores.py`
- `code/steps_vs_lpips_sweep.py`
- `code/neuromorphic_prompt_translator.py`
- `code/neuromorphic_benchmark_suite.py`

### `human_eval/`

Use this area for:

- evaluation protocol clarifications
- fixes to `evaluation_form.html`
- metadata or pairing corrections in `eval_pairs.json`
- documentation updates in `human_eval/README.md`

### `paper/`

Use this area for:

- wording fixes
- citation cleanup
- figure/table references
- supplementary material updates that match the current results

If you change claims in `paper/`, update the matching supporting material in `data/` or `code/` in the same PR.

## Local Setup

### Prerequisites

- Python 3.10+
- `pip`
- enough local disk space for data artifacts and intermediate outputs

### Install Dependencies

The README currently documents this baseline environment:

```bash
pip install torch lpips webp sentence-transformers transformers Pillow matplotlib
```

If your change introduces a new dependency, document why it is needed and update the relevant reproduction instructions.

## Reproduction Workflow

Use the current repository steps as the source of truth:

```bash
python code/run_lpips_fvd.py
python code/compute_clip_image_text.py
python code/compute_clip_scores.py
python code/steps_vs_lpips_sweep.py
```

Not every contribution needs the full pipeline. Run the narrowest check that matches your change.

## Validation

Before opening a PR, run the fastest relevant checks.

For Python-only changes:

```bash
python3 -m py_compile code/*.py
```

For documentation-only changes:

```bash
git diff --check
```

For HTML or evaluation-material changes, also open `human_eval/evaluation_form.html` in a browser and confirm the page still loads cleanly.

For LaTeX-only changes, keep edits scoped and note whether you compiled locally or only performed source review.

## Data and Results

- Do not silently overwrite result files in `data/`
- If you regenerate a result artifact, explain the command, model/config change, and reason in the PR
- Keep filenames stable unless there is a strong reason to rename them

## Pull Requests

A good PR for this repo should include:

- the exact area changed (`code/`, `human_eval/`, or `paper/`)
- why the change is needed
- what you ran to validate it
- whether any data files or reported numbers changed

Examples of good small PRs:

- fix one metric script bug
- improve one reproduction instruction
- clarify one evaluation step
- correct one paper reference or wording issue

## Style Notes

- prefer explicit, research-friendly naming over shorthand
- keep analysis scripts readable and reproducible
- avoid mixing paper edits, code changes, and data regeneration unless they are directly linked
- document any assumption that would affect reproducibility

## Questions

If a result, protocol, or paper claim is unclear, open an issue or explain the uncertainty in your PR instead of guessing.
