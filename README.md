[![BCOS Certified](https://img.shields.io/badge/BCOS-Certified-brightgreen?style=flat)](BCOS.md)

# Emotional Vocabulary as Semantic Grounding

**How Language Register Affects Diffusion Efficiency in Image-to-Video Generation**

*Accepted at CVPR 2026 GRAIL-V Workshop*

**Scott Boudreaux** — [Elyan Labs](https://elyanlabs.com)

## Abstract

We investigate whether the **semantic register** of prompt language — emotional vs. literal — affects diffusion efficiency in image-to-video generation. Through systematic A/B testing on LTX-2 (35 matched pairs, 7 emotional arcs, 5 seeds), we show that emotional vocabulary **maintains perceptual quality at 20% fewer diffusion steps** (30→24) for solo portraits (LPIPS = 0.011 ± 0.005, n=15, p < 10⁻¹⁹). A controlled ablation with identical parameters confirms the effect is prompt-driven (p = 0.014). Embedding topology analysis reveals emotional vocabulary forms **16% tighter clusters** in Gemma 3 embedding space.

## Key Finding

Emotional prompts ("eyes brighten with quiet realization") achieve the same visual quality as literal prompts ("subtle head movement, slight smile") using 20% fewer diffusion steps — because emotional vocabulary occupies denser regions in the text encoder's embedding space.

## Results Summary

| Metric | STOCK (literal) | NEURO (emotional) | Finding |
|--------|----------------|-------------------|---------|
| Steps needed | 30 | 24 | **20% reduction** |
| Solo LPIPS | — | 0.011 ± 0.005 | Perceptually equivalent |
| Embedding radius | 0.269 | 0.225 | **16% tighter** |
| CLIP (solo) | 0.204 | 0.231 | **+13.5% alignment** |
| CLIP (complex) | 0.296 | 0.244 | -17.4% (trade-off) |

## Repository Structure

```
├── code/
│   ├── neuromorphic_prompt_translator.py  # Prompt Translator module
│   ├── compute_clip_scores.py             # Text embedding analysis
│   ├── compute_clip_image_text.py         # CLIP image-text similarity
│   ├── run_lpips_fvd.py                   # LPIPS computation
│   ├── steps_vs_lpips_sweep.py            # Convergence analysis
│   ├── neuromorphic_benchmark_suite.py    # Full benchmark pipeline
│   ├── verify_paper_stats.py              # Re-derive the paper's statistics
│   └── grail_config.py                    # Paths / endpoints (env-overridable)
├── data/
│   ├── lpips_results.json                 # Frame-level LPIPS for all 35 pairs
│   ├── clip_image_text_scores.json        # CLIP ViT-B/32 scores
│   ├── clip_text_similarity.json          # Embedding topology analysis
│   └── stock_realization_convergence.json # Steps-vs-LPIPS sweep
├── human_eval/
│   ├── evaluation_form.html               # Self-contained 2AFC evaluation form
│   ├── eval_pairs.json                    # All prompt conditions per arc
│   └── README.md                          # Evaluation protocol
├── tests/                                 # pytest suite for code/
├── figures/                               # Paper figures
└── paper/
    ├── grail_v_paper.tex                  # Main paper (LaTeX)
    └── supplementary.tex                  # Supplementary materials
```

## Prompt Translator

The Prompt Translator automatically converts literal motion descriptors to emotionally-grounded prompts. The full 22-entry dictionary is Table 3 of `paper/supplementary.tex`; a sample:

| Literal Input | Emotional Output |
|--------------|-----------------|
| head movement | subtle shift in attention |
| slight smile | knowing warmth emerging |
| hand movement | gesture carrying emotional weight |
| stares | intensity building |
| frowns | concern deepening |

```bash
python code/neuromorphic_prompt_translator.py   # demo: translation + salience
```

## Requirements

```bash
pip install -r requirements.txt
```

## Configuration

The analysis scripts take every path and endpoint from `code/grail_config.py`.
Defaults are repo-relative; override them with environment variables:

| Variable | Default | Used by |
|----------|---------|---------|
| `GRAIL_BENCHMARK_DIR` | `benchmark_results/outputs/` | LPIPS + CLIP image-text |
| `GRAIL_DATA_DIR` | `data/` | all metric outputs |
| `GRAIL_FIGURES_DIR` | `figures/` | convergence sweep figure |
| `GRAIL_SWEEP_DIR` | `steps_sweep_renders/` | convergence sweep renders |
| `GRAIL_SOURCE_IMAGE_DIR` | `source_images/` | benchmark suite |
| `GRAIL_COMFYUI_SERVER` | `http://127.0.0.1:8188` | render pipelines |
| `GRAIL_CLIP_INDEX` | `00001` | which clip per seed is measured |

## Reproduction

```bash
# 0. Point the scripts at your rendered clips
export GRAIL_BENCHMARK_DIR=/path/to/benchmark_results/outputs

# 1. Compute LPIPS between STOCK/NEURO render pairs   -> data/lpips_results.json
python code/run_lpips_fvd.py

# 2. Compute CLIP image-text alignment      -> data/clip_image_text_scores.json
python code/compute_clip_image_text.py

# 3. Analyze embedding topology              -> data/clip_text_similarity.json
python code/compute_clip_scores.py

# 4. Run convergence sweep (requires ComfyUI + LTX-2)
GRAIL_COMFYUI_SERVER=http://your-comfyui:8188 python code/steps_vs_lpips_sweep.py

# 5. Check the paper's headline numbers against the LPIPS results
python code/verify_paper_stats.py
```

Rendering the clips in step 0 is `python code/neuromorphic_benchmark_suite.py full`
(7 arcs x 5 seeds x STOCK/NEURO), which needs ComfyUI with LTX-2 and the source
images in `GRAIL_SOURCE_IMAGE_DIR`. It exits non-zero if ComfyUI did not accept
every job, and prints why — a run where nothing was queued produces no renders,
so the later steps would otherwise just report "no pairs found".

Step 5 needs no GPU and no renders: it re-derives the published mean, standard
deviation, one-sample *t*, *p* and Cohen's *d* from the committed
`data/lpips_results.json` and prints each next to the value in the paper.
Aggregates across seeds use the sample standard deviation (ddof=1), which is
what the paper reports; `numpy`'s default (ddof=0) yields *t* = −72.03 instead
of the published −69.59.

## Tests

```bash
pip install pytest
python -m pytest tests/ -q
```

The suite checks the analysis code against the published artifacts: the
motion-to-emotion dictionary is compared entry by entry with Table 3 of
`paper/supplementary.tex`, and the benchmark seeds, arc names and render
filenames are compared with `data/clip_image_text_scores.json`. No GPU,
network or model download is required.

## Human Evaluation

Open `human_eval/evaluation_form.html` in any browser. See `human_eval/README.md` for the full protocol.

## Citation

```bibtex
@inproceedings{boudreaux2026emotional,
  title={Emotional Vocabulary as Semantic Grounding: How Language Register Affects Diffusion Efficiency in Image-to-Video Generation},
  author={Boudreaux, Scott},
  booktitle={CVPR 2026 Workshop on Generative Models for Computer Vision (GRAIL-V)},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

This research was conducted at Elyan Labs, an independent research lab. Compute infrastructure includes IBM POWER8 S824 (512GB RAM), Tesla V100 32GB, and vintage PowerPC systems. No institutional funding was received.