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
│   └── neuromorphic_benchmark_suite.py    # Full benchmark pipeline
├── data/
│   ├── lpips_results.json                 # Frame-level LPIPS for all 35 pairs
│   ├── clip_image_text_scores.json        # CLIP ViT-B/32 scores
│   ├── clip_text_similarity.json          # Embedding topology analysis
│   └── stock_realization_convergence.json # Steps-vs-LPIPS sweep
├── human_eval/
│   ├── evaluation_form.html               # Self-contained 2AFC evaluation form
│   ├── eval_pairs.json                    # All prompt conditions per arc
│   └── README.md                          # Evaluation protocol
├── figures/                               # Paper figures
└── paper/
    ├── grail_v_paper.tex                  # Main paper (LaTeX)
    └── supplementary.tex                  # Supplementary materials
```

## Prompt Translator

The Prompt Translator automatically converts literal motion descriptors to emotionally-grounded prompts:

| Literal Input | Emotional Output |
|--------------|-----------------|
| head movement | subtle shift in attention |
| slight smile | knowing smile forming |
| hand movement | gesture carrying emotional weight |
| stares | intensity building |
| frowns | concern deepening |

## Requirements

```bash
pip install torch lpips webp sentence-transformers transformers Pillow matplotlib
```

## Reproduction

```bash
# 1. Compute LPIPS between STOCK/NEURO render pairs
python code/run_lpips_fvd.py

# 2. Compute CLIP image-text alignment
python code/compute_clip_image_text.py

# 3. Analyze embedding topology
python code/compute_clip_scores.py

# 4. Run convergence sweep (requires ComfyUI + LTX-2)
python code/steps_vs_lpips_sweep.py
```

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
