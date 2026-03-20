# GRAIL-V Human Evaluation Study

Human evaluation form for the CVPR 2026 GRAIL-V paper:
*Emotional Vocabulary as Semantic Grounding for Video Generation.*

## Directory Structure

```
human_eval/
  evaluation_form.html   # Self-contained evaluation form (open in any browser)
  README.md              # This file
  videos/                # Place generated videos here (create this directory)
```

## Adding Video Files

### 1. Create the videos directory

```bash
mkdir -p /home/scott/grail_paper/human_eval/videos
```

### 2. Naming convention

Each video file must follow this pattern:

```
{arc}_{seed}_{condition}.mp4
```

Where:
- **arc**: one of `realization`, `contemplation`, `determination`, `confidence`, `respect`, `passion`, `tension`
- **seed**: either `42424242` or `42425242`
- **condition**: either `stock` or `neuro` (lowercase)

### 3. Full file list (14 STOCK + 14 NEURO = 28 files)

```
videos/realization_42424242_stock.mp4
videos/realization_42424242_neuro.mp4
videos/realization_42425242_stock.mp4
videos/realization_42425242_neuro.mp4
videos/contemplation_42424242_stock.mp4
videos/contemplation_42424242_neuro.mp4
videos/contemplation_42425242_stock.mp4
videos/contemplation_42425242_neuro.mp4
videos/determination_42424242_stock.mp4
videos/determination_42424242_neuro.mp4
videos/determination_42425242_stock.mp4
videos/determination_42425242_neuro.mp4
videos/confidence_42424242_stock.mp4
videos/confidence_42424242_neuro.mp4
videos/confidence_42425242_stock.mp4
videos/confidence_42425242_neuro.mp4
videos/respect_42424242_stock.mp4
videos/respect_42424242_neuro.mp4
videos/respect_42425242_stock.mp4
videos/respect_42425242_neuro.mp4
videos/passion_42424242_stock.mp4
videos/passion_42424242_neuro.mp4
videos/passion_42425242_stock.mp4
videos/passion_42425242_neuro.mp4
videos/tension_42424242_stock.mp4
videos/tension_42424242_neuro.mp4
videos/tension_42425242_stock.mp4
videos/tension_42425242_neuro.mp4
```

If video files are missing, the form shows a text placeholder with the expected filename.

## Sending to Evaluators

### Option A: Local file (simplest)

1. Place all 28 `.mp4` files in the `videos/` subdirectory.
2. Zip the entire `human_eval/` folder:
   ```bash
   cd /home/scott/grail_paper
   zip -r grail_v_eval.zip human_eval/
   ```
3. Send the zip to each evaluator. They unzip and open `evaluation_form.html` in a browser.

### Option B: Host on a web server

1. Upload the `human_eval/` directory to any static host (GitHub Pages, S3, a VPS).
2. Send evaluators the URL to `evaluation_form.html`.
3. CORS is not an issue since everything is self-contained.

### Option C: Google Drive / Dropbox

1. Upload the folder. Share with evaluators.
2. They must download and open locally (streaming from Drive will not load relative video paths).

## Collecting Results

1. Each evaluator clicks "Submit" after answering all 42 questions (14 pairs x 3 questions).
2. The form generates a JSON blob displayed in a text area.
3. The evaluator copies the JSON and sends it back (email, Slack, etc.).
4. Save each response as `results/evaluator_{ID}.json`.

## Analyzing Results

The JSON output includes:
- **Per-pair responses**: raw A/B choice and decoded condition name (STOCK/NEURO/TIE)
- **Summary counts**: aggregated wins for STOCK, NEURO, and TIE across all 3 dimensions

### Quick analysis with Python

```python
import json, glob

files = glob.glob("results/evaluator_*.json")
totals = {"STOCK": {"quality": 0, "emotion": 0, "preference": 0},
          "NEURO": {"quality": 0, "emotion": 0, "preference": 0},
          "TIE":   {"quality": 0, "emotion": 0, "preference": 0}}

for f in files:
    data = json.load(open(f))
    for cond in ["STOCK", "NEURO", "TIE"]:
        for dim in ["quality", "emotion", "preference"]:
            totals[cond][dim] += data["summary"][cond][dim]

n_evaluators = len(files)
n_pairs = 14
total_judgments = n_evaluators * n_pairs

print(f"Evaluators: {n_evaluators}")
print(f"Total judgments per dimension: {total_judgments}")
print()
for dim in ["quality", "emotion", "preference"]:
    s = totals["STOCK"][dim]
    n = totals["NEURO"][dim]
    t = totals["TIE"][dim]
    print(f"{dim:>12s}:  STOCK={s}  NEURO={n}  TIE={t}  "
          f"(NEURO win rate: {n/(s+n)*100:.1f}% excl. ties)")
```

### Statistical significance

Use a two-sided binomial test on (NEURO wins) vs (STOCK wins), excluding ties:

```python
from scipy.stats import binomtest

for dim in ["quality", "emotion", "preference"]:
    s = totals["STOCK"][dim]
    n = totals["NEURO"][dim]
    result = binomtest(n, n + s, 0.5, alternative='two-sided')
    print(f"{dim}: p={result.pvalue:.4f}")
```

## A/B Randomization

The A/B assignment is deterministic, seeded with `20260320` (the date). All evaluators see the same mapping. The mapping is embedded in the HTML and also recorded in each JSON output under `pairs[].a_condition` and `pairs[].b_condition`.

## Design Notes

- No external dependencies. All CSS and JS are inline.
- Works offline once videos are in place.
- The form prevents submission until all 42 questions are answered.
- Progress bar tracks completion.
- The decoded condition (STOCK/NEURO) is recorded alongside raw A/B so analysis scripts do not need to reconstruct the mapping.
