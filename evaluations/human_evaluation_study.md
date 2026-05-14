# Human Evaluation Study Design

## GRAIL-V 2026: Emotional Vocabulary as Semantic Grounding

**Study Title:** Two-Alternative Forced Choice (2AFC) Evaluation of Emotional vs. Literal Prompting in Image-to-Video Generation

**Associated Paper:** "Emotional Vocabulary as Semantic Grounding: How Language Register Affects Diffusion Efficiency in Image-to-Video Generation" — CVPR 2026 GRAIL-V Workshop (Accepted)

**Author:** Scott Boudreaux, Elyan Labs

**Document Version:** 1.0

---

## 1. Research Objectives

### 1.1 Primary Objective

Determine whether videos generated from **emotionally-grounded prompts** (NEURO condition) are preferred over those from **literal motion descriptors** (STOCK condition) in blinded human evaluation, across three dimensions:

1. **Visual Quality** — Perceptual fidelity, temporal coherence, and artifact absence
2. **Emotional Accuracy** — Degree to which the generated motion matches the intended emotional arc
3. **Overall Preference** — Holistic viewer preference irrespective of specific criteria

### 1.2 Secondary Objectives

- Measure **inter-rater agreement** (Fleiss' κ) to validate consistency across evaluators
- Compare human preference rates against automated metrics (CLIP score, LPIPS)
- Assess whether emotional grounding provides measurable perceptual advantages at reduced diffusion steps (24 vs. 30)

### 1.3 Hypotheses

| ID | Hypothesis | Direction |
|----|-----------|-----------|
| H1 | NEURO videos are preferred over STOCK videos for overall visual quality | NEURO > STOCK |
| H2 | NEURO videos better match the target emotional description | NEURO > STOCK |
| H3 | Overall preference favors NEURO-generated videos | NEURO > STOCK |
| H4 | Inter-rater agreement exceeds chance (κ > 0) | κ > 0 |

---

## 2. Methodology: Two-Alternative Forced Choice (2AFC)

### 2.1 Rationale

2AFC is the gold-standard psychophysical method for comparative quality assessment (Mantiuk et al., 2012; Perez-Ortiz et al., 2019). Key advantages for this study:

- **Forced choice eliminates non-committal responses**, increasing statistical power
- **Blinded presentation prevents expectation bias** — evaluators cannot identify which condition generated which video
- **Tie option ("No difference") is included** to avoid random guessing when differences are genuinely imperceptible
- **Each pair is self-contained** — no absolute quality reference needed

### 2.2 Stimuli

- **28 videos** across 14 matched pairs (7 emotional arcs × 2 seeds)
- Each pair shares the same source image, seed, and model (LTX-2 19B with Gemma 3 text encoder)
- The **only manipulated variable** is the prompt text:
  - **STOCK:** Literal motion descriptors (e.g., "subtle head movement, slight smile")
  - **NEURO:** Emotional state descriptors (e.g., "eyes brighten with quiet realization")
- Videos are 2-second animated clips rendered at matching resolution

### 2.3 Arc and Seed Conditions

| Arc | Semantic Target | Seeds |
|-----|----------------|-------|
| realization | Quiet epiphany, understanding dawning | 42424242, 42425242 |
| contemplation | Deep thought, reflection | 42424242, 42425242 |
| determination | Resolute focus, resolve forming | 42424242, 42425242 |
| confidence | Self-assurance, poise | 42424242, 42425242 |
| respect | Reverence, acknowledgment | 42424242, 42425242 |
| passion | Intensity, emotional engagement | 42424242, 42425242 |
| tension | Strain, emotional pressure | 42424242, 42425242 |

### 2.4 Blinding and Randomization

- **A/B position** for each pair is deterministically randomized using seed `20260320`
- Condition assignment (STOCK/NEURO → A/B) is fixed across all evaluators for consistency
- **Evaluators never see condition labels** during the evaluation — only "Video A" and "Video B"
- Condition mapping is embedded in the form and recorded in output JSON for analysis

---

## 3. Evaluation Dimensions

### 3.1 Dimension 1: Visual Quality

**Question:** "Which video has higher visual quality?"

Evaluators assess:
- **Temporal coherence:** Smooth, natural motion without flickering or frame inconsistencies
- **Artifact absence:** No distortion, blurring, or unnatural deformation
- **Perceptual fidelity:** The video looks like a plausible, high-quality animation
- **Detail preservation:** Fine details from the source image are maintained

### 3.2 Dimension 2: Emotional Accuracy

**Question:** "Which video better matches the emotional description?"

Evaluators receive the target emotional arc name (e.g., "realization") and assess:
- **Semantic alignment:** The motion conveys the intended emotional state
- **Natural expression:** The emotion is expressed in a believable, non-exaggerated way
- **Subtlety calibration:** The emotional intensity feels appropriate for the arc

### 3.3 Dimension 3: Overall Preference

**Question:** "Which video do you prefer overall?"

Evaluators provide a holistic judgment incorporating:
- Any combination of the above factors
- Personal aesthetic preference
- Intuitive "which looks better" assessment

### 3.4 Response Options

Each question offers three choices:
- **A** — Video A is superior
- **B** — Video B is superior
- **No difference** — Videos are perceptually equivalent on this dimension

---

## 4. Evaluator Recruitment Criteria

### 4.1 Minimum Qualifications (≥2 required)

Evaluators must meet **at least two** of the following criteria:

1. **AI/ML Experience:** Experience with LLM development, prompt engineering, or generative AI systems
2. **Computer Vision Expertise:** Experience with computer vision, video generation, or diffusion models
3. **Visual Media Background:** Background in visual design, animation, or video production
4. **Research Contributions:** Published research or verifiable contributions to AI/ML projects
5. **Active Developer Profile:** GitHub account older than 6 months with real, substantive contributions

### 4.2 Rationale for Criteria

CVPR reviewers expect evaluators who can distinguish subtle visual quality differences in generated content. Domain expertise ensures:
- Informed judgments about temporal coherence and artifact detection
- Understanding of what constitutes natural vs. artificial motion
- Ability to assess emotional expression in animated content

### 4.3 Target Sample Size

- **Minimum:** 3 evaluators
- **Target:** 5 evaluators
- **Maximum:** 5 evaluators (budget constraint: 250 RTC total)

### 4.4 Compensation

- **50 RTC per qualified evaluator** (RustChain Token, ~$5.00 USD reference rate)
- Payment upon submission of complete, valid JSON results
- Evaluators acknowledged in supplementary materials

---

## 5. Sample Size Calculation and Statistical Power

### 5.1 Power Analysis for 2AFC Design

Under the null hypothesis H₀: p = 0.5 (no preference), a 2AFC design requires:

**Per-dimension analysis (binomial test):**

| Evaluator Count | Pairs per Evaluator | Total Judgments | Detectable Effect (80% power, α=0.05) |
|----------------|--------------------|-----------------|---------------------------------------|
| 3 | 14 | 42 | p ≥ 0.69 (Δ ≥ 0.19) |
| 5 | 14 | 70 | p ≥ 0.63 (Δ ≥ 0.13) |
| 5 | 14 | 70 | p ≥ 0.60 (Δ ≥ 0.10) at α=0.10 |

**Formula:**
```
n = (Z_α/2 + Z_β)² × p₀(1-p₀) / (p₁ - p₀)²
```
Where p₀ = 0.5 (null), p₁ = expected NEURO win rate, α = 0.05, β = 0.20.

### 5.2 Inter-Rater Agreement Power

For Fleiss' κ with 5 raters, 14 items, 3 categories:

- Chance agreement (Pₑ) = 1/3 ≈ 0.333
- Detectable κ ≥ 0.4 (moderate agreement) at 80% power
- Below 3 raters, κ estimates become unreliable

### 5.3 Practical Considerations

- **14 pairs × 3 dimensions = 42 judgments per evaluator** — manageable in ~30 minutes
- **Evaluation fatigue:** 42 forced-choice comparisons remain below the recommended 50-pair threshold for maintaining attention (ITU-R BT.500-13)
- **Order effects mitigated** by randomized pair presentation within the form

---

## 6. Evaluation Procedure and Scoring

### 6.1 Pre-Evaluation Setup

1. Distribute the evaluation package (ZIP containing `evaluation_form.html` + `videos/` directory)
2. Confirm evaluator meets ≥2 qualification criteria
3. Assign unique evaluator ID (e.g., `evaluator_001`)
4. Provide brief orientation:
   - "You will view 14 pairs of short animated videos (2 seconds each)"
   - "For each pair, answer three questions about visual quality, emotional accuracy, and overall preference"
   - "Choose A, B, or 'No difference' for each question"
   - "There are no trick questions — report what you genuinely perceive"

### 6.2 Evaluation Flow

```
┌─────────────────────────────────────┐
│  1. Evaluator opens form in browser  │
│  2. Enters Evaluator ID              │
│  3. Reads brief instructions         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  For each of 14 pairs:               │
│  ├─ Play Video A (2s loop)           │
│  ├─ Play Video B (2s loop)           │
│  ├─ Q1: Visual quality? (A/B/Tie)   │
│  ├─ Q2: Emotional accuracy? (A/B/Tie)│
│  └─ Q3: Overall preference? (A/B/Tie)│
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  4. Progress bar: 42/42 complete     │
│  5. Submit → JSON output generated   │
│  6. Copy JSON → Send to researchers  │
└─────────────────────────────────────┘
```

### 6.3 Scoring Protocol

For each pair and dimension:
- Response recorded as raw choice: `A`, `B`, or `TIE`
- Response decoded to condition: `STOCK`, `NEURO`, or `TIE`
- **No partial credit** — each judgment is independent
- **Ties are valid responses**, not treated as missing data

### 6.4 Quality Control Checks

Before accepting a submission:
1. **Completeness:** All 42 judgments answered (14 pairs × 3 dimensions)
2. **Consistency check:** No identical responses for all 14 pairs on any dimension (suggests inattention)
3. **Response time:** Minimum 10 minutes total (flags rushed evaluations)
4. **Evaluator ID** present and unique

---

## 7. Statistical Analysis Plan

### 7.1 Primary Analysis: Preference Rates

For each dimension, compute:

```
NEURO win rate = NEURO_wins / (NEURO_wins + STOCK_wins)  # excluding ties
Tie rate = TIE_count / total_judgments
```

**Statistical test:** Two-sided exact binomial test (Clopper-Pearson)

```python
from scipy.stats import binomtest

# H0: p = 0.5 (no preference between conditions)
result = binomtest(neuro_wins, neuro_wins + stock_wins, 0.5, alternative='two-sided')
```

### 7.2 Confidence Intervals

- **Wilson score interval** for binomial proportions (preferred over normal approximation for small samples)
- Report 95% CI for NEURO win rate per dimension

```python
from statsmodels.stats.proportion import proportion_confint
ci_low, ci_high = proportion_confint(neuro_wins, total_decisive, alpha=0.05, method='wilson')
```

### 7.3 Inter-Rater Agreement: Fleiss' κ

Assess consistency across evaluators:

```python
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters

# Build matrix: rows = pairs, columns = [n_STOCK, n_NEURO, n_TIE]
# For each dimension separately
matrix = build_fleiss_matrix(responses, dimension='quality')
kappa = fleiss_kappa(matrix)
```

**Interpretation (Landis & Koch, 1977):**

| κ Range | Agreement Level |
|---------|----------------|
| < 0.00 | Poor |
| 0.00–0.20 | Slight |
| 0.21–0.40 | Fair |
| 0.41–0.60 | Moderate |
| 0.61–0.80 | Substantial |
| 0.81–1.00 | Almost Perfect |

### 7.4 Per-Arc Analysis

Disaggregate by emotional arc to identify whether certain arcs drive the effect:

```python
for arc in ARCS:
    arc_responses = filter_by_arc(all_responses, arc)
    neuro_rate = compute_win_rate(arc_responses)
    p_value = binomtest(neuro_wins, total_decisive, 0.5)
```

### 7.5 Correlation with Automated Metrics

Compare human preference with existing automated metrics:

| Human Dimension | Automated Metric | Expected Correlation |
|----------------|-----------------|---------------------|
| Visual Quality | LPIPS (lower = better) | Positive |
| Emotional Accuracy | CLIP image-text score | Positive |
| Overall Preference | Weighted combination | Moderate |

**Method:** Spearman rank correlation between per-pair human preference rate and automated metric values.

### 7.6 Multiple Comparison Correction

- **Primary test:** 3 binomial tests (one per dimension) → Bonferroni correction: α_adj = 0.05/3 = 0.0167
- **Per-arc analysis:** 7 arcs × 3 dimensions = 21 tests → report uncorrected p-values with FDR-BH adjusted q-values

---

## 8. Results Reporting Template

### 8.1 Aggregate Results Table

| Dimension | STOCK Wins | NEURO Wins | Ties | NEURO Win Rate (excl. ties) | 95% CI | p-value (binomial) |
|-----------|-----------|-----------|------|---------------------------|--------|-------------------|
| Visual Quality | _/70 | _/70 | _/70 | _% | [_, _] | _ |
| Emotional Accuracy | _/70 | _/70 | _/70 | _% | [_, _] | _ |
| Overall Preference | _/70 | _/70 | _/70 | _% | [_, _] | _ |

(Filled per 5 evaluators × 14 pairs = 70 total judgments per dimension)

### 8.2 Inter-Rater Agreement

| Dimension | Fleiss' κ | 95% CI | Agreement Level |
|-----------|----------|--------|----------------|
| Visual Quality | _ | [_, _] | _ |
| Emotional Accuracy | _ | [_, _] | _ |
| Overall Preference | _ | [_, _] | _ |

### 8.3 Per-Arc Breakdown

| Arc | NEURO Win Rate (Quality) | NEURO Win Rate (Emotion) | NEURO Win Rate (Preference) |
|-----|------------------------|-------------------------|---------------------------|
| realization | _% | _% | _% |
| contemplation | _% | _% | _% |
| determination | _% | _% | _% |
| confidence | _% | _% | _% |
| respect | _% | _% | _% |
| passion | _% | _% | _% |
| tension | _% | _% | _% |

### 8.4 Human vs. Automated Metric Correlation

| Human Dimension | Automated Metric | Spearman ρ | p-value |
|----------------|-----------------|-----------|---------|
| Visual Quality | LPIPS | _ | _ |
| Emotional Accuracy | CLIP image-text | _ | _ |
| Overall Preference | CLIP text similarity | _ | _ |

---

## 9. Ethical Considerations

### 9.1 Informed Consent

- Evaluators receive clear description of the study purpose and procedure
- Estimated time commitment (~30 minutes) disclosed upfront
- Compensation structure (50 RTC) stated clearly
- Evaluators may withdraw at any time before submission

### 9.2 Data Privacy

- **No personal data collected** beyond a self-chosen evaluator ID
- Video content is non-sensitive (Victorian-era portraits, landscapes, architecture)
- Individual responses are not published — only aggregated statistics
- JSON results stored locally by evaluators until voluntarily shared

### 9.3 Conflict of Interest

- Evaluators must not be affiliated with the author's lab (Elyan Labs)
- Evaluators must not have contributed to the prompt design or model fine-tuning
- Compensation is fixed per evaluator (no outcome-dependent payment)

---

## 10. Timeline

| Phase | Dates | Activity |
|-------|-------|----------|
| Recruitment | Ongoing | Accept qualified evaluators (max 5) |
| Evaluation | Upon assignment | Evaluators complete 42 judgments (~30 min) |
| Collection | Upon submission | Receive and validate JSON results |
| Analysis | After 3+ submissions | Compute preference rates, Fleiss' κ, correlations |
| Reporting | Camera-ready deadline | Include results in supplementary materials |

---

## 11. Deliverables

1. **Evaluation form:** `evaluations/evaluation_form.html` — self-contained, offline-capable HTML form
2. **This document:** `evaluations/human_evaluation_study.md` — complete study design
3. **Results template:** Section 8 above — pre-formatted tables for reporting
4. **Analysis script:** See `human_eval/README.md` for Python analysis code

---

## References

1. Mantiuk, R. K., Tomaszewska, A., & Mantiuk, R. (2012). Comparison of four subjective methods for image quality assessment. *Computer Graphics Forum*, 31(8), 2478–2491.
2. Perez-Ortiz, M., Mikhailiuk, A., Zerman, E., Hulusic, V., Valenzise, G., & Mantiuk, R. K. (2019). From pairwise comparisons and rating to a unified quality scale. *IEEE Transactions on Image Processing*, 29, 1132–1145.
3. Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for categorical data. *Biometrics*, 33(1), 159–174.
4. ITU-R BT.500-13 (2012). Methodology for the subjective assessment of the quality of television pictures. *International Telecommunication Union*.
