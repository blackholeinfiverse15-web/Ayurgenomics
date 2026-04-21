# CRISPR RL v4 — Ayurgenomic CRISPR Intelligence Engine

> *The first RL-based gRNA optimiser guided not just by molecular efficiency, but by bioethical harmony derived from Ayurvedic biological intelligence.*

[![CI](https://github.com/rutuja/crispr-rl-v4/actions/workflows/ci.yml/badge.svg)](https://github.com/rutuja/crispr-rl-v4/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com)

---

## North Star

Create the first **Ayurgenomic CRISPR Intelligence Engine** — where RL-based gRNA optimisation is guided not just by molecular efficiency, but by bioethical harmony derived from Ayurvedic frameworks like *prakriti*, *dosha* balance, and *marma*-linked gene clusters.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Running the API](#running-the-api)
4. [API Reference](#api-reference)
5. [Ayurvedic Mapping Layer](#ayurvedic-mapping-layer)
6. [AyurHarmonyScore](#ayurharmonyscore)
7. [RL Agent](#rl-agent)
8. [Testing](#testing)
9. [Visualisation](#visualisation)
10. [Docker Deployment](#docker-deployment)
11. [Integration Hooks](#integration-hooks)
12. [Metrics & Monitoring](#metrics--monitoring)
13. [Reflection Block](#reflection-block)
14. [References](#references)

---

## Project Structure

```
crispr_rl_v4/
├── crispr_rl/
│   ├── __init__.py              # Package exports
│   ├── api.py                   # FastAPI microservice
│   ├── features.py              # PAM scanner, GC%, feature extraction
│   ├── rl_agent.py              # Q-learning agent + reward shaping
│   └── ayur_layer/
│       ├── __init__.py
│       ├── ayur_map.json        # Ayurvedic–genomic ontology
│       ├── mapping.py           # Dosha / marma / prakriti lookups
│       └── weights.py           # AyurHarmonyScore computation
├── tests/
│   ├── test_features.py         # PAM scanner & molecular features
│   ├── test_rl.py               # Reward shaping & reproducibility
│   └── test_ayur.py             # AyurHarmonyScore logic
├── scripts/
│   └── visualize_scores.py      # Baseline vs RL vs AyurHarmony chart
├── .github/workflows/ci.yml     # GitHub Actions CI/CD
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## Installation

### From source (recommended for development)

```bash
git clone https://github.com/rutuja/crispr-rl-v4.git
cd crispr_rl_v4

# Install with dev + viz extras
pip install -e ".[all]"
```

### Minimal install (API only)

```bash
pip install -e .
```

---

## Running the API

```bash
uvicorn crispr_rl.api:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## API Reference

### `GET /health`
Liveness probe.
```json
{"status": "ok", "version": "4.0.0"}
```

---

### `POST /crispr/design`

Accept gene IDs + Ayurvedic profile → returns AyurHarmonyScore-ranked gRNA candidates.

**Request body:**
```json
{
  "run_id": "run-001",
  "targets": [
    {
      "gene_id": "BDNF",
      "sequence": "ATGACCATCCTTTTCCTTAC...",
      "pam_type": "NGG"
    }
  ],
  "ayur_profile": "vata_dominant",
  "top_k": 3
}
```

**Supported `ayur_profile` values:**
| Profile | Dosha Mix | Therapeutic Focus |
|---|---|---|
| `vata_dominant` | Vata 60% | Neural stability, grounding |
| `pitta_dominant` | Pitta 60% | Metabolic balance, clarity |
| `kapha_dominant` | Kapha 60% | Growth regulation, immunity |
| `vata_pitta` | Vata+Pitta 45% each | Stress resilience |
| `pitta_kapha` | Pitta+Kapha 45% each | Metabolic-structural balance |
| `tridoshic` | Equal thirds | Full systemic harmony |

**Example response:**
```json
{
  "run_id": "run-001",
  "request_id": "uuid-...",
  "ayur_profile": "vata_dominant",
  "latency_ms": 42.1,
  "avg_harmony_index": 0.7234,
  "results": [
    {
      "gene_id": "BDNF",
      "top_candidates": [...],
      "rl_selected": {
        "gene_id": "BDNF",
        "grna_sequence": "ACGTACGTACGTACGTACGT",
        "efficiency_score": 0.82,
        "specificity_score": 0.78,
        "harmony_index": 0.7643,
        "caution_level": "medium",
        "notes": [...]
      },
      "rl_reward": 0.4231
    }
  ]
}
```

---

### `POST /crispr/feedback`

Submit human expert feedback to improve the RL agent.

```json
{
  "grna_sequence": "ACGTACGTACGTACGTACGT",
  "human_score": 0.9,
  "notes": "Excellent on-target activity observed in HEK293T cells"
}
```

`human_score` ∈ [-1, 1]. Positive scores reinforce the agent; negative scores penalise.

---

### `GET /crispr/metrics`

Returns aggregate performance metrics.

```json
{
  "status": "ok",
  "metrics": {
    "total_episodes": 147,
    "avg_reward": 0.3821,
    "avg_harmony_index": 0.7105,
    "avg_efficiency": 0.7234,
    "efficiency_uplift_pct": 31.5,
    "failed_calls": 0,
    "epsilon": 0.0612,
    "q_table_states": 24,
    "baseline_efficiency": 0.55
  }
}
```

---

### `GET /crispr/config`

Returns agent configuration and available prakriti profiles.

---

## Ayurvedic Mapping Layer

The `ayur_layer/` module encodes Vedic biological intelligence as structured, machine-readable ontology:

### Dosha–Gene Correlations

| Dosha | Biological Domain | Example Genes |
|---|---|---|
| **Vata** | Neural signalling, movement | BDNF, SCN1A, DRD2, GABRA1 |
| **Pitta** | Metabolism, inflammation, enzymes | PPARG, TNF, CYP1A2, NFKB1 |
| **Kapha** | Growth, structure, immunity | COL1A1, TGFB1, IGF1, VEGFA |

### Marma–Organ–Gene Linkage

Seven *marma* points are mapped to organs and gene clusters:

| Marma | Organ | Key Genes | Edit Sensitivity |
|---|---|---|---|
| Hridaya | Heart | MYH7, TNNT2, SCN5A | 0.95 (Critical) |
| Sthapani | Frontal brain | BDNF, NTRK2, SIRT1 | 0.90 (Critical) |
| Nabhi | Gut-metabolic axis | PPARG, FTO, MC4R | 0.75 (High) |
| Basti | Reproductive | BRCA1, BRCA2, TP53 | 0.85 (High) |
| Gulpha | Adrenal/endocrine | NR3C1, CRH, HSD11B1 | 0.80 (High) |
| Talahrida | Liver/detox | CYP3A4, CYP2E1, UGT1A1 | 0.70 (Medium) |
| Krikatika | Thyroid/cervical | TSHR, DIO2, TPO | 0.78 (High) |

---

## AyurHarmonyScore

```
AHS = 0.35 × efficiency + 0.30 × specificity + 0.20 × marma_alignment + 0.15 × prakriti_congruence
```

Where:
- **efficiency** — on-target prediction (GC-content adjusted)
- **specificity** — 1 − off-target risk
- **marma_alignment** — sensitivity-weighted match between gene and patient's priority marma points
- **prakriti_congruence** — cosine-like similarity between gene's dosha signature and patient's dosha mix (with priority gene boost)

### Caution Levels

| Level | Condition |
|---|---|
| `critical` | Marma sensitivity ≥ 0.90 **or** AHS < 0.40 |
| `high` | Marma sensitivity ≥ 0.80 **or** AHS < 0.55 |
| `medium` | Marma sensitivity ≥ 0.70 **or** AHS < 0.70 |
| `low` | AHS ≥ 0.70 and max sensitivity < 0.70 |

---

## RL Agent

CRISPR RL uses a **tabular Q-learning** agent (no GPU required):

- **State:** `(GC bin, seed-GC bin, PAM type)` — 75 discrete states
- **Action:** index into top-5 ranked candidates
- **Reward:**
  ```
  R = efficiency_uplift + 0.5×specificity + 0.3×harmony − caution_penalty
  ```
- **ε-greedy exploration** with decay (ε₀=0.15 → εmin=0.02)
- Human feedback incorporated via `/crispr/feedback`

Checkpoints saved/loaded as JSON for full reproducibility.

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=crispr_rl --cov-report=term-missing

# Individual suites
pytest tests/test_features.py -v   # PAM scanner, GC%
pytest tests/test_rl.py -v         # Reward shaping & reproducibility
pytest tests/test_ayur.py -v       # AyurHarmonyScore logic
```

---

## Visualisation

```bash
# Install viz extras first
pip install -e ".[viz]"

# Generate comparison chart
python scripts/visualize_scores.py --n 20 --output plots/comparison.png

# Custom prakriti profiles
python scripts/visualize_scores.py \
  --n 25 \
  --profiles vata_dominant pitta_dominant tridoshic \
  --output plots/my_comparison.png
```

Produces a two-panel chart comparing:
- Baseline (efficiency only)
- RL-adjusted score
- AyurHarmonyScore per prakriti profile

---

## Docker Deployment

### Local (single container)

```bash
docker build -t crispr-rl-v4 .
docker run -p 8000:8000 -v $(pwd)/logs:/app/logs crispr-rl-v4
```

### Docker Compose (with Akash backend integration)

```bash
docker compose up --build
```

The service exposes:
- API: `http://localhost:8000`
- Logs: `./logs/requests.jsonl`

---

## Integration Hooks

| Partner | Endpoints | Notes |
|---|---|---|
| **Akash** | `/crispr/design`, `/crispr/config`, `/crispr/metrics` | Backend test integration via Docker Compose |
| **Tejaswi** | `ayur_profile` field in `/crispr/design` | Prakriti profile drives agent personalisation (balance, calm, focus) |
| **Dashboard** | `/crispr/metrics` → `avg_harmony_index` | Add "AyurHarmony Index" panel alongside RL metrics |

---

## Metrics & Monitoring

Structured JSONL logs written to `logs/requests.jsonl`:

```jsonl
{"run_id": "...", "request_id": "...", "ayur_profile": "vata_dominant", "harmony_score": 0.74, "latency_ms": 38.2, "timestamp_ms": 1720000000000, "n_genes": 2}
```

Live metrics via `GET /crispr/metrics` include:
- `efficiency_uplift_pct` — % improvement vs random-pick baseline (target ≥ 15%)
- `avg_harmony_index` — AyurHarmony composite score
- `failed_calls` — error count
- `epsilon` — exploration rate (indicates learning maturity)

---

## Reflection Block

### 🌿 Humility — One Limitation

The current RL agent uses **tabular Q-learning** with a coarse state space (GC bin × PAM type). This severely limits generalisation: two gRNAs with the same GC bin but very different seed sequences may be treated identically. A more honest approach would use a neural policy (DQN or PPO) that directly encodes the full 20-nt sequence. This is the most significant technical debt in v4.

Additionally, the Ayurvedic-genomic mappings, while grounded in published literature, are **approximations**. Marma sensitivity scores were hand-curated — they express a philosophical caution, not empirically validated edit-risk probabilities.

### 🙏 Gratitude — Acknowledgement

This work stands on the shoulders of:
- The researchers behind *"Prakriti-based Genomics and Personalized Medicine"* (Frontiers in Bioscience, 2020) for demonstrating that Ayurvedic constitution types have measurable genomic correlates.
- The FastAPI and Pydantic communities for making production-grade Python APIs accessible.
- Every open-source CRISPR computational tool (especially the Doench 2016 efficiency model) that this heuristic system learns from.
- Tejaswi and Akash for their integration work and questions that kept the design honest.

### 🔍 Honesty — One Truth About What Didn't Work

The original plan included a **Bayesian off-target risk model** that would ingest actual genomic coordinates and query the UCSC genome browser for known SNP overlaps in seed regions. This was scoped out during Day 3–4 because the CORS-restricted browser API proved impractical within the sprint timeline without a dedicated backend proxy. The current `estimate_off_target_risk()` is rule-based only and will over-estimate specificity for genes in repeat-rich genomic contexts. This should be the first thing upgraded in v5.

---

## References

1. Bhalerao, S. et al. (2021). *Mapping Ayurveda to Modern Genetics.* NCBI/PMC.
2. Mallick, H.N. et al. (2019). *Marma Points: Neurophysiological Correlates.* Journal of Integrative Medicine, 17(5), 371–379.
3. Prasher, B. et al. (2020). *Prakriti-based Genomics and Personalized Medicine.* Frontiers in Bioscience, 25, 1064–1075.
4. [FastAPI Official Docs](https://fastapi.tiangolo.com/)
5. [Pytest Quickstart](https://docs.pytest.org/en/latest/)

---

*Built with 🌿 by Rutuja — CRISPR Intelligence & Bioinformatics | CRISPR RL v4.0.0*
#   A y u r g e n o m i c s  
 #   A y u r g e n o m i c s  
 # Ayurgenomics
