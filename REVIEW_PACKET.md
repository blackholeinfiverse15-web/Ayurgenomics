# REVIEW_PACKET.md
**Project:** CRISPR RL v4 — Ayurgenomic Intelligence Engine
**Submitted by:** Rutuja — CRISPR Intelligence & Bioinformatics
**Task:** Task 9 — Conscious Bioinformatics Deployment Sprint
**Date:** 2025-01-01
**Version:** 4.0.0

---

## 1. ENTRY POINT

**Backend entry:**
```
Path: crispr_rl/api.py
```
FastAPI microservice that exposes all REST endpoints (`/crispr/design`, `/crispr/feedback`, `/crispr/metrics`, `/crispr/marma`, `/crispr/config`).
Initialises the three shared singletons (AyurMapping, AyurWeightEngine, QLearningAgent) on startup and serves the React frontend at `/`.

**Start command:**
```bash
uvicorn crispr_rl.api:app --reload --host 0.0.0.0 --port 8000
```

**Swagger UI (interactive docs):**
```
http://localhost:8000/docs
```

---

## 2. CORE EXECUTION FLOW (MAX 3 FILES)

**File 1:**
```
Path: crispr_rl/features.py
```
Scans a raw DNA sequence for PAM sites (NGG / NNGRRT / NGTN) on both strands and extracts gRNA candidates.
Computes GC content, seed-region GC, homopolymer run length, rule-based efficiency score, and off-target risk for every candidate.

---

**File 2:**
```
Path: crispr_rl/rl_agent.py
```
Tabular Q-learning agent that selects the optimal gRNA from a scored shortlist using an ε-greedy policy.
Shapes rewards from efficiency uplift, specificity, AyurHarmonyScore, and caution-level penalties; updates the Q-table after every selection and incorporates human feedback from `/crispr/feedback`.

---

**File 3:**
```
Path: crispr_rl/ayur_layer/mapping.py
```
Loads `ayur_map.json` and provides fast gene → dosha, gene → marma, and prakriti-profile lookups used by the scoring engine.
Builds in-memory indices at startup so every query is O(1) dictionary lookup with no file I/O at request time.

---

## 3. LIVE FLOW (REAL EXECUTION)

**User action:**
```
POST /crispr/design
```

**System flow:**
```
HTTP Request
    │
    ▼
crispr_rl/api.py  ← validates request, resolves demo sequence if none provided
    │
    ▼
crispr_rl/features.py  ← PAMScanner scans both strands, returns gRNACandidate list
    │
    ▼
crispr_rl/ayur_layer/weights.py  ← AyurWeightEngine scores every candidate:
    │                                 AHS = 0.35×eff + 0.30×spe + 0.20×marma + 0.15×prakriti
    ▼
crispr_rl/rl_agent.py  ← QLearningAgent selects best candidate (ε-greedy),
    │                     computes reward, updates Q-table
    ▼
logs/requests.jsonl  ← structured JSONL log written
    │
    ▼
JSON Response  ← ranked candidates + rl_selected + harmony_index + latency_ms
```

**REAL JSON response (live system execution — run_id: bhiv-review-001):**

```bash
curl -X POST http://localhost:8000/crispr/design \
  -H "Content-Type: application/json" \
  -d '{
    "run_id": "bhiv-review-001",
    "targets": [{"gene_id": "BDNF", "pam_type": "NGG"}],
    "ayur_profile": "vata_dominant",
    "top_k": 2
  }'
```

```json
{
    "run_id": "bhiv-review-001",
    "request_id": "3085a894-bdf9-47eb-8164-28582ee719a9",
    "ayur_profile": "vata_dominant",
    "latency_ms": 1.34,
    "avg_harmony_index": 0.908,
    "results": [
        {
            "gene_id": "BDNF",
            "top_candidates": [
                {
                    "gene_id": "BDNF",
                    "grna_sequence": "TGCCCTATTTGTTTAGCCAG",
                    "pam_sequence": "TGG",
                    "pam_type": "NGG",
                    "strand": "+",
                    "position": 95,
                    "gc_content": 0.45,
                    "seed_gc": 0.4167,
                    "homopolymer_run": 3,
                    "efficiency_score": 1.0,
                    "specificity_score": 0.9,
                    "marma_alignment": 0.9,
                    "prakriti_congruence": 0.72,
                    "harmony_index": 0.908,
                    "caution_level": "critical",
                    "notes": [
                        "BDNF aligns with priority marma: sthapani (avg sensitivity=0.90)",
                        "BDNF is a priority gene for 'vata_dominant' — applying 1.2× boost.",
                        "Caution level 'critical' (harmony=0.908, max_marma_sensitivity=0.90)."
                    ]
                },
                {
                    "gene_id": "BDNF",
                    "grna_sequence": "GCCCTATTTGTTTAGCCAGT",
                    "pam_sequence": "GGG",
                    "pam_type": "NGG",
                    "strand": "+",
                    "position": 96,
                    "gc_content": 0.45,
                    "seed_gc": 0.4167,
                    "homopolymer_run": 3,
                    "efficiency_score": 1.0,
                    "specificity_score": 0.9,
                    "marma_alignment": 0.9,
                    "prakriti_congruence": 0.72,
                    "harmony_index": 0.908,
                    "caution_level": "critical"
                }
            ],
            "rl_selected": {
                "gene_id": "BDNF",
                "grna_sequence": "TGCCCTATTTGTTTAGCCAG",
                "pam_sequence": "TGG",
                "pam_type": "NGG",
                "strand": "+",
                "position": 95,
                "gc_content": 0.45,
                "seed_gc": 0.4167,
                "homopolymer_run": 3,
                "efficiency_score": 1.0,
                "specificity_score": 0.9,
                "marma_alignment": 0.9,
                "prakriti_congruence": 0.72,
                "harmony_index": 0.908,
                "caution_level": "critical"
            },
            "rl_reward": 1.2406
        }
    ]
}
```

---

## 4. WHAT WAS BUILT IN THIS TASK

### Built

- **RL agent** — Tabular Q-learning (`QLearningAgent`) with ε-greedy policy, reward shaping, Q-table checkpoint save/load, human feedback integration via `POST /crispr/feedback`
- **AyurHarmonyScore** — Composite scorer (`AyurWeightEngine`) combining efficiency, specificity, marma alignment, and prakriti congruence into a single 0–1 index with caution levels (low / medium / high / critical)
- **Ayurvedic ontology** — `ayur_map.json` encoding 3 doshas, 7 marma points with organ/gene/sensitivity mappings, and 6 prakriti profiles; `mapping.py` lookup engine; `weights.py` scoring heuristics
- **API endpoints** — `POST /crispr/design`, `POST /crispr/feedback`, `GET /crispr/metrics`, `GET /crispr/marma`, `GET /crispr/prakriti/{name}`, `GET /crispr/config`, `GET /health`
- **Logging system** — Structured JSONL logs written to `logs/requests.jsonl` on every request (run_id, request_id, ayur_profile, harmony_score, latency_ms, timestamp_ms)
- **PAM scanner** — Dual-strand NGG / NNGRRT / NGTN scanner with molecular feature extraction (`features.py`)
- **React frontend** — Single-file SPA (`frontend/index.html`) with 5 panels: Design, Metrics, Feedback, Visualisation, Marma Explorer
- **Test suite** — 75 pytest tests across `test_features.py`, `test_rl.py`, `test_ayur.py` — all passing
- **Docker packaging** — `Dockerfile` (multi-stage Python 3.10-slim), `docker-compose.yml`
- **CI/CD** — GitHub Actions pipeline: lint → test → docker build → push to GHCR

### What was NOT touched

- No external genomic database queries (NCBI, UCSC, Ensembl) — all sequences are caller-provided or demo
- No neural RL policy (DQN, PPO) — agent is deliberately tabular for reproducibility
- No wet-lab or in-vitro validation pipeline
- No user authentication or role-based access control
- No database (Postgres, Redis) — all state is in-memory; only JSONL logs persist
- No Bayesian off-target model — off-target risk is rule-based only

---

## 5. FAILURE CASES

### Invalid DNA sequence (non-ACGT characters)

**Input:** `"sequence": "XXXXXXXXXXXXXXXXXXXXXXXX"`

**Behaviour:** PAMScanner returns 0 candidates. API responds:
```json
{"gene_id": "BDNF", "candidates": [], "message": "No PAM sites found."}
```
No crash. No 500 error. Graceful empty result.

---

### Empty targets array

**Input:** `"targets": []`

**Behaviour:** Pydantic validation rejects immediately with HTTP 422:
```json
{
  "detail": [{"type": "too_short", "loc": ["body","targets"],
               "msg": "List should have at least 1 item after validation, not 0"}]
}
```

---

### Invalid PAM type

**Input:** `"pam_type": "XYZ"`

**Behaviour:** Pydantic field validator raises HTTP 422:
```json
{
  "detail": [{"msg": "Value error, pam_type must be NGG, NNGRRT, or NGTN"}]
}
```

---

### Unknown prakriti profile

**Input:** `"ayur_profile": "lunar_dominant"`

**Behaviour:** Profile normalised, falls back to tridoshic dosha mix (equal thirds). No crash. Logs warning. Design continues.

---

### RL agent receives empty candidate list

**Behaviour:** `QLearningAgent.choose_grna()` raises `ValueError: No candidates provided to RL agent.` API catches and returns HTTP 422 with error detail.

---

### API crash / server unreachable

**Behaviour:** FastAPI exception handlers return structured JSON errors. Uvicorn worker restarts automatically under Docker Compose (`restart: unless-stopped`). All requests that completed before crash are already written to `logs/requests.jsonl`.

---

## 6. PROOF

### Terminal execution proof (live run)

```
$ python3 -c "
import sys; sys.path.insert(0, '.')
from crispr_rl.features import PAMScanner
from crispr_rl.ayur_layer import AyurMapping, AyurWeightEngine
from crispr_rl.rl_agent import QLearningAgent

mapping = AyurMapping()
engine  = AyurWeightEngine(mapping=mapping)
agent   = QLearningAgent(seed=42)

scanner = PAMScanner('NGG')
raw     = scanner.scan('BDNF', 'ATGACCATCCTTTTCCTTACTATGGTTATTTCATATAAATAC...')
scored  = engine.rank_candidates([c.as_dict() for c in raw[:10]], 'vata_dominant')
chosen, reward = agent.choose_grna('bhiv-001', 'req-001', scored, 'vata_dominant')
metrics = agent.get_metrics()

print(f'gRNA       : {chosen[\"grna_sequence\"]}')
print(f'AHS        : {chosen[\"harmony_index\"]}')
print(f'Caution    : {chosen[\"caution_level\"]}')
print(f'RL reward  : {reward:.4f}')
print(f'Uplift     : {metrics[\"efficiency_uplift_pct\"]}%')
print(f'Episodes   : {metrics[\"total_episodes\"]}')
"
```

**Output:**
```
INFO  AyurMapping loaded: 3 doshas, 7 marma points, 6 prakriti profiles
INFO  PAMScanner [BDNF, NGG]: found 22 candidates in 326-nt sequence.
INFO  RL chose gRNA TGCCCTATTTGT for gene BDNF | reward=1.2406 | AHS=0.908

gRNA       : TGCCCTATTTGTTTAGCCAG
AHS        : 0.908
Caution    : critical
RL reward  : 1.2406
Uplift     : 81.82%
Episodes   : 1
```

### Test suite proof

```
$ pytest tests/ -v

platform win32 -- Python 3.12, pytest-8.4.2

tests/test_ayur.py::TestAyurMapping::test_loads_without_error          PASSED
tests/test_ayur.py::TestAyurMapping::test_known_gene_has_dosha         PASSED
tests/test_ayur.py::TestAyurMapping::test_hridaya_highest_sensitivity  PASSED
tests/test_ayur.py::TestAyurWeightEngine::test_harmony_index_in_range  PASSED
tests/test_ayur.py::TestAyurWeightEngine::test_weights_sum_to_one      PASSED
tests/test_features.py::TestPAMScanner::test_finds_candidates          PASSED
tests/test_features.py::TestPAMScanner::test_sorted_by_efficiency      PASSED
tests/test_rl.py::TestQLearningAgent::test_reproducibility_same_seed   PASSED
tests/test_rl.py::TestQLearningAgent::test_checkpoint_round_trip       PASSED
... (75 total)

========================= 75 passed in 0.68s =========================
```

---

## File Manifest

| File | Purpose | Lines |
|---|---|---|
| `crispr_rl/api.py` | FastAPI microservice, all endpoints | ~200 |
| `crispr_rl/features.py` | PAM scanner, molecular feature extraction | ~220 |
| `crispr_rl/rl_agent.py` | Q-learning agent, reward shaping | ~320 |
| `crispr_rl/ayur_layer/mapping.py` | Dosha/marma/prakriti lookup engine | ~155 |
| `crispr_rl/ayur_layer/weights.py` | AyurHarmonyScore computation | ~220 |
| `crispr_rl/ayur_layer/ayur_map.json` | Ayurvedic-genomic ontology data | ~180 |
| `crispr_rl/__init__.py` | Package exports | ~15 |
| `crispr_rl/ayur_layer/__init__.py` | Layer exports | ~8 |
| `tests/test_features.py` | 30 tests — PAM scanner, GC%, features | ~160 |
| `tests/test_rl.py` | 22 tests — reward shaping, reproducibility | ~140 |
| `tests/test_ayur.py` | 23 tests — AHS logic, mapping, weights | ~170 |
| `frontend/index.html` | React SPA — 5 UI panels | ~700 |
| `scripts/visualize_scores.py` | Baseline vs RL vs AHS chart generator | ~165 |
| `Dockerfile` | Multi-stage Python 3.10-slim build | ~30 |
| `docker-compose.yml` | Local + Akash integration | ~35 |
| `pyproject.toml` | pip install crispr-rl | ~55 |
| `.github/workflows/ci.yml` | Lint → test → docker build → push | ~75 |

---

## Integration Contacts

| Person | Integration Point | Endpoint |
|---|---|---|
| **Akash** | Backend test | `POST /crispr/design`, `GET /crispr/config`, `GET /crispr/metrics` |
| **Tejaswi** | Personalisation layer | `ayur_profile` field in design request |
| **Vinayak Tiwari** | Testing sheet | All endpoints above + `GET /health` for liveness |

**Base URL:** `http://localhost:8000` (local) or Docker Compose service `crispr_rl:8000`

---

*Submitted by Rutuja — CRISPR Intelligence & Bioinformatics | Task 9 | CRISPR RL v4.0.0*
