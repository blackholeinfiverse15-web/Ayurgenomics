"""
api.py — FastAPI Microservice for CRISPR RL v4

Endpoints
---------
POST /crispr/design      — Accept gene IDs + ayur_profile → ranked gRNA designs
POST /crispr/feedback    — Receive human feedback on a gRNA selection
GET  /crispr/metrics     — Return uplift %, latency, AyurHarmony index
GET  /crispr/config      — Return current agent config and prakriti profiles
GET  /crispr/marma       — Return full marma point map
GET  /crispr/prakriti/{profile} — Return single prakriti profile details
GET  /health             — Health check
GET  /                   — Serve React frontend

Run with:
    uvicorn crispr_rl.api:app --reload --host 0.0.0.0 --port 8000
"""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .ayur_layer import AyurMapping, AyurWeightEngine
from .features import PAMScanner
from .rl_agent import QLearningAgent

# ------------------------------------------------------------------ #
# Logging
# ------------------------------------------------------------------ #

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
logger = logging.getLogger("crispr_rl.api")
_jsonl_log_path = LOG_DIR / "requests.jsonl"


def _log_jsonl(record: dict) -> None:
    with open(_jsonl_log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ------------------------------------------------------------------ #
# Singletons
# ------------------------------------------------------------------ #

_mapping = AyurMapping()
_weight_engine = AyurWeightEngine(mapping=_mapping)
_agent = QLearningAgent(seed=42)

_DEMO_SEQUENCES: dict[str, str] = {
    "BDNF":  "ATGACCATCCTTTTCCTTACTATGGTTATTTCATATAAATACTATAATGTTTTTTCTCAAAAAATGAAAAGTTTATTTGTTTCCACTTTGCCTTCTGCCCTATTTGTTTAGCCAGTGGGGAAAGCAAACATGTCTATGAGGGTTCGGAGAGGTGAGAAGAGTTGCAGTACGTAACACTGTCACTGTGTTGTGGAGTAGTCGCCAGCATCTTTTGGTTGCTTGTTCTCTGTGTAATTTGAAAAGAAAGAGCCTGTGTGTAAAGTGTGAATCAGTTCATTGTGTTGAATAAAGTTTTGATCAATCAAATGTTATGTTTTTATAAAGGG",
    "PPARG": "ATGGGTGAAACTCTGGGAGATTCTCCTATTGACCCAGAAAGCGATTCCTTCACTGATCGAACTCCAGCTTCCATCCCCAGCTTCAGGCCCAGCCAAAGCCCTTATAATGGAAGACAAGGACTCAGACTATGATCGATTCTATGAAAATCGCTTCATGTCCTCGCAGGTGGAGATCGACAGCAACAAGATTGAGCTGGTGAAGGCGGAGAAG",
    "SCN1A": "ATGGGCAATCGGAGCGGCAGCGGCAGCGGCAGCGGCCATGGGGAGCAGCGGCAGCGGCAGCAGCAGCAGCAGCAAGAAGAGGAGAAGGGGAAGGCGGAGCAGAGCAGCAGCAGCAGCAGCAGCGGTGGCAGCAGCATGAGCAGCAGCAGCAGCGGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAG",
    "BRCA1": "ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATCTTAGAGTGTCCCATCTGTCTGGAGTTGATCAAGGAACCTGTCTCCACAAAGTGTGACCACATATTTTGCAAATTTTGCATGCTGAAACTTCTCAACCAGAAGAAAGGGCCTTCACAGTGTCCTTTATGTAAGAATGATATAACCAAAAG",
    "TP53":  "ATGGAGGAGCCGCAGTCAGATCCTAGCGTTGAGTCAGAACATGGCAGAAACACGCTTCCGAAAATGTTTGGGAAGGGACAGAAACACGTAGATTGGGGGAGGAGACGGAACAGCTTTGAGGTGCGTGTTTGTGCCTGTCCTGGGAGAGACCGGCGCACAGAGGAAGAGAATCTCCGCAAGAAAGTGGAGCCTCACCATGAGCGACTGCCC",
    "MYH7":  "ATGGATGCAGACGAGAATGAGAAGAAGATCAAGCAGCTGAAGGAGGAGCTGGACAAGGTCATCACCAAGGAGATCCGGGAGCAGCTGGAGAAGATGCAGATCACGCAGAGCAAGGCCCAGTTCCTGCAGGACCTGGCCAACAAGCTGCAGCAGCTGAACAACAACAAGGGCGTGCAGGACATCTACGAGATGGAGAACATCGAGATGG",
    "TGFB1": "ATGCCGCCCTCCGGGCTGCGGCTGCGGGCGGCGCTGCTGCTGCTGTGCCTGGGGCTGCTGGCGGCGGGGCAGCAGGCGGCGGCGGCGGCGGCGGCAGCCGCCGCCGCCGCCGCCGCGGCGGCGGCTGCGGCCCCGGCAGCCCAGCGCCGGCCCCGGCCGCGGCGGCGGCCGCAGCCGCCGCAGAGCAGCGCCGCGGCGGCGGCGG",
}


def _get_sequence(gene_id: str, provided: Optional[str]) -> str:
    if provided:
        return provided
    gene_upper = gene_id.upper()
    if gene_upper in _DEMO_SEQUENCES:
        return _DEMO_SEQUENCES[gene_upper]
    import hashlib
    seed_bytes = hashlib.sha256(gene_upper.encode()).digest()
    return "".join("ACGT"[b % 4] for b in seed_bytes * 10)


# ------------------------------------------------------------------ #
# Pydantic models
# ------------------------------------------------------------------ #

class GeneTarget(BaseModel):
    gene_id: str = Field(..., description="Gene symbol e.g. 'BDNF'")
    sequence: Optional[str] = Field(None)
    pam_type: str = Field("NGG")

    @field_validator("pam_type")
    @classmethod
    def validate_pam(cls, v):
        if v.upper() not in {"NGG", "NNGRRT", "NGTN"}:
            raise ValueError("pam_type must be NGG, NNGRRT, or NGTN")
        return v.upper()


class DesignRequest(BaseModel):
    run_id: Optional[str] = None
    targets: list[GeneTarget] = Field(..., min_length=1, max_length=10)
    ayur_profile: str = Field("tridoshic")
    top_k: int = Field(3, ge=1, le=10)


class FeedbackRequest(BaseModel):
    grna_sequence: str
    human_score: float = Field(..., ge=-1.0, le=1.0)
    notes: Optional[str] = None


# ------------------------------------------------------------------ #
# App
# ------------------------------------------------------------------ #

app = FastAPI(title="CRISPR RL v4 — Ayurgenomic Intelligence Engine", version="4.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = str(round((time.time() - start) * 1000, 2))
    return response


# ------------------------------------------------------------------ #
# Routes
# ------------------------------------------------------------------ #

@app.get("/", include_in_schema=False)
def serve_frontend():
    index = _FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Place index.html in the frontend/ folder."}


@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "version": "4.0.0"}


@app.get("/crispr/config", tags=["System"])
def get_config():
    return {
        "agent": {"alpha": _agent.alpha, "gamma": _agent.gamma, "epsilon": round(_agent.epsilon, 4), "seed": _agent.seed},
        "prakriti_profiles": _mapping.list_prakriti_profiles(),
        "marma_points": [m["key"] for m in _mapping.list_marma_points()],
        "supported_pam_types": ["NGG", "NNGRRT", "NGTN"],
        "demo_genes": list(_DEMO_SEQUENCES.keys()),
    }


@app.get("/crispr/marma", tags=["Ayurveda"])
def get_marma_map():
    return {"marma_points": _mapping.list_marma_points(), "total": len(_mapping.list_marma_points())}


@app.get("/crispr/prakriti/{profile_name}", tags=["Ayurveda"])
def get_prakriti_profile(profile_name: str):
    profile = _mapping.get_prakriti_profile(profile_name)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_name}' not found.")
    return {
        "name": profile_name,
        "dosha_mix": profile["doshas"],
        "therapeutic_focus": profile["therapeutic_focus"],
        "priority_genes": profile["gene_priority_boost"],
        "priority_marma": profile["marma_priority"],
    }


@app.post("/crispr/design", tags=["CRISPR"])
def design_grna(body: DesignRequest):
    t0 = time.time()
    run_id = body.run_id or str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    profile = body.ayur_profile.lower().replace(" ", "_").replace("-", "_")

    results = []
    for target in body.targets:
        gene_id = target.gene_id.upper()
        seq = _get_sequence(gene_id, target.sequence)
        try:
            scanner = PAMScanner(pam_type=target.pam_type)
            raw_candidates = scanner.scan(gene_id, seq)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        if not raw_candidates:
            results.append({"gene_id": gene_id, "candidates": [], "message": "No PAM sites found."})
            continue

        scored = _weight_engine.rank_candidates([c.as_dict() for c in raw_candidates[:20]], profile)
        chosen, reward = _agent.choose_grna(run_id, request_id, scored, profile)
        results.append({"gene_id": gene_id, "top_candidates": scored[:body.top_k], "rl_selected": chosen, "rl_reward": round(reward, 4)})

    latency_ms = round((time.time() - t0) * 1000, 2)
    all_harmony = [r["rl_selected"].get("harmony_index", 0.0) for r in results if "rl_selected" in r]
    avg_harmony = sum(all_harmony) / len(all_harmony) if all_harmony else 0.0

    _log_jsonl({"run_id": run_id, "request_id": request_id, "ayur_profile": profile,
                "harmony_score": round(avg_harmony, 4), "latency_ms": latency_ms,
                "timestamp_ms": int(time.time() * 1000), "n_genes": len(body.targets)})

    return {"run_id": run_id, "request_id": request_id, "ayur_profile": profile,
            "latency_ms": latency_ms, "avg_harmony_index": round(avg_harmony, 4), "results": results}


@app.post("/crispr/feedback", tags=["CRISPR"])
def submit_feedback(body: FeedbackRequest):
    _agent.receive_feedback(body.grna_sequence, body.human_score)
    return {"status": "feedback_received", "grna_sequence": body.grna_sequence, "human_score": body.human_score}


@app.get("/crispr/metrics", tags=["CRISPR"])
def get_metrics():
    return {"status": "ok", "metrics": _agent.get_metrics()}
