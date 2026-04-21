"""
features.py — CRISPR Feature Extraction

Provides molecular feature extraction utilities:
  - PAM site scanning (NGG, NNGRRT, NGTN)
  - GC content calculation
  - Seed region analysis
  - Basic off-target risk estimation
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# PAM patterns (5'→3' on non-template strand)
PAM_PATTERNS: dict[str, str] = {
    "NGG": r"(?=.GG)",         # SpCas9
    "NNGRRT": r"(?=..[AG][AG]T)",  # SaCas9
    "NGTN": r"(?=.[GT][ACGT].)",   # Cas12a (simplified)
}

# Seed region = last 12 nt adjacent to PAM (most mismatch-sensitive)
SEED_REGION_LEN = 12
GRNA_LEN = 20


@dataclass
class gRNACandidate:
    """Represents a single gRNA candidate with extracted features."""
    gene_id: str
    grna_sequence: str          # 20-nt protospacer
    pam_sequence: str           # PAM immediately 3' of protospacer
    pam_type: str               # e.g. "NGG"
    strand: str                 # "+" or "-"
    position: int               # 0-based start in target sequence
    gc_content: float           # fraction 0–1
    seed_gc: float              # GC in seed region (last 12 nt)
    homopolymer_run: int        # longest run of identical nt
    predicted_efficiency: float # rule-based estimate 0–1
    off_target_risk: float      # rule-based risk estimate 0–1
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "gene_id": self.gene_id,
            "grna_sequence": self.grna_sequence,
            "pam_sequence": self.pam_sequence,
            "pam_type": self.pam_type,
            "strand": self.strand,
            "position": self.position,
            "gc_content": round(self.gc_content, 4),
            "seed_gc": round(self.seed_gc, 4),
            "homopolymer_run": self.homopolymer_run,
            "efficiency_score": round(self.predicted_efficiency, 4),
            "specificity_score": round(1.0 - self.off_target_risk, 4),
            "notes": self.notes,
        }


# ------------------------------------------------------------------ #
# Low-level feature functions
# ------------------------------------------------------------------ #

def calc_gc(sequence: str) -> float:
    """Return GC content as fraction 0–1."""
    seq = sequence.upper()
    if not seq:
        return 0.0
    return sum(1 for nt in seq if nt in "GC") / len(seq)


def calc_seed_gc(grna: str, seed_len: int = SEED_REGION_LEN) -> float:
    """Return GC fraction of the seed region (last N nt of protospacer)."""
    seed = grna.upper()[-seed_len:]
    return calc_gc(seed)


def calc_homopolymer_run(sequence: str) -> int:
    """Return length of the longest homopolymer run in sequence."""
    if not sequence:
        return 0
    max_run = 1
    current_run = 1
    seq = sequence.upper()
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run


def estimate_efficiency(
    gc: float,
    seed_gc: float,
    homopolymer: int,
    pam_type: str = "NGG",
) -> float:
    """
    Rule-based efficiency estimate (0–1).
    Penalties for:
      - GC outside 40–70%
      - Seed GC outside 40–70%
      - Homopolymer runs ≥ 4
      - Non-NGG PAM types
    """
    score = 1.0

    # GC penalty
    if not (0.40 <= gc <= 0.70):
        score *= 0.80

    # Seed GC penalty
    if not (0.40 <= seed_gc <= 0.75):
        score *= 0.85

    # Homopolymer penalty
    if homopolymer >= 5:
        score *= 0.60
    elif homopolymer >= 4:
        score *= 0.80

    # PAM type penalty
    pam_factors = {"NGG": 1.0, "NNGRRT": 0.85, "NGTN": 0.75}
    score *= pam_factors.get(pam_type, 0.70)

    return round(max(0.0, min(1.0, score)), 4)


def estimate_off_target_risk(
    gc: float,
    seed_gc: float,
    homopolymer: int,
) -> float:
    """
    Rule-based off-target risk estimate (0–1, 0 = minimal risk).
    High GC, high seed GC, and short homopolymers increase risk.
    """
    risk = 0.1  # baseline

    if gc > 0.65:
        risk += 0.10
    if seed_gc > 0.70:
        risk += 0.15
    if homopolymer >= 4:
        risk += 0.05  # might cause secondary structure

    return round(max(0.0, min(1.0, risk)), 4)


# ------------------------------------------------------------------ #
# PAM Scanner
# ------------------------------------------------------------------ #

class PAMScanner:
    """
    Scans a DNA sequence for PAM sites and extracts gRNA candidates.

    Supports NGG (SpCas9), NNGRRT (SaCas9), and NGTN (Cas12a).
    """

    def __init__(self, pam_type: str = "NGG"):
        pam_type = pam_type.upper()
        if pam_type not in PAM_PATTERNS:
            raise ValueError(
                f"Unknown PAM '{pam_type}'. Supported: {list(PAM_PATTERNS)}"
            )
        self.pam_type = pam_type
        self._pattern = re.compile(PAM_PATTERNS[pam_type], re.IGNORECASE)

    def scan(self, gene_id: str, dna_sequence: str) -> list[gRNACandidate]:
        """
        Find all valid gRNA candidates in dna_sequence.

        Searches both strands. Returns a list of gRNACandidate objects,
        sorted by predicted_efficiency descending.
        """
        seq = dna_sequence.upper().replace(" ", "").replace("\n", "")
        candidates: list[gRNACandidate] = []

        # Forward strand
        candidates.extend(self._scan_strand(gene_id, seq, strand="+"))

        # Reverse complement
        rc = _reverse_complement(seq)
        candidates.extend(self._scan_strand(gene_id, rc, strand="-"))

        candidates.sort(key=lambda c: c.predicted_efficiency, reverse=True)
        logger.info(
            "PAMScanner [%s, %s]: found %d candidates in %d-nt sequence.",
            gene_id, self.pam_type, len(candidates), len(seq),
        )
        return candidates

    def _scan_strand(
        self, gene_id: str, seq: str, strand: str
    ) -> list[gRNACandidate]:
        results = []
        pam_len = {"NGG": 3, "NNGRRT": 6, "NGTN": 4}.get(self.pam_type, 3)

        for m in self._pattern.finditer(seq):
            pam_start = m.start()
            grna_start = pam_start - GRNA_LEN
            if grna_start < 0:
                continue

            protospacer = seq[grna_start:pam_start]
            pam_seq = seq[pam_start: pam_start + pam_len]

            if len(protospacer) != GRNA_LEN:
                continue

            gc = calc_gc(protospacer)
            seed_gc = calc_seed_gc(protospacer)
            hp = calc_homopolymer_run(protospacer)
            eff = estimate_efficiency(gc, seed_gc, hp, self.pam_type)
            risk = estimate_off_target_risk(gc, seed_gc, hp)

            notes = []
            if hp >= 4:
                notes.append(f"Homopolymer run of {hp} detected — may reduce efficiency.")

            results.append(
                gRNACandidate(
                    gene_id=gene_id,
                    grna_sequence=protospacer,
                    pam_sequence=pam_seq,
                    pam_type=self.pam_type,
                    strand=strand,
                    position=grna_start,
                    gc_content=gc,
                    seed_gc=seed_gc,
                    homopolymer_run=hp,
                    predicted_efficiency=eff,
                    off_target_risk=risk,
                    notes=notes,
                )
            )
        return results


# ------------------------------------------------------------------ #
# Helper
# ------------------------------------------------------------------ #

def _reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    complement = str.maketrans("ACGTN", "TGCAN")
    return seq.translate(complement)[::-1]
