"""
weights.py — Ayurvedic Scoring Heuristics

Computes the AyurHarmonyScore (AHS) for a gRNA candidate by blending:
  - Molecular efficiency (on-target prediction)
  - Specificity (off-target penalty)
  - Marma alignment (sensitivity-aware edit placement)
  - Prakriti congruence (dosha-mix match to therapeutic intent)

AHS = f(specificity, efficiency, marma_alignment, prakriti_congruence)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from .mapping import AyurMapping

logger = logging.getLogger(__name__)


@dataclass
class AyurHarmonyScore:
    """
    Structured result of the AyurWeightEngine scoring pipeline.

    Fields
    ------
    gene_id : str
        Target gene symbol.
    grna_sequence : str
        The 20-nt gRNA protospacer sequence.
    efficiency_score : float
        Raw on-target efficiency (0–1, from RL agent).
    specificity_score : float
        Off-target adjusted specificity (0–1).
    marma_alignment : float
        0–1; 1.0 = gene is in top-priority marma for given prakriti.
    prakriti_congruence : float
        0–1; measures dosha-mix fit between gene and patient profile.
    harmony_index : float
        Final composite AyurHarmonyScore (0–1).
    caution_level : str
        Derived edit risk category: low / medium / high / critical.
    notes : list[str]
        Human-readable reasoning notes.
    """

    gene_id: str
    grna_sequence: str
    efficiency_score: float
    specificity_score: float
    marma_alignment: float
    prakriti_congruence: float
    harmony_index: float
    caution_level: str
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "gene_id": self.gene_id,
            "grna_sequence": self.grna_sequence,
            "efficiency_score": round(self.efficiency_score, 4),
            "specificity_score": round(self.specificity_score, 4),
            "marma_alignment": round(self.marma_alignment, 4),
            "prakriti_congruence": round(self.prakriti_congruence, 4),
            "harmony_index": round(self.harmony_index, 4),
            "caution_level": self.caution_level,
            "notes": self.notes,
        }


class AyurWeightEngine:
    """
    Applies Ayurvedic heuristics to score gRNA candidates.

    Weights
    -------
    w_eff : float   — efficiency component weight   (default 0.35)
    w_spe : float   — specificity component weight  (default 0.30)
    w_mar : float   — marma alignment weight        (default 0.20)
    w_pra : float   — prakriti congruence weight    (default 0.15)
    """

    # Component weights (must sum to 1.0)
    W_EFF: float = 0.35
    W_SPE: float = 0.30
    W_MAR: float = 0.20
    W_PRA: float = 0.15

    def __init__(self, mapping: Optional[AyurMapping] = None):
        self.mapping = mapping or AyurMapping()
        assert abs(self.W_EFF + self.W_SPE + self.W_MAR + self.W_PRA - 1.0) < 1e-6, (
            "Ayurvedic weights must sum to 1.0"
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def score(
        self,
        gene_id: str,
        grna_sequence: str,
        efficiency_score: float,
        specificity_score: float,
        prakriti_profile: str = "tridoshic",
    ) -> AyurHarmonyScore:
        """
        Compute the AyurHarmonyScore for a single gRNA candidate.

        Parameters
        ----------
        gene_id : str            Gene symbol (e.g. "BDNF")
        grna_sequence : str      20-nt protospacer
        efficiency_score : float On-target efficiency 0–1
        specificity_score : float Off-target specificity 0–1
        prakriti_profile : str   Patient's prakriti name
        """
        gene_id = gene_id.upper()
        notes: list[str] = []

        # --- Marma alignment ---
        marma_alignment = self._compute_marma_alignment(
            gene_id, prakriti_profile, notes
        )

        # --- Prakriti congruence ---
        prakriti_congruence = self._compute_prakriti_congruence(
            gene_id, prakriti_profile, notes
        )

        # --- GC content modifier (gentle penalty for extreme GC) ---
        gc_mod = self._gc_modifier(grna_sequence, notes)
        eff_adj = min(1.0, efficiency_score * gc_mod)

        # --- Composite harmony index ---
        harmony_index = (
            self.W_EFF * eff_adj
            + self.W_SPE * specificity_score
            + self.W_MAR * marma_alignment
            + self.W_PRA * prakriti_congruence
        )

        # Clamp
        harmony_index = max(0.0, min(1.0, harmony_index))

        # --- Caution level ---
        caution_level = self._derive_caution(
            gene_id, harmony_index, prakriti_profile, notes
        )

        logger.debug(
            "AHS for %s/%s: eff=%.3f spe=%.3f mar=%.3f pra=%.3f → AHS=%.4f [%s]",
            gene_id, grna_sequence[:8],
            eff_adj, specificity_score, marma_alignment,
            prakriti_congruence, harmony_index, caution_level,
        )

        return AyurHarmonyScore(
            gene_id=gene_id,
            grna_sequence=grna_sequence,
            efficiency_score=eff_adj,
            specificity_score=specificity_score,
            marma_alignment=marma_alignment,
            prakriti_congruence=prakriti_congruence,
            harmony_index=harmony_index,
            caution_level=caution_level,
            notes=notes,
        )

    def rank_candidates(
        self,
        candidates: list[dict],
        prakriti_profile: str = "tridoshic",
    ) -> list[dict]:
        """
        Rank a list of gRNA candidate dicts by their AyurHarmonyScore.

        Each candidate dict must have:
            gene_id, grna_sequence, efficiency_score, specificity_score
        Returns sorted list (highest harmony_index first) with AHS fields merged.
        """
        scored = []
        for c in candidates:
            ahs = self.score(
                gene_id=c["gene_id"],
                grna_sequence=c["grna_sequence"],
                efficiency_score=c["efficiency_score"],
                specificity_score=c["specificity_score"],
                prakriti_profile=prakriti_profile,
            )
            scored.append({**c, **ahs.as_dict()})

        scored.sort(key=lambda x: x["harmony_index"], reverse=True)
        return scored

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _compute_marma_alignment(
        self, gene_id: str, prakriti_profile: str, notes: list[str]
    ) -> float:
        """
        Marma alignment: 1.0 if gene is linked to a priority marma for
        the patient's prakriti, scaled by marma sensitivity.
        """
        linked_marma = self.mapping.get_marma_for_gene(gene_id)
        priority_marma = self.mapping.get_priority_marma(prakriti_profile)

        if not linked_marma:
            notes.append(f"{gene_id} has no known marma linkage — neutral alignment.")
            return 0.5  # neutral

        overlap = [m for m in linked_marma if m in priority_marma]
        if overlap:
            # Use average sensitivity of overlapping marma points
            sensitivities = [
                self.mapping.get_marma_sensitivity(m) for m in overlap
            ]
            alignment = sum(sensitivities) / len(sensitivities)
            notes.append(
                f"{gene_id} aligns with priority marma: "
                f"{', '.join(overlap)} (avg sensitivity={alignment:.2f})"
            )
        else:
            # Gene is in some marma, just not priority ones
            sensitivities = [
                self.mapping.get_marma_sensitivity(m) for m in linked_marma
            ]
            alignment = 0.5 * sum(sensitivities) / len(sensitivities)
            notes.append(
                f"{gene_id} linked to non-priority marma: "
                f"{', '.join(linked_marma)} — partial alignment."
            )
        return alignment

    def _compute_prakriti_congruence(
        self, gene_id: str, prakriti_profile: str, notes: list[str]
    ) -> float:
        """
        Prakriti congruence: cosine-like similarity between gene's
        dosha signature and patient's dosha mix.
        """
        gene_doshas = self.mapping.get_doshas_for_gene(gene_id)
        if not gene_doshas:
            notes.append(f"{gene_id} has no known dosha affinity — neutral congruence.")
            return 0.5

        dosha_mix = self.mapping.get_dosha_mix(prakriti_profile)
        priority_genes = self.mapping.get_priority_genes(prakriti_profile)

        # Gene dosha vector (normalised)
        all_doshas = ["vata", "pitta", "kapha"]
        gene_vec = [1.0 if d in gene_doshas else 0.0 for d in all_doshas]
        norm = math.sqrt(sum(x**2 for x in gene_vec)) or 1.0
        gene_vec = [x / norm for x in gene_vec]

        # Patient dosha vector
        patient_vec = [dosha_mix.get(d, 0.0) for d in all_doshas]

        # Dot product (cosine similarity since patient vec already sums to 1)
        congruence = sum(a * b for a, b in zip(gene_vec, patient_vec))

        # Boost if gene is explicitly boosted for this prakriti
        if gene_id in priority_genes:
            congruence = min(1.0, congruence * 1.2)
            notes.append(
                f"{gene_id} is a priority gene for '{prakriti_profile}' "
                f"— applying 1.2× boost."
            )
        else:
            notes.append(
                f"{gene_id} doshas={gene_doshas} vs prakriti congruence={congruence:.2f}."
            )

        return max(0.0, min(1.0, congruence))

    @staticmethod
    def _gc_modifier(grna_sequence: str, notes: list[str]) -> float:
        """
        Penalise gRNAs with extreme GC content (optimal 40–70%).
        Returns a multiplier ∈ (0.7, 1.0].
        """
        seq = grna_sequence.upper().replace(" ", "")
        if not seq:
            return 1.0
        gc = sum(1 for nt in seq if nt in "GC") / len(seq)
        if gc < 0.40 or gc > 0.70:
            mod = 0.70 + 0.30 * (1 - abs(gc - 0.55) / 0.45)
            notes.append(
                f"GC content {gc*100:.1f}% is outside optimal range "
                f"(40–70%); applying modifier {mod:.2f}."
            )
            return max(0.70, mod)
        return 1.0

    def _derive_caution(
        self,
        gene_id: str,
        harmony_index: float,
        prakriti_profile: str,
        notes: list[str],
    ) -> str:
        """
        Derive caution level from harmony index and marma sensitivity.
        """
        marmas = self.mapping.get_marma_for_gene(gene_id)
        max_sensitivity = max(
            (self.mapping.get_marma_sensitivity(m) for m in marmas),
            default=0.5,
        )

        if max_sensitivity >= 0.90 or harmony_index < 0.40:
            level = "critical"
        elif max_sensitivity >= 0.80 or harmony_index < 0.55:
            level = "high"
        elif max_sensitivity >= 0.70 or harmony_index < 0.70:
            level = "medium"
        else:
            level = "low"

        notes.append(
            f"Caution level '{level}' (harmony={harmony_index:.3f}, "
            f"max_marma_sensitivity={max_sensitivity:.2f})."
        )
        return level
