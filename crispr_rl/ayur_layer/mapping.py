"""
mapping.py — Ayurvedic–Genomic Ontology Layer

Provides gene-to-dosha and gene-to-marma lookups.
Encodes prakriti-based priority boosts for CRISPR target ranking.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_AYUR_MAP_PATH = Path(__file__).parent / "ayur_map.json"


class AyurMapping:
    """
    Central lookup engine for Ayurvedic–genomic correspondences.

    Loads ayur_map.json on initialisation and provides query methods
    for dosha affinity, marma sensitivity, and prakriti profile data.
    """

    def __init__(self, map_path: Optional[Path] = None):
        path = map_path or _AYUR_MAP_PATH
        with open(path, "r") as f:
            self._data = json.load(f)

        self._doshas = self._data["doshas"]
        self._marma = self._data["marma_organ_gene_map"]
        self._prakriti = self._data["prakriti_profiles"]

        # Build fast gene → dosha index
        self._gene_dosha_index: dict[str, list[str]] = {}
        for dosha, info in self._doshas.items():
            for gene in info["gene_categories"]:
                self._gene_dosha_index.setdefault(gene.upper(), []).append(dosha)

        # Build fast gene → marma index
        self._gene_marma_index: dict[str, list[str]] = {}
        for marma_key, info in self._marma.items():
            for gene in info["linked_genes"]:
                self._gene_marma_index.setdefault(gene.upper(), []).append(marma_key)

        logger.info(
            "AyurMapping loaded: %d doshas, %d marma points, %d prakriti profiles",
            len(self._doshas),
            len(self._marma),
            len(self._prakriti),
        )

    # ------------------------------------------------------------------ #
    # Dosha queries
    # ------------------------------------------------------------------ #

    def get_doshas_for_gene(self, gene_id: str) -> list[str]:
        """Return list of doshas associated with a given gene symbol."""
        return self._gene_dosha_index.get(gene_id.upper(), [])

    def get_dosha_stability_weight(self, dosha: str) -> float:
        """Return the stability weight multiplier for a given dosha."""
        info = self._doshas.get(dosha.lower())
        if info is None:
            logger.warning("Unknown dosha '%s'; returning weight 1.0", dosha)
            return 1.0
        return info["stability_weight"]

    def get_dosha_caution(self, dosha: str) -> str:
        """Return edit caution level for a dosha: high / medium / low."""
        info = self._doshas.get(dosha.lower(), {})
        return info.get("edit_caution", "medium")

    # ------------------------------------------------------------------ #
    # Marma queries
    # ------------------------------------------------------------------ #

    def get_marma_for_gene(self, gene_id: str) -> list[str]:
        """Return marma point keys linked to a gene symbol."""
        return self._gene_marma_index.get(gene_id.upper(), [])

    def get_marma_sensitivity(self, marma_key: str) -> float:
        """
        Return the edit_sensitivity score (0–1) for a marma point.
        Higher = more caution required.
        """
        info = self._marma.get(marma_key)
        if info is None:
            return 0.5  # neutral default
        return info["edit_sensitivity"]

    def get_marma_dosha_affinity(self, marma_key: str) -> str:
        """Return the dosha affinity string for a marma point."""
        info = self._marma.get(marma_key, {})
        return info.get("dosha_affinity", "tridoshic")

    def list_marma_points(self) -> list[dict]:
        """Return all marma points with their metadata."""
        return [
            {"key": k, **{kk: vv for kk, vv in v.items()}}
            for k, v in self._marma.items()
        ]

    # ------------------------------------------------------------------ #
    # Prakriti queries
    # ------------------------------------------------------------------ #

    def get_prakriti_profile(self, profile_name: str) -> Optional[dict]:
        """Return prakriti profile dict or None if not found."""
        return self._prakriti.get(profile_name.lower().replace(" ", "_").replace("-", "_"))

    def get_priority_genes(self, profile_name: str) -> list[str]:
        """Return genes that receive a boost for the given prakriti profile."""
        profile = self.get_prakriti_profile(profile_name)
        if profile is None:
            return []
        return [g.upper() for g in profile.get("gene_priority_boost", [])]

    def get_priority_marma(self, profile_name: str) -> list[str]:
        """Return prioritised marma points for a prakriti profile."""
        profile = self.get_prakriti_profile(profile_name)
        if profile is None:
            return []
        return profile.get("marma_priority", [])

    def get_dosha_mix(self, profile_name: str) -> dict[str, float]:
        """Return dosha proportion dict for a prakriti profile."""
        profile = self.get_prakriti_profile(profile_name)
        if profile is None:
            return {"vata": 0.33, "pitta": 0.33, "kapha": 0.34}
        return profile["doshas"]

    def list_prakriti_profiles(self) -> list[str]:
        """Return all available prakriti profile names."""
        return list(self._prakriti.keys())

    # ------------------------------------------------------------------ #
    # Convenience: full gene report
    # ------------------------------------------------------------------ #

    def gene_report(self, gene_id: str) -> dict:
        """Return a combined Ayurvedic context report for a gene."""
        gene = gene_id.upper()
        doshas = self.get_doshas_for_gene(gene)
        marmas = self.get_marma_for_gene(gene)
        return {
            "gene_id": gene,
            "doshas": doshas,
            "marma_points": marmas,
            "avg_marma_sensitivity": (
                sum(self.get_marma_sensitivity(m) for m in marmas) / len(marmas)
                if marmas
                else 0.5
            ),
            "stability_weights": [
                self.get_dosha_stability_weight(d) for d in doshas
            ],
        }
