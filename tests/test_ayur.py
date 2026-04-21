"""
test_ayur.py — Unit tests for AyurHarmonyScore logic, mapping, and weight engine.
"""

import pytest

from crispr_rl.ayur_layer.mapping import AyurMapping
from crispr_rl.ayur_layer.weights import AyurWeightEngine, AyurHarmonyScore


# ------------------------------------------------------------------ #
# AyurMapping
# ------------------------------------------------------------------ #

class TestAyurMapping:
    @pytest.fixture
    def mapping(self):
        return AyurMapping()

    def test_loads_without_error(self, mapping):
        assert mapping is not None

    def test_known_gene_has_dosha(self, mapping):
        doshas = mapping.get_doshas_for_gene("BDNF")
        assert len(doshas) > 0
        assert "vata" in doshas

    def test_unknown_gene_returns_empty(self, mapping):
        doshas = mapping.get_doshas_for_gene("FAKEGENE999")
        assert doshas == []

    def test_case_insensitive(self, mapping):
        assert mapping.get_doshas_for_gene("bdnf") == mapping.get_doshas_for_gene("BDNF")

    def test_dosha_stability_weight(self, mapping):
        w = mapping.get_dosha_stability_weight("vata")
        assert isinstance(w, float)
        assert w > 0

    def test_unknown_dosha_returns_1(self, mapping):
        w = mapping.get_dosha_stability_weight("unknown_dosha")
        assert w == 1.0

    def test_marma_for_known_gene(self, mapping):
        marmas = mapping.get_marma_for_gene("BDNF")
        assert "sthapani" in marmas

    def test_marma_sensitivity_range(self, mapping):
        for marma_key in ["hridaya", "nabhi", "sthapani", "basti", "gulpha"]:
            s = mapping.get_marma_sensitivity(marma_key)
            assert 0.0 <= s <= 1.0

    def test_marma_sensitivity_unknown_returns_half(self, mapping):
        s = mapping.get_marma_sensitivity("nonexistent_marma")
        assert s == 0.5

    def test_hridaya_highest_sensitivity(self, mapping):
        """Heart marma should have sensitivity ≥ 0.9."""
        assert mapping.get_marma_sensitivity("hridaya") >= 0.9

    def test_prakriti_profile_exists(self, mapping):
        profile = mapping.get_prakriti_profile("vata_dominant")
        assert profile is not None
        assert "doshas" in profile

    def test_unknown_prakriti_returns_none(self, mapping):
        profile = mapping.get_prakriti_profile("lunar_dominant")
        assert profile is None

    def test_dosha_mix_sums_to_one(self, mapping):
        for name in mapping.list_prakriti_profiles():
            mix = mapping.get_dosha_mix(name)
            total = sum(mix.values())
            assert total == pytest.approx(1.0, abs=0.01)

    def test_gene_report_structure(self, mapping):
        report = mapping.gene_report("SCN1A")
        assert "gene_id" in report
        assert "doshas" in report
        assert "marma_points" in report

    def test_list_marma_points(self, mapping):
        points = mapping.list_marma_points()
        assert len(points) >= 5
        keys = [p["key"] for p in points]
        assert "hridaya" in keys

    def test_list_prakriti_profiles(self, mapping):
        profiles = mapping.list_prakriti_profiles()
        assert "tridoshic" in profiles
        assert len(profiles) >= 4


# ------------------------------------------------------------------ #
# AyurWeightEngine
# ------------------------------------------------------------------ #

class TestAyurWeightEngine:
    @pytest.fixture
    def engine(self):
        return AyurWeightEngine()

    def test_score_returns_harmony_score(self, engine):
        ahs = engine.score(
            gene_id="BDNF",
            grna_sequence="A" * 10 + "G" * 10,
            efficiency_score=0.75,
            specificity_score=0.80,
            prakriti_profile="vata_dominant",
        )
        assert isinstance(ahs, AyurHarmonyScore)

    def test_harmony_index_in_range(self, engine):
        ahs = engine.score(
            gene_id="PPARG",
            grna_sequence="G" * 14 + "ATACG" + "C",
            efficiency_score=0.60,
            specificity_score=0.70,
            prakriti_profile="pitta_dominant",
        )
        assert 0.0 <= ahs.harmony_index <= 1.0

    def test_caution_level_valid_values(self, engine):
        ahs = engine.score(
            gene_id="SCN1A",
            grna_sequence="ACGTACGTACGTACGTACGT",
            efficiency_score=0.50,
            specificity_score=0.50,
            prakriti_profile="tridoshic",
        )
        assert ahs.caution_level in {"low", "medium", "high", "critical"}

    def test_hridaya_gene_gets_high_caution(self, engine):
        """MYH7 is linked to hridaya (sensitivity=0.95) → critical caution."""
        ahs = engine.score(
            gene_id="MYH7",
            grna_sequence="ACGTACGTACGTACGTACGT",
            efficiency_score=0.90,
            specificity_score=0.90,
            prakriti_profile="tridoshic",
        )
        assert ahs.caution_level in {"critical", "high"}

    def test_priority_gene_gets_boost(self, engine):
        """BDNF is a priority gene for vata_dominant — should score higher."""
        ahs_vata = engine.score(
            gene_id="BDNF",
            grna_sequence="ACGTACGTACGTACGTACGT",
            efficiency_score=0.70,
            specificity_score=0.70,
            prakriti_profile="vata_dominant",
        )
        ahs_pitta = engine.score(
            gene_id="BDNF",
            grna_sequence="ACGTACGTACGTACGTACGT",
            efficiency_score=0.70,
            specificity_score=0.70,
            prakriti_profile="pitta_dominant",
        )
        assert ahs_vata.prakriti_congruence >= ahs_pitta.prakriti_congruence

    def test_extreme_gc_modifies_efficiency(self, engine):
        """All-G sequence (100% GC) should reduce efficiency component."""
        ahs_bad = engine.score(
            gene_id="PPARG",
            grna_sequence="G" * 20,
            efficiency_score=0.80,
            specificity_score=0.80,
            prakriti_profile="tridoshic",
        )
        ahs_good = engine.score(
            gene_id="PPARG",
            grna_sequence="ACGTACGTACGTGCATGCAT",  # ~50% GC
            efficiency_score=0.80,
            specificity_score=0.80,
            prakriti_profile="tridoshic",
        )
        assert ahs_good.efficiency_score >= ahs_bad.efficiency_score

    def test_notes_are_populated(self, engine):
        ahs = engine.score(
            gene_id="BDNF",
            grna_sequence="ACGTACGTACGTACGTACGT",
            efficiency_score=0.70,
            specificity_score=0.70,
            prakriti_profile="vata_dominant",
        )
        assert len(ahs.notes) > 0

    def test_as_dict_serialisable(self, engine):
        import json
        ahs = engine.score("BDNF", "ACGTACGTACGTACGTACGT", 0.7, 0.7, "tridoshic")
        d = ahs.as_dict()
        json.dumps(d)  # must not raise

    def test_rank_candidates_sorted(self, engine):
        candidates = [
            {
                "gene_id": "BDNF",
                "grna_sequence": "ACGTACGTACGTACGTACGT",
                "efficiency_score": 0.60,
                "specificity_score": 0.60,
            },
            {
                "gene_id": "BDNF",
                "grna_sequence": "GCATGCATGCATGCATGCAT",
                "efficiency_score": 0.85,
                "specificity_score": 0.85,
            },
        ]
        ranked = engine.rank_candidates(candidates, "vata_dominant")
        assert ranked[0]["harmony_index"] >= ranked[1]["harmony_index"]

    def test_weights_sum_to_one(self):
        w = AyurWeightEngine
        total = w.W_EFF + w.W_SPE + w.W_MAR + w.W_PRA
        assert total == pytest.approx(1.0)
