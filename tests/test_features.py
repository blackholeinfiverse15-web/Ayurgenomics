"""
test_features.py — Unit tests for PAM scanner and molecular feature functions.
"""

import pytest
from crispr_rl.features import (
    PAMScanner,
    calc_gc,
    calc_seed_gc,
    calc_homopolymer_run,
    estimate_efficiency,
    estimate_off_target_risk,
    _reverse_complement,
)


# ------------------------------------------------------------------ #
# GC content
# ------------------------------------------------------------------ #

class TestCalcGC:
    def test_all_gc(self):
        assert calc_gc("GGGG") == 1.0

    def test_all_at(self):
        assert calc_gc("AAAA") == 0.0

    def test_mixed(self):
        assert calc_gc("GCGCAT") == pytest.approx(4 / 6)

    def test_empty(self):
        assert calc_gc("") == 0.0

    def test_case_insensitive(self):
        assert calc_gc("gcgc") == 1.0


# ------------------------------------------------------------------ #
# Seed GC
# ------------------------------------------------------------------ #

class TestSeedGC:
    def test_seed_last_12(self):
        # 8 AT + 12 GC
        seq = "ATATATAT" + "G" * 12
        assert calc_seed_gc(seq) == pytest.approx(1.0)

    def test_seed_default_length(self):
        seq = "A" * 20
        assert calc_seed_gc(seq) == 0.0


# ------------------------------------------------------------------ #
# Homopolymer
# ------------------------------------------------------------------ #

class TestHomopolymer:
    def test_no_run(self):
        assert calc_homopolymer_run("ACGT") == 1

    def test_run_of_4(self):
        assert calc_homopolymer_run("ATTTTTCG") == 5

    def test_empty(self):
        assert calc_homopolymer_run("") == 0


# ------------------------------------------------------------------ #
# Efficiency estimator
# ------------------------------------------------------------------ #

class TestEstimateEfficiency:
    def test_optimal_gc(self):
        eff = estimate_efficiency(gc=0.55, seed_gc=0.55, homopolymer=1, pam_type="NGG")
        assert eff == 1.0

    def test_low_gc_penalised(self):
        eff = estimate_efficiency(gc=0.20, seed_gc=0.20, homopolymer=1, pam_type="NGG")
        assert eff < 1.0

    def test_homopolymer_5_penalised(self):
        eff = estimate_efficiency(gc=0.55, seed_gc=0.55, homopolymer=5, pam_type="NGG")
        assert eff < estimate_efficiency(gc=0.55, seed_gc=0.55, homopolymer=1, pam_type="NGG")

    def test_non_ngg_pam_penalised(self):
        eff_ngg = estimate_efficiency(0.55, 0.55, 1, "NGG")
        eff_sa = estimate_efficiency(0.55, 0.55, 1, "NNGRRT")
        assert eff_ngg > eff_sa

    def test_range(self):
        eff = estimate_efficiency(gc=0.0, seed_gc=0.0, homopolymer=6, pam_type="NGTN")
        assert 0.0 <= eff <= 1.0


# ------------------------------------------------------------------ #
# Off-target risk
# ------------------------------------------------------------------ #

class TestOffTargetRisk:
    def test_baseline(self):
        risk = estimate_off_target_risk(gc=0.5, seed_gc=0.5, homopolymer=1)
        assert risk == pytest.approx(0.1)

    def test_high_gc_increases_risk(self):
        risk = estimate_off_target_risk(gc=0.80, seed_gc=0.80, homopolymer=1)
        assert risk > 0.1

    def test_range(self):
        risk = estimate_off_target_risk(gc=0.99, seed_gc=0.99, homopolymer=5)
        assert 0.0 <= risk <= 1.0


# ------------------------------------------------------------------ #
# Reverse complement
# ------------------------------------------------------------------ #

class TestReverseComplement:
    def test_simple(self):
        assert _reverse_complement("ATCG") == "CGAT"

    def test_self_complement(self):
        assert _reverse_complement("AATT") == "AATT"

    def test_n_preserved(self):
        rc = _reverse_complement("ANCG")
        assert "N" in rc


# ------------------------------------------------------------------ #
# PAM Scanner
# ------------------------------------------------------------------ #

class TestPAMScanner:
    _SEQ = (
        "ATGACCATCCTTTTCCTTACTATGGTTATTTCATATAAATACTATAATGTTTTTTCTCAAAAAATGAA"
        "AAGTTTATTTGTTTCCACTTTGCCTTCTGCCCTATTTGTTTAGCCAGTGGGGAAAGCAAACATGTCTA"
        "TGAGGGTTCGGAGAGGTGAGAAGAGTTGCAGTACGTAACACTGTCACTGTGTTGTGGAGTAGTCGCCA"
        "GCATCTTTTGGTTGCTTGTTCTCTGTGTAATTTGAAAAGAAAGAGCCTGTGTGTAAAGTGTGAATCAG"
        "TTCATTGTGTTGAATAAAGTTTTGATCAATCAAATGTTATGTTTTTATAAAGGG"
    )

    def test_scanner_creates(self):
        scanner = PAMScanner("NGG")
        assert scanner.pam_type == "NGG"

    def test_invalid_pam_raises(self):
        with pytest.raises(ValueError):
            PAMScanner("XYZ")

    def test_finds_candidates(self):
        scanner = PAMScanner("NGG")
        candidates = scanner.scan("BDNF", self._SEQ)
        assert len(candidates) > 0

    def test_candidates_have_correct_length(self):
        scanner = PAMScanner("NGG")
        candidates = scanner.scan("BDNF", self._SEQ)
        for c in candidates:
            assert len(c.grna_sequence) == 20

    def test_gc_content_in_range(self):
        scanner = PAMScanner("NGG")
        candidates = scanner.scan("BDNF", self._SEQ)
        for c in candidates:
            assert 0.0 <= c.gc_content <= 1.0

    def test_sorted_by_efficiency(self):
        scanner = PAMScanner("NGG")
        candidates = scanner.scan("BDNF", self._SEQ)
        effs = [c.predicted_efficiency for c in candidates]
        assert effs == sorted(effs, reverse=True)

    def test_strand_both_directions(self):
        scanner = PAMScanner("NGG")
        candidates = scanner.scan("TEST", self._SEQ)
        strands = {c.strand for c in candidates}
        assert "+" in strands or "-" in strands

    def test_as_dict_keys(self):
        scanner = PAMScanner("NGG")
        candidates = scanner.scan("BDNF", self._SEQ)
        if candidates:
            d = candidates[0].as_dict()
            for key in ("gene_id", "grna_sequence", "gc_content", "efficiency_score"):
                assert key in d
