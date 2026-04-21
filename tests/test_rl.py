"""
test_rl.py — Unit tests for RL agent: reward shaping & reproducibility.
"""

import json
import tempfile
from pathlib import Path

import pytest

from crispr_rl.rl_agent import (
    QLearningAgent,
    compute_reward,
    encode_state,
    gc_bin,
    BASELINE_EFFICIENCY,
)


# ------------------------------------------------------------------ #
# Reward shaping
# ------------------------------------------------------------------ #

class TestComputeReward:
    def test_perfect_candidate_positive(self):
        r = compute_reward(
            efficiency=1.0,
            specificity=1.0,
            harmony_index=1.0,
            caution_level="low",
        )
        assert r > 0

    def test_bad_candidate_negative(self):
        r = compute_reward(
            efficiency=0.10,
            specificity=0.10,
            harmony_index=0.10,
            caution_level="critical",
        )
        assert r < 0

    def test_baseline_efficiency_neutral(self):
        r = compute_reward(
            efficiency=BASELINE_EFFICIENCY,
            specificity=0.5,
            harmony_index=0.5,
            caution_level="medium",
        )
        # efficiency uplift is 0, so reward driven by specificity/harmony/caution
        # Should be moderate
        assert -0.5 < r < 1.0

    def test_critical_caution_penalised(self):
        r_low = compute_reward(1.0, 1.0, 1.0, "low")
        r_crit = compute_reward(1.0, 1.0, 1.0, "critical")
        assert r_low > r_crit

    def test_reward_clamped(self):
        for eff, spe, har, caut in [
            (2.0, 2.0, 2.0, "low"),
            (-5.0, -5.0, -5.0, "critical"),
        ]:
            r = compute_reward(eff, spe, har, caut)
            assert -2.0 <= r <= 2.0


# ------------------------------------------------------------------ #
# State encoding
# ------------------------------------------------------------------ #

class TestEncodeState:
    def test_deterministic(self):
        s1 = encode_state(2, 3, "NGG")
        s2 = encode_state(2, 3, "NGG")
        assert s1 == s2

    def test_different_pam_different_state(self):
        assert encode_state(2, 2, "NGG") != encode_state(2, 2, "NNGRRT")


class TestGCBin:
    def test_zero_gc(self):
        assert gc_bin(0.0) == 0

    def test_full_gc(self):
        assert gc_bin(1.0) == 4  # min(4, int(1.0 * 5))

    def test_range(self):
        for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
            b = gc_bin(v)
            assert 0 <= b <= 4


# ------------------------------------------------------------------ #
# Q-Learning Agent
# ------------------------------------------------------------------ #

def _make_candidates(n: int = 5) -> list[dict]:
    return [
        {
            "gene_id": "BDNF",
            "grna_sequence": f"ACGT" * 5,  # 20 nt
            "pam_type": "NGG",
            "gc_content": 0.50 + i * 0.02,
            "seed_gc": 0.50,
            "efficiency_score": 0.60 + i * 0.03,
            "specificity_score": 0.80,
            "harmony_index": 0.70 + i * 0.01,
            "caution_level": "medium",
        }
        for i in range(n)
    ]


class TestQLearningAgent:
    def test_initialises(self):
        agent = QLearningAgent(seed=0)
        assert agent.epsilon == 0.15

    def test_choose_grna_returns_candidate(self):
        agent = QLearningAgent(seed=42)
        candidates = _make_candidates()
        chosen, reward = agent.choose_grna("run-1", "req-1", candidates, "tridoshic")
        assert "gene_id" in chosen
        assert isinstance(reward, float)

    def test_reward_recorded(self):
        agent = QLearningAgent(seed=42)
        candidates = _make_candidates()
        agent.choose_grna("run-1", "req-1", candidates, "tridoshic")
        assert len(agent._episode_history) == 1

    def test_epsilon_decays(self):
        agent = QLearningAgent(seed=42, epsilon=0.5)
        initial_eps = agent.epsilon
        candidates = _make_candidates()
        for i in range(20):
            agent.choose_grna(f"run-{i}", f"req-{i}", candidates, "tridoshic")
        assert agent.epsilon < initial_eps

    def test_metrics_after_episodes(self):
        agent = QLearningAgent(seed=42)
        candidates = _make_candidates()
        for i in range(5):
            agent.choose_grna(f"run-{i}", f"req-{i}", candidates, "tridoshic")
        m = agent.get_metrics()
        assert m["total_episodes"] == 5
        assert "avg_harmony_index" in m
        assert "efficiency_uplift_pct" in m

    def test_empty_metrics_before_episodes(self):
        agent = QLearningAgent(seed=0)
        m = agent.get_metrics()
        assert m["total_episodes"] == 0

    def test_feedback_does_not_crash(self):
        agent = QLearningAgent(seed=42)
        candidates = _make_candidates()
        chosen, _ = agent.choose_grna("run-1", "req-1", candidates, "tridoshic")
        # Should not raise
        agent.receive_feedback(chosen["grna_sequence"], 0.8)

    # ---------------------------------------------------------------- #
    # Reproducibility
    # ---------------------------------------------------------------- #

    def test_reproducibility_same_seed(self):
        """Same seed → same sequence of actions."""
        def run_agent():
            agent = QLearningAgent(seed=99, epsilon=0.0)  # greedy
            candidates = _make_candidates(5)
            selections = []
            for i in range(10):
                chosen, _ = agent.choose_grna(f"run-{i}", f"req-{i}", candidates, "tridoshic")
                selections.append(chosen["gc_content"])
            return selections

        assert run_agent() == run_agent()

    def test_checkpoint_round_trip(self):
        """Save and reload agent → identical Q-table."""
        agent = QLearningAgent(seed=42)
        candidates = _make_candidates()
        for i in range(5):
            agent.choose_grna(f"run-{i}", f"req-{i}", candidates, "tridoshic")

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "ckpt.json"
            agent.save_checkpoint(ckpt)

            agent2 = QLearningAgent(seed=0)
            agent2.load_checkpoint(ckpt)

            # Q-tables should be equal
            for state, actions in agent._q_table.items():
                for action, q in actions.items():
                    assert agent2._q_table[state][action] == pytest.approx(q)

    def test_no_candidates_raises(self):
        agent = QLearningAgent(seed=0)
        with pytest.raises((ValueError, IndexError)):
            agent.choose_grna("r", "q", [], "tridoshic")
