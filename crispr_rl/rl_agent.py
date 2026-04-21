"""
rl_agent.py — Reinforcement Learning gRNA Optimisation Agent

Implements a tabular Q-learning agent that learns to rank gRNA candidates
by maximising a reward signal composed of:
  - Molecular efficiency
  - Specificity
  - AyurHarmonyScore

The agent is intentionally lightweight (no GPU required) so the full
crispr_rl package can run on minimal infrastructure.
"""

import json
import logging
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Reward shaping
# ------------------------------------------------------------------ #

BASELINE_EFFICIENCY = 0.55   # random-pick efficiency benchmark


def compute_reward(
    efficiency: float,
    specificity: float,
    harmony_index: float,
    caution_level: str,
) -> float:
    """
    Compute a shaped reward for the RL agent.

    Components:
      - efficiency uplift vs baseline (range ~-1 to +1)
      - specificity bonus (range 0–0.5)
      - harmony bonus  (range 0–0.3)
      - caution penalty (range 0 to -0.3)
    """
    efficiency_uplift = (efficiency - BASELINE_EFFICIENCY) / BASELINE_EFFICIENCY
    specificity_bonus = 0.5 * specificity
    harmony_bonus = 0.3 * harmony_index
    caution_penalties = {"low": 0.0, "medium": -0.05, "high": -0.15, "critical": -0.30}
    caution_penalty = caution_penalties.get(caution_level, -0.10)

    reward = efficiency_uplift + specificity_bonus + harmony_bonus + caution_penalty
    return round(max(-2.0, min(2.0, reward)), 6)


# ------------------------------------------------------------------ #
# State / Action encoding
# ------------------------------------------------------------------ #

def encode_state(gc_bin: int, seed_gc_bin: int, pam_type: str) -> str:
    """Encode agent state as a discrete string key."""
    pam_map = {"NGG": 0, "NNGRRT": 1, "NGTN": 2}
    return f"{gc_bin}:{seed_gc_bin}:{pam_map.get(pam_type, 3)}"


def gc_bin(gc: float, bins: int = 5) -> int:
    """Discretise GC content into N bins."""
    return min(bins - 1, int(gc * bins))


# ------------------------------------------------------------------ #
# Q-Learning Agent
# ------------------------------------------------------------------ #

@dataclass
class Episode:
    """Record of a single training episode."""
    run_id: str
    request_id: str
    gene_id: str
    prakriti_profile: str
    selected_grna: str
    reward: float
    harmony_index: float
    efficiency: float
    specificity: float
    caution_level: str
    timestamp_ms: int

    def as_dict(self) -> dict:
        return asdict(self)


class QLearningAgent:
    """
    Tabular Q-learning agent for gRNA candidate selection.

    State: (GC bin, seed-GC bin, PAM type)
    Action: index into ranked candidate list (0 = top-ranked)
    """

    def __init__(
        self,
        alpha: float = 0.1,       # learning rate
        gamma: float = 0.9,       # discount factor
        epsilon: float = 0.15,    # exploration rate
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.02,
        seed: int = 42,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.seed = seed

        random.seed(seed)
        self._q_table: dict[str, dict[int, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._episode_history: list[Episode] = []
        self._step_count = 0

    # ------------------------------------------------------------------ #
    # Core RL methods
    # ------------------------------------------------------------------ #

    def select_action(self, state: str, n_actions: int) -> int:
        """ε-greedy action selection."""
        if n_actions == 0:
            return 0
        if random.random() < self.epsilon:
            return random.randint(0, n_actions - 1)
        q_vals = self._q_table[state]
        if not q_vals:
            return 0
        return max(range(n_actions), key=lambda a: q_vals.get(a, 0.0))

    def update(self, state: str, action: int, reward: float, next_state: str) -> float:
        """
        Update Q-table with the Bellman equation.
        Returns the TD error magnitude.
        """
        current_q = self._q_table[state][action]
        next_max = max(self._q_table[next_state].values(), default=0.0)
        target = reward + self.gamma * next_max
        td_error = target - current_q
        self._q_table[state][action] = current_q + self.alpha * td_error

        self._step_count += 1
        # Decay epsilon
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )
        return abs(td_error)

    # ------------------------------------------------------------------ #
    # High-level API
    # ------------------------------------------------------------------ #

    def choose_grna(
        self,
        run_id: str,
        request_id: str,
        candidates: list[dict],
        prakriti_profile: str,
    ) -> tuple[dict, float]:
        """
        Select the best gRNA from scored candidates using RL policy.

        Returns (chosen_candidate_dict, reward)
        """
        if not candidates:
            raise ValueError("No candidates provided to RL agent.")

        # Build states from top-5 candidates
        shortlist = candidates[:5]
        states = [
            encode_state(
                gc_bin(c["gc_content"]),
                gc_bin(c.get("seed_gc", c["gc_content"])),
                c.get("pam_type", "NGG"),
            )
            for c in shortlist
        ]

        action = self.select_action(states[0], len(shortlist))
        chosen = shortlist[action]

        reward = compute_reward(
            efficiency=chosen["efficiency_score"],
            specificity=chosen["specificity_score"],
            harmony_index=chosen.get("harmony_index", 0.5),
            caution_level=chosen.get("caution_level", "medium"),
        )

        # Update Q-table
        next_state = states[min(action + 1, len(states) - 1)]
        self.update(states[action], action, reward, next_state)

        # Log episode
        episode = Episode(
            run_id=run_id,
            request_id=request_id,
            gene_id=chosen["gene_id"],
            prakriti_profile=prakriti_profile,
            selected_grna=chosen["grna_sequence"],
            reward=reward,
            harmony_index=chosen.get("harmony_index", 0.5),
            efficiency=chosen["efficiency_score"],
            specificity=chosen["specificity_score"],
            caution_level=chosen.get("caution_level", "medium"),
            timestamp_ms=int(time.time() * 1000),
        )
        self._episode_history.append(episode)
        logger.info(
            "RL chose gRNA %s for gene %s | reward=%.4f | AHS=%.4f",
            chosen["grna_sequence"][:10],
            chosen["gene_id"],
            reward,
            chosen.get("harmony_index", 0.5),
        )
        return chosen, reward

    def receive_feedback(self, grna_sequence: str, human_score: float) -> None:
        """
        Incorporate explicit human feedback as an additional reward signal.
        human_score should be in [-1, 1].
        """
        logger.info(
            "Human feedback received for %s: score=%.2f",
            grna_sequence[:10], human_score,
        )
        # Locate last episode with this gRNA
        for ep in reversed(self._episode_history):
            if ep.selected_grna == grna_sequence:
                synthetic_state = encode_state(2, 2, "NGG")  # neutral state
                self.update(synthetic_state, 0, human_score * 0.5, synthetic_state)
                break

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #

    def get_metrics(self) -> dict:
        """Return aggregate metrics over all episodes."""
        if not self._episode_history:
            return {
                "total_episodes": 0,
                "avg_reward": 0.0,
                "avg_harmony_index": 0.0,
                "avg_efficiency": 0.0,
                "efficiency_uplift_pct": 0.0,
                "failed_calls": 0,
                "epsilon": round(self.epsilon, 4),
                "q_table_states": len(self._q_table),
            }

        rewards = [ep.reward for ep in self._episode_history]
        harmonies = [ep.harmony_index for ep in self._episode_history]
        efficiencies = [ep.efficiency for ep in self._episode_history]

        avg_eff = sum(efficiencies) / len(efficiencies)
        uplift_pct = (avg_eff - BASELINE_EFFICIENCY) / BASELINE_EFFICIENCY * 100

        return {
            "total_episodes": len(self._episode_history),
            "avg_reward": round(sum(rewards) / len(rewards), 4),
            "avg_harmony_index": round(sum(harmonies) / len(harmonies), 4),
            "avg_efficiency": round(avg_eff, 4),
            "efficiency_uplift_pct": round(uplift_pct, 2),
            "failed_calls": 0,
            "epsilon": round(self.epsilon, 4),
            "q_table_states": len(self._q_table),
            "baseline_efficiency": BASELINE_EFFICIENCY,
        }

    def save_checkpoint(self, path: Path) -> None:
        """Persist Q-table and config to JSON."""
        checkpoint = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "seed": self.seed,
            "step_count": self._step_count,
            "q_table": {
                state: dict(actions)
                for state, actions in self._q_table.items()
            },
        }
        path.write_text(json.dumps(checkpoint, indent=2))
        logger.info("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: Path) -> None:
        """Load Q-table and config from JSON."""
        data = json.loads(path.read_text())
        self.alpha = data["alpha"]
        self.gamma = data["gamma"]
        self.epsilon = data["epsilon"]
        self.seed = data["seed"]
        self._step_count = data["step_count"]
        self._q_table = defaultdict(
            lambda: defaultdict(float),
            {
                state: defaultdict(float, {int(a): q for a, q in actions.items()})
                for state, actions in data["q_table"].items()
            },
        )
        logger.info("Checkpoint loaded from %s (%d states)", path, len(self._q_table))
