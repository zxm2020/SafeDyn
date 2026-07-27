"""
policies/frozen_vln_policy.py
Phase 0: Wrapper that wraps any policy as a "frozen VLN" source.
In the full system, this would be replaced by an actual VLN model.
"""

import numpy as np
from typing import Dict, Any, Optional

from .policy_interface import Policy
from .goal_following_policy import GoalFollowingPolicy


class FrozenVLNPolicy(Policy):
    """
    Wraps a base policy and labels its output as a VLN intent.

    The system should NEVER execute the raw output of this policy directly.
    All outputs must pass through the safety guard certification pipeline.

    This class is a thin wrapper that adds metadata to indicate the action
    came from the frozen VLN policy.
    """

    def __init__(self, base_policy: Optional[Policy] = None):
        self._base = base_policy or GoalFollowingPolicy()

    def act(self, obs: Dict[str, Any]) -> Dict[str, float]:
        action = self._base.act(obs)
        action["source"] = "frozen_vln"
        return action

    def set_goal(self, goal: np.ndarray) -> None:
        if hasattr(self._base, "set_goal"):
            self._base.set_goal(goal)

    def reset(self) -> None:
        self._base.reset()

    @property
    def name(self) -> str:
        return f"FrozenVLN({self._base.name})"
