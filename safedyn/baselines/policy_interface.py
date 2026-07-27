"""
policies/policy_interface.py
Phase 0: Abstract policy interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class Policy(ABC):
    """
    Abstract interface for all policies.
    """

    @abstractmethod
    def act(self, obs: Dict[str, Any]) -> Dict[str, float]:
        """
        Given an observation, return an action dict.

        Args:
            obs: dict with keys like "robot_state", "entities", "goal", etc.

        Returns:
            dict with action values, e.g.
                {"linear_velocity": v, "angular_velocity": omega}
                or {"v_forward": v, "v_left": 0.0, "omega": omega}
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset internal policy state (e.g. for a new episode)."""
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__
