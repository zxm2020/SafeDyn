"""
policies/etpnav_wrapper.py
Stage F3G: Minimal ETPNav Policy Wrapper

Loads ETPNav checkpoint but provides fallback if full model instantiation fails.
Does NOT claim baseline reproduction if fallback is used.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

# Add ETPNav to path
PROJECT_ROOT = Path(__file__).parent.parent
ETPNAV_ROOT = PROJECT_ROOT / "external" / "ETPNav"
if str(ETPNAV_ROOT) not in sys.path:
    sys.path.insert(0, str(ETPNAV_ROOT))


class ETPNavPolicyWrapper:
    """
    Minimal ETPNav policy wrapper with checkpoint loading.
    
    Modes:
    - "checkpoint_only": Checkpoint loaded but model not instantiated
    - "full_model": Full ETPNav model instantiated (requires habitat-lab)
    - "fallback": Using fallback action generation
    
    Honest reporting:
    - Always sets checkpoint_loaded=True if checkpoint loads
    - Sets model_instantiated=False if habitat-lab not available
    - Never claims baseline reproduction if using fallback
    """
    
    def __init__(
        self,
        repo_root: str = "external/ETPNav",
        checkpoint_path: str = "external/ETPNav/pretrained/ETP/model_step_82500.pt",
        clip_checkpoint_path: str = "external/ETPNav/ViT-B-32.pt",
        device: str = "cuda",
        allow_fallback: bool = True,
        goal: Optional[np.ndarray] = None,
    ):
        """
        Initialize ETPNav wrapper.
        
        Args:
            repo_root: Path to ETPNav repo
            checkpoint_path: Path to ETP checkpoint
            clip_checkpoint_path: Path to CLIP checkpoint
            device: Device for inference
            allow_fallback: Allow fallback to dummy policy if model fails
            goal: Goal position [x, z] for fallback
        """
        self.policy_mode = "etpnav_wrapper"
        self.policy_repo = "ETPNav"
        self.repo_root = Path(repo_root)
        self.checkpoint_path = Path(checkpoint_path)
        self.clip_checkpoint_path = Path(clip_checkpoint_path)
        self.device = device
        self.allow_fallback = allow_fallback
        self.goal = goal
        
        # Status flags
        self.checkpoint_loaded = False
        self.clip_checkpoint_loaded = False
        self.checkpoint_key_count = 0
        self.checkpoint_key_prefixes = []
        self.policy_model_instantiated = False
        self.wrapper_mode = "uninitialized"
        self.blockers = []
        self.imported_modules = []
        self.missing_modules = []
        
        # Model (if instantiated)
        self.model = None
        self.checkpoint_data = None
        
        # Initialize
        self._load_checkpoint_metadata()
        self._try_import_etpnav()
        self._determine_wrapper_mode()
    
    def _load_checkpoint_metadata(self) -> None:
        """Load checkpoint and extract metadata without full model instantiation."""
        try:
            import torch
            
            # Load main checkpoint
            if self.checkpoint_path.exists():
                self.checkpoint_data = torch.load(
                    self.checkpoint_path, 
                    map_location="cpu",
                    weights_only=False
                )
                self.checkpoint_loaded = True
                
                if isinstance(self.checkpoint_data, dict):
                    self.checkpoint_key_count = len(self.checkpoint_data)
                    self.checkpoint_key_prefixes = self._extract_key_prefixes(
                        list(self.checkpoint_data.keys())
                    )
                else:
                    self.blockers.append(f"Checkpoint is {type(self.checkpoint_data)}, not dict")
            else:
                self.blockers.append(f"Checkpoint not found: {self.checkpoint_path}")
            
            # Load CLIP checkpoint
            if self.clip_checkpoint_path.exists():
                clip_data = torch.load(
                    self.clip_checkpoint_path,
                    map_location="cpu",
                    weights_only=False
                )
                self.clip_checkpoint_loaded = True
            else:
                self.blockers.append(f"CLIP checkpoint not found: {self.clip_checkpoint_path}")
                
        except Exception as e:
            self.blockers.append(f"Checkpoint load error: {e}")
    
    def _extract_key_prefixes(self, keys: List[str], max_prefixes: int = 10) -> List[str]:
        """Extract unique key prefixes from checkpoint keys."""
        prefixes = set()
        for key in keys:
            parts = key.split('.')
            if len(parts) >= 2:
                prefix = '.'.join(parts[:2])
            else:
                prefix = parts[0] if parts else key
            prefixes.add(prefix)
        return sorted(list(prefixes))[:max_prefixes]
    
    def _try_import_etpnav(self) -> None:
        """Try to import ETPNav modules and record what works."""
        modules_to_test = [
            ("vlnce_baselines", "vlnce_baselines"),
            ("vlnce_baselines.models", "from vlnce_baselines import models"),
            ("vlnce_baselines.models.policy", "from vlnce_baselines.models import policy"),
            ("Policy_ViewSelection_ETP", "from vlnce_baselines.models import Policy_ViewSelection_ETP"),
            ("ss_trainer_ETP", "from vlnce_baselines import ss_trainer_ETP"),
            ("habitat_extensions", "habitat_extensions"),
        ]
        
        for name, import_stmt in modules_to_test:
            try:
                if "import " in import_stmt:
                    exec(import_stmt)
                else:
                    __import__(import_stmt)
                self.imported_modules.append(name)
            except ModuleNotFoundError as e:
                self.missing_modules.append(f"{name}: {e}")
            except Exception as e:
                self.missing_modules.append(f"{name}: {type(e).__name__}: {e}")
    
    def _determine_wrapper_mode(self) -> None:
        """Determine what mode the wrapper should operate in."""
        if not self.checkpoint_loaded:
            self.wrapper_mode = "checkpoint_load_failed"
            self.policy_model_instantiated = False
            return
        
        # Check if we can instantiate full model
        etp_available = "Policy_ViewSelection_ETP" in self.imported_modules
        habitat_available = "habitat_extensions" in self.imported_modules
        
        if etp_available and habitat_available:
            # Try to instantiate full model
            try:
                self._instantiate_full_model()
                self.wrapper_mode = "full_model"
                self.policy_model_instantiated = True
            except Exception as e:
                self.blockers.append(f"Model instantiation failed: {e}")
                if self.allow_fallback:
                    self.wrapper_mode = "checkpoint_only_fallback"
                    self.policy_model_instantiated = False
                else:
                    self.wrapper_mode = "checkpoint_only_no_fallback"
                    self.policy_model_instantiated = False
        else:
            # Cannot instantiate full model due to missing dependencies
            if not etp_available:
                self.blockers.append("Policy_ViewSelection_ETP not importable (missing dependencies)")
            if not habitat_available:
                self.blockers.append("habitat_extensions not importable (habitat-lab not integrated)")
            
            if self.allow_fallback:
                self.wrapper_mode = "checkpoint_only_fallback"
            else:
                self.wrapper_mode = "checkpoint_only_no_fallback"
            self.policy_model_instantiated = False
    
    def _instantiate_full_model(self) -> None:
        """
        Instantiate full ETPNav model.
        
        This requires:
        - habitat-lab installed and configured
        - All ETPNav dependencies (gym, etc.)
        - Model configuration
        - Checkpoint weight loading
        """
        # This would be implemented when habitat-lab is fully integrated
        # For now, mark as requiring full integration
        raise NotImplementedError(
            "Full ETPNav model instantiation requires habitat-lab integration. "
            "See experiments/F3G_ETPNAV_WRAPPER_AUDIT.md for details."
        )
    
    def act(self, observation: Dict[str, Any], instruction: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate action from observation.
        
        If full model is instantiated, use real ETPNav inference.
        Otherwise, use fallback (if allowed).
        
        Args:
            observation: Current observation dict with robot_state, etc.
            instruction: Navigation instruction (for VLN)
        
        Returns:
            Action dict with proposal_action, metadata
        """
        if self.wrapper_mode in ["full_model"] and self.policy_model_instantiated:
            return self._act_full_model(observation, instruction)
        elif self.allow_fallback:
            return self._act_fallback(observation, instruction)
        else:
            return self._act_failure()
    
    def _act_full_model(self, observation: Dict[str, Any], instruction: Optional[str]) -> Dict[str, Any]:
        """Generate action using full ETPNav model."""
        # Would implement full forward pass here
        raise NotImplementedError("Full model inference not yet implemented")
    
    def _act_fallback(self, observation: Dict[str, Any], instruction: Optional[str]) -> Dict[str, Any]:
        """
        Generate fallback action when full model not available.
        
        Uses goal-following as fallback.
        Does NOT claim to be ETPNav baseline reproduction.
        """
        robot_state = observation.get("robot_state", {})
        rx = robot_state.get("x", 0.0)
        rz = robot_state.get("z", 0.0)
        yaw = robot_state.get("yaw", 0.0)
        
        # Goal following
        if self.goal is not None:
            dx = self.goal[0] - rx
            dz = self.goal[1] - rz
            dist = np.sqrt(dx**2 + dz**2)
            
            if dist < 0.5:
                v = 0.0
                omega = 0.0
            else:
                target_yaw = np.arctan2(dx, dz) + np.pi/2
                yaw_error = target_yaw - yaw
                yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
                v = min(0.4, dist * 0.5)
                omega = np.clip(yaw_error * 2.0, -1.0, 1.0)
        else:
            v = 0.0
            omega = 0.0
        
        return {
            "linear_velocity": float(v),
            "angular_velocity": float(omega),
            "action_source": "etpnav_wrapper_fallback",
            "policy_mode": self.policy_mode,
            "policy_repo": self.policy_repo,
            "policy_checkpoint_path": str(self.checkpoint_path),
            "policy_checkpoint_loaded": self.checkpoint_loaded,
            "policy_checkpoint_key_count": self.checkpoint_key_count,
            "policy_model_instantiated": self.policy_model_instantiated,
            "wrapper_mode": self.wrapper_mode,
            "blockers": self.blockers,
            "baseline_reproduction": False,
            "preliminary": True,
        }
    
    def _act_failure(self) -> Dict[str, Any]:
        """Return failure when no fallback allowed and model not ready."""
        return {
            "linear_velocity": 0.0,
            "angular_velocity": 0.0,
            "action_source": "etpnav_wrapper_failure",
            "policy_mode": self.policy_mode,
            "policy_checkpoint_loaded": self.checkpoint_loaded,
            "policy_model_instantiated": self.policy_model_instantiated,
            "wrapper_mode": self.wrapper_mode,
            "blockers": self.blockers,
            "error": "Model not instantiated and fallback not allowed",
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Return current wrapper status."""
        return {
            "policy_mode": self.policy_mode,
            "policy_repo": self.policy_repo,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_exists": self.checkpoint_path.exists(),
            "clip_checkpoint_path": str(self.clip_checkpoint_path),
            "clip_checkpoint_exists": self.clip_checkpoint_path.exists(),
            "checkpoint_loaded": self.checkpoint_loaded,
            "checkpoint_key_count": self.checkpoint_key_count,
            "checkpoint_key_prefixes": self.checkpoint_key_prefixes,
            "clip_checkpoint_loaded": self.clip_checkpoint_loaded,
            "imported_modules": self.imported_modules,
            "missing_modules": self.missing_modules,
            "policy_model_instantiated": self.policy_model_instantiated,
            "wrapper_mode": self.wrapper_mode,
            "blockers": self.blockers,
            "can_generate_proposal": self.wrapper_mode in ["full_model", "checkpoint_only_fallback"],
            "baseline_reproduction": False,
            "preliminary": True,
        }
    
    def reset(self):
        """Reset wrapper state."""
        if self.model is not None:
            # Would reset model state if instantiated
            pass


def create_etpnav_wrapper(
    checkpoint_path: Optional[str] = None,
    goal: Optional[np.ndarray] = None,
    allow_fallback: bool = True,
) -> ETPNavPolicyWrapper:
    """
    Factory function to create ETPNav wrapper.
    
    Args:
        checkpoint_path: Override default checkpoint path
        goal: Goal position for fallback
        allow_fallback: Allow fallback actions
    
    Returns:
        ETPNavPolicyWrapper instance
    """
    if checkpoint_path is None:
        checkpoint_path = "external/ETPNav/pretrained/ETP/model_step_82500.pt"
    
    return ETPNavPolicyWrapper(
        checkpoint_path=checkpoint_path,
        goal=goal,
        allow_fallback=allow_fallback,
    )
