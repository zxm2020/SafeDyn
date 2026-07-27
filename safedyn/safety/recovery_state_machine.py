"""
safety/recovery_state_machine.py
Explicit Recovery and Restart State Machine for SafeDyn-VLN Guard.

Paper Reference: Section 4.1, Figure 2 - State Machine
Implements explicit RECOVERY and RESTART states as described in paper.

State Transitions:
  NORMAL_EXECUTION → (reject) → CERTIFIED_REJECT → CERTIFIED_BACKUP_SEARCH
  CERTIFIED_BACKUP_SEARCH → (backup found) → CERTIFIED_YIELD_OR_HOLD
  CERTIFIED_BACKUP_SEARCH → (no backup) → FAIL_CLOSED
  CERTIFIED_YIELD_OR_HOLD → (hazard clears) → CERTIFIED_RESTART
  CERTIFIED_RESTART → (safe) → CERTIFIED_RESUME → NORMAL_EXECUTION
  CERTIFIED_RESTART → (unsafe) → FAIL_CLOSED

All transitions are logged with explicit transition reasons.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import time


class RecoveryState(Enum):
    """
    Explicit recovery states as described in paper Section 4.1.

    NORMAL_EXECUTION: Robot executing certified actions
    CERTIFIED_REJECT: Nominal action rejected, entering recovery
    CERTIFIED_BACKUP_SEARCH: Searching for safe backup action
    CERTIFIED_YIELD_OR_HOLD: Executing yield/hold backup
    CERTIFIED_RESTART: Attempting to restart after hazard clears
    CERTIFIED_RESUME: Resuming normal execution
    FAIL_CLOSED: Terminal fail-safe state (emergency stop)
    DEADLOCK_RECOVERY: Spin-stuck deadlock detected, attempting recovery
    """
    NORMAL_EXECUTION = "normal_execution"
    CERTIFIED_REJECT = "certified_reject"
    CERTIFIED_BACKUP_SEARCH = "certified_backup_search"
    CERTIFIED_YIELD_OR_HOLD = "certified_yield_or_hold"
    CERTIFIED_RESTART = "certified_restart"
    CERTIFIED_RESUME = "certified_resume"
    FAIL_CLOSED = "fail_closed"
    DEADLOCK_RECOVERY = "deadlock_recovery"


@dataclass
class RecoveryContext:
    """
    Context for recovery decision making.
    Passed to state machine at each step.
    """
    # Current decision
    proposal_accepted: bool = False
    proposal_rejected: bool = False
    proposal_modified: bool = False

    # Backup information
    backup_available: bool = False
    backup_type: Optional[str] = None
    safe_backup_count: int = 0

    # Hazard status
    hazard_cleared: bool = False
    min_clearance: float = float('inf')
    time_since_reject: float = 0.0

    # Restart conditions
    restart_safe: bool = False
    restart_conditions_met: bool = False

    # Fail-closed triggers
    deadline_miss: bool = False
    zero_slack_violation: bool = False
    visibility_violation: bool = False
    no_safe_action: bool = False

    # Deadlock detection fields
    certified_spin_stuck: bool = False
    spin_streak: int = 0
    no_progress_streak: int = 0
    robot_displacement_window: float = 0.0
    dist_to_goal_delta_window: float = 0.0
    min_dynamic_entity_distance: float = float('inf')
    min_dynamic_entity_distance_delta: float = 0.0
    min_ttc: float = float('inf')
    recovery_action_available: bool = False
    recovery_action_type: Optional[str] = None


@dataclass
class RecoveryDecision:
    """
    Output of state machine step.
    Records state, transition, and recovery actions.
    """
    # Current state
    current_state: RecoveryState = RecoveryState.NORMAL_EXECUTION
    previous_state: Optional[RecoveryState] = None

    # Transition information
    transition_occurred: bool = False
    transition_reason: str = ""
    transition_timestamp: float = 0.0

    # Recovery actions
    recovery_after_reject: bool = False
    fallback_type: Optional[str] = None
    restart_attempted: bool = False
    restart_success: bool = False

    # Terminal state
    fail_closed: bool = False
    fail_closed_reason: str = ""

    # Audit trail
    recovery_state_history: List[str] = field(default_factory=list)
    transition_count: int = 0


class RecoveryStateMachine:
    """
    Explicit Recovery and Restart State Machine.

    Implements paper Section 4.1 state machine with explicit
    RECOVERY (backup search + yield/hold) and RESTART states.

    Usage:
        machine = RecoveryStateMachine()
        context = RecoveryContext(proposal_rejected=True, backup_available=True)
        decision = machine.step(context)

        # decision.current_state = CERTIFIED_BACKUP_SEARCH
        # decision.transition_reason = "proposal_rejected_entering_recovery"
    """

    def __init__(self):
        self.current_state = RecoveryState.NORMAL_EXECUTION
        self.state_history: List[RecoveryState] = [RecoveryState.NORMAL_EXECUTION]
        self.transition_count = 0
        self.reject_timestamp: Optional[float] = None
        self.recovery_start_timestamp: Optional[float] = None

    def reset(self):
        """Reset state machine to initial state."""
        self.current_state = RecoveryState.NORMAL_EXECUTION
        self.state_history = [RecoveryState.NORMAL_EXECUTION]
        self.transition_count = 0
        self.reject_timestamp = None
        self.recovery_start_timestamp = None

    def step(self, context: RecoveryContext) -> RecoveryDecision:
        """
        Execute one state machine step.

        Args:
            context: Current decision context

        Returns:
            RecoveryDecision with state, transition, and actions
        """
        decision = RecoveryDecision(
            current_state=self.current_state,
            previous_state=self.current_state,
            transition_timestamp=time.time(),
            recovery_state_history=[s.value for s in self.state_history],
            transition_count=self.transition_count,
        )

        # State transition logic
        previous_state = self.current_state

        if self.current_state == RecoveryState.NORMAL_EXECUTION:
            self._handle_normal_execution(context, decision)

        elif self.current_state == RecoveryState.CERTIFIED_REJECT:
            self._handle_certified_reject(context, decision)

        elif self.current_state == RecoveryState.CERTIFIED_BACKUP_SEARCH:
            self._handle_backup_search(context, decision)

        elif self.current_state == RecoveryState.CERTIFIED_YIELD_OR_HOLD:
            self._handle_yield_or_hold(context, decision)

        elif self.current_state == RecoveryState.CERTIFIED_RESTART:
            self._handle_restart(context, decision)

        elif self.current_state == RecoveryState.CERTIFIED_RESUME:
            self._handle_resume(context, decision)

        elif self.current_state == RecoveryState.FAIL_CLOSED:
            self._handle_fail_closed(context, decision)

        elif self.current_state == RecoveryState.DEADLOCK_RECOVERY:
            self._handle_deadlock_recovery(context, decision)

        # Record transition if state changed
        if self.current_state != previous_state:
            decision.transition_occurred = True
            decision.previous_state = previous_state
            self.state_history.append(self.current_state)
            self.transition_count += 1
            decision.transition_count = self.transition_count

        decision.current_state = self.current_state
        decision.recovery_state_history = [s.value for s in self.state_history]

        return decision

    def _handle_normal_execution(self, context: RecoveryContext, decision: RecoveryDecision):
        """Handle NORMAL_EXECUTION state."""
        # Check for fail-closed triggers
        if context.deadline_miss:
            self.current_state = RecoveryState.FAIL_CLOSED
            decision.transition_reason = "deadline_miss_fail_closed"
            decision.fail_closed = True
            decision.fail_closed_reason = "deadline_miss"
            return

        if context.zero_slack_violation:
            self.current_state = RecoveryState.FAIL_CLOSED
            decision.transition_reason = "zero_slack_violation_fail_closed"
            decision.fail_closed = True
            decision.fail_closed_reason = "zero_slack_violation"
            return

        if context.visibility_violation:
            self.current_state = RecoveryState.FAIL_CLOSED
            decision.transition_reason = "visibility_violation_fail_closed"
            decision.fail_closed = True
            decision.fail_closed_reason = "visibility_violation"
            return

        # Check for deadlock (spin-stuck with entity approaching)
        if context.certified_spin_stuck:
            self.current_state = RecoveryState.DEADLOCK_RECOVERY
            decision.transition_reason = "certified_spin_stuck_deadlock_detected"
            self.recovery_start_timestamp = time.time()
            return

        # Normal operation: check if rejected
        if context.proposal_rejected:
            self.current_state = RecoveryState.CERTIFIED_REJECT
            decision.transition_reason = "proposal_rejected_entering_recovery"
            self.reject_timestamp = time.time()
            self.recovery_start_timestamp = time.time()
        else:
            # Stay in normal execution
            decision.transition_reason = "normal_execution_continuing"

    def _handle_certified_reject(self, context: RecoveryContext, decision: RecoveryDecision):
        """Handle CERTIFIED_REJECT state."""
        # Immediately transition to backup search
        self.current_state = RecoveryState.CERTIFIED_BACKUP_SEARCH
        decision.transition_reason = "reject_to_backup_search"
        decision.recovery_after_reject = True

    def _handle_backup_search(self, context: RecoveryContext, decision: RecoveryDecision):
        """Handle CERTIFIED_BACKUP_SEARCH state (RECOVERY)."""
        decision.recovery_after_reject = True

        if context.backup_available:
            # Backup found → yield/hold
            self.current_state = RecoveryState.CERTIFIED_YIELD_OR_HOLD
            decision.transition_reason = "backup_found_entering_yield"
            decision.fallback_type = context.backup_type
        else:
            # No backup → fail-closed
            self.current_state = RecoveryState.FAIL_CLOSED
            decision.transition_reason = "no_backup_available_fail_closed"
            decision.fail_closed = True
            decision.fail_closed_reason = "no_safe_backup"

    def _handle_yield_or_hold(self, context: RecoveryContext, decision: RecoveryDecision):
        """Handle CERTIFIED_YIELD_OR_HOLD state (RECOVERY continued)."""
        decision.recovery_after_reject = True
        decision.fallback_type = context.backup_type

        # Check if hazard cleared and restart conditions met
        if context.hazard_cleared and context.restart_conditions_met:
            self.current_state = RecoveryState.CERTIFIED_RESTART
            decision.transition_reason = "hazard_cleared_attempting_restart"
            decision.restart_attempted = True
        elif context.no_safe_action:
            # Backup became infeasible
            self.current_state = RecoveryState.FAIL_CLOSED
            decision.transition_reason = "backup_infeasible_fail_closed"
            decision.fail_closed = True
            decision.fail_closed_reason = "backup_became_infeasible"
        else:
            # Continue yielding/holding
            decision.transition_reason = "yielding_waiting_for_hazard_clear"

    def _handle_restart(self, context: RecoveryContext, decision: RecoveryDecision):
        """Handle CERTIFIED_RESTART state."""
        decision.restart_attempted = True

        if context.restart_safe:
            # Restart safe → resume
            self.current_state = RecoveryState.CERTIFIED_RESUME
            decision.transition_reason = "restart_safe_resuming"
            decision.restart_success = True
        else:
            # Restart unsafe → fail-closed
            self.current_state = RecoveryState.FAIL_CLOSED
            decision.transition_reason = "restart_unsafe_fail_closed"
            decision.fail_closed = True
            decision.fail_closed_reason = "restart_unsafe"
            decision.restart_success = False

    def _handle_resume(self, context: RecoveryContext, decision: RecoveryDecision):
        """Handle CERTIFIED_RESUME state."""
        # Resume → normal execution
        self.current_state = RecoveryState.NORMAL_EXECUTION
        decision.transition_reason = "resumed_to_normal_execution"
        decision.restart_success = True
        self.reject_timestamp = None
        self.recovery_start_timestamp = None

    def _handle_fail_closed(self, context: RecoveryContext, decision: RecoveryDecision):
        """Handle FAIL_CLOSED state (terminal)."""
        # Terminal state - no transitions out
        decision.fail_closed = True
        decision.transition_reason = "fail_closed_terminal"
        if not decision.fail_closed_reason:
            decision.fail_closed_reason = "terminal_fail_safe"

    def _handle_deadlock_recovery(self, context: RecoveryContext, decision: RecoveryDecision):
        """Handle DEADLOCK_RECOVERY state — spin-stuck deadlock detected.

        Recovery hierarchy:
          1. Recovery motion action if available and passes safety checks
          2. Emergency stop if no safe recovery action exists
        """
        decision.recovery_after_reject = True

        if context.recovery_action_available and context.recovery_action_type:
            # Recovery action available — stay in DEADLOCK_RECOVERY
            # The caller will use the recovery action instead of the spin
            decision.transition_reason = f"deadlock_recovery_action_{context.recovery_action_type}"
            decision.fallback_type = f"deadlock_{context.recovery_action_type}"
        elif context.no_safe_action:
            # No safe recovery — fail-closed
            self.current_state = RecoveryState.FAIL_CLOSED
            decision.transition_reason = "deadlock_recovery_no_safe_action_fail_closed"
            decision.fail_closed = True
            decision.fail_closed_reason = "deadlock_recovery_no_safe_action"
        else:
            # Still in deadlock recovery, trying to find action
            decision.transition_reason = "deadlock_recovery_searching"

    def get_state_summary(self) -> Dict[str, Any]:
        """Get current state machine summary."""
        return {
            "current_state": self.current_state.value,
            "state_history": [s.value for s in self.state_history],
            "transition_count": self.transition_count,
            "in_recovery": self.current_state in [
                RecoveryState.CERTIFIED_REJECT,
                RecoveryState.CERTIFIED_BACKUP_SEARCH,
                RecoveryState.CERTIFIED_YIELD_OR_HOLD,
                RecoveryState.DEADLOCK_RECOVERY,
            ],
            "in_restart": self.current_state == RecoveryState.CERTIFIED_RESTART,
            "fail_closed": self.current_state == RecoveryState.FAIL_CLOSED,
            "in_deadlock_recovery": self.current_state == RecoveryState.DEADLOCK_RECOVERY,
        }
