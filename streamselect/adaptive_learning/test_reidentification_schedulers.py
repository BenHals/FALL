from copy import deepcopy
from typing import Dict, List, Optional

from streamselect.adaptive_learning.reidentification_schedulers import (
    DriftDetectionCheck,
    DriftInfo,
    DriftType,
    PeriodicCheck,
    ReidentificationSchedule,
)


def test_drift_detection_check() -> None:
    """Test that the drift detection check correctly schedules checks a constant number
    of timesteps after a real drift."""
    check_delay = 50
    real_drifts = [
        DriftInfo(100, DriftType.DriftDetectorTriggered),
        DriftInfo(500, DriftType.DriftDetectorTriggered),
        DriftInfo(1000, DriftType.DriftDetectorTriggered),
    ]
    real_drifts_copy = deepcopy(real_drifts)
    schedule = ReidentificationSchedule()
    schedule.add_scheduler(DriftDetectionCheck(check_delay))

    drift_timeline: List[DriftInfo] = []
    for t in range(real_drifts[-1].drift_timestep + check_delay + 1):
        real_drift: Optional[DriftInfo] = None
        if len(real_drifts) > 0 and real_drifts[0].drift_timestep == t:
            real_drift = real_drifts.pop(0)

        triggered_drifts = schedule.get_scheduled_reidentifications(t)

        if real_drift:
            schedule.schedule_reidentification(real_drift)
        for triggered_drift in triggered_drifts:
            schedule.schedule_reidentification(triggered_drift)

        if real_drift:
            drift_timeline.append(real_drift)
        for triggered_drift in triggered_drifts:
            drift_timeline.append(triggered_drift)

    assert len(drift_timeline) == 6
    assert drift_timeline[0] == real_drifts_copy[0]
    assert drift_timeline[1].drift_timestep == real_drifts_copy[0].drift_timestep + check_delay
    assert drift_timeline[1].drift_type == DriftType.ScheduledOne
    assert drift_timeline[2] == real_drifts_copy[1]
    assert drift_timeline[3].drift_timestep == real_drifts_copy[1].drift_timestep + check_delay
    assert drift_timeline[3].drift_type == DriftType.ScheduledOne
    assert drift_timeline[4] == real_drifts_copy[2]
    assert drift_timeline[5].drift_timestep == real_drifts_copy[2].drift_timestep + check_delay
    assert drift_timeline[5].drift_type == DriftType.ScheduledOne


def test_drift_detection_check_transition() -> None:
    """Test that the drift detection check correctly schedules checks a constant number of timesteps after
    a real drift.
    Check transitions are correctly handled."""
    check_delay = 50
    real_drifts = [
        DriftInfo(100, DriftType.DriftDetectorTriggered),
        DriftInfo(500, DriftType.DriftDetectorTriggered),
        DriftInfo(1000, DriftType.DriftDetectorTriggered),
    ]
    real_drifts_copy = deepcopy(real_drifts)
    schedule = ReidentificationSchedule()
    schedule.add_scheduler(DriftDetectionCheck(check_delay))
    schedule.initialize(0)

    drift_timeline: List[DriftInfo] = []
    for t in range(real_drifts[-1].drift_timestep + check_delay + 1):
        real_drift: Optional[DriftInfo] = None
        if len(real_drifts) > 0 and real_drifts[0].drift_timestep == t:
            real_drift = real_drifts.pop(0)

        triggered_drifts = schedule.get_scheduled_reidentifications(t)

        if real_drift:
            schedule.schedule_reidentification(real_drift)
        for triggered_drift in triggered_drifts:
            schedule.schedule_reidentification(triggered_drift)

        # In the drift check case, the transition_reset should have no effect,
        # since we schedule the exact same check after the transition.
        # Note that we should call transition reset after the drift has been added!
        if t == 500:
            schedule.transition_reset(t)

        if real_drift:
            drift_timeline.append(real_drift)
        for triggered_drift in triggered_drifts:
            drift_timeline.append(triggered_drift)

    assert len(drift_timeline) == 6
    assert drift_timeline[0] == real_drifts_copy[0]
    assert drift_timeline[1].drift_timestep == real_drifts_copy[0].drift_timestep + check_delay
    assert drift_timeline[1].drift_type == DriftType.ScheduledOne
    assert drift_timeline[2] == real_drifts_copy[1]
    assert drift_timeline[3].drift_timestep == real_drifts_copy[1].drift_timestep + check_delay
    assert drift_timeline[3].drift_type == DriftType.ScheduledOne
    assert drift_timeline[4] == real_drifts_copy[2]
    assert drift_timeline[5].drift_timestep == real_drifts_copy[2].drift_timestep + check_delay
    assert drift_timeline[5].drift_type == DriftType.ScheduledOne


def test_drift_detection_check_transition_cancel() -> None:
    """Test that the drift detection check correctly schedules checks a constant number of timesteps after
    a real drift.
    Check transitions are correctly handled."""
    check_delay = 50
    real_drifts = [
        DriftInfo(490, DriftType.DriftDetectorTriggered),
        DriftInfo(500, DriftType.DriftDetectorTriggered),
        DriftInfo(1000, DriftType.DriftDetectorTriggered),
    ]
    real_drifts_copy = deepcopy(real_drifts)
    schedule = ReidentificationSchedule()
    schedule.add_scheduler(DriftDetectionCheck(check_delay))
    schedule.initialize(0)

    drift_timeline: List[DriftInfo] = []
    for t in range(real_drifts[-1].drift_timestep + check_delay + 1):
        real_drift: Optional[DriftInfo] = None
        if len(real_drifts) > 0 and real_drifts[0].drift_timestep == t:
            real_drift = real_drifts.pop(0)

        triggered_drifts = schedule.get_scheduled_reidentifications(t)

        if real_drift:
            schedule.schedule_reidentification(real_drift)
        for triggered_drift in triggered_drifts:
            schedule.schedule_reidentification(triggered_drift)

        # A transition should cancel pending checks which have not triggered.
        if t == 500:
            schedule.transition_reset(t)

        if real_drift:
            drift_timeline.append(real_drift)
        for triggered_drift in triggered_drifts:
            drift_timeline.append(triggered_drift)

    assert len(drift_timeline) == 5
    assert drift_timeline[0] == real_drifts_copy[0]
    assert drift_timeline[1] == real_drifts_copy[1]
    assert drift_timeline[2].drift_timestep == real_drifts_copy[1].drift_timestep + check_delay
    assert drift_timeline[2].drift_type == DriftType.ScheduledOne
    assert drift_timeline[3] == real_drifts_copy[2]
    assert drift_timeline[4].drift_timestep == real_drifts_copy[2].drift_timestep + check_delay
    assert drift_timeline[4].drift_type == DriftType.ScheduledOne


def test_periodic_check() -> None:
    """Test that the periodic check correctly schedules checks periodically."""
    check_period = 50
    real_drifts = [
        DriftInfo(100, DriftType.DriftDetectorTriggered),
        DriftInfo(500, DriftType.DriftDetectorTriggered),
        DriftInfo(1000, DriftType.DriftDetectorTriggered),
    ]
    real_drifts_copy = deepcopy(real_drifts)
    schedule = ReidentificationSchedule()
    schedule.add_scheduler(PeriodicCheck(check_period))
    schedule.initialize(0)

    drift_timeline: Dict[int, List[DriftInfo]] = {}
    total_timesteps = real_drifts[-1].drift_timestep + check_period + 1
    for t in range(total_timesteps):
        real_drift: Optional[DriftInfo] = None
        if len(real_drifts) > 0 and real_drifts[0].drift_timestep == t:
            real_drift = real_drifts.pop(0)

        triggered_drifts = schedule.get_scheduled_reidentifications(t)

        if real_drift:
            schedule.schedule_reidentification(real_drift)
        for triggered_drift in triggered_drifts:
            schedule.schedule_reidentification(triggered_drift)

        if real_drift:
            drift_timeline.setdefault(t, []).append(real_drift)
        for triggered_drift in triggered_drifts:
            drift_timeline.setdefault(t, []).append(triggered_drift)

    total_periodic_checks = total_timesteps // check_period
    all_captured_drifts = [d for drifts in drift_timeline.values() for d in drifts]
    assert len(all_captured_drifts) == total_periodic_checks + len(real_drifts_copy)
    for triggered_ts in range(check_period, total_timesteps, check_period):
        assert drift_timeline[triggered_ts][-1].drift_timestep == triggered_ts
        assert drift_timeline[triggered_ts][-1].drift_type == DriftType.ScheduledPeriodic


def test_periodic_check_reset() -> None:
    """Test that the periodic check correctly schedules checks periodically.
    Check that transition resets are handled appropriatelly, i.e., reset the period."""
    check_period = 50
    real_drifts = [
        DriftInfo(100, DriftType.DriftDetectorTriggered),
        DriftInfo(225, DriftType.DriftDetectorTriggered),
        DriftInfo(500, DriftType.DriftDetectorTriggered),
    ]
    schedule = ReidentificationSchedule()
    schedule.add_scheduler(PeriodicCheck(check_period))
    schedule.initialize(0)

    drift_timeline: Dict[int, List[DriftInfo]] = {}
    total_timesteps = real_drifts[-1].drift_timestep + check_period + 1
    for t in range(total_timesteps):
        real_drift: Optional[DriftInfo] = None
        if len(real_drifts) > 0 and real_drifts[0].drift_timestep == t:
            real_drift = real_drifts.pop(0)

        triggered_drifts = schedule.get_scheduled_reidentifications(t)

        if real_drift:
            schedule.schedule_reidentification(real_drift)
        for triggered_drift in triggered_drifts:
            schedule.schedule_reidentification(triggered_drift)

        if real_drift:
            schedule.transition_reset(t)

        if real_drift:
            drift_timeline.setdefault(t, []).append(real_drift)
        for triggered_drift in triggered_drifts:
            drift_timeline.setdefault(t, []).append(triggered_drift)

    periodic_ts = [50, 100, 150, 200, 275, 325, 375, 425, 475, 550]
    real_ts = [100, 225, 500]
    all_captured_drifts = [d for drifts in drift_timeline.values() for d in drifts]
    assert len(all_captured_drifts) == len(periodic_ts) + len(real_ts)
    for triggered_ts in periodic_ts:
        assert triggered_ts in drift_timeline
        assert drift_timeline[triggered_ts][-1].drift_timestep == triggered_ts
        assert drift_timeline[triggered_ts][-1].drift_type == DriftType.ScheduledPeriodic
    for detected_ts in real_ts:
        assert detected_ts in drift_timeline
        assert drift_timeline[detected_ts][0].drift_timestep == detected_ts
        assert drift_timeline[detected_ts][0].drift_type == DriftType.DriftDetectorTriggered
