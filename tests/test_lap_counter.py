from __future__ import annotations

from vla_driving.planning.lap_counter import LapCounter


def test_lap_counter_requires_arming_before_crossing() -> None:
    counter = LapCounter(
        gate_a=(0.0, -1.0),
        gate_b=(0.0, 1.0),
        forward_yaw=0.0,
        total_laps=3,
        arm_distance_m=1.0,
    )

    state = counter.update(x=-0.1, y=0.0, yaw=0.0, timestamp_s=0.0, lap_progress=1.0)
    assert state.lap_count == 0

    state = counter.update(x=2.0, y=0.0, yaw=0.0, timestamp_s=1.0, lap_progress=0.2)
    assert state.armed

    state = counter.update(x=-0.1, y=0.0, yaw=0.0, timestamp_s=4.5, lap_progress=1.0)
    assert state.lap_count == 1
    assert not state.finished


def test_lap_counter_finishes_after_total_laps() -> None:
    counter = LapCounter(
        gate_a=(0.0, -1.0),
        gate_b=(0.0, 1.0),
        forward_yaw=0.0,
        total_laps=2,
        cooldown_s=0.0,
    )

    counter.update(x=2.0, y=0.0, yaw=0.0, timestamp_s=0.0, lap_progress=1.0)
    counter.update(x=-0.1, y=0.0, yaw=0.0, timestamp_s=1.0, lap_progress=1.0)
    counter.update(x=2.0, y=0.0, yaw=0.0, timestamp_s=2.0, lap_progress=1.0)
    state = counter.update(x=-0.1, y=0.0, yaw=0.0, timestamp_s=3.0, lap_progress=1.0)

    assert state.lap_count == 2
    assert state.laps_remaining == 0
    assert state.finished
