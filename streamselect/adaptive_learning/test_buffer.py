""" Testing the buffer """

import numpy as np
from river.stream import iter_array, simulate_qa

from streamselect.adaptive_learning.buffer import (
    Observation,
    ObservationBuffer,
    SupervisedUnsupervisedBuffer,
)


# pylint: disable=too-many-statements, duplicate-code, R0801
def test_observation() -> None:
    """Test the observation class."""
    x = {"x1": 1}
    y = 0
    sample_weight = 1.0
    seen_at = 0
    active_state_id = 1
    observation = Observation(x=x, y=y, active_state_id=active_state_id, sample_weight=sample_weight, seen_at=seen_at)
    assert observation.x == x
    assert observation.y == y
    assert observation.sample_weight == sample_weight
    assert observation.seen_at == seen_at
    assert observation.active_state_id == active_state_id
    assert not observation.predictions

    observation.add_prediction(1, 0)
    observation.add_prediction(0, 1)

    assert observation.predictions == {0: 1, 1: 0}


def test_observation_buffer_1() -> None:
    """Test observation buffer with window_size 1"""
    window_size = 1
    buffer_timeout = 0
    buffer = ObservationBuffer(window_size=window_size)
    x_list = [{"x": 0}, {"x": 0}, {"x": 1}, {"x": 0}, {"x": 2}]
    y_list = [0, 1, 1, 0, 1]
    sample_weights = [1.0, 1.0, 0.0, 1.0, 1.0]
    timesteps = list(range(len(x_list)))
    for timestep, (x, y) in enumerate(zip(x_list, y_list)):
        stable_timestep = max(timestep - buffer_timeout, -1)
        active_state_id = 1
        _ = buffer.buffer_data(
            x,
            y,
            active_state_id=active_state_id,
            sample_weight=sample_weights[timestep],
            current_timestamp=timestep,
            stable_timestamp=stable_timestep,
        )

        assert [o.y for o in buffer.active_window] == y_list[timestep + 1 - window_size : timestep + 1]
        assert [o.x for o in buffer.active_window] == x_list[timestep + 1 - window_size : timestep + 1]
        assert [o.sample_weight for o in buffer.active_window] == sample_weights[
            timestep + 1 - window_size : timestep + 1
        ]
        assert [o.seen_at for o in buffer.active_window] == timesteps[timestep + 1 - window_size : timestep + 1]

        assert len(buffer.buffer) == 0

        assert [o.y for o in buffer.stable_window] == y_list[timestep + 1 - window_size : timestep + 1]
        assert [o.x for o in buffer.stable_window] == x_list[timestep + 1 - window_size : timestep + 1]
        assert [o.sample_weight for o in buffer.stable_window] == sample_weights[
            timestep + 1 - window_size : timestep + 1
        ]
        assert [o.seen_at for o in buffer.stable_window] == timesteps[timestep + 1 - window_size : timestep + 1]


def test_observation_buffer_2() -> None:
    """Test observation buffer with window_size 3"""
    window_size = 3
    buffer_timeout = 0
    buffer = ObservationBuffer(window_size=window_size)
    x_list = [{"x": 0}, {"x": 0}, {"x": 1}, {"x": 0}, {"x": 2}]
    y_list = [0, 1, 1, 0, 1]
    sample_weights = [1.0, 1.0, 0.0, 1.0, 1.0]
    timesteps = list(range(len(x_list)))
    for timestep, (x, y) in enumerate(zip(x_list, y_list)):
        stable_timestep = max(timestep - buffer_timeout, -1)
        active_state_id = 1
        _ = buffer.buffer_data(
            x,
            y,
            active_state_id=active_state_id,
            sample_weight=sample_weights[timestep],
            current_timestamp=timestep,
            stable_timestamp=stable_timestep,
        )

        assert [o.y for o in buffer.active_window] == y_list[max(timestep + 1 - window_size, 0) : timestep + 1]
        assert [o.x for o in buffer.active_window] == x_list[max(timestep + 1 - window_size, 0) : timestep + 1]
        assert [o.sample_weight for o in buffer.active_window] == sample_weights[
            max(timestep + 1 - window_size, 0) : timestep + 1
        ]
        assert [o.seen_at for o in buffer.active_window] == timesteps[
            max(timestep + 1 - window_size, 0) : timestep + 1
        ]
        assert len(buffer.active_window) == min(timestep + 1, window_size)

        assert len(buffer.buffer) == 0

        assert [o.y for o in buffer.stable_window] == y_list[max(timestep + 1 - window_size, 0) : timestep + 1]
        assert [o.x for o in buffer.stable_window] == x_list[max(timestep + 1 - window_size, 0) : timestep + 1]
        assert [o.sample_weight for o in buffer.stable_window] == sample_weights[
            max(timestep + 1 - window_size, 0) : timestep + 1
        ]
        assert [o.seen_at for o in buffer.stable_window] == timesteps[
            max(timestep + 1 - window_size, 0) : timestep + 1
        ]
        assert len(buffer.stable_window) == min(timestep + 1, window_size)


def test_observation_buffer_3() -> None:
    """Test observation buffer with window_size 3 and buffer 1"""
    window_size = 3
    buffer_timeout = 1
    buffer = ObservationBuffer(window_size=window_size)
    x_list = [{"x": 0}, {"x": 0}, {"x": 1}, {"x": 0}, {"x": 2}]
    y_list = [0, 1, 1, 0, 1]
    sample_weights = [1.0, 1.0, 0.0, 1.0, 1.0]
    timesteps = list(range(len(x_list)))
    for timestep, (x, y) in enumerate(zip(x_list, y_list)):
        stable_timestep = max(timestep - buffer_timeout, -1)
        active_state_id = 1
        _ = buffer.buffer_data(
            x,
            y,
            active_state_id=active_state_id,
            sample_weight=sample_weights[timestep],
            current_timestamp=timestep,
            stable_timestamp=stable_timestep,
        )

        assert [o.y for o in buffer.active_window] == y_list[max(timestep + 1 - window_size, 0) : timestep + 1]
        assert [o.x for o in buffer.active_window] == x_list[max(timestep + 1 - window_size, 0) : timestep + 1]
        assert [o.sample_weight for o in buffer.active_window] == sample_weights[
            max(timestep + 1 - window_size, 0) : timestep + 1
        ]
        assert [o.seen_at for o in buffer.active_window] == timesteps[
            max(timestep + 1 - window_size, 0) : timestep + 1
        ]
        assert len(buffer.active_window) == min(timestep + 1, window_size)

        assert len(buffer.buffer) == buffer_timeout

        assert [o.y for o in buffer.stable_window] == y_list[
            max(stable_timestep + 1 - window_size, 0) : stable_timestep + 1
        ]
        assert [o.x for o in buffer.stable_window] == x_list[
            max(stable_timestep + 1 - window_size, 0) : stable_timestep + 1
        ]
        assert [o.sample_weight for o in buffer.stable_window] == sample_weights[
            max(stable_timestep + 1 - window_size, 0) : stable_timestep + 1
        ]
        assert [o.seen_at for o in buffer.stable_window] == timesteps[
            max(stable_timestep + 1 - window_size, 0) : stable_timestep + 1
        ]
        assert len(buffer.stable_window) == min(stable_timestep + 1, window_size)

    assert [o.y for o in buffer.active_window] == [1, 0, 1]
    assert [o.y for o in buffer.stable_window] == [1, 1, 0]
    assert [o.y for o in buffer.buffer] == [1]


def test_observation_buffer_4() -> None:
    """Test observation buffer with window_size 3 and buffer 2"""
    window_size = 3
    buffer_timeout = 2
    buffer = ObservationBuffer(window_size=window_size)
    x_list = [{"x": 0}, {"x": 0}, {"x": 1}, {"x": 0}, {"x": 2}]
    y_list = [0, 1, 1, 0, 1]
    sample_weights = [1.0, 1.0, 0.0, 1.0, 1.0]
    timesteps = list(range(len(x_list)))
    collected_stable_obs = []
    for timestep, (x, y) in enumerate(zip(x_list, y_list)):
        stable_timestep = max(timestep - buffer_timeout, -1)
        active_state_id = 1
        stable_observations = buffer.buffer_data(
            x,
            y,
            active_state_id=active_state_id,
            sample_weight=sample_weights[timestep],
            current_timestamp=timestep,
            stable_timestamp=stable_timestep,
        )
        collected_stable_obs += stable_observations

        assert [o.y for o in buffer.active_window] == y_list[max(timestep + 1 - window_size, 0) : timestep + 1]
        assert [o.x for o in buffer.active_window] == x_list[max(timestep + 1 - window_size, 0) : timestep + 1]
        assert [o.sample_weight for o in buffer.active_window] == sample_weights[
            max(timestep + 1 - window_size, 0) : timestep + 1
        ]
        assert [o.seen_at for o in buffer.active_window] == timesteps[
            max(timestep + 1 - window_size, 0) : timestep + 1
        ]
        assert len(buffer.active_window) == min(timestep + 1, window_size)

        assert len(buffer.buffer) == min(timestep + 1, buffer_timeout)

        assert [o.y for o in buffer.stable_window] == y_list[
            max(stable_timestep + 1 - window_size, 0) : stable_timestep + 1
        ]
        assert [o.x for o in buffer.stable_window] == x_list[
            max(stable_timestep + 1 - window_size, 0) : stable_timestep + 1
        ]
        assert [o.sample_weight for o in buffer.stable_window] == sample_weights[
            max(stable_timestep + 1 - window_size, 0) : stable_timestep + 1
        ]
        assert [o.seen_at for o in buffer.stable_window] == timesteps[
            max(stable_timestep + 1 - window_size, 0) : stable_timestep + 1
        ]
        assert len(buffer.stable_window) == min(stable_timestep + 1, window_size)
        assert list(buffer.stable_window) == collected_stable_obs

    assert [o.y for o in buffer.active_window] == [1, 0, 1]
    assert [o.y for o in buffer.stable_window] == [0, 1, 1]
    assert [o.y for o in buffer.buffer] == [0, 1]


def test_observation_buffer_drift_reset() -> None:
    """Test resetting observation buffer with window_size 3 and buffer 2"""
    window_size = 3
    buffer_timeout = 2
    x_list = [{"x": 0}, {"x": 0}, {"x": 1}, {"x": 0}, {"x": 2}]
    y_list = [0, 1, 1, 0, 1]
    sample_weights = [1.0, 1.0, 0.0, 1.0, 1.0]
    timesteps = list(range(len(x_list)))

    buffer = ObservationBuffer(window_size=window_size)
    for timestep, (x, y) in enumerate(zip(x_list, y_list)):
        stable_timestep = max(timestep - buffer_timeout, -1)
        active_state_id = 1
        _ = buffer.buffer_data(
            x,
            y,
            active_state_id=active_state_id,
            sample_weight=sample_weights[timestep],
            current_timestamp=timestep,
            stable_timestamp=stable_timestep,
        )

    assert [o.y for o in buffer.active_window] == [1, 0, 1]
    assert [o.y for o in buffer.stable_window] == [0, 1, 1]
    assert [o.y for o in buffer.buffer] == [0, 1]

    buffer.reset_on_drift(timesteps[-1])
    assert len(buffer.buffer) == 0
    assert len(buffer.active_window) == 0
    assert len(buffer.stable_window) == 0

    buffer = ObservationBuffer(window_size=window_size)
    for timestep, (x, y) in enumerate(zip(x_list, y_list)):
        stable_timestep = max(timestep - buffer_timeout, -1)
        active_state_id = 1
        _ = buffer.buffer_data(
            x,
            y,
            active_state_id=active_state_id,
            sample_weight=sample_weights[timestep],
            current_timestamp=timestep,
            stable_timestamp=stable_timestep,
        )

    assert [o.y for o in buffer.active_window] == [1, 0, 1]
    assert [o.y for o in buffer.stable_window] == [0, 1, 1]
    assert [o.y for o in buffer.buffer] == [0, 1]

    drift_timestamp = timesteps[-2]
    buffer.reset_on_drift(drift_timestamp)
    assert [o.y for o in buffer.active_window] == [1]
    assert all(o.seen_at > drift_timestamp for o in buffer.active_window)
    assert not [o.y for o in buffer.stable_window]
    assert all(o.seen_at > drift_timestamp for o in buffer.stable_window)
    assert [o.y for o in buffer.buffer] == [1]
    assert all(o.seen_at > drift_timestamp for o in buffer.buffer)

    buffer = ObservationBuffer(window_size=window_size)
    for timestep, (x, y) in enumerate(zip(x_list, y_list)):
        stable_timestep = max(timestep - buffer_timeout, -1)
        active_state_id = 1
        _ = buffer.buffer_data(
            x,
            y,
            active_state_id=active_state_id,
            sample_weight=sample_weights[timestep],
            current_timestamp=timestep,
            stable_timestamp=stable_timestep,
        )

    assert [o.y for o in buffer.active_window] == [1, 0, 1]
    assert [o.y for o in buffer.stable_window] == [0, 1, 1]
    assert [o.y for o in buffer.buffer] == [0, 1]

    drift_timestamp = timesteps[0]
    buffer.reset_on_drift(drift_timestamp)
    assert [o.y for o in buffer.active_window] == [1, 0, 1]
    assert all(o.seen_at > drift_timestamp for o in buffer.active_window)
    assert [o.y for o in buffer.stable_window] == [1, 1]
    assert all(o.seen_at > drift_timestamp for o in buffer.stable_window)
    assert [o.y for o in buffer.buffer] == [0, 1]
    assert all(o.seen_at > drift_timestamp for o in buffer.buffer)


def test_supervised_unsupervised_buffer() -> None:
    """Test the supervised/unsupervised buffer."""
    window_size = 2
    buffer_timeout = 2
    x_list = [{"x": 0}, {"x": 0}, {"x": 1}, {"x": 0}, {"x": 2}]
    y_list = [0, 1, 1, 0, 1]
    sample_weights = [1.0, 1.0, 0.0, 1.0, 1.0]

    buffer = SupervisedUnsupervisedBuffer(window_size, -1.0, buffer_timeout)
    for timestep, (x, y) in enumerate(zip(x_list, y_list)):
        active_state_id = 1
        buffer.buffer_supervised(x, y, active_state_id=active_state_id, sample_weight=sample_weights[timestep])
        buffer.buffer_unsupervised(x, active_state_id=active_state_id, sample_weight=sample_weights[timestep])

    collected_supervised = buffer.collect_stable_supervised()
    assert list(o.y for o in collected_supervised) == [0, 1, 1]
    assert list(o.x for o in collected_supervised) == [{"x": 0}, {"x": 0}, {"x": 1}]
    collected_unsupervised = buffer.collect_stable_unsupervised()
    assert list(o.x for o in collected_unsupervised) == [{"x": 0}, {"x": 0}, {"x": 1}]


def test_supervised_unsupervised_buffer_incremental() -> None:
    """Test the supervised/unsupervised buffer collecting stable incrementally."""
    window_size = 2
    buffer_timeout = 2
    x_list = [{"x": 0}, {"x": 0}, {"x": 1}, {"x": 0}, {"x": 2}]
    y_list = [0, 1, 1, 0, 1]
    sample_weights = [1.0, 1.0, 0.0, 1.0, 1.0]

    collected_supervised = []
    collected_unsupervised = []
    buffer = SupervisedUnsupervisedBuffer(window_size, -1.0, buffer_timeout)
    for timestep, (x, y) in enumerate(zip(x_list, y_list)):
        active_state_id = 1
        buffer.buffer_supervised(x, y, active_state_id=active_state_id, sample_weight=sample_weights[timestep])
        buffer.buffer_unsupervised(x, active_state_id=active_state_id, sample_weight=sample_weights[timestep])
        collected_unsupervised += buffer.collect_stable_unsupervised()
        collected_supervised += buffer.collect_stable_supervised()

    assert list(o.y for o in collected_supervised) == [0, 1, 1]
    assert list(o.x for o in collected_supervised) == [{"x": 0}, {"x": 0}, {"x": 1}]
    assert list(o.x for o in collected_unsupervised) == [{"x": 0}, {"x": 0}, {"x": 1}]


def test_supervised_unsupervised_buffer_qa() -> None:
    """Test the supervised/unsupervised buffer in the qa format."""

    window_size = 2
    buffer_timeout = 2
    x_list = [{"x": 0}, {"x": 0}, {"x": 1}, {"x": 0}, {"x": 2}]
    x_vals_list = [[x["x"]] for x in x_list]
    y_list = [0, 1, 1, 0, 1]

    for delay in [0, 1, 2, 3, 4, 5]:
        dataset = iter_array(np.array(x_vals_list), np.array(y_list), feature_names=["x"])
        collected_supervised = []
        collected_unsupervised = []
        buffer = SupervisedUnsupervisedBuffer(window_size, -1.0, buffer_timeout)
        #: Error in river typing, should allow None
        for i, x, y in simulate_qa(dataset, moment=None, delay=delay):  # type: ignore
            print(i, x, y)
            active_state_id = 1
            supervised = y is not None
            if supervised:
                buffer.buffer_supervised(x, y, active_state_id=active_state_id, current_timestamp=i)
                collected_supervised += buffer.collect_stable_supervised()
            else:
                buffer.buffer_unsupervised(x, active_state_id=active_state_id, current_timestamp=i)
                collected_unsupervised += buffer.collect_stable_unsupervised()

        collected_supervised += buffer.collect_stable_supervised()
        collected_unsupervised += buffer.collect_stable_unsupervised()
        print(collected_supervised)
        print(collected_unsupervised)
        assert list(o.y for o in collected_supervised) == [0, 1, 1]
        assert list(o.x for o in collected_supervised) == [{"x": 0}, {"x": 0}, {"x": 1}]
        assert list(o.x for o in collected_unsupervised) == [{"x": 0}, {"x": 0}, {"x": 1}]


def test_supervised_unsupervised_buffer_qa_supervised() -> None:
    """Test the supervised/unsupervised buffer in the qa format.
    Using the supervised mode, observations are released from the buffer
    when they are older than buffer_timeout from the current newest supervised observation.
    The idea here is that if supervised drift detection is used, we will detect a drift and
    clear the buffer before these observations are learned from."""

    window_size = 2
    buffer_timeout = 2
    x_list = [{"x": 0}, {"x": 0}, {"x": 1}, {"x": 0}, {"x": 2}]
    x_vals_list = [[x["x"]] for x in x_list]
    y_list = [0, 1, 1, 0, 1]

    delay = 3
    dataset = iter_array(np.array(x_vals_list), np.array(y_list), feature_names=["x"])
    collected_supervised = []
    collected_unsupervised = []
    buffer = SupervisedUnsupervisedBuffer(window_size, -1.0, buffer_timeout)
    t = 0
    #: Error in river typing, should allow None
    for i, x, y in simulate_qa(dataset, moment=None, delay=delay):  # type: ignore
        print(i, x, y)
        active_state_id = 1
        supervised = y is not None
        if supervised:
            buffer.buffer_supervised(x, y, active_state_id=active_state_id, current_timestamp=i)
        else:
            buffer.buffer_unsupervised(x, active_state_id=active_state_id, current_timestamp=i)
        collected_supervised += buffer.collect_stable_supervised()
        collected_unsupervised += buffer.collect_stable_unsupervised()
        if t in [0, 1, 2, 3, 4, 5, 6]:
            assert not list(o.y for o in collected_supervised)
            assert not list(o.x for o in collected_supervised)
            assert not list(o.x for o in collected_unsupervised)
        if t == 7:
            assert list(o.y for o in collected_supervised) == [0]
            assert list(o.x for o in collected_supervised) == [{"x": 0}]
            assert list(o.x for o in collected_unsupervised) == [{"x": 0}]
        if t == 8:
            assert list(o.y for o in collected_supervised) == [0, 1]
            assert list(o.x for o in collected_supervised) == [{"x": 0}, {"x": 0}]
            assert list(o.x for o in collected_unsupervised) == [{"x": 0}, {"x": 0}]

        t += 1

    collected_supervised += buffer.collect_stable_supervised()
    collected_unsupervised += buffer.collect_stable_unsupervised()
    print(collected_supervised)
    print(collected_unsupervised)
    assert list(o.y for o in collected_supervised) == [0, 1, 1]
    assert list(o.x for o in collected_supervised) == [{"x": 0}, {"x": 0}, {"x": 1}]
    assert list(o.x for o in collected_unsupervised) == [{"x": 0}, {"x": 0}, {"x": 1}]


def test_supervised_unsupervised_buffer_qa_unsupervised() -> None:
    """Test the supervised/unsupervised buffer in the qa format.
    Using the unsupervised mode, observations are released from the buffer
    when they are older than buffer_timeout from the current newest unsupervised observation.
    The idea here is that if unsupervised drift detection is used, we will detect a drift and
    clear the buffer before these observations are learned from."""
    # pylint: disable="too-many-statements"
    window_size = 2
    buffer_timeout = 2
    x_list = [{"x": 0}, {"x": 0}, {"x": 1}, {"x": 0}, {"x": 2}]
    x_vals_list = [[x["x"]] for x in x_list]
    y_list = [0, 1, 1, 0, 1]

    delay = 3
    dataset = iter_array(np.array(x_vals_list), np.array(y_list), feature_names=["x"])
    collected_supervised = []
    collected_unsupervised = []
    buffer = SupervisedUnsupervisedBuffer(window_size, buffer_timeout, buffer_timeout, release_strategy="unsupervised")
    t = 0
    #: Error in river typing, should allow None
    for i, x, y in simulate_qa(dataset, moment=None, delay=delay):  # type: ignore
        print(i, x, y)
        active_state_id = 1
        supervised = y is not None
        if supervised:
            buffer.buffer_supervised(x, y, active_state_id=active_state_id, current_timestamp=i)
        else:
            buffer.buffer_unsupervised(x, active_state_id=active_state_id, current_timestamp=i)
        collected_supervised += buffer.collect_stable_supervised()
        collected_unsupervised += buffer.collect_stable_unsupervised()
        if t in [0, 1]:
            assert not list(o.y for o in collected_supervised)
            assert not list(o.x for o in collected_supervised)
            assert not list(o.x for o in collected_unsupervised)
        if t in [2]:
            assert not list(o.y for o in collected_supervised)
            assert not list(o.x for o in collected_supervised)
            assert list(o.x for o in collected_unsupervised) == [{"x": 0}]
        if t in [3]:
            assert list(o.y for o in collected_supervised) == [0]
            assert list(o.x for o in collected_supervised) == [{"x": 0}]
            assert list(o.x for o in collected_unsupervised) == [{"x": 0}]
        if t in [4]:
            assert list(o.y for o in collected_supervised) == [0]
            assert list(o.x for o in collected_supervised) == [{"x": 0}]
            assert list(o.x for o in collected_unsupervised) == [{"x": 0}, {"x": 0}]
        if t in [5]:
            assert list(o.y for o in collected_supervised) == [0, 1]
            assert list(o.x for o in collected_supervised) == [{"x": 0}, {"x": 0}]
            assert list(o.x for o in collected_unsupervised) == [{"x": 0}, {"x": 0}]
        if t in [6]:
            assert list(o.y for o in collected_supervised) == [0, 1]
            assert list(o.x for o in collected_supervised) == [{"x": 0}, {"x": 0}]
            assert list(o.x for o in collected_unsupervised) == [{"x": 0}, {"x": 0}, {"x": 1}]
        if t in [7]:
            assert list(o.y for o in collected_supervised) == [0, 1, 1]
            assert list(o.x for o in collected_supervised) == [{"x": 0}, {"x": 0}, {"x": 1}]
            assert list(o.x for o in collected_unsupervised) == [{"x": 0}, {"x": 0}, {"x": 1}]

        t += 1

    collected_supervised += buffer.collect_stable_supervised()
    collected_unsupervised += buffer.collect_stable_unsupervised()
    print(collected_supervised)
    print(collected_unsupervised)
    assert list(o.y for o in collected_supervised) == [0, 1, 1]
    assert list(o.x for o in collected_supervised) == [{"x": 0}, {"x": 0}, {"x": 1}]
    assert list(o.x for o in collected_unsupervised) == [{"x": 0}, {"x": 0}, {"x": 1}]


def test_supervised_unsupervised_buffer_qa_independent() -> None:
    """Test the supervised/unsupervised buffer in the qa format.
    Using the independent mode, observations are released from the buffer
    when they are older than buffer_timeout from the current newest observation in their
    respective buffer. Separate timeouts are possible for each type.
    The idea here is that if supervised and drift detection is used, we will detect a drift and
    clear the buffer before these observations are learned from."""
    # pylint: disable="too-many-statements"
    window_size = 2
    buffer_timeout = 2
    x_list = [{"x": 0}, {"x": 0}, {"x": 1}, {"x": 0}, {"x": 2}]
    x_vals_list = [[x["x"]] for x in x_list]
    y_list = [0, 1, 1, 0, 1]

    delay = 3
    dataset = iter_array(np.array(x_vals_list), np.array(y_list), feature_names=["x"])
    collected_supervised = []
    collected_unsupervised = []
    buffer = SupervisedUnsupervisedBuffer(window_size, buffer_timeout, buffer_timeout, release_strategy="independent")
    t = 0
    #: Error in river typing, should allow None
    for i, x, y in simulate_qa(dataset, moment=None, delay=delay):  # type: ignore
        print(i, x, y)
        active_state_id = 1
        supervised = y is not None
        if supervised:
            buffer.buffer_supervised(x, y, active_state_id=active_state_id, current_timestamp=i)
        else:
            buffer.buffer_unsupervised(x, active_state_id=active_state_id, current_timestamp=i)
        collected_supervised += buffer.collect_stable_supervised()
        collected_unsupervised += buffer.collect_stable_unsupervised()
        if t in [0, 1]:
            assert not list(o.y for o in collected_supervised)
            assert not list(o.x for o in collected_supervised)
            assert not list(o.x for o in collected_unsupervised)
        if t in [2]:
            assert not list(o.y for o in collected_supervised)
            assert not list(o.x for o in collected_supervised)
            assert list(o.x for o in collected_unsupervised) == [{"x": 0}]
        if t in [3]:
            assert not list(o.y for o in collected_supervised)
            assert not list(o.x for o in collected_supervised)
            assert list(o.x for o in collected_unsupervised) == [{"x": 0}]
        if t in [4]:
            assert not list(o.y for o in collected_supervised)
            assert not list(o.x for o in collected_supervised)
            assert list(o.x for o in collected_unsupervised) == [{"x": 0}, {"x": 0}]
        if t in [5]:
            assert not list(o.y for o in collected_supervised)
            assert not list(o.x for o in collected_supervised)
            assert list(o.x for o in collected_unsupervised) == [{"x": 0}, {"x": 0}]
        if t in [6]:
            assert not list(o.y for o in collected_supervised)
            assert not list(o.x for o in collected_supervised)
            assert list(o.x for o in collected_unsupervised) == [{"x": 0}, {"x": 0}, {"x": 1}]
        if t in [7]:
            assert list(o.y for o in collected_supervised) == [0]
            assert list(o.x for o in collected_supervised) == [{"x": 0}]
            assert list(o.x for o in collected_unsupervised) == [{"x": 0}, {"x": 0}, {"x": 1}]
        if t in [8]:
            assert list(o.y for o in collected_supervised) == [0, 1]
            assert list(o.x for o in collected_supervised) == [{"x": 0}, {"x": 0}]
            assert list(o.x for o in collected_unsupervised) == [{"x": 0}, {"x": 0}, {"x": 1}]

        t += 1

    collected_supervised += buffer.collect_stable_supervised()
    collected_unsupervised += buffer.collect_stable_unsupervised()
    print(collected_supervised)
    print(collected_unsupervised)
    assert list(o.y for o in collected_supervised) == [0, 1, 1]
    assert list(o.x for o in collected_supervised) == [{"x": 0}, {"x": 0}, {"x": 1}]
    assert list(o.x for o in collected_unsupervised) == [{"x": 0}, {"x": 0}, {"x": 1}]


def test_supervised_unsupervised_buffer_reset_supervised() -> None:
    """Test the supervised/unsupervised buffer reset in supervised mode."""

    window_size = 2
    buffer_timeout = 3
    x_list = [{"x": 0}, {"x": 0}, {"x": 1}, {"x": 0}, {"x": 2}]
    x_vals_list = [[x["x"]] for x in x_list]
    y_list = [0, 1, 1, 0, 1]

    delay = 3
    dataset = iter_array(np.array(x_vals_list), np.array(y_list), feature_names=["x"])
    collected_supervised = []
    collected_unsupervised = []
    buffer = SupervisedUnsupervisedBuffer(window_size, -1.0, buffer_timeout)
    #: Error in river typing, should allow None
    for i, x, y in simulate_qa(dataset, moment=None, delay=delay):  # type: ignore
        print(i, x, y)
        active_state_id = 1
        supervised = y is not None
        if supervised:
            buffer.buffer_supervised(x, y, active_state_id=active_state_id, current_timestamp=i)
        else:
            buffer.buffer_unsupervised(x, active_state_id=active_state_id, current_timestamp=i)
        collected_supervised += buffer.collect_stable_supervised()
        collected_unsupervised += buffer.collect_stable_unsupervised()

    # Test that initially, things are what we would expect for an initialized system
    assert len(buffer.supervised_buffer.stable_window) == window_size
    assert len(buffer.supervised_buffer.active_window) == window_size
    assert len(buffer.supervised_buffer.buffer) == buffer_timeout
    assert len(buffer.unsupervised_buffer.stable_window) == window_size
    assert len(buffer.unsupervised_buffer.active_window) == window_size
    assert len(buffer.unsupervised_buffer.buffer) == buffer_timeout
    buffer.reset_on_drift()

    # with the reset, we should keep the active window, since that is how we have selected the
    # new active state so it should be valid.
    # But we should delete items in the buffer and stable window which are not in the active window.
    assert len(buffer.supervised_buffer.stable_window) == 0
    assert len(buffer.supervised_buffer.active_window) == window_size
    assert len(buffer.supervised_buffer.buffer) == 2
    assert len(buffer.unsupervised_buffer.stable_window) == 0
    assert len(buffer.unsupervised_buffer.active_window) == window_size
    assert len(buffer.unsupervised_buffer.buffer) == 2


def test_supervised_unsupervised_buffer_reset_supervised_2() -> None:
    """Test the supervised/unsupervised buffer reset in supervised mode."""

    window_size = 2
    buffer_timeout = 1
    x_list = [{"x": 0}, {"x": 0}, {"x": 1}, {"x": 0}, {"x": 2}]
    x_vals_list = [[x["x"]] for x in x_list]
    y_list = [0, 1, 1, 0, 1]

    delay = 3
    dataset = iter_array(np.array(x_vals_list), np.array(y_list), feature_names=["x"])
    collected_supervised = []
    collected_unsupervised = []
    buffer = SupervisedUnsupervisedBuffer(window_size, -1.0, buffer_timeout)
    #: Error in river typing, should allow None
    for i, x, y in simulate_qa(dataset, moment=None, delay=delay):  # type: ignore
        print(i, x, y)
        active_state_id = 1
        supervised = y is not None
        if supervised:
            buffer.buffer_supervised(x, y, active_state_id=active_state_id, current_timestamp=i)
        else:
            buffer.buffer_unsupervised(x, active_state_id=active_state_id, current_timestamp=i)
        collected_supervised += buffer.collect_stable_supervised()
        collected_unsupervised += buffer.collect_stable_unsupervised()

    # Test that initially, things are what we would expect for an initialized system
    assert len(buffer.supervised_buffer.stable_window) == window_size
    assert len(buffer.supervised_buffer.active_window) == window_size
    assert len(buffer.supervised_buffer.buffer) == buffer_timeout
    assert len(buffer.unsupervised_buffer.stable_window) == window_size
    assert len(buffer.unsupervised_buffer.active_window) == window_size
    assert len(buffer.unsupervised_buffer.buffer) == buffer_timeout
    buffer.reset_on_drift()

    # with the reset, we should keep the active window, since that is how we have selected the
    # new active state so it should be valid.
    # But we should delete items in the buffer and stable window which are not in the active window.
    # In this case, the stable window shares the first element with the active window
    assert len(buffer.supervised_buffer.stable_window) == 1
    assert len(buffer.supervised_buffer.active_window) == window_size
    assert len(buffer.supervised_buffer.buffer) == 1
    assert len(buffer.unsupervised_buffer.stable_window) == 1
    assert len(buffer.unsupervised_buffer.active_window) == window_size
    assert len(buffer.unsupervised_buffer.buffer) == 1


def test_supervised_unsupervised_buffer_reset_supervised_3() -> None:
    """Test the supervised/unsupervised buffer reset in supervised mode."""

    window_size = 2
    buffer_timeout = 1
    x_list = [{"x": 0}, {"x": 0}, {"x": 1}, {"x": 0}, {"x": 2}]
    x_vals_list = [[x["x"]] for x in x_list]
    y_list = [0, 1, 1, 0, 1]

    delay = 3
    dataset = iter_array(np.array(x_vals_list), np.array(y_list), feature_names=["x"])
    collected_supervised = []
    collected_unsupervised = []
    buffer = SupervisedUnsupervisedBuffer(window_size, -1.0, buffer_timeout)
    t = 0
    #: Error in river typing, should allow None
    for i, x, y in simulate_qa(dataset, moment=None, delay=delay):  # type: ignore
        print(i, x, y)
        active_state_id = 1
        supervised = y is not None
        if supervised:
            buffer.buffer_supervised(x, y, active_state_id=active_state_id, current_timestamp=i)
        else:
            buffer.buffer_unsupervised(x, active_state_id=active_state_id, current_timestamp=i)
        collected_supervised += buffer.collect_stable_supervised()
        collected_unsupervised += buffer.collect_stable_unsupervised()
        t += 1
        if t == 6:
            break

    # Test that initially, things are what we would expect for an initialized system
    assert len(buffer.supervised_buffer.stable_window) == 1
    assert len(buffer.supervised_buffer.active_window) == window_size
    assert len(buffer.supervised_buffer.buffer) == buffer_timeout
    assert len(buffer.unsupervised_buffer.stable_window) == 1
    assert len(buffer.unsupervised_buffer.active_window) == window_size
    assert len(buffer.unsupervised_buffer.buffer) == 3
    buffer.reset_on_drift()

    # with the reset, we should keep the active window, since that is how we have selected the
    # new active state so it should be valid.
    # But we should delete items in the buffer and stable window which are not in the active window.
    # In this case, the stable window shares the first element with the active window
    # In this case, since we are basing drift on the supervised data, the drift timestep is set to 0
    # this means all data occured after the drift, so should all be retained.
    assert len(buffer.supervised_buffer.stable_window) == 1
    assert len(buffer.supervised_buffer.active_window) == window_size
    assert len(buffer.supervised_buffer.buffer) == buffer_timeout
    assert len(buffer.unsupervised_buffer.stable_window) == 1
    assert len(buffer.unsupervised_buffer.active_window) == window_size
    assert len(buffer.unsupervised_buffer.buffer) == 3


def test_supervised_unsupervised_buffer_reset_unsupervised() -> None:
    """Test the supervised/unsupervised buffer reset in supervised mode."""

    window_size = 2
    buffer_timeout = 1
    x_list = [{"x": 0}, {"x": 0}, {"x": 1}, {"x": 0}, {"x": 2}]
    x_vals_list = [[x["x"]] for x in x_list]
    y_list = [0, 1, 1, 0, 1]

    delay = 3
    dataset = iter_array(np.array(x_vals_list), np.array(y_list), feature_names=["x"])
    collected_supervised = []
    collected_unsupervised = []
    buffer = SupervisedUnsupervisedBuffer(window_size, buffer_timeout, buffer_timeout, release_strategy="unsupervised")
    t = 0
    #: Error in river typing, should allow None
    for i, x, y in simulate_qa(dataset, moment=None, delay=delay):  # type: ignore
        print(i, x, y)
        active_state_id = 1
        supervised = y is not None
        if supervised:
            buffer.buffer_supervised(x, y, active_state_id=active_state_id, current_timestamp=i)
        else:
            buffer.buffer_unsupervised(x, active_state_id=active_state_id, current_timestamp=i)
        print(buffer.unsupervised_buffer.buffer)
        collected_supervised += buffer.collect_stable_supervised()
        collected_unsupervised += buffer.collect_stable_unsupervised()
        t += 1
        if t == 6:
            break

    # Test that initially, things are what we would expect for an initialized system
    assert len(buffer.supervised_buffer.stable_window) == window_size
    assert len(buffer.supervised_buffer.active_window) == window_size
    assert len(buffer.supervised_buffer.buffer) == 0
    assert len(buffer.unsupervised_buffer.stable_window) == window_size
    assert len(buffer.unsupervised_buffer.active_window) == window_size
    assert len(buffer.unsupervised_buffer.buffer) == buffer_timeout
    buffer.reset_on_drift()

    # with the reset, we should keep the active window, since that is how we have selected the
    # new active state so it should be valid.
    # But we should delete items in the buffer and stable window which are not in the active window.
    # In this case, the stable window shares the first element with the active window
    # In this case, since we are basing drift on the unsupervised data, the drift timestep is set to 1
    # this means all supervised data occured before the drift, so should all be discarded.
    assert len(buffer.supervised_buffer.stable_window) == 0
    assert len(buffer.supervised_buffer.active_window) == 0
    assert len(buffer.supervised_buffer.buffer) == 0
    assert len(buffer.unsupervised_buffer.stable_window) == 1
    assert len(buffer.unsupervised_buffer.active_window) == window_size
    assert len(buffer.unsupervised_buffer.buffer) == buffer_timeout


def test_supervised_unsupervised_buffer_reset_independent() -> None:
    """Test the supervised/unsupervised buffer reset in independent mode."""

    window_size = 2
    buffer_timeout = 1
    x_list = [{"x": 0}, {"x": 0}, {"x": 1}, {"x": 0}, {"x": 2}]
    x_vals_list = [[x["x"]] for x in x_list]
    y_list = [0, 1, 1, 0, 1]

    delay = 3
    dataset = iter_array(np.array(x_vals_list), np.array(y_list), feature_names=["x"])
    collected_supervised = []
    collected_unsupervised = []
    buffer = SupervisedUnsupervisedBuffer(window_size, buffer_timeout, buffer_timeout, release_strategy="independent")
    t = 0
    #: Error in river typing, should allow None
    for i, x, y in simulate_qa(dataset, moment=None, delay=delay):  # type: ignore
        print(i, x, y)
        active_state_id = 1
        supervised = y is not None
        if supervised:
            buffer.buffer_supervised(x, y, active_state_id=active_state_id, current_timestamp=i)
        else:
            buffer.buffer_unsupervised(x, active_state_id=active_state_id, current_timestamp=i)
        print(buffer.unsupervised_buffer.buffer)
        collected_supervised += buffer.collect_stable_supervised()
        collected_unsupervised += buffer.collect_stable_unsupervised()
        t += 1
        if t == 6:
            break

    # Test that initially, things are what we would expect for an initialized system
    assert len(buffer.supervised_buffer.stable_window) == 1
    assert len(buffer.supervised_buffer.active_window) == window_size
    assert len(buffer.supervised_buffer.buffer) == buffer_timeout
    assert len(buffer.unsupervised_buffer.stable_window) == window_size
    assert len(buffer.unsupervised_buffer.active_window) == window_size
    assert len(buffer.unsupervised_buffer.buffer) == buffer_timeout
    buffer.reset_on_drift()

    # with the reset, we should keep the active window, since that is how we have selected the
    # new active state so it should be valid.
    # But we should delete items in the buffer and stable window which are not in the active window.
    # In this case, the stable window shares the first element with the active window
    # In this case, since we are basing drift idenpendently, each should act as in the supervised or unsupervised case
    # independentely.
    assert len(buffer.supervised_buffer.stable_window) == 1
    assert len(buffer.supervised_buffer.active_window) == window_size
    assert len(buffer.supervised_buffer.buffer) == buffer_timeout
    assert len(buffer.unsupervised_buffer.stable_window) == 1
    assert len(buffer.unsupervised_buffer.active_window) == window_size
    assert len(buffer.unsupervised_buffer.buffer) == buffer_timeout


# %%
