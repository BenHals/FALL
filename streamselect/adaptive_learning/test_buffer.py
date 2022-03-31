""" Testing the buffer """

from streamselect.adaptive_learning.buffer import (
    Observation,
    ObservationBuffer,
    SupervisedUnsupervisedBuffer,
)


def test_observation() -> None:
    """Test the observation class."""
    x = {"x1": 1}
    y = 0
    sample_weight = 1.0
    seen_at = 0
    observation = Observation(x, y, sample_weight, seen_at)
    assert observation.x == x
    assert observation.y == y
    assert observation.sample_weight == sample_weight
    assert observation.seen_at == seen_at
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
        stable_timestep = timestep - buffer_timeout
        _ = buffer.buffer_data(
            x, y, sample_weight=sample_weights[timestep], current_timestamp=timestep, stable_timestamp=stable_timestep
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
        stable_timestep = timestep - buffer_timeout
        _ = buffer.buffer_data(
            x, y, sample_weight=sample_weights[timestep], current_timestamp=timestep, stable_timestamp=stable_timestep
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
        stable_timestep = timestep - buffer_timeout
        _ = buffer.buffer_data(
            x, y, sample_weight=sample_weights[timestep], current_timestamp=timestep, stable_timestamp=stable_timestep
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
        stable_observations = buffer.buffer_data(
            x, y, sample_weight=sample_weights[timestep], current_timestamp=timestep, stable_timestamp=stable_timestep
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
        _ = buffer.buffer_data(
            x, y, sample_weight=sample_weights[timestep], current_timestamp=timestep, stable_timestamp=stable_timestep
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
        _ = buffer.buffer_data(
            x, y, sample_weight=sample_weights[timestep], current_timestamp=timestep, stable_timestamp=stable_timestep
        )

    assert [o.y for o in buffer.active_window] == [1, 0, 1]
    assert [o.y for o in buffer.stable_window] == [0, 1, 1]
    assert [o.y for o in buffer.buffer] == [0, 1]

    drift_timestamp = timesteps[-2]
    buffer.reset_on_drift(drift_timestamp)
    assert [o.y for o in buffer.active_window] == [1]
    assert all(o.seen_at > drift_timestamp for o in buffer.active_window)
    assert [o.y for o in buffer.stable_window] == []
    assert all(o.seen_at > drift_timestamp for o in buffer.stable_window)
    assert [o.y for o in buffer.buffer] == [1]
    assert all(o.seen_at > drift_timestamp for o in buffer.buffer)

    buffer = ObservationBuffer(window_size=window_size)
    for timestep, (x, y) in enumerate(zip(x_list, y_list)):
        stable_timestep = max(timestep - buffer_timeout, -1)
        _ = buffer.buffer_data(
            x, y, sample_weight=sample_weights[timestep], current_timestamp=timestep, stable_timestamp=stable_timestep
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
        buffer.buffer_supervised(x, y, sample_weight=sample_weights[timestep])
        buffer.buffer_unsupervised(x, sample_weight=sample_weights[timestep])

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
        buffer.buffer_supervised(x, y, sample_weight=sample_weights[timestep])
        buffer.buffer_unsupervised(x, sample_weight=sample_weights[timestep])
        collected_unsupervised += buffer.collect_stable_unsupervised()
        collected_supervised += buffer.collect_stable_supervised()

    assert list(o.y for o in collected_supervised) == [0, 1, 1]
    assert list(o.x for o in collected_supervised) == [{"x": 0}, {"x": 0}, {"x": 1}]
    assert list(o.x for o in collected_unsupervised) == [{"x": 0}, {"x": 0}, {"x": 1}]
