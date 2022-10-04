import math
from collections import deque
from itertools import chain
from typing import Generator, List, Optional

import numpy as np
from river import datasets

Vector = List[float]


class PollutionEmission:
    def __init__(
        self, x: float, y: float, r: float, initial_strength: float, spread_factor: float, min_max_range: Vector
    ):
        self.x = x
        self.y = y
        self.r = r
        self.initial_strength = initial_strength
        self.strength = initial_strength
        self.spread_factor = spread_factor
        self.min_max_range = min_max_range

        self.alive = True

        self.child: Optional[PollutionEmission] = None
        self.first_emitted = False
        self.last_time_step = -1.0

    def propagate(self, wind_vec: Vector, ts: float) -> None:
        if ts == self.last_time_step:
            return
        if self.alive:
            self.x += wind_vec[0]
            self.y += wind_vec[1]

            self.r *= self.spread_factor

            self.strength *= (1 / self.spread_factor) ** 2

        if any(
            (
                self.strength <= 10,
                self.min_max_range[1] < self.x,
                self.x < self.min_max_range[0],
                self.min_max_range[1] < self.y,
                self.y < self.min_max_range[0],
            )
        ):
            self.alive = False
            if not self.child is None:
                self.child.first_emitted = True
        self.last_time_step = ts

    def draw(self, world: np.ndarray, grid_square_size: int) -> np.ndarray:
        x1 = int(self.x - (self.r / 2))
        y1 = int(self.y - (self.r / 2))
        x2 = int(self.x + (self.r / 2))
        y2 = int(self.y + (self.r / 2))
        c1 = world_to_grid(x1, y1, grid_square_size)
        c2 = world_to_grid(x2, y2, grid_square_size)
        world[c1[0] : c2[0] + 1, c1[1] : c2[1] + 1] = world[c1[0] : c2[0] + 1, c1[1] : c2[1] + 1] + self.strength
        return world


class PollutionSource:
    def __init__(
        self,
        x: float,
        y: float,
        strength: float,
        size: float,
        spread_factor: float,
        seed: int,
        min_max_range: Vector,
        concept_id: int,
    ):
        self.x = x
        self.y = y
        self.strength = strength
        self.spread_factor = spread_factor
        self.initial_radius = size
        self.last_emitted: Optional[PollutionEmission] = None
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.min_max_range = min_max_range
        self.concept_id = concept_id
        self.emission_wait = 1 + 4 * self.concept_id
        self.seen = 0

    def update(self) -> bool:
        self.seen += 1
        if self.seen % self.emission_wait == 0:
            return True
        return False

    def emit(self) -> PollutionEmission:
        s = self.strength + self.rng.random() * self.strength
        e = PollutionEmission(self.x, self.y, self.initial_radius, s, self.spread_factor, self.min_max_range)
        if not self.last_emitted is None:
            self.last_emitted.child = e
        else:
            e.first_emitted = True
        self.last_emitted = e
        return e


class WindSimGenerator(datasets.base.SyntheticDataset):
    def __init__(
        self,
        concept: int = 2,
        produce_image: bool = False,
        num_sensors: int = 8,
        sensor_pattern: str = "circle",
        sample_random_state_init: Optional[int] = None,
    ):

        self.n_classes = 21
        self.n_targets = 1
        self.target_names = ["class"]
        self.target_values = [i for i in range(self.n_classes)]
        self.n_num_features = num_sensors * 2
        self.n_cat_features = 0
        self.n_categories_per_cat_feature = 0
        self.n_features = self.n_num_features + self.n_cat_features * self.n_categories_per_cat_feature
        self.feature_names = ["att_num_" + str(i) for i in range(self.n_num_features)]
        super().__init__(
            n_features=self.n_num_features + self.n_cat_features,
            n_classes=self.n_classes,
            n_outputs=1,
            task=datasets.base.MULTI_CLF,
        )
        if sample_random_state_init is None:
            self.sample_random_state = np.random.randint(0, 10000)
        else:
            self.sample_random_state = sample_random_state_init
        self.anchor = [0, 0]

        # Treat as meters, i.e 1000 = a 1000x1000 m window
        self.window_width = 150
        self.window_center = (int(self.window_width / 2), int(self.window_width / 2))

        # How many grid squares. window_width / grid_n = size of grid square in meters.
        self.grid_n = 75

        self.grid_square_width = int(self.window_width / self.grid_n)

        self.emitter_distance = 60
        self.base_emitter_size = 10
        self.base_emitter_strength = 50
        self.variable_emitter_strength = 50
        self.base_emitter_spread_factor = 1.005
        self.variable_emitter_spread_factor = 0.01

        self.sensor_distance = 40

        self.sources: list[PollutionSource] = []
        self.pollution: set[PollutionEmission] = set()
        self.pollution_chain: list[PollutionEmission] = []
        self.wind_direction = 0.0
        self.wind_strength = 2.0
        self.init_concept = concept
        self.concept = concept
        self.set_concept(concept)

        center_sensor_loc = self.window_center
        self.optimal_sensor_locs: list[tuple[float, float]] = [center_sensor_loc]
        if sensor_pattern == "circle":
            radius = self.sensor_distance / 2
            angle = 0.0
            while angle < 360:
                px = center_sensor_loc[0] + math.cos(math.radians(angle)) * radius
                py = center_sensor_loc[1] + math.sin(math.radians(angle)) * radius
                self.optimal_sensor_locs.append((px, py))
                angle += 360 / num_sensors
            radius = self.sensor_distance
            angle = 0
            while angle < 360:
                px = center_sensor_loc[0] + math.cos(math.radians(angle)) * radius
                py = center_sensor_loc[1] + math.sin(math.radians(angle)) * radius
                self.optimal_sensor_locs.append((px, py))
                angle += 360 / num_sensors
        elif sensor_pattern == "grid":
            num_sensors_x = math.ceil(math.sqrt(num_sensors))
            num_sensors_y = math.ceil(math.sqrt(num_sensors))
            sensor_x_gap = self.window_width / (num_sensors_x + 1)
            sensor_y_gap = self.window_width / (num_sensors_y + 1)

            for c in range(num_sensors_x):
                for r in range(num_sensors_y):
                    px = (c + 1) * sensor_x_gap
                    py = (r + 1) * sensor_y_gap
                    self.optimal_sensor_locs.append((px, py))
        elif sensor_pattern == "town":
            self.sensor_locs = [
                (4286, 1995),
                (734, 773),
                (1949, 1462),
                (1926, 2479),
                (3219, 1758),
                (4218, 3532),
                (3604, 1469),
                (3676, 2798),
                (714, 1654),
                (1158, 3263),
                (2947, 4227),
                (2515, 3419),
                (2865, 2577),
            ]
            self.sensor_square_locs = []
            for sx, sy in self.sensor_locs:
                self.sensor_square_locs.append((int(sx / self.grid_square_width), int(sy / self.grid_square_width)))
        else:
            raise ValueError("No valid sensor pattern")

        self.n_features = len(self.optimal_sensor_locs) - 1
        # Timestep in seconds
        self.timestep = 60 * 10

        self.produce_image = produce_image
        self.last_update_image: Optional[np.ndarray] = None

        self.emitted_values: list[deque[float]] = [deque()]
        self.last_y = 0
        # The number of timesteps a prediction is ahead of X.
        # I.E the y value received with a given X is the y value
        # y_lag ahead of the reveived X values.
        # For this sim, it should be 10 minutes.
        self.y_lag = 2
        self.x_trail = 1

        self.reading_period = 5

        self.prepared = False

        self.world = np.zeros(shape=(self.grid_n, self.grid_n), dtype=float)

        self.ex = 0

        self.prepare_for_use()

    def set_concept(self, concept_id: int) -> None:
        self.concept = concept_id
        self.set_wind(concept_id=concept_id)

        self.sources = []

        x = self.window_center[0] - math.cos(self.wind_direction_radians) * self.emitter_distance
        y = self.window_center[1] - math.sin(self.wind_direction_radians) * self.emitter_distance
        strength = self.base_emitter_strength + self.variable_emitter_strength * (
            get_circle_proportion(self.wind_direction_radians)
        )
        size = self.base_emitter_size
        spread_factor = 1.0
        self.sources.append(
            PollutionSource(
                x, y, strength, size, spread_factor, self.sample_random_state, [0, self.window_width], self.concept
            )
        )
        spread_factor = self.base_emitter_spread_factor + self.variable_emitter_spread_factor * (
            get_circle_proportion(self.wind_direction_radians)
        )

        x2 = x - math.cos(self.orth_wind_direction_radians) * (self.emitter_distance * 0.4)
        y2 = y - math.sin(self.orth_wind_direction_radians) * (self.emitter_distance * 0.4)
        strength = strength * 2.25
        size = self.base_emitter_size
        spread_factor += 0.02
        self.sources.append(
            PollutionSource(
                x2,
                y2,
                strength,
                size,
                spread_factor,
                self.sample_random_state + 1,
                [0, self.window_width],
                self.concept,
            )
        )

        x3 = x + math.cos(self.orth_wind_direction_radians) * (self.emitter_distance * 0.4)
        y3 = y + math.sin(self.orth_wind_direction_radians) * (self.emitter_distance * 0.4)
        # strength = strength * 2.25
        size = self.base_emitter_size
        # spread_factor += 0.02
        self.sources.append(
            PollutionSource(
                x3,
                y3,
                strength,
                size,
                spread_factor,
                self.sample_random_state + 2,
                [0, self.window_width],
                self.concept,
            )
        )

    def get_direction_from_concept(self, concept_id: int) -> float:
        return 90.0 * concept_id

    def set_wind(self, concept_id: int = 0, direc: Optional[float] = None, strength: Optional[float] = None) -> None:
        self.concept = concept_id
        wind_direction = self.get_direction_from_concept(concept_id)
        if direc is not None:
            wind_direction = direc
        # In knots: 1 knot = 0.514 m/s
        # Data average is around 2.2
        self.wind_strength = strength if strength is not None else self.wind_strength
        self.wind_direction = wind_direction % 360.0

        # Wind direction is a bearing, want a unit circle degree.
        wind_direction_corrected = (self.wind_direction - 90) % 360
        # print(f"Wind direction corrected: {wind_direction_corrected} degrees")
        self.wind_direction_radians = math.radians(wind_direction_corrected)
        # print(f"Wind direction corrected: {self.wind_direction_radians} radians")
        self.orth_wind_direction_radians = math.radians(self.wind_direction)

        self.wind_strength_x = math.cos(self.wind_direction_radians) * self.wind_strength
        self.orth_wind_strength_x = math.cos(self.orth_wind_direction_radians) * self.wind_strength
        self.wind_strength_y = math.sin(self.wind_direction_radians) * self.wind_strength
        self.orth_wind_strength_y = math.sin(self.orth_wind_direction_radians) * self.wind_strength

    def update_simulation(self) -> None:
        self.ex += 1
        del_set = []
        for p in self.pollution:
            p.propagate([self.wind_strength_x, self.wind_strength_y], self.ex)

            if not p.alive:
                del_set.append(p)
        for p in del_set:
            self.pollution.remove(p)

        for i, s in enumerate(self.sources):
            e = s.update()
            if e:
                emission = s.emit()
                self.pollution.add(emission)

    def update_world(self) -> np.ndarray:
        world = np.zeros((self.grid_n, self.grid_n))
        for p in self.pollution:
            world = p.draw(world, self.grid_square_width)
        return world

    def collect_sensor_readings(self, world: np.ndarray) -> tuple[list[float], float]:
        sensor_windows = []
        for x, y in self.optimal_sensor_locs:
            sensor_sum = 0

            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    noise_x_pos = int(x + dx)
                    noise_y_pos = int(y + dy)
                    grid_coords = world_to_grid(noise_x_pos, noise_y_pos, self.grid_square_width)
                    pollution_amount = world[grid_coords[0], grid_coords[1]]
                    sensor_sum += pollution_amount
            sensor_windows.append(sensor_sum)
        return (list(map(lambda x: x, sensor_windows[1:])), sensor_windows[0])

    def set_sensor_locs(self, world: np.ndarray) -> np.ndarray:
        for x, y in self.optimal_sensor_locs:
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    grid_coords = world_to_grid(x + dx, y + dy, self.grid_square_width)
                    world[grid_coords[0], grid_coords[1]] = 255
        return world

    def update(
        self, ts: int, collect_readings: bool = True, draw_intermediary: bool = False
    ) -> list[tuple[np.ndarray, float]]:
        num_updates_to_process = ts - self.ex
        update_readings = []
        drawn = False
        for i in range(num_updates_to_process):
            self.update_simulation()
            if draw_intermediary:
                world = self.update_world()
                drawn = True
            if self.ex % self.reading_period == 0 and collect_readings:
                if not drawn:
                    world = self.update_world()
                    drawn = True
                readings = self.collect_sensor_readings(world)
                self.add_emissions(readings)
                X, y = self.get_current_sample()
                update_readings.append((X, y))
            if drawn:
                world = self.set_sensor_locs(world)
                self.last_update_image = world
        return update_readings

    def get_last_image(self) -> Optional[np.ndarray]:
        return self.last_update_image

    def add_emissions(self, readings: tuple[list[float], float]) -> None:
        X, y = readings
        for index, emit in enumerate(chain([y], X)):
            self.emitted_values[index].append(emit)

    def prepare_for_use(self) -> None:
        self.emitted_values = [deque(maxlen=self.y_lag)]
        for i in range(self.n_features):
            self.emitted_values.append(deque(maxlen=self.x_trail + self.y_lag))
        # Skip first 50 values while pollution blows across to sensors
        self.update(self.ex + 50, collect_readings=False)

        # # Need to set up y values for X values y_lag behind.
        # for i in range(1 + self.x_trail + self.y_lag):
        #     self.add_emissions()
        self.update(self.ex + (1 + self.x_trail + self.y_lag) * self.reading_period)

        # print("prepared")
        # print(self.emitted_values)
        self.prepared = True

    def _prepare_for_use(self) -> None:
        self.prepare_for_use()

    def get_current_sample(self) -> tuple[np.ndarray, float]:
        trail_features = []
        for i, x_emissions in enumerate(self.emitted_values[1:]):
            trail_features.append(np.array(x_emissions)[: self.x_trail])

        X = np.hstack(trail_features).flatten()
        current_y = self.emitted_values[0][-1]
        y = self.quantize_y(current_y)
        return X, y

    def quantize_y(self, y: float) -> int:
        return int(min(y // 50, 20))

    def next_sample(self, batch_size: int = 1) -> tuple[np.ndarray, np.ndarray]:
        if not self.prepared:
            self.prepare_for_use()
        x_vals = []
        y_vals = []
        for b in range(batch_size):
            readings = self.update(self.ex + 1)
            while len(readings) == 0:
                readings = self.update(self.ex + 1)
            X, y = readings[0]
            x_vals.append(X)
            y_vals.append(y)
            # print(y)
        return (np.array(x_vals), np.array(y_vals))

    def probe_sample(self, draw_intermediary: bool = False) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        readings = self.update(self.ex + 1, draw_intermediary=draw_intermediary)
        if len(readings) > 0:
            X, y = readings[0]
            return np.array([X]), np.array([y])
        return None, None

    def get_info(self, concept: Optional[int] = None, strength: Optional[float] = None) -> str:
        c = concept if concept is not None else self.concept
        s = strength if strength is not None else self.wind_strength
        return f"WIND: Direction: {self.get_direction_from_concept(c)}, Speed: {s}"

    def __iter__(self) -> Generator:
        while True:
            x = dict()
            batch_x_vec, batch_y = self.next_sample()
            x_vec = batch_x_vec[0]
            y = batch_y[0]
            for i, x_val in enumerate(x_vec):
                x[f"x_num_{i}"] = x_val

            yield x, y


def get_circle_proportion(radians: float) -> float:
    """Get proportion of the way around a circle an amount of radians is"""
    return (radians % (2 * math.pi)) / (2 * math.pi)


def world_to_grid(world_x: float, world_y: float, grid_width: float) -> tuple[int, int]:
    return (int(round(world_x / grid_width)), int(round(world_y / grid_width)))
