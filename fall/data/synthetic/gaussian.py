from typing import Generator, Optional

import numpy as np
from river import datasets


class GaussianGenerator(datasets.base.SyntheticDataset):
    def __init__(
        self,
        concept: int = 1,
        sample_random_state_init: Optional[int] = None,
    ):

        self.n_classes = 100
        self.n_targets = 1
        self.target_names = ["class"]
        self.target_values = list(range(self.n_classes))
        self.n_num_features = 1
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
            self.sample_random_state = np.random.default_rng(seed=np.random.randint(0, 10000))
        else:
            self.sample_random_state = np.random.default_rng(seed=sample_random_state_init)

        self.concept = concept
        state_rng = np.random.default_rng(seed=concept)
        self.mu = state_rng.random() * 100
        self.sigma = state_rng.random() * 25

        self.ex = 0
        self.prepared = True

        self.prepare_for_use()

    def prepare_for_use(self) -> None:
        self.prepared = True

    def _prepare_for_use(self) -> None:
        self.prepare_for_use()

    def next_sample(self, batch_size: int = 1) -> tuple[np.ndarray, np.ndarray]:
        if not self.prepared:
            self.prepare_for_use()
        x_vals = []
        y_vals = []
        for _ in range(batch_size):
            X = [self.sample_random_state.random()]
            y = quantize_y(self.sample_random_state.normal(loc=self.mu, scale=self.sigma))
            x_vals.append(X)
            y_vals.append(y)
        return (np.array(x_vals), np.array(y_vals))

    def get_info(self, concept: Optional[int] = None, strength: Optional[float] = None) -> str:
        c = concept if concept is not None else self.concept
        s = strength if strength is not None else self.wind_strength
        return f"WIND: Direction: {self.get_direction_from_concept(c)}, Speed: {s}"

    def __iter__(self) -> Generator:
        while True:
            x = {}
            batch_x_vec, batch_y = self.next_sample()
            x_vec = batch_x_vec[0]
            y = batch_y[0]
            for i, x_val in enumerate(x_vec):
                x[f"x_num_{i}"] = x_val

            yield x, y


def quantize_y(y: float) -> int:
    return int(min(y, 99))
