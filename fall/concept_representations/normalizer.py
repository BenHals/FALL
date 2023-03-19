from .meta_feature_distributions import GaussianDistribution


class MetaFeatureNormalizer:
    """Collects and records the overall distribution of meta-features for normalization."""

    def __init__(self) -> None:
        self.meta_feature_distributions: list[GaussianDistribution] = []
        self.initialized = False

    def initialize(self, meta_features: list[float]) -> None:
        """Initialize with information specific to the observation schema.
        We do this when we first encounter an observation.
        """

        # We have all metafeatures calculated by the RollingTimeseries, for y, p, e and each feature.
        self.n_meta_featues = len(meta_features)
        self.meta_feature_distributions = [GaussianDistribution(memory_size=-1) for i in range(self.n_meta_featues)]
        self.initialized = True

    def learn_one(self, meta_features: list[float]) -> None:
        if not self.initialized:
            self.initialize(meta_features)
        for i, mf in enumerate(meta_features):
            self.meta_feature_distributions[i].learn_one(mf)

    def min_max_normalize(self, meta_features: list[float]) -> list[float]:
        transformed_meta_features = []
        for i, mf in enumerate(meta_features):
            transformed_meta_features.append(self.meta_feature_distributions[i].min_max_normalize(mf))

        return transformed_meta_features

    def standardize(self, meta_features: list[float]) -> list[float]:
        transformed_meta_features = []
        for i, mf in enumerate(meta_features):
            transformed_meta_features.append(self.meta_feature_distributions[i].standardize(mf))

        return transformed_meta_features
