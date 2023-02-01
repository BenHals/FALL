from collections import deque
from math import sqrt
from typing import Optional

import numpy as np

# Based on code from the 'rolling' python library at https://github.com/ajcr/rolling
# Adapted to one class here.


class RollingTimeseries:
    def __init__(self, window_size: int, ddof: int = 1) -> None:
        self.window_size = window_size
        self.ddof = ddof
        self.statistic_names = ["mean", "stdev", "skew", "kurtosis", "turning_point_rate", "FI"]
        self.timeseries: deque = deque()
        self.np_timeseries: np.ndarray = np.zeros(1)
        self.cache: dict = {}
        self._nobs: int = 0
        self._sum = 0.0
        self._v_mean = 0.0  # mean of values
        self._v_sslm = 0.0  # sum of squared values less the mean

        self._x1 = 0.0
        self._x2 = 0.0
        self._x3 = 0.0
        self._x4 = 0.0

    def get_np_timeseries(self) -> np.ndarray:
        if self.np_timeseries is not None:
            return self.np_timeseries
        self.np_timeseries = np.array(self.timeseries)
        return self.np_timeseries

    def get_mean(self) -> float:
        if self._nobs < 1:
            return 0
        if "mean" in self.cache:
            return self.cache["mean"]
        mean = self._sum / self._nobs
        self.cache["mean"] = mean
        return mean

    def get_variance(self) -> float:
        if self._nobs < 2:
            return 0
        if "var" in self.cache:
            return self.cache["var"]
        try:
            var = self._v_sslm / (self._nobs - self.ddof)
        except Exception as e:
            print(self._v_sslm)
            print(self._nobs)
            print(self.ddof)
            raise e
        self.cache["var"] = var
        return var

    def get_stdev(self) -> float:
        if self._nobs < 2:
            return 0
        if "stdev" in self.cache:
            return self.cache["stdev"]
        try:
            # Due to instability can sometimes be a very small negative
            var = max(self.get_variance(), 0)
            stdev = sqrt(var)
        except Exception as e:
            print(self._v_sslm)
            print(self._nobs)
            print(self.ddof)
            print(self._v_sslm / (self._nobs - self.ddof))
            raise e
        self.cache["stdev"] = stdev
        return stdev

    def get_skew(self) -> float:
        if self._nobs < 2:
            return 0
        if "skew" in self.cache:
            return self.cache["skew"]
        N = self._nobs

        # compute moments
        A = self._x1 / N
        B = self._x2 / N - A * A
        C = self._x3 / N - A * A * A - 3 * A * B

        if B <= 1e-14:
            return 0.0

        R = sqrt(B)

        # If correcting for bias
        # skew = (sqrt(N * (N - 0)) * C) / ((N - 1) * R * R * R)
        # Otherwise
        skew = C / (R * R * R)
        self.cache["skew"] = skew
        return skew

    def get_kurtosis(self) -> float:
        if self._nobs < 2:
            # -3 is the kurtosis for a normal distribution,
            # (with fisher sdjustment), so we use as default
            return -3
        if "kurtosis" in self.cache:
            return self.cache["kurtosis"]
        N = self._nobs

        # compute moments
        A = self._x1 / N
        R = A * A

        B = self._x2 / N - R
        R *= A

        C = self._x3 / N - R - 3 * A * B
        R *= A

        D = self._x4 / N - R - 6 * B * A * A - 4 * C * A

        if B <= 1e-14:
            return -3

        # If correcting for bias
        # K = (N * N - 1) * D / (B * B) - 3 * ((N - 1) ** 2)
        # kurtosis = K / ((N - 2) * (N - 3))
        # Otherwise
        K = D / (B * B)
        kurtosis = K
        fisher_style = True
        kurtosis = kurtosis - 3 if fisher_style else kurtosis
        self.cache["kurtosis"] = kurtosis
        return kurtosis

    def get_turning_point_rate(self) -> float:
        if self._nobs < 3:
            return 0
        if "turning_point_rate" in self.cache:
            return self.cache["turning_point_rate"]
        np_timeseries = self.get_np_timeseries()
        if np_timeseries.dtype == np.bool_:
            np_timeseries = np_timeseries.astype(np.int_)
        dx = np.diff(np_timeseries)
        turning_point_rate = np.sum(dx[1:] * dx[:-1] < 0)
        # turning_point_rate = len([*argrelmin(np_timeseries), *argrelmax(np_timeseries)])
        self.cache["turning_point_rate"] = turning_point_rate
        return turning_point_rate

    def update(self, new_val: float) -> None:
        last_np_timeseries = self.np_timeseries
        self.np_timeseries = np.zeros(1)
        self.cache = {}
        self.timeseries.append(new_val)

        self._nobs += 1
        self._sum += new_val

        # update parameters for variance
        delta = new_val - self._v_mean
        self._v_mean += delta / self._nobs
        self._v_sslm += delta * (new_val - self._v_mean)

        # update parameters for moments
        sq = new_val * new_val
        self._x1 += new_val
        self._x2 += sq
        self._x3 += sq * new_val
        self._x4 += sq**2

        if len(self.timeseries) > self.window_size:
            self._remove_old()
            if last_np_timeseries is not None:
                self.np_timeseries = last_np_timeseries
                self.np_timeseries[:-1] = self.np_timeseries[1:]
                self.np_timeseries[-1] = new_val

    def _remove_old(self) -> None:
        self.np_timeseries = np.zeros(1)
        self.cache = {}
        old = self.timeseries.popleft()
        self._nobs -= 1
        self._sum -= old

        # Update parameters for variance
        delta = old - self._v_mean
        self._v_mean -= delta / self._nobs
        self._v_sslm -= delta * (old - self._v_mean)

        # update parameters for moments
        sq = old * old
        self._x1 -= old
        self._x2 -= sq
        self._x3 -= sq * old
        self._x4 -= sq**2

    def get_stats(self, FI: Optional[float] = None) -> list[float]:
        """Calculates a set of statistics for current data."""
        stats = []
        timeseries = self.get_np_timeseries()
        with np.errstate(divide="ignore", invalid="ignore"):
            stats.append(self.get_mean())
            stats.append(self.get_stdev())
            stats.append(self.get_skew())
            stats.append(self.get_kurtosis())

            tp = int(self.get_turning_point_rate())
            if len(timeseries) > 0:
                tp_rate = tp / len(timeseries)
            else:
                tp_rate = 0
            stats.append(tp_rate)

            stats.append(FI if FI is not None else 0)

        return stats
