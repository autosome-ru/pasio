import numpy as np
import scipy.special

# Works only with non-negative integer values
class LogComputer:
    def __init__(self, shift = 0, cache_size = 1048576):
        self.cache_size = cache_size
        self.shift = shift
        self.precomputed = np.log(np.arange(self.cache_size) + shift)

    def compute_for_number(self, x):
        if x < self.cache_size:
            return self.precomputed[x]
        else:
            return np.log(x + self.shift)

    # uses fast algorithm if maximal value of x is specified and doesn't exceed cache size
    def compute_for_array(self, x, max_value):
        if max_value < self.cache_size:
            return self.precomputed[x]
        else:
            return self.compute_for_array_unbound(x)

    def compute_for_array_unbound(self, x):
        result = np.zeros(x.shape)
        is_small = x < self.cache_size
        result[is_small] = self.precomputed[x[is_small]]
        result[~is_small] = np.log(x[~is_small] + self.shift)
        return result

# Works only with non-negative integer values
class LogGammaComputer:
    def __init__(self, shift = 0, cache_size = 1048576):
        self.cache_size = cache_size
        self.shift = shift
        self.precomputed = scipy.special.gammaln(np.arange(self.cache_size) + shift)

    def compute_for_number(self, x):
        if x < self.cache_size:
            return self.precomputed[x]
        else:
            return scipy.special.gammaln(x + self.shift)

    # uses fast algorithm if maximal value of x is specified and doesn't exceed cache size
    def compute_for_array(self, x, max_value):
        if max_value < self.cache_size:
            return self.precomputed[x]
        else:
            return self.compute_for_array_unbound(x)

    def compute_for_array_unbound(self, x):
        result = np.zeros(x.shape)
        is_small = x < self.cache_size
        result[is_small] = self.precomputed[x[is_small]]
        result[~is_small] = scipy.special.gammaln(x[~is_small] + self.shift)
        return result
