# cython: boundscheck=False, wraparound=False, initializedcheck=False
import numpy as np
import scipy.special
cimport cython
cimport numpy as np
cimport libc.math
cimport scipy.special.cython_special as scipy_special_python

# Works only with non-negative integer values
cdef class LogComputer:
    def __init__(self, shift = 0, cache_size = 1048576):
        self.cache_size = cache_size
        self.shift = shift
        self.precomputed = np.log(np.arange(cache_size) + shift)
        self.precomputed_view = self.precomputed

    cpdef double compute_for_number(self, Py_ssize_t x):
        if x < self.cache_size:
            return self.precomputed_view[x]
        else:
            return libc.math.log(<double>x + self.shift)

    # uses fast algorithm if maximal value of x is specified and doesn't exceed cache size
    cpdef compute_for_array(self, np.ndarray xs, double max_value):
        if max_value < self.cache_size:
            return self.precomputed[xs]
        else:
            return self.compute_for_array_unbound(xs)

    cpdef object compute_for_array_unbound(self, np.ndarray xs):
        cdef Py_ssize_t i
        cdef Py_ssize_t val
        cdef double[::1] result_view
        cdef Py_ssize_t[::1] xs_view = xs
        len_x = len(xs_view)
        result = np.empty(len_x, dtype=float)
        result_view = result
        for i in range(len_x):
            val = xs_view[i]
            if val < self.cache_size:
                result_view[i] = self.precomputed_view[val]
            else:
                result_view[i] = np.log(<double>val + self.shift)
        return result

# Works only with non-negative integer values
cdef class LogGammaComputer:
    def __init__(self, shift = 0, cache_size = 1048576):
        self.cache_size = cache_size
        self.shift = shift
        self.precomputed = scipy.special.gammaln(np.arange(self.cache_size) + shift)
        self.precomputed_view = self.precomputed

    cpdef double compute_for_number(self, Py_ssize_t x):
        if x < self.cache_size:
            return self.precomputed_view[x]
        else:
            return scipy_special_python.gammaln(<double>x + self.shift)

    # uses fast algorithm if maximal value of x is specified and doesn't exceed cache size
    cpdef compute_for_array(self, np.ndarray x, double max_value):
        if max_value < self.cache_size:
            return self.precomputed[x]
        else:
            return self.compute_for_array_unbound(x)

    cpdef object compute_for_array_unbound(self, np.ndarray xs):
        cdef Py_ssize_t i
        cdef Py_ssize_t val
        cdef double[::1] result_view
        cdef Py_ssize_t[::1] xs_view = xs
        len_x = len(xs_view)
        result = np.empty(len_x, dtype=float)
        result_view = result
        for i in range(len_x):
            val = xs_view[i]
            if val < self.cache_size:
                result_view[i] = self.precomputed_view[val]
            else:
                result_view[i] = scipy_special_python.gammaln(<double>val + self.shift)
        return result
