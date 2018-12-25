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

    # Note that it's faster to access view directly without method call (even though it's inlined)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline double compute_for_number_cached(self, int x) nogil:
        return self.precomputed_view[x]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double compute_for_number(self, int x):
        if x < self.cache_size:
            return self.precomputed_view[x]
        else:
            return libc.math.log(<double>x + self.shift)

    # This method is the same as compute_for_number but doesn't add overhead of python wrapper
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline double compute_for_number_cython(self, int x) nogil:
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
        cdef int i
        cdef int val
        cdef double[::1] result_view
        cdef long[::1] xs_view = xs
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

    # Note that it's faster to access view directly without method call (even though it's inlined)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline double compute_for_number_cached(self, int x) nogil:
        return self.precomputed_view[x]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double compute_for_number(self, int x):
        if x < self.cache_size:
            return self.precomputed_view[x]
        else:
            return scipy_special_python.gammaln(<double>x + self.shift)

    # This method is the same as compute_for_number but doesn't add overhead of python wrapper
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline double compute_for_number_cython(self, int x) nogil:
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
        cdef int i
        cdef int val
        cdef double[::1] result_view
        cdef long[::1] xs_view = xs
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
