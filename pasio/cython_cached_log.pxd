# cython: boundscheck=False, wraparound=False, initializedcheck=False
cimport numpy as np
cimport cython
cimport libc.math
cimport scipy.special.cython_special as scipy_special_python

cdef class LogComputer:
    cdef readonly int cache_size
    cdef readonly double shift
    cdef np.ndarray precomputed
    cdef double[::1] precomputed_view

    # This method is the same as `compute_for_number` but doesn't add overhead of python wrapper
    cdef inline double compute_for_number_cython(self, int x) nogil:
        if x < self.cache_size:
            return self.precomputed_view[x]
        else:
            return libc.math.log(<double>x + self.shift)

    # Note that it's faster to access view directly without method call (even though it's inlined)
    cdef inline double compute_for_number_cached(self, int x) nogil:
        return self.precomputed_view[x]

    cpdef double compute_for_number(self, int x)
    cpdef compute_for_array(self, np.ndarray x, double max_value)
    cpdef compute_for_array_unbound(self, np.ndarray xs)

cdef class LogGammaComputer:
    cdef readonly int cache_size
    cdef readonly double shift
    cdef np.ndarray precomputed
    cdef double[::1] precomputed_view

    cdef inline double compute_for_number_cython(self, int x) nogil:
        if x < self.cache_size:
            return self.precomputed_view[x]
        else:
            return scipy_special_python.gammaln(<double>x + self.shift)

    cdef inline double compute_for_number_cached(self, int x) nogil:
        return self.precomputed_view[x]

    cpdef double compute_for_number(self, int x)
    cpdef compute_for_array(self, np.ndarray x, double max_value)
    cpdef compute_for_array_unbound(self, np.ndarray xs)
