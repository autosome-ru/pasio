cimport numpy as np

cdef class LogComputer:
    cdef readonly int cache_size
    cdef readonly double shift
    cdef readonly np.ndarray precomputed
    cdef double[::1] precomputed_view

    cpdef compute_for_number(self, int x)
    cpdef compute_for_array(self, np.ndarray x, double max_value)
    cpdef compute_for_array_unbound(self, np.ndarray xs)

cdef class LogGammaComputer:
    cdef readonly int cache_size
    cdef readonly double shift
    cdef readonly np.ndarray precomputed
    cdef double[::1] precomputed_view

    cpdef compute_for_number(self, int x)
    cpdef compute_for_array(self, np.ndarray x, double max_value)
    cpdef compute_for_array_unbound(self, np.ndarray xs)
