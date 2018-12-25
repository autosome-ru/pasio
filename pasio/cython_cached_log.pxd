cimport numpy as np

cdef class LogComputer:
    cdef readonly int cache_size
    cdef readonly double shift
    cdef np.ndarray precomputed
    cdef double[::1] precomputed_view

    cpdef double compute_for_number(self, int x)
    cdef  inline double compute_for_number_cython(self, int x) nogil
    cdef  inline double compute_for_number_cached(self, int x) nogil
    cpdef compute_for_array(self, np.ndarray x, double max_value)
    cpdef compute_for_array_unbound(self, np.ndarray xs)

cdef class LogGammaComputer:
    cdef readonly int cache_size
    cdef readonly double shift
    cdef np.ndarray precomputed
    cdef double[::1] precomputed_view

    cpdef double compute_for_number(self, int x)
    cdef  inline double compute_for_number_cython(self, int x) nogil
    cdef  inline double compute_for_number_cached(self, int x) nogil
    cpdef compute_for_array(self, np.ndarray x, double max_value)
    cpdef compute_for_array_unbound(self, np.ndarray xs)
