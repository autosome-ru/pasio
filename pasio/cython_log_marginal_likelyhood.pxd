cimport numpy as np
from .cython_cached_log cimport LogComputer, LogGammaComputer

cdef class LogMarginalLikelyhoodComputer:
    cdef object alpha
    cdef LogComputer log_computer
    cdef LogGammaComputer log_gamma_computer
    cdef LogGammaComputer log_gamma_alpha_computer
    cdef np.ndarray split_candidates
    cdef np.ndarray cumsum, logfac_cumsum
    cdef readonly double segment_creation_cost
    cpdef np.ndarray all_suffixes_self_score(self, int stop)

    # Empty implementation to be redefined in subclasses
    cpdef void all_suffixes_self_score_in_place(self, int stop, double[::1] result_view)

cdef class LogMarginalLikelyhoodIntAlphaComputer(LogMarginalLikelyhoodComputer):
    cdef int int_alpha
    cpdef void all_suffixes_self_score_in_place(self, int stop, double[::1] result_view)

cdef class LogMarginalLikelyhoodRealAlphaComputer(LogMarginalLikelyhoodComputer):
    cdef double real_alpha
    cpdef void all_suffixes_self_score_in_place(self, int stop, double[::1] result_view)
