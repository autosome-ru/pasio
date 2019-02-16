cimport numpy as np
from .cython_cached_log cimport LogComputer, LogGammaComputer

cdef class BasicLogMarginalLikelyhoodComputer:
    cdef public double segment_creation_cost # it's public to make it easy to create mocks
    cpdef np.ndarray all_suffixes_self_score(self, Py_ssize_t stop)
    # Empty implementation to be redefined in subclasses
    cpdef void all_suffixes_self_score_in_place(self, Py_ssize_t stop, double[::1] result_view)

cdef class LogMarginalLikelyhoodComputer(BasicLogMarginalLikelyhoodComputer):
    cdef object alpha
    cdef LogComputer log_computer
    cdef LogGammaComputer log_gamma_computer
    cdef LogGammaComputer log_gamma_alpha_computer
    cdef np.ndarray split_candidates
    cdef np.ndarray cumsum, logfac_cumsum
    cdef long[::1] split_candidates_view
    cdef long long[::1] cumsum_view

cdef class LogMarginalLikelyhoodIntAlphaComputer(LogMarginalLikelyhoodComputer):
    cdef unsigned int int_alpha
    cpdef void all_suffixes_self_score_in_place(self, Py_ssize_t stop, double[::1] result_view)

cdef class LogMarginalLikelyhoodRealAlphaComputer(LogMarginalLikelyhoodComputer):
    cdef double real_alpha
    cpdef void all_suffixes_self_score_in_place(self, Py_ssize_t stop, double[::1] result_view)
