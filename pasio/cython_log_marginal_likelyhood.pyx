from __future__ import division
import numpy as np
cimport numpy as np
cimport cython
from .cython_cached_log import LogComputer, LogGammaComputer
from .cython_cached_log cimport LogComputer, LogGammaComputer

class ScorerFactory:
    def __init__(self, alpha, beta):
        assert alpha >= 0
        assert beta >= 0
        if isinstance(alpha, float) and alpha.is_integer():
            self.alpha = int(alpha)
        else:
            self.alpha = alpha
        self.beta = beta
        self.log_gamma_computer = LogGammaComputer()
        self.log_gamma_alpha_computer = LogGammaComputer(shift=alpha)
        self.log_computer = LogComputer(shift = beta)

    def __call__(self, counts, split_candidates):
        if isinstance(self.alpha, int):
            return LogMarginalLikelyhoodIntAlphaComputer(counts, self.alpha, self.beta, split_candidates,
                                                log_computer=self.log_computer,
                                                log_gamma_computer=self.log_gamma_computer,
                                                log_gamma_alpha_computer=self.log_gamma_alpha_computer)
        else:
            return LogMarginalLikelyhoodRealAlphaComputer(counts, self.alpha, self.beta, split_candidates,
                                                log_computer=self.log_computer,
                                                log_gamma_computer=self.log_gamma_computer,
                                                log_gamma_alpha_computer=self.log_gamma_alpha_computer)

def assert_correct_counts(counts):
    assert isinstance(counts, np.ndarray)
    assert counts.dtype == int
    assert np.all(counts >= 0)
    assert len(counts) > 0

def assert_correct_split_candidates(split_candidates, counts):
    assert isinstance(split_candidates, np.ndarray)
    assert split_candidates[0] == 0
    assert split_candidates[-1] == len(counts)
    assert np.all(split_candidates[1:] > split_candidates[:-1]) # strictly ascending

# Indexing of LogMarginalLikelyhoodComputer iterates over split candidates, not counts
cdef class LogMarginalLikelyhoodComputer:
    cdef object alpha
    cdef LogComputer log_computer
    cdef LogGammaComputer log_gamma_computer
    cdef LogGammaComputer log_gamma_alpha_computer
    cdef np.ndarray split_candidates
    cdef np.ndarray cumsum, logfac_cumsum
    cdef readonly double segment_creation_cost

    def __init__(self, counts, alpha, beta, split_candidates, log_computer=None, log_gamma_computer=None, log_gamma_alpha_computer=None):
        self.alpha = alpha

        self.log_computer = log_computer if log_computer else LogComputer(shift=beta)
        self.log_gamma_computer = log_gamma_computer if log_gamma_computer else LogGammaComputer()
        self.log_gamma_alpha_computer = log_gamma_alpha_computer if log_gamma_alpha_computer else LogGammaComputer(shift=alpha)

        assert_correct_counts(counts)
        assert_correct_split_candidates(split_candidates, counts)
        self.split_candidates = split_candidates

        self.cumsum = np.hstack([0, np.cumsum(counts)])[split_candidates]

        count_logfacs = self.log_gamma_computer.compute_for_array_unbound(counts + 1)
        self.logfac_cumsum = np.hstack([0, np.cumsum(count_logfacs)])[split_candidates]

        self.segment_creation_cost = alpha * self.log_computer.compute_for_number(0) - self.log_gamma_alpha_computer.compute_for_number(0)

    def total_sum_logfac(self):
        return self.logfac_cumsum[-1]

    def scores(self):
        segment_lengths = np.diff(self.split_candidates)
        segment_counts = np.diff(self.cumsum)
        shifted_segment_counts = segment_counts + self.alpha
        add = self.log_gamma_alpha_computer.compute_for_array_unbound(segment_counts)
        sub = shifted_segment_counts * self.log_computer.compute_for_array_unbound(segment_lengths)
        self_scores = add - sub
        return self_scores + self.segment_creation_cost

    def log_marginal_likelyhoods(self):
        segment_sum_logfacs = np.diff(self.logfac_cumsum)
        return self.scores() - segment_sum_logfacs

    def mean_counts(self):
        segment_lengths = np.diff(self.split_candidates)
        segment_counts = np.diff(self.cumsum)
        return segment_counts / segment_lengths

    def score(self, start, stop):
        return self.self_score(start, stop) + self.segment_creation_cost

    def self_score(self, start, stop):
        segment_count = self.cumsum[stop] - self.cumsum[start]
        shifted_segment_count = segment_count + self.alpha
        segment_length = self.split_candidates[stop] - self.split_candidates[start]
        add = self.log_gamma_alpha_computer.compute_for_number(segment_count)
        sub = shifted_segment_count * self.log_computer.compute_for_number(segment_length)
        return add - sub

    def self_score_no_splits(self):
        return self.self_score(0, len(self.split_candidates) - 1)
    def score_no_splits(self):
        return self.self_score_no_splits() + self.segment_creation_cost

    # marginal likelihoods for segments [i, stop) for all i < stop.
    # [i, stop) means that segment boundaries are ... i - 1][i ...... stop - 1][stop ...
    # These scores are not corrected for constant penalty for segment creation
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray all_suffixes_self_score(self, int stop):
        cdef np.ndarray result = np.empty(stop, dtype=float)
        self.all_suffixes_self_score_in_place(stop, result)
        return result

    cpdef void all_suffixes_self_score_in_place(self, int stop, double[::1] result_view):
        raise NotImplementedError()


cdef class LogMarginalLikelyhoodIntAlphaComputer(LogMarginalLikelyhoodComputer):
    cdef int int_alpha
    def __init__(self, counts, alpha, beta, split_candidates, log_computer=None, log_gamma_computer=None, log_gamma_alpha_computer=None):
        super(LogMarginalLikelyhoodIntAlphaComputer, self).__init__(counts, alpha, beta, split_candidates, log_computer, log_gamma_computer, log_gamma_alpha_computer)
        self.int_alpha = alpha

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void all_suffixes_self_score_in_place(self, int stop, double[::1] result_view):
        cdef int i
        cdef long[::1] cumsum_view = self.cumsum
        cdef long[::1] split_candidates_view = self.split_candidates

        cdef int cumsum_last = cumsum_view[stop]
        cdef int cumsum_last_shifted = self.int_alpha + cumsum_last
        cdef int split_candidates_last = split_candidates_view[stop]
        cdef int shifted_segment_count, segment_length
        cdef double add, sub

        cdef bint log_gamma_fully_cached = (self.int_alpha + cumsum_last < self.log_gamma_computer.cache_size)
        cdef bint log_fully_cached = (split_candidates_last < self.log_computer.cache_size)

        # direct memory access is much more efficient than even inlined method call
        cdef double[::1] log_gamma_view = self.log_gamma_computer.precomputed_view
        cdef double[::1] log_view = self.log_computer.precomputed_view

        if log_gamma_fully_cached and log_fully_cached:
            for i in range(stop):
                shifted_segment_count = cumsum_last_shifted - cumsum_view[i]
                segment_length = split_candidates_last - split_candidates_view[i]
                add = log_gamma_view[shifted_segment_count]
                sub = shifted_segment_count * log_view[segment_length]
                result_view[i] = add - sub
        elif log_gamma_fully_cached and not log_fully_cached:
            for i in range(stop):
                shifted_segment_count = cumsum_last_shifted - cumsum_view[i]
                segment_length = split_candidates_last - split_candidates_view[i]
                add = log_gamma_view[shifted_segment_count]
                sub = shifted_segment_count * self.log_computer.compute_for_number_cython(segment_length)
                result_view[i] = add - sub
        elif not log_gamma_fully_cached and log_fully_cached:
            for i in range(stop):
                shifted_segment_count = cumsum_last_shifted - cumsum_view[i]
                segment_length = split_candidates_last - split_candidates_view[i]
                add = self.log_gamma_computer.compute_for_number_cython(shifted_segment_count)
                sub = shifted_segment_count * log_view[segment_length]
                result_view[i] = add - sub
        elif not log_gamma_fully_cached and not log_fully_cached:
            for i in range(stop):
                shifted_segment_count = cumsum_last_shifted - cumsum_view[i]
                segment_length = split_candidates_last - split_candidates_view[i]
                add = self.log_gamma_computer.compute_for_number_cython(shifted_segment_count)
                sub = shifted_segment_count * self.log_computer.compute_for_number_cython(segment_length)
                result_view[i] = add - sub

cdef class LogMarginalLikelyhoodRealAlphaComputer(LogMarginalLikelyhoodComputer):
    cdef double real_alpha
    def __init__(self, counts, alpha, beta, split_candidates, log_computer=None, log_gamma_computer=None, log_gamma_alpha_computer=None):
        super(LogMarginalLikelyhoodRealAlphaComputer, self).__init__(counts, alpha, beta, split_candidates, log_computer, log_gamma_computer, log_gamma_alpha_computer)
        self.real_alpha = alpha

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void all_suffixes_self_score_in_place(self, int stop, double[::1] result_view):
        cdef int i
        cdef long[::1] cumsum_view = self.cumsum
        cdef long[::1] split_candidates_view = self.split_candidates

        cdef int cumsum_last = cumsum_view[stop]
        cdef double cumsum_last_shifted = self.real_alpha + cumsum_last
        cdef int split_candidates_last = split_candidates_view[stop]
        cdef int segment_length
        cdef int segment_count
        cdef double shifted_segment_count
        cdef double add, sub

        cdef bint log_gamma_alpha_fully_cached = (cumsum_last < self.log_gamma_alpha_computer.cache_size)
        cdef bint log_fully_cached = (split_candidates_last < self.log_computer.cache_size)

        # direct memory access is much more efficient than even inlined method call
        cdef double[::1] log_gamma_alpha_view = self.log_gamma_alpha_computer.precomputed_view
        cdef double[::1] log_view = self.log_computer.precomputed_view

        if log_gamma_alpha_fully_cached and log_fully_cached:
            for i in range(stop):
                segment_count = cumsum_last - cumsum_view[i]
                shifted_segment_count = cumsum_last_shifted - cumsum_view[i]
                segment_length = split_candidates_last - split_candidates_view[i]
                add = log_gamma_alpha_view[segment_count]
                sub = shifted_segment_count * log_view[segment_length]
                result_view[i] = add - sub
        elif log_gamma_alpha_fully_cached and not log_fully_cached:
            for i in range(stop):
                segment_count = cumsum_last - cumsum_view[i]
                shifted_segment_count = cumsum_last_shifted - cumsum_view[i]
                segment_length = split_candidates_last - split_candidates_view[i]
                add = log_gamma_alpha_view[segment_count]
                sub = shifted_segment_count * self.log_computer.compute_for_number_cython(segment_length)
                result_view[i] = add - sub
        elif not log_gamma_alpha_fully_cached and log_fully_cached:
            for i in range(stop):
                segment_count = cumsum_last - cumsum_view[i]
                shifted_segment_count = cumsum_last_shifted - cumsum_view[i]
                segment_length = split_candidates_last - split_candidates_view[i]
                add = self.log_gamma_alpha_computer.compute_for_number_cython(segment_count)
                sub = shifted_segment_count * log_view[segment_length]
                result_view[i] = add - sub
        elif not log_gamma_alpha_fully_cached and not log_fully_cached:
            for i in range(stop):
                segment_count = cumsum_last - cumsum_view[i]
                shifted_segment_count = cumsum_last_shifted - cumsum_view[i]
                segment_length = split_candidates_last - split_candidates_view[i]
                add = self.log_gamma_alpha_computer.compute_for_number_cython(segment_count)
                sub = shifted_segment_count * self.log_computer.compute_for_number_cython(segment_length)
                result_view[i] = add - sub
