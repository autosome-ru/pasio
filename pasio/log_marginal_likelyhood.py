from __future__ import division
import numpy as np
from .cached_log import LogComputer, LogGammaComputer

class ScorerFactory:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.log_gamma_computer = LogGammaComputer()
        self.log_computer = LogComputer()

    def __call__(self, counts, split_candidates):
        return LogMarginalLikelyhoodComputer(counts, self.alpha, self.beta, split_candidates,
                                            log_computer=self.log_computer,
                                            log_gamma_computer=self.log_gamma_computer)

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
class LogMarginalLikelyhoodComputer:
    def __init__(self, counts, alpha, beta, split_candidates, log_computer=None, log_gamma_computer=None):
        assert isinstance(alpha, int)
        assert isinstance(beta, int)
        assert alpha >= 0
        assert beta >= 0
        self.alpha = alpha
        self.beta = beta

        self.log_computer = log_computer if log_computer else LogComputer()
        self.log_gamma_computer = log_gamma_computer if log_gamma_computer else LogGammaComputer()

        assert_correct_counts(counts)
        self.counts = counts

        assert_correct_split_candidates(split_candidates, counts)
        self.split_candidates = split_candidates

        self.cumsum = np.hstack([0, np.cumsum(counts)])[split_candidates]

        count_logfacs = self.log_gamma_computer.compute_for_array_unbound(counts + 1)
        self.logfac_cumsum = np.hstack([0, np.cumsum(count_logfacs)])[split_candidates]

        self.segment_creation_cost = alpha * self.log_computer.compute_for_number(beta) - self.log_gamma_computer.compute_for_number(alpha)

    def total_sum_logfac(self):
        return self.logfac_cumsum[-1]

    def scores(self):
        segment_lengths = np.diff(self.split_candidates)
        segment_counts = np.diff(self.cumsum)
        shifted_segment_counts = segment_counts + self.alpha
        shifted_segment_lengths = segment_lengths + self.beta
        add = self.log_gamma_computer.compute_for_array_unbound(shifted_segment_counts)
        sub = shifted_segment_counts * self.log_computer.compute_for_array_unbound(shifted_segment_lengths)
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
        shifted_segment_length = segment_length + self.beta
        add = self.log_gamma_computer.compute_for_number(shifted_segment_count)
        sub = shifted_segment_count * self.log_computer.compute_for_number(shifted_segment_length)
        return add - sub

    def self_score_no_splits(self):
        return self.self_score(0, len(self.split_candidates) - 1)
    def score_no_splits(self):
        return self.self_score_no_splits() + self.segment_creation_cost

    # marginal likelihoods for segments [i, stop) for all i < stop.
    # [i, stop) means that segment boundaries are ... i - 1][i ...... stop - 1][stop ...
    # These scores are not corrected for constant penalty for segment creation
    def all_suffixes_self_score(self, stop):
        # segment_count + alpha
        shifted_segment_count_vec = (self.alpha + self.cumsum[stop]) - self.cumsum[0:stop]
        # it's more efficient to add up numbers, then add result to vector
        #   (alternative is to add numbers to a vector one-by-one)

        # segment_length + beta
        shifted_segment_length_vec = (self.beta + self.split_candidates[stop]) - self.split_candidates[:stop]

        add_vec = self.log_gamma_computer.compute_for_array(shifted_segment_count_vec, max_value=(self.alpha + self.cumsum[stop]))
        sub_vec = shifted_segment_count_vec * self.log_computer.compute_for_array(shifted_segment_length_vec, max_value=(self.beta + self.split_candidates[stop]))

        return add_vec - sub_vec
