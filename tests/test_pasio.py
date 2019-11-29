import pytest
from pasio.log_marginal_likelyhood import LogMarginalLikelyhoodIntAlphaComputer
from pasio.cached_log import LogGammaComputer
from pasio.splitters import *
from pasio.dto.sliding_window import SlidingWindow
from pasio.process_bedgraph import parse_bedgraph
import numpy as np
import math
import functools
import operator

def test_stat_split_into_segments_square():
    def split_on_two_segments_or_not(counts, scorer_factory):
        scorer = scorer_factory(counts, np.arange(len(counts) + 1))
        best_score = scorer.score(0, len(counts))
        split_point = 0
        for i in range(len(counts)):
            current_score = scorer.score(0, i) + scorer.score(i, len(counts))
            if current_score > best_score:
                split_point = i
                best_score = current_score
        return best_score, split_point

    np.random.seed(4)
    scorer_factory = lambda counts, split_candidates: LogMarginalLikelyhoodIntAlphaComputer(counts, 1, 1, split_candidates)
    for repeat in range(5):
        counts = np.concatenate([np.random.poisson(15, 100),
                                 np.random.poisson(20, 100)])

        optimal_score, optimal_split = SquareSplitter(scorer_factory).split(counts, np.arange(len(counts) + 1))

        two_split_score, two_split_point = split_on_two_segments_or_not(counts, scorer_factory)

        assert optimal_score >= two_split_score
        assert two_split_point in optimal_split

        scorer = scorer_factory(counts, optimal_split)
        split_score = np.sum(scorer.scores())
        assert np.allclose(optimal_score, split_score)
        if (two_split_point is None):
            assert np.array_equal(optimal_split, [0,200])
        else:
            assert abs(two_split_point - 100) < 10

def test_log_marginal_likelyhood_exact():
    def exact_function(counts, alpha, beta):
        counts_facproduct = functools.reduce(operator.mul, map(np.math.factorial, counts), 1)
        cs = sum(counts)
        ns = len(counts)
        return np.log(
            (beta**alpha) * math.gamma(cs+alpha) / (
                math.gamma(alpha) * counts_facproduct * ((ns+beta)**(cs+alpha))
            )
        )
    scorer = LogMarginalLikelyhoodIntAlphaComputer(np.array([0]), 3, 5, split_candidates=np.array([0,1]))
    assert np.allclose(scorer.log_marginal_likelyhoods(), exact_function(np.array([0]), 3, 5))

    scorer = LogMarginalLikelyhoodIntAlphaComputer(np.array([0, 1]), 3, 5, split_candidates=np.array([0,2]))
    assert np.allclose(scorer.log_marginal_likelyhoods(), exact_function(np.array([0, 1]), 3, 5))

    scorer = LogMarginalLikelyhoodIntAlphaComputer(np.array([4, 0, 1, 3]), 5, 2, split_candidates=np.array([0,4]))
    assert np.allclose(scorer.log_marginal_likelyhoods(), exact_function(np.array([4, 0, 1, 3]), 5, 2))

    scorer = LogMarginalLikelyhoodIntAlphaComputer(np.array([4, 0, 1, 3]), 1, 1, split_candidates=np.array([0,4]))
    assert np.allclose(scorer.log_marginal_likelyhoods(), exact_function(np.array([4, 0, 1, 3]), 1, 1))


class SimpleScorer:
    def __init__(self, sequence, split_candidates):
        self.sequence = sequence
        self.split_candidates = split_candidates
        self.segment_creation_cost = 0

    def score(self, start, stop):
        return self.self_score(start, stop)

    def self_score(self, start, stop):
        start = self.split_candidates[start]
        stop = self.split_candidates[stop]
        if len(set(self.sequence[start:stop])) == 1:
            return (stop-start)**2
        return stop-start

    def all_suffixes_self_score(self, stop):
        return np.array([self.self_score(i, stop) for i in range(stop)], dtype='float64')

simple_scorer_factory = lambda counts, split_candidates: SimpleScorer(counts, split_candidates)

def test_split_into_segments_square():
    sequence = 'A'
    optimal_score, optimal_split = SquareSplitter(simple_scorer_factory).split(sequence, np.arange(len(sequence) + 1))
    assert np.array_equal(optimal_split, [0,1])
    assert optimal_score == 1

    sequence = 'AAA'
    optimal_score, optimal_split = SquareSplitter(simple_scorer_factory).split(sequence, np.arange(len(sequence) + 1))
    assert np.array_equal(optimal_split, [0,3])
    assert optimal_score == 9

    sequence = 'AAABBB'
    optimal_score, optimal_split = SquareSplitter(simple_scorer_factory).split(sequence, np.arange(len(sequence) + 1))
    assert np.array_equal(optimal_split, [0,3,6])
    assert optimal_score == 9+9

    sequence = 'AAABBBC'
    optimal_score, optimal_split = SquareSplitter(simple_scorer_factory).split(sequence, np.arange(len(sequence) + 1))
    assert np.array_equal(optimal_split, [0,3,6,7])
    assert optimal_score == 9+9+1

    sequence = 'ABBBC'
    optimal_score, optimal_split = SquareSplitter(simple_scorer_factory).split(sequence, np.arange(len(sequence) + 1))
    assert np.array_equal(optimal_split, [0,1,4,5])
    assert optimal_score == 1+9+1


def test_split_into_segments_candidates():

    sequence = 'AAABBB'
    optimal_score, optimal_split = SquareSplitter(simple_scorer_factory).split(sequence, np.array([0,1,2,3,5,6]))
    assert np.array_equal(optimal_split, [0,3,6])
    assert optimal_score == 9+9

    sequence = 'AAABBB'
    optimal_score, optimal_split = SquareSplitter(simple_scorer_factory).split(sequence, np.array([0,3,5,6]))
    assert np.array_equal(optimal_split, [0,3,6])
    assert optimal_score == 9+9

    sequence = 'AAABBBC'
    optimal_score, optimal_split = SquareSplitter(simple_scorer_factory).split(sequence, np.array([0,3,7]))
    assert np.array_equal(optimal_split, [0,3,7])
    assert optimal_score == 9+4

    sequence = 'AAABBBC'
    optimal_score, optimal_split = SquareSplitter(simple_scorer_factory).split(sequence, np.array([0,3,7]))
    assert np.array_equal(optimal_split, [0,3,7])
    assert optimal_score == 9+4

    sequence = 'AAAAAA'
    optimal_score, optimal_split = SquareSplitter(simple_scorer_factory).split(sequence, np.array([0,3,6]))
    assert np.array_equal(optimal_split, [0,6])
    assert optimal_score == 36

def test_split_with_split_num_regularization():
    # score of split 'AAA|B|AA' = 9+1+4 = 14
    # with regularization = 9+1+4 - 3*2 = 8
    # alternative split: 'AAA|BAA' gives score = 9+3-3*1 = 9
    sequence = 'AAABAA'
    splitter = SquareSplitter(SimpleScorer,
                                    split_number_regularization_multiplier = 3,
                                    split_number_regularization_function = lambda x:x)
    optimal_score, optimal_split = splitter.split(sequence, np.arange(len(sequence) + 1))
    assert np.array_equal(optimal_split, [0,3,6])
    assert optimal_score == 9

def test_split_with_length_regularization():
    # score of split 'AAA|B|AA' = 9+1+4 = 14
    # with regularization = 9+1+4 - 1.5*(1/log(3+1)+1/log(1+1)+1/log(2+1)) = 9.38
    # alternative split: 'AAA|BAA' gives score = 9+3 - 1.5*(1/log(3+1)+1/log(3+1)) = 9.83
    sequence = 'AAABAA'
    splitter = SquareSplitter(SimpleScorer,
                                    length_regularization_multiplier = 1.5,
                                    length_regularization_function = lambda x:1/np.log(1+x))
    optimal_score, optimal_split = splitter.split(sequence, np.arange(len(sequence) + 1))

    assert np.array_equal(optimal_split, [0, 3, 6])
    assert optimal_score == 9+3 - 1.5*(1/np.log(3+1)+1/np.log(3+1))

    # limiting possible splits
    splitter = SquareSplitter(SimpleScorer,
                                    length_regularization_multiplier = 1.5,
                                    length_regularization_function = lambda x:1/np.log(1+x))
    optimal_score, optimal_split = splitter.split(sequence, np.array([0,4,5,6]))

    assert np.array_equal(optimal_split, [0, 4, 6])
    assert optimal_score == 4+4 - 1.5*(1/np.log(4+1)+1/np.log(2+1))


def test_approximate_log_gamma():
    log_gamma_computer = LogGammaComputer()
    tol = 1e-8
    assert np.abs(np.log(np.arange(1, 256+1)).sum() - log_gamma_computer.compute_for_number(256+1)) < tol
    assert np.abs(np.log(np.arange(1, 4095+1)).sum() - log_gamma_computer.compute_for_number(4095+1)) < tol
    assert np.allclose(np.log(np.arange(1, 4096+1)).sum(),
                       log_gamma_computer.compute_for_number(4096+1), atol=tol)
    assert np.abs(np.log(np.arange(1, 4097+1)).sum() - log_gamma_computer.compute_for_number(4097+1)) < tol
    assert np.abs(np.log(np.arange(1, 10000+1)).sum() - log_gamma_computer.compute_for_number(10000+1)) < tol

    array_to_count = np.array([0,1,20,1024,10000])
    logfac_array = np.array([np.log(np.arange(1, x+1)).sum() for x in array_to_count])
    assert np.allclose(log_gamma_computer.compute_for_array_unbound(array_to_count+1), logfac_array, atol=tol)


def test_suffixes_scores():
    np.random.seed(2)
    counts = np.concatenate([np.random.poisson(15, 100),
                             np.random.poisson(20, 100)])

    scorer = LogMarginalLikelyhoodIntAlphaComputer(counts, 1, 1, np.arange(len(counts) + 1))
    suffixes_scores = [scorer.self_score(i, 150) for i in range(150)]
    assert np.allclose(scorer.all_suffixes_self_score(150), np.array(suffixes_scores))

    counts = np.array([0,0,1,0,0,2,2,2,10,11,100,1,0,0,1,0], dtype='int64')
    scorer = LogMarginalLikelyhoodIntAlphaComputer(counts, 1, 1, np.arange(len(counts) + 1))
    suffixes_scores = [scorer.self_score(i, len(counts)-1) for i in range(len(counts)-1)]
    assert np.allclose(scorer.all_suffixes_self_score(len(counts)-1), np.array(suffixes_scores))

def test_suffixes_scores_with_candidates():
    np.random.seed(2)
    counts = np.arange(1,10)
    scorer = LogMarginalLikelyhoodIntAlphaComputer(counts, 1, 1, np.arange(len(counts) + 1))
    candidates = np.array([0,1,3,4,5,6,7,8,9])
    scorer_with_candidates = LogMarginalLikelyhoodIntAlphaComputer(counts, 1, 1, candidates)
    candidate_suffixes = scorer.all_suffixes_self_score(9)[candidates[:-1]]
    suffixes_just_candidates = scorer_with_candidates.all_suffixes_self_score(8)
    assert np.allclose(candidate_suffixes, suffixes_just_candidates)

    counts = np.concatenate([np.random.poisson(15, 100),
                             np.random.poisson(20, 100)])
    scorer = LogMarginalLikelyhoodIntAlphaComputer(counts, 1, 1, np.arange(len(counts) + 1))
    candidates = np.array([0,1,10,20,21,30,40, 149,200])
    scorer_with_candidates = LogMarginalLikelyhoodIntAlphaComputer(counts, 1, 1, candidates)
    candidate_suffixes = scorer.all_suffixes_self_score(200)[candidates[:-1]]
    suffixes_just_candidates = scorer_with_candidates.all_suffixes_self_score(len(candidates)-1)
    assert np.allclose(candidate_suffixes, suffixes_just_candidates)

def compute_log_marginal_likelyhood2(scorer, length):
    scorer.score(0, length)


def test_collect_split_points():
    assert SquareSplitter.collect_split_points([0,0,0]) == [0, 2]
    assert SquareSplitter.collect_split_points([0,0,1,2,3,4]) == [0,1,2,3,4,5]
    assert SquareSplitter.collect_split_points([0,0,0,0,0,2,3,4]) == [0,4,7]
    assert SquareSplitter.collect_split_points([0,0,0,2,1,4]) == [0,1,4,5]
    assert SquareSplitter.collect_split_points([0,0,0,2,1,3]) == [0,2,3,5]

def test_bedgraph_reader(tmpdir):
    bedgraph_file = tmpdir.mkdir("sub").join("test.bedgraph")
    bedgraph_file.write(
        '''chr1 0 10 0
        chr1 10 22 21
        chr1 22 23 30
        chr1 23 50 0
        chr2 0 15 0
        chr2 15 50 2
        chr2 50 60 0
        ''')
    chromosomes = {k:v for (k,v,l) in parse_bedgraph(str(bedgraph_file))}
    print(list(parse_bedgraph(str(bedgraph_file))))
    assert len(chromosomes) == 2
    assert len(chromosomes['chr1']) == 50
    assert np.all(chromosomes['chr1'][0:10] == 0)
    assert np.all(chromosomes['chr1'][10:22] == 21)
    assert chromosomes['chr1'][22] == 30
    assert np.all(chromosomes['chr1'][23:50] == 0)

    assert len(chromosomes['chr2']) == 60
    assert np.all(chromosomes['chr2'][0:15] == 0)
    assert np.all(chromosomes['chr2'][15:50] == 2)
    assert np.all(chromosomes['chr2'][50:60] == 0)


def test_split_into_segments_slidingwindow():
    A = 'AAAAAAAAAAAAAAAA'
    B = 'BBBBBBBBBBBBBBBBB'
    sequence = A + B
    sliding_window = SlidingWindow(window_size=10, window_shift=5)

    base_splitter = SquareSplitter(simple_scorer_factory)
    sliding_window_reducer = SlidingWindowReducer(sliding_window, base_reducer=base_splitter)
    splitter = ReducerCombiner(sliding_window_reducer, base_splitter)
    score, splits = splitter.split(sequence, np.arange(len(sequence) + 1))
    assert np.array_equal(splits, [0, len(A), len(sequence)])
    assert score == len(A)**2+len(B)**2

    base_splitter = SquareSplitter(simple_scorer_factory, split_number_regularization_multiplier=2)
    sliding_window_reducer = SlidingWindowReducer(sliding_window, base_reducer=base_splitter)
    splitter = ReducerCombiner(sliding_window_reducer, base_splitter)
    score, splits = splitter.split(sequence, np.arange(len(sequence) + 1))
    assert np.array_equal(splits, [0, len(A), len(sequence)])
    assert score == len(A)**2 + len(B)**2 - 2


class SimpleGreedyScorer:
    def __init__(self, sequence, split_candidates):
        self.sequence = sequence
        self.split_candidates = split_candidates
        self.segment_creation_cost = 0

    def score(self, start, stop):
        return self.self_score(start, stop)

    def self_score(self, start, stop):
        start = self.split_candidates[start]
        stop = self.split_candidates[stop]
        return (stop-start)**0.5

    def all_suffixes_self_score(self, stop):
        return np.array([self.self_score(i, stop) for i in range(stop)], dtype='float64')

simple_greedy_scorer_factory = lambda counts, split_candidates: SimpleGreedyScorer(counts, split_candidates)


def test_not_constant_splitter():
    sequence = np.array([1,1,1,2,2,2,2])
    splitter = ReducerCombiner(NotZeroReducer(), SquareSplitter(simple_scorer_factory))
    score, splits = splitter.split(sequence, np.arange(len(sequence) + 1))
    assert np.array_equal(splits, [0, 3, 7])

    splitter = ReducerCombiner(NotZeroReducer(),SquareSplitter(simple_greedy_scorer_factory))
    score, splits = splitter.split(sequence, np.arange(len(sequence) + 1))
    assert np.array_equal(splits, list(range(len(sequence) + 1)))

    splitter = ReducerCombiner(NotConstantReducer(), SquareSplitter(simple_greedy_scorer_factory))
    score, splits = splitter.split(sequence, np.arange(len(sequence) + 1))
    assert np.array_equal(splits, [0, 3, 7])

    splitter = ReducerCombiner(NotConstantReducer(), SquareSplitter(simple_greedy_scorer_factory))
    score, splits = splitter.split(sequence, np.array([0,1,2,3,4,5,7]))
    assert np.array_equal(splits, [0, 3, 7])

    assert np.array_equal(NotConstantReducer().reduce_candidate_list(sequence, np.array([0, 3, 7])), [0, 3, 7])
    assert np.array_equal(NotConstantReducer().reduce_candidate_list(sequence, np.arange(8)), [0, 3, 7])
    assert np.array_equal(NotConstantReducer().reduce_candidate_list(sequence, np.array([0, 3, 5, 7])), [0, 3, 7])
    assert np.array_equal(NotConstantReducer().reduce_candidate_list(sequence, np.array([0, 5, 7])), [0, 7])
