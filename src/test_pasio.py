import pytest
import pasio
import numpy as np
import random
import tempfile
import math


def test_stat_split_into_segments_square():
    np.random.seed(4)
    scorer_factory = lambda counts, split_candidates=None: pasio.LogMarginalLikelyhoodComputer(
        counts, 1, 1, split_candidates)
    for repeat in range(5):
        counts = np.concatenate([np.random.poisson(15, 100),
                                 np.random.poisson(20, 100)])

        optimal_split = pasio.split_into_segments_square(counts, scorer_factory)

        two_split = pasio.split_on_two_segments_or_not(counts, scorer_factory)

        assert optimal_split[0] >= two_split[0]
        assert two_split[1] in optimal_split[1]
        assert optimal_split[0] == pasio.compute_score_from_splits(
            counts, optimal_split[1], scorer_factory)
        if (two_split[1] is None):
            assert optimal_split[1] == [0,200]
        else:
            assert abs(two_split[1]-100) < 10

def test_log_marginal_likelyhood_exact():
    def exact_function(counts, alpha, beta):
        counts_facproduct = np.prod(np.array(map(np.math.factorial, counts)))
        return np.log(
            (beta**alpha*math.gamma(sum(counts)+alpha)) / (
                math.gamma(alpha)*counts_facproduct*((len(counts)+beta)**(sum(counts)+alpha))
                )
        )
    scorer = pasio.LogMarginalLikelyhoodComputer(np.array([0]), 3, 5, None)
    assert np.allclose(scorer(), exact_function(np.array([0]), 3, 5))

    scorer = pasio.LogMarginalLikelyhoodComputer(np.array([0, 1]), 3, 5, None)
    assert np.allclose(scorer(), exact_function(np.array([0, 1]), 3, 5))

    scorer = pasio.LogMarginalLikelyhoodComputer(np.array([4, 0, 1, 3]), 5, 2, None)
    assert np.allclose(scorer(), exact_function(np.array([4, 0, 1, 3]), 5, 2))

    scorer = pasio.LogMarginalLikelyhoodComputer(np.array([4, 0, 1, 3]), 1, 1, None)
    assert np.allclose(scorer(), exact_function(np.array([4, 0, 1, 3]), 1, 1))


class SimpleScorer:
    def __init__(self, sequence, split_candidates=None):
        self.sequence = sequence
        if split_candidates is None:
            split_candidates = range(len(self.sequence)+1)
        self.split_candidates = split_candidates
    def __call__(self, start=0, stop=None):
        start = self.split_candidates[start]
        if stop is None:
            stop = self.split_candidates[-1]
        else:
            stop = self.split_candidates[stop]
        if len(set(self.sequence[start:stop])) == 1:
            return (stop-start)**2
        return stop-start
    def all_suffixes_score(self, stop):
        return np.array([self(i, stop) for i in range(stop)])

simple_scorer_factory = lambda counts, split_candidates=None: SimpleScorer(counts, split_candidates)

def test_split_into_segments_square():
    sequence = 'A'
    optimal_split = pasio.split_into_segments_square(sequence,
                                                     simple_scorer_factory)
    assert optimal_split[1] == [0]
    assert optimal_split[0] == 1

    sequence = 'AAA'
    optimal_split = pasio.split_into_segments_square(sequence,
                                                     simple_scorer_factory)
    assert optimal_split[1] == [0]
    assert optimal_split[0] == 9

    sequence = 'AAABBB'
    optimal_split = pasio.split_into_segments_square(sequence,
                                                     simple_scorer_factory)
    assert optimal_split[1] == [0,3]
    assert optimal_split[0] == 9+9

    sequence = 'AAABBBC'
    optimal_split = pasio.split_into_segments_square(sequence,
                                                     simple_scorer_factory)
    assert optimal_split[1] == [0,3,6]
    assert optimal_split[0] == 9+9+1

    sequence = 'ABBBC'
    optimal_split = pasio.split_into_segments_square(sequence,
                                                     simple_scorer_factory)
    assert optimal_split[1] == [0,1,4]
    assert optimal_split[0] == 1+9+1


def test_split_into_segments_candidates():

    sequence = 'AAABBB'
    optimal_split = pasio.split_into_segments_square(sequence,
                                                     simple_scorer_factory,
                                                     split_candidates=[0,1,2,3,5,6])
    assert optimal_split[1] == [0,3]
    assert optimal_split[0] == 9+9

    sequence = 'AAABBB'
    optimal_split = pasio.split_into_segments_square(sequence,
                                                     simple_scorer_factory,
                                                     split_candidates=[0,3,5,6])
    assert optimal_split[1] == [0,3]
    assert optimal_split[0] == 9+9

    sequence = 'AAABBBC'
    optimal_split = pasio.split_into_segments_square(sequence,
                                                     simple_scorer_factory,
                                                     split_candidates=[0,3,7])
    assert optimal_split[1] == [0,3]
    assert optimal_split[0] == 9+4

    sequence = 'AAABBBC'
    optimal_split = pasio.split_into_segments_square(sequence,
                                                     simple_scorer_factory,
                                                     split_candidates=[0,3])
    assert optimal_split[1] == [0,3]
    assert optimal_split[0] == 9+4

    sequence = 'AAAAAA'
    optimal_split = pasio.split_into_segments_square(sequence,
                                                     simple_scorer_factory,
                                                     split_candidates=[0,3])
    assert optimal_split[1] == [0]
    assert optimal_split[0] == 36

def test_split_with_regularisation():
    # score of split 'AAA|B|AA' = 9+1+4 = 14
    # with regularisation = 9+1+4 - 3*2 = 8
    # alternative split: 'AAA|BAA' gives score = 9+3-3*1 = 9
    sequence = 'AAABAA'
    optimal_split = pasio.split_into_segments_square(sequence,
                                                     SimpleScorer,
                                                     regularisation_multiplyer = 3,
                                                     regularisation_function = lambda x:x)
    assert optimal_split[1] == [0,3]
    assert optimal_split[0] == 9

def test_split_with_length_regularisation():
    # score of split 'AAA|B|AA' = 9+1+4 = 14
    # with regularisation = 9+1+4 - 1.5*(1/log(3+1)+1/log(1+1)+1/log(2+1)) = 9.38
    # alternative split: 'AAA|BAA' gives score = 9+3 - 1.5*(1/log(3+1)+1/log(3+1)) = 9.83
    sequence = 'AAABAA'
    optimal_split = pasio.split_into_segments_square(sequence,
                                                     SimpleScorer,
                                                     regularisation_multiplyer = 0,
                                                     length_regularisation_multiplyer = 1.5,
                                                     length_regularisation_function = lambda x:1/np.log(1+x))
                                                     #length_regularisation_function = lambda x:x)
    assert optimal_split[1] == [0, 3]
    assert optimal_split[0] == 9+3 - 1.5*(1/np.log(3+1)+1/np.log(3+1))


def test_approximate_log_factorial():
    assert np.log(np.arange(1, 256+1)).sum() == pasio.log_factorial(256)
    assert np.log(np.arange(1, 4095+1)).sum() == pasio.log_factorial(4095)
    assert np.allclose(np.log(np.arange(1, 4096+1)).sum(),
                       pasio.log_factorial(4096))
    assert np.log(np.arange(1, 4097+1)).sum() == pasio.log_factorial(4097)
    assert np.log(np.arange(1, 10000+1)).sum() == pasio.log_factorial(10000)

    array_to_count = np.array([0,1,20,1024,10000])
    logfac_array = np.array([np.log(np.arange(1, x+1)).sum() for x in array_to_count])
    assert np.allclose(pasio.log_factorial(array_to_count), logfac_array)


def test_suffixes_scores():
    np.random.seed(2)
    counts = np.concatenate([np.random.poisson(15, 100),
                             np.random.poisson(20, 100)])

    scorer = pasio.LogMarginalLikelyhoodComputer(counts, 1, 1)
    suffixes_scores = [scorer(i, 150) for i in range(150)]
    assert np.allclose(scorer.all_suffixes_score(150), np.array(suffixes_scores))

    counts = np.array([0,0,1,0,0,2,2,2,10,11,100,1,0,0,1,0], dtype='int64')
    scorer = pasio.LogMarginalLikelyhoodComputer(counts, 1, 1)
    suffixes_scores = [scorer(i, len(counts)-1) for i in range(len(counts)-1)]
    assert np.allclose(scorer.all_suffixes_score(len(counts)-1), np.array(suffixes_scores))

def test_suffixes_scores_with_candidates():
    np.random.seed(2)
    counts = np.arange(1,10)
    scorer = pasio.LogMarginalLikelyhoodComputer(counts, 1, 1)
    candidates = np.array([0,1,3,4,5,6,7,8,9])
    scorer_with_candidates = pasio.LogMarginalLikelyhoodComputer(
        counts, 1, 1,
        split_candidates = candidates)
    candidate_suffixes = scorer.all_suffixes_score(9)[candidates[:-1]]
    suffixes_just_candidates = scorer_with_candidates.all_suffixes_score(8)
    assert np.allclose(candidate_suffixes, suffixes_just_candidates)

    counts = np.concatenate([np.random.poisson(15, 100),
                             np.random.poisson(20, 100)])
    scorer = pasio.LogMarginalLikelyhoodComputer(counts, 1, 1)
    candidates = np.array([0,1,10,20,21,30,40, 149])
    scorer_with_candidates = pasio.LogMarginalLikelyhoodComputer(
        counts, 1, 1,
        split_candidates = candidates)
    candidate_suffixes = scorer.all_suffixes_score(149)[candidates[:-1]]
    suffixes_just_candidates = scorer_with_candidates.all_suffixes_score(len(candidates)-1)
    assert np.allclose(candidate_suffixes, suffixes_just_candidates)

def compute_log_marginal_likelyhood2(scorer, length):
    scorer(0, length)


def test_collect_split_points():
    assert pasio.collect_split_points([0,0,0]) == [0]
    assert pasio.collect_split_points([0,1,2,3,4]) == [0,1,2,3,4]
    assert pasio.collect_split_points([0,0,0,0,2,3,4]) == [0,4]
    assert pasio.collect_split_points([0,0,2,1,4]) == [0,1,4]
    assert pasio.collect_split_points([0,0,2,1,3]) == [0,2,3]

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
    chromosomes = {k:v for (k,v,l) in pasio.parse_bedgrah(str(bedgraph_file))}
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
    sequence = A+B
    splits = pasio.split_into_segments_slidingwindow(sequence, simple_scorer_factory, 10, 5)
    assert splits[1] == [0, len(A)]
    assert splits[0] == len(A)**2+len(B)**2

    splits = pasio.split_into_segments_slidingwindow(sequence, simple_scorer_factory, 10, 5, 2)
    assert splits[1] == [0, len(A)]
    assert splits[0] == len(A)**2+len(B)**2-2
