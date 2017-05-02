import pytest
import pasio
import numpy as np
import random
import tempfile


def test_stat_split_into_segments_square():
    np.random.seed(2)
    scorer_factory = lambda c:pasio.LogMarginalLikelyhoodComputer(c, 1, 1)
    for repeat in range(10):
        counts = np.concatenate([np.random.poisson(15, 100),
                                 np.random.poisson(20, 100)])

        optimal_split = pasio.split_into_segments_square(counts, scorer_factory)

        two_split = pasio.split_on_two_segments_or_not(counts, scorer_factory)

        assert optimal_split[0] >= two_split[0]
        assert two_split[1] in optimal_split[1]
        if (two_split[1] is None):
            assert optimal_split[1] == [0,200]
        else:
            assert abs(two_split[1]-100) < 10

def test_split_into_segments_square():
    class SimpleScorer:
        def __init__(self, sequence):
            self.sequence = sequence
        def __call__(self, start=0, stop=None):
            if stop is None:
                stop = len(self.sequence)
            if len(set(self.sequence[start:stop])) == 1:
                return (stop-start)**2
            return 0
        def all_suffixes_score(self, stop):
            return np.array([self(i, stop) for i in range(stop)])

    sequence = 'AAA'
    optimal_split = pasio.split_into_segments_square(sequence,
                                                     SimpleScorer)
    assert optimal_split[1] == [0]
    assert optimal_split[0] == 9

    sequence = 'AAABBBC'
    optimal_split = pasio.split_into_segments_square(sequence,
                                                     SimpleScorer)
    assert optimal_split[1] == [0,3,6]
    assert optimal_split[0] == 9+9+1

def test_split_with_regularisation():
    class SimpleScorer:
        def __init__(self, sequence):
            self.sequence = sequence
        def __call__(self, start=0, stop=None):
            if stop is None:
                stop = len(self.sequence)
            if len(set(self.sequence[start:stop])) == 1:
                return (stop-start)**2
            return stop-start
        def all_suffixes_score(self, stop):
            return np.array([self(i, stop) for i in range(stop)])

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
    chromosomes = pasio.parse_bedgrah(str(bedgraph_file))
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

