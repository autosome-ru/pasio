import pytest
import pasio
import numpy as np
import random


def test_split_into_segments_square():
    np.random.seed(2)
    score_fn = lambda c: pasio.log_marginal_likelyhood(c, 1, 1)
    for repeat in range(10):
        counts = np.concatenate([np.random.poisson(15, 100),
                                 np.random.poisson(20, 100)])

        optimal_split = pasio.split_into_segments_square(counts, score_fn)

        two_split = pasio.split_on_two_segments_or_not(counts, score_fn)

        assert optimal_split[0] >= two_split[0]
        assert two_split[1] in optimal_split[1]
        if (two_split[1] is None):
            assert optimal_split[1] == [0,200]
        else:
            assert abs(two_split[1]-100) < 10


