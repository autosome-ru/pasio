import time
import numpy as np
import pasio


def compute_log_marginal_likelyhood2(scorer, length):
    scorer(0, length)

def segmentation200(counts, scorer):
    optimal_split = pasio.split_into_segments_square(counts, scorer)

def test_benchmark_segmentation(benchmark):
    np.random.seed(2)

    counts = np.concatenate([np.random.poisson(15, 50),
                             np.random.poisson(20, 50)])

    scorer = pasio.LogMarginalLikelyhoodComputer(counts, 1, 1)
    result = benchmark(segmentation200, counts, scorer)

def test_benchmark_log_marginal_likehood(benchmark):
    counts = np.concatenate([np.random.poisson(200, 50),
                             np.random.poisson(20, 50)])
    scorer = pasio.LogMarginalLikelyhoodComputer(counts, 1, 1)
    result = benchmark(compute_log_marginal_likelyhood2,
                                scorer,len(counts))
