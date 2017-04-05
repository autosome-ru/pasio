import time
import numpy as np
import pasio


def compute_log_marginal_likelyhood():
    counts = np.concatenate([np.random.poisson(15, 50),
                             np.random.poisson(20, 50)])
    pasio.log_marginal_likelyhood(counts, 1, 1)

def compute_log_marginal_likelyhood2():
    counts = np.concatenate([np.random.poisson(200, 50),
                             np.random.poisson(20, 50)])
    pasio.log_marginal_likelyhood(counts, 1, 1)

def segmentation200():
    np.random.seed(2)
    score_fn = lambda c: pasio.log_marginal_likelyhood(c, 1, 1)
    counts = np.concatenate([np.random.poisson(15, 50),
                             np.random.poisson(20, 50)])

    optimal_split = pasio.split_into_segments_square(counts, score_fn)

def test_benchmark_segmentation(benchmark):
    result = benchmark(segmentation200)

def test_benchmark_log_marginal_likehood(benchmark):
    result = benchmark(segmentation200)
