import time
import numpy as np
import pasio
import random


def compute_log_marginal_likelyhood2(scorer, length):
    scorer(0, length)

def segmentation200(counts, scorer):
    optimal_split = pasio.split_into_segments_square(counts, scorer)

def parse_bedgraph(filename):
    {k:v for (k,v,_) in pasio.parse_bedgrah(filename)}

def test_benchmark_segmentation(benchmark):
    np.random.seed(2)

    counts = np.concatenate([np.random.poisson(15, 50),
                             np.random.poisson(20, 50)])

    scorer_factory = lambda counts, split_candidates=None : pasio.LogMarginalLikelyhoodComputer(
        counts, 1, 1, split_candidates)
    result = benchmark(segmentation200, counts, scorer_factory)

def test_benchmark_log_marginal_likehood(benchmark):
    counts = np.concatenate([np.random.poisson(200, 50),
                             np.random.poisson(20, 50)])
    scorer = pasio.LogMarginalLikelyhoodComputer(counts, 1, 1)

    result = benchmark(compute_log_marginal_likelyhood2,
                       scorer, len(counts))

def test_benchmark_bedgraph_parser(benchmark, tmpdir):
    bedgraph_file = tmpdir.mkdir("sub").join("test.bedgraph")
    random.seed(0)
    previous_i = 0
    for i in range(1, 100000):
        if random.randint(0,10)==5:
            bedgraph_file.write('chr1 %d %d %d\n' % (previous_i, i, 1000))

    chromosomes = benchmark(parse_bedgraph, str(bedgraph_file))
