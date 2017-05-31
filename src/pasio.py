import numpy as np
import argparse
import array
import logging

logger = logging.getLogger(__name__)
stderr = logging.StreamHandler()
stderr.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stderr.setFormatter(formatter)
logger.addHandler(stderr)
logger.setLevel(logging.INFO)


class LogFactorialComputer:
    def approximate_log_factorial(self, x):
        return (x+1./2)*np.log(x) - x + (1./2)*np.log(2*np.pi) + 1./(12*x)
    def __init__(self):
        self.precomputed = np.zeros(4096)
        for i in range(4096):
            self.precomputed[i] = np.log(np.arange(1, i+1)).sum()
    def __call__(self, x):
        if type(x) is np.ndarray:
            log_factorial = np.zeros(x.shape)
            is_small = x < 4096
            log_factorial[is_small] = self.precomputed[x[is_small]]
            log_factorial[~is_small] = self.approximate_log_factorial(x[~is_small])
            return log_factorial
        else:
            if x < 4096:
                return self.precomputed[x]
            else:
                return self.approximate_log_factorial(x)


log_factorial = LogFactorialComputer()

class LogMarginalLikelyhoodComputer:
    def __init__(self, counts, alpha, beta, split_candidates=None):
        self.counts = counts
        self.alpha = alpha
        self.beta = beta
        if split_candidates is None:
            self.split_candidates = np.arange(len(counts)+1)
        else:
            assert split_candidates[0] == 0
            self.split_candidates = split_candidates
        self.cumsum = np.hstack([0, np.cumsum(counts)])[self.split_candidates]
        self.logfac_cumsum = np.hstack([0, np.cumsum(log_factorial(counts))])[self.split_candidates]

    def __call__(self, start=None, stop=None):
        if start is None:
            start = 0
        if stop is None:
            stop = len(self.split_candidates)-1
        num_counts = self.split_candidates[stop]-self.split_candidates[start]
        if stop == 0:
            sum_counts = 0
            sum_log_fac = 0
        elif start == 0:
            sum_counts = self.cumsum[stop]
            sum_log_fac = self.logfac_cumsum[stop]
        else:
            sum_counts = self.cumsum[stop]-self.cumsum[start]
            sum_log_fac = self.logfac_cumsum[stop]-self.logfac_cumsum[start]
        add1 = log_factorial(sum_counts+self.alpha)
        sub1 = sum_log_fac
        sub2 = (sum_counts+self.alpha+1)*np.log(num_counts+self.beta)
        return add1-sub1-sub2

    def all_suffixes_score(self, stop):
        suffixes_score = np.zeros(stop, dtype='float64')

        counts_cumsum = np.zeros(stop)
        counts_cumsum += self.cumsum[stop]-self.cumsum[0:stop]

        logfac_counts_cumsum = np.zeros(stop)
        logfac_counts_cumsum += (self.logfac_cumsum[stop]-
                                     self.logfac_cumsum[0:stop])

        counts_cumsum = np.zeros(stop, dtype='int')
        counts_cumsum += self.cumsum[stop]-self.cumsum[0:stop]

        add1_vec = log_factorial(counts_cumsum+self.alpha)

        sub2_vec = (counts_cumsum+self.alpha+1)*np.log(
            self.split_candidates[stop]-self.split_candidates[:stop]+self.beta)

        suffixes_score = add1_vec - logfac_counts_cumsum - sub2_vec

        return suffixes_score

def compute_score_from_splits(counts, splits, scorer_factory):
    scorer = scorer_factory(counts)
    sum_scores = 0
    for start, stop in zip(splits, splits[1:]):
        sum_scores += scorer(start, stop)
    sum_scores+=scorer(start = splits[-1])
    return sum_scores

def split_on_two_segments_or_not(counts, scorer_factory):
    scorer = scorer_factory(counts)
    best_score = scorer(0, len(counts))
    split_point = 0
    for i in range(len(counts)):
        current_score = scorer(stop=i)
        current_score += scorer(start=i)
        if current_score > best_score:
            split_point = i
            best_score = current_score
    return best_score, split_point

def collect_split_points(right_borders):
    split_point = right_borders[-1]
    split_points_collected = [split_point]
    while split_point != 0:
        split_point -= 1
        split_point = right_borders[split_point]
        split_points_collected.append(split_point)
    return split_points_collected[::-1]

def split_into_segments_square(counts, score_computer_factory,
                               regularisation_multiplyer=0,
                               regularisation_function=None,
                               split_candidates=None):
    if regularisation_function is None:
        regularisation_function = lambda x: x
    if split_candidates is None:
        split_candidates = np.arange(len(counts)+1)
    else:
        if split_candidates[-1] == len(counts):
            split_candidates = np.array(split_candidates, dtype=int)
        else:
            split_candidates = np.hstack([np.array(split_candidates, dtype=int),
                                          len(counts)])

    score_computer = score_computer_factory(counts,
                                            split_candidates=split_candidates)
    split_scores = np.zeros((len(split_candidates),))
    right_borders = np.zeros((len(split_candidates),), dtype=int)
    num_splits = np.zeros((len(split_candidates),))
    split_scores[0] = 0
    split_scores[1] = score_computer(0, 1)
    for i, split in enumerate(split_candidates[1:], 1):
        score_if_split_at_ = score_computer.all_suffixes_score(i).astype('float64')
        score_if_split_at_ += split_scores[:i]
        score_if_split_at_[:] -= regularisation_multiplyer*(regularisation_function(num_splits[:i]+1))
        score_if_split_at_[0] += regularisation_multiplyer*regularisation_function(1)
        right_borders[i] = np.argmax(score_if_split_at_)
        if right_borders[i] != 0:
            num_splits[i]= num_splits[right_borders[i]] + 1
        split_scores[i] = score_if_split_at_[right_borders[i]]
    return split_scores[-1], [split_candidates[i] for i in collect_split_points(right_borders[1:])]

def split_into_segments_slidingwindow(
        counts, score_computer_factory,
        window_size, window_shift,
        regularisation_multiplyer=0,
        regularisation_function=None):
    split_points = set([0])
    for start in range(0, len(counts), window_shift):
        logger.info('Processing window at start:%d (%.2f %s of chrom)' % (start, 100*start/float(len(counts)), '%'))
        stop = min(start+window_size, len(counts))
        segment_score, segment_split_points = split_into_segments_square(
            counts[start:stop], score_computer_factory,
            regularisation_multiplyer,
            regularisation_function=None)
        split_points.update([start+s for s in segment_split_points])
    return split_into_segments_square(
            counts, score_computer_factory,
            regularisation_multiplyer,
            regularisation_function,
            split_candidates=sorted(split_points))

def parse_bedgrah(filename):
    chromosome_data = None
    previous_chrom = None
    with open(filename) as bedgrpah_file:
        for line in bedgrpah_file:
            if line.strip() == '':
                continue
            chrom, start, stop, coverage = line.strip().split()
            start = int(start)
            stop = int(stop)
            coverage = int(coverage)
            if chrom != previous_chrom:
                if previous_chrom is not None:
                    yield previous_chrom, np.array(chromosome_data)
                chromosome_data = array.array('l')
            chromosome_data.extend([coverage]*(stop-start))
            previous_chrom = chrom
        yield chrom, np.array(chromosome_data)

def split_bedgraph(in_filename, out_filename, scorer_factory,
                   regularisation_multiplyer, split_function):
    with open(out_filename, 'w') as outfile:
        logger.info('Reading input file %s' % (in_filename))
        for chrom, counts in parse_bedgrah(in_filename):
            logger.info('Starting chrom %s of length %d' % (chrom, len(counts)))
            score, splits = split_function(counts, scorer_factory, regularisation_multiplyer)
            logger.info('chrom %s finished, score %f' % (chrom, score))
            scorer = scorer_factory(counts, splits+[len(counts)])
            logger.info('Starting output of chrom %s' % (chrom))
            for i, (start, stop) in enumerate(zip(splits, splits[1:])):
                outfile.write('%s\t%d\t%d\t%f\n' % (chrom, start, stop, scorer(i, i+1)))
            outfile.write('%s\t%d\t%d\t%f\n' % (chrom, splits[-1], len(counts), scorer(len(splits)-1)))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Pasio")
    argparser.add_argument('--algorithm', choices=['slidingwindow', 'exact'],
                           help="Algorithm to use")
    argparser.add_argument('--bedgraph', required=True, help="Input bedgraph path")
    argparser.add_argument('-o', '--out_bedgraph', help="Output begraph path")
    argparser.add_argument('--alpha', type=int, required=True,
                           help="alpha parameter of gamma distribution")
    argparser.add_argument('--beta', type=float, required=True,
                           help="beta parameter of gamma distribution")
    argparser.add_argument('--regularisation', type=float, default=0, help="Penalty for each split")
    argparser.add_argument('--window_size', type=int, help="Size of window fo split with exact algorithm")
    argparser.add_argument('--window_shift', type=int, help = "Shift in one step")

    args = argparser.parse_args()
    print args
    scorer_factory = lambda counts, split_candidates=None: LogMarginalLikelyhoodComputer(
        counts, args.alpha, args.beta, split_candidates = split_candidates)

    if args.algorithm == 'slidingwindow':
        split_function = lambda counts, factory, regularisation: split_into_segments_slidingwindow(
            counts, factory,
            window_size=args.window_size, window_shift=args.window_shift,
            regularisation_multiplyer=regularisation,
            regularisation_function=None)
    elif args.algorithm=='exact':
        split_function = lambda counts, factory, regularisation: split_into_segments_square(
            counts, factory,
            regularisation_multiplyer=regularisation,
            regularisation_function=None)

    logger.info('Starting')
    split_bedgraph(args.bedgraph, args.out_bedgraph, scorer_factory,
                   args.regularisation, split_function)

