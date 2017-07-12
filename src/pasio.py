import numpy as np
import argparse
import array
import logging
import sys

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

    def __init__(self, cache_size = 1048576):
        self.cache_size = cache_size
        self.precomputed = np.zeros(self.cache_size)
        for i in range(min(4096, self.cache_size)):
            self.precomputed[i] = np.log(np.arange(1, i+1)).sum()
        for i in range(min(4096, self.cache_size), self.cache_size):
            self.precomputed[i] = self.approximate_log_factorial(i)

    def __call__(self, x):
        if type(x) is np.ndarray:
            log_factorial = np.zeros(x.shape)
            is_small = x < self.cache_size
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

        self.constant = alpha*np.log(beta)-log_factorial(alpha-1)

        # buffers
        self.suffixes_score_ = np.zeros(len(self.split_candidates), dtype='float64')
        self.counts_cumsum_ = np.zeros(len(self.split_candidates), dtype='int')
        self.logfac_counts_cumsum_ = np.zeros(len(self.split_candidates), dtype='float64')
        self.add1_vec_ = np.zeros(len(self.split_candidates), dtype='float64')
        self.sub2_vec_ = np.zeros(len(self.split_candidates), dtype='float64')

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
        add1 = log_factorial(sum_counts+self.alpha-1)
        sub1 = sum_log_fac
        sub2 = (sum_counts+self.alpha)*np.log(num_counts+self.beta)
        return add1-sub1-sub2+self.constant

    def all_suffixes_score(self, stop):
        self.counts_cumsum_[:stop] = self.cumsum[stop]-self.cumsum[0:stop]

        self.logfac_counts_cumsum_[:stop] = (self.logfac_cumsum[stop]-
                                     self.logfac_cumsum[0:stop])

        self.add1_vec_[:stop] = log_factorial(self.counts_cumsum_[:stop]+self.alpha-1)

        self.sub2_vec_[:stop] = (self.counts_cumsum_[:stop]+self.alpha)*np.log(
            self.split_candidates[stop]-self.split_candidates[:stop]+self.beta)

        self.suffixes_score_[:stop] = self.add1_vec_[:stop] - self.logfac_counts_cumsum_[:stop] - self.sub2_vec_[:stop]

        return self.suffixes_score_[:stop]+self.constant

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
                               regularization_multiplyer=0,
                               regularization_function=None,
                               length_regularization_multiplyer=0,
                               length_regularization_function=None,
                               split_candidates=None):
    if regularization_function is None:
        regularization_function = lambda x: x

    if length_regularization_function is None:
        length_regularization_function = lambda x: x

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
    split_scores = np.zeros(len(split_candidates))
    right_borders = np.zeros(len(split_candidates), dtype=int)
    num_splits = np.zeros(len(split_candidates))
    splits_length_regularization = np.zeros(len(split_candidates))
    split_scores[0] = 0
    split_scores[1] = score_computer(0, 1)

    for i, split in enumerate(split_candidates[1:], 1):
        score_if_split_at_ = score_computer.all_suffixes_score(i).astype('float64')
        score_if_split_at_ += split_scores[:i]

        score_if_split_at_[:] -= regularization_multiplyer*(
            regularization_function(num_splits[:i]+1))
        score_if_split_at_[0] += regularization_multiplyer*regularization_function(1)

        last_segment_length_regularization = (length_regularization_multiplyer*
                                              length_regularization_function(
                                                  split - split_candidates[:i]))
        score_if_split_at_[:] -= last_segment_length_regularization[:i]

        right_borders[i] = np.argmax(score_if_split_at_)
        if right_borders[i] != 0:
            num_splits[i] = num_splits[right_borders[i]] + 1
        splits_length_regularization[i] = (splits_length_regularization[right_borders[i]]+
                                           last_segment_length_regularization[right_borders[i]])
        split_scores[i] = score_if_split_at_[right_borders[i]]

    return split_scores[-1], [split_candidates[i] for i in collect_split_points(right_borders[1:])]

def split_into_segments_if_not_all_zero(counts, score_computer_factory,
                                        regularization_multiplyer=0,
                                        regularization_function=None,
                                        split_candidates=None):
    if np.all(counts == 0):
        logger.info('Window contains just zeros. Skipping.')
        scorer = score_computer_factory(counts, split_candidates)
        return scorer(), [0, len(counts)]
    logger.info('Not zeros. Spliting.')
    return split_into_segments_square(counts, score_computer_factory,
                                      regularization_multiplyer=regularization_multiplyer,
                                      regularization_function=regularization_function,
                                      split_candidates=split_candidates)

def split_into_segments_slidingwindow(
        counts, score_computer_factory,
        window_size, window_shift,
        regularization_multiplyer=0,
        regularization_function=None):
    split_points = set([0])
    for start in range(0, len(counts), window_shift):
        logger.info('Processing window at start:%d (%.2f %s of chrom)' % (start, 100*start/float(len(counts)), '%'))
        stop = min(start+window_size, len(counts))
        segment_score, segment_split_points = split_into_segments_if_not_all_zero(
            counts[start:stop], score_computer_factory,
            regularization_multiplyer,
            regularization_function=None)
        split_points.update([start+s for s in segment_split_points])
    logger.info('Final split of chromosome with %d split points' % (len(split_points)))
    return split_into_segments_square(
            counts, score_computer_factory,
            regularization_multiplyer,
            regularization_function,
            split_candidates=sorted(split_points))

def split_into_segments_rounds(
        counts, score_computer_factory,
        window_size, window_shift,
        regularization_multiplyer=0,
        regularization_function=None,
        num_rounds=None):
    possible_split_points = np.arange(len(counts)+1)
    if num_rounds is None:
        num_rounds = len(counts)
    for round_ in range(num_rounds):
        new_split_points = set([0])
        logger.info('Starting split round %d, num_candidates %d' % (round_, len(possible_split_points)))
        for start_index in range(0, len(possible_split_points), window_shift):
            stop_index = min(start_index+window_size, len(possible_split_points)-1)
            start = possible_split_points[start_index]
            stop = possible_split_points[stop_index]
            logger.info('Round:%d Splitting window [%d, %d], %d points, (%.2f %s of round complete)' % (
                round_, start, stop, len(possible_split_points[start_index:stop_index]),
                float(start_index)/len(possible_split_points)*100, '%'))
            segment_score, segment_split_points = split_into_segments_if_not_all_zero(
                counts[start:stop], score_computer_factory,
                regularization_multiplyer,
                regularization_function=None,
                split_candidates = np.array(
                    [p-start for p in possible_split_points[start_index:stop_index]]
                )
            )
            new_split_points.update([start+s for s in segment_split_points])
        new_split_points = np.array(sorted(new_split_points))
        # last possible split point is the last point
        if np.all(new_split_points == possible_split_points[:-1]):
            logger.info('Round:%d No split points removed. Finishing round' % round_)
            # So no split points removed
            break
        else:
            assert len(new_split_points) < len(possible_split_points)
        possible_split_points = np.hstack([new_split_points, len(counts)])
    final_score = compute_score_from_splits(counts, new_split_points, score_computer_factory)

    logger.info('Splitting finished in %d rounds. Score %f Number of split points %d' % (round_,
                                                                                         final_score,
                                                                                         len(new_split_points)))
    return (final_score, list(new_split_points))

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
                    yield previous_chrom, np.array(chromosome_data), chromosome_start
                chromosome_data = array.array('l')
                chromosome_start = start
            chromosome_data.extend([coverage]*(stop-start))
            previous_chrom = chrom
        yield chrom, np.array(chromosome_data), chromosome_start


def split_bedgraph(in_filename, out_filename, scorer_factory,
                   regularization_multiplyer, split_function):
    with open(out_filename, 'w') as outfile:
        logger.info('Reading input file %s' % (in_filename))
        for chrom, counts, chrom_start in parse_bedgrah(in_filename):
            logger.info('Starting chrom %s of length %d' % (chrom, len(counts)))
            score, splits = split_function(counts, scorer_factory,
                                           regularization_multiplyer)
            logger.info('chrom %s finished, score %f, number of splits %d' % (chrom, score, len(splits)))
            scorer = scorer_factory(counts, splits+[len(counts)])
            logger.info('Starting output of chrom %s' % (chrom))
            for i, (start, stop) in enumerate(zip(splits, splits[1:])):
                outfile.write('%s\t%d\t%d\t%f\t%d\t%f\n' % (chrom,
                                                            start+chrom_start,
                                                            stop+chrom_start,
                                                            counts[start:stop].mean(),
                                                            stop-start,
                                                            scorer(i, i+1)))
            outfile.write('%s\t%d\t%d\t%f\t%d\t%f\n' % (chrom,
                                                        splits[-1]+chrom_start,
                                                        len(counts)+chrom_start,
                                                        counts[splits[-1]:].mean(),
                                                        len(counts)-splits[-1],
                                                        scorer(len(splits)-1)))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Pasio", formatter_class=argparse.RawTextHelpFormatter)
    argparser.add_argument('--algorithm', choices=['slidingwindow', 'exact', 'rounds'],
                           required=True,
                           help="Algorithm to use")
    argparser.add_argument('--bedgraph', required=True, help="Input bedgraph path")
    argparser.add_argument('-o', '--out_bedgraph', help="Output begraph path")
    argparser.add_argument('--alpha', type=int, required=True,
                           help="alpha parameter of gamma distribution")
    argparser.add_argument('--beta', type=float, required=True,
                           help="beta parameter of gamma distribution")
    argparser.add_argument('--split_regularization', type=float, default=0,
                           help="Penalty multiplyer for each split")
    argparser.add_argument('--length_regularization', type=float, default=0,
                           help="Penalty multiplyer for length of each segment")
    argparser.add_argument('--length_regularization_function', type=str, default='none',
                           choices=['none', 'revlog', 'neglog'],
                           help='''Penalty function for length of segments.:
                           none: no length regulatization
                           revlog: 1/log(1+l)
                           neglog: -log(l)
                           ''')
    argparser.add_argument('--window_size', type=int,
                           help="Size of window fo split with exact algorithm")
    argparser.add_argument('--window_shift', type=int,
                           help = "Shift in one step")
    argparser.add_argument('--num_rounds', type=int,
                           help = "Number of rounds for round algorithm")

    args = argparser.parse_args()
    print args
    if args.algorithm in ['slidingwindow', 'rounds']:
        if args.window_shift is None:
            sys.exit('Argument --window_shift is required for algorithms slidingwingow and rounds')
        if args.window_size is None:
            sys.exit('Argument --window_size is required for algorithms slidingwingow and rounds')
    scorer_factory = lambda counts, split_candidates=None: LogMarginalLikelyhoodComputer(
        counts, args.alpha, args.beta, split_candidates = split_candidates)

    if args.algorithm == 'slidingwindow':
        split_function = lambda counts, factory, regularization: split_into_segments_slidingwindow(
            counts, factory,
            window_size=args.window_size, window_shift=args.window_shift,
            regularization_multiplyer=regularization,
            regularization_function=None)
    elif args.algorithm=='exact':
        split_function = lambda counts, factory, regularization: split_into_segments_square(
            counts, factory,
            regularization_multiplyer=regularization,
            regularization_function=None)
    elif args.algorithm=='rounds':
        split_function = lambda counts, factory, regularization: split_into_segments_rounds(
            counts, factory,
            window_size=args.window_size, window_shift=args.window_shift,
            regularization_multiplyer=regularization,
            regularization_function=None,
            num_rounds=args.num_rounds)

    logger.info('Starting Pasio wis args'+str(args))
    split_bedgraph(args.bedgraph, args.out_bedgraph, scorer_factory,
                   args.regularization, split_function)

