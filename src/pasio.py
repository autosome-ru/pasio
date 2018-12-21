from __future__ import print_function, division
from builtins import range
import numpy as np
import argparse
import logging
import sys
import scipy.special

logger = logging.getLogger(__name__)
stderr = logging.StreamHandler()
stderr.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stderr.setFormatter(formatter)
logger.addHandler(stderr)
logger.setLevel(logging.INFO)

class LoggingContextFilter:
    def __init__(self):
        self.context = {}

    def put_to_context(self, key, value):
        self.context[key] = value

    def remove_from_context(self, key):
        del self.context[key]

    def filter(self, record):
        if 'window' in self.context:
            record.msg = 'Window %s: %s' % (self.context['window'], record.msg)
        if 'round' in self.context:
            record.msg = 'Round %d: %s' % (self.context['round'], record.msg)
        return True

logging_filter = LoggingContextFilter()
logger.addFilter(logging_filter)

# Works only with non-negative integer values
class LogComputer:
    def __init__(self, cache_size = 1048576):
        self.cache_size = cache_size
        self.precomputed = np.log(np.arange(self.cache_size))

    def compute_for_number(self, x):
        if x < self.cache_size:
            return self.precomputed[x]
        else:
            return np.log(x)

    # uses fast algorithm if maximal value of x is specified and doesn't exceed cache size
    def compute_for_array(self, x, max_value = float('inf')):
        if max_value < self.cache_size:
            return self.precomputed[x]
        else:
            return self.compute_for_array_non_cached(x)

    def compute_for_array_non_cached(self, x):
        result = np.zeros(x.shape)
        is_small = x < self.cache_size
        result[is_small] = self.precomputed[x[is_small]]
        result[~is_small] = np.log(x[~is_small])
        return result

# Works only with non-negative integer values
class LogGammaComputer:
    def __init__(self, cache_size = 1048576):
        self.cache_size = cache_size
        self.precomputed = scipy.special.gammaln(np.arange(self.cache_size))

    def compute_for_number(self, x):
        if x < self.cache_size:
            return self.precomputed[x]
        else:
            return scipy.special.gammaln(x)

    # uses fast algorithm if maximal value of x is specified and doesn't exceed cache size
    def compute_for_array(self, x, max_value = float('inf')):
        if max_value < self.cache_size:
            return self.precomputed[x]
        else:
            return self.compute_for_array_non_cached(x)

    def compute_for_array_non_cached(self, x):
        result = np.zeros(x.shape)
        is_small = x < self.cache_size
        result[is_small] = self.precomputed[x[is_small]]
        result[~is_small] = scipy.special.gammaln(x[~is_small])
        return result

log_computer = LogComputer()
log_gamma_computer = LogGammaComputer()

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
    def __init__(self, counts, alpha, beta, split_candidates):

        assert isinstance(alpha, int)
        assert isinstance(beta, int)
        assert alpha >= 0
        assert beta >= 0
        self.alpha = alpha
        self.beta = beta

        assert_correct_counts(counts)
        self.counts = counts

        assert_correct_split_candidates(split_candidates, counts)
        self.split_candidates = split_candidates

        self.cumsum = np.hstack([0, np.cumsum(counts)])[split_candidates]

        count_logfacs = log_gamma_computer.compute_for_array(counts + 1)
        self.logfac_cumsum = np.hstack([0, np.cumsum(count_logfacs)])[split_candidates]

        self.segment_creation_cost = alpha * log_computer.compute_for_number(beta) - log_gamma_computer.compute_for_number(alpha)

    def total_sum_logfac(self):
        return self.logfac_cumsum[-1]

    def scores(self):
        segment_lengths = np.diff(self.split_candidates)
        segment_counts = np.diff(self.cumsum)
        shifted_segment_counts = segment_counts + self.alpha
        shifted_segment_lengths = segment_lengths + self.beta
        add = log_gamma_computer.compute_for_array(shifted_segment_counts)
        sub = shifted_segment_counts * log_computer.compute_for_array(shifted_segment_lengths)
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
        add = log_gamma_computer.compute_for_number(shifted_segment_count)
        sub = shifted_segment_count * log_computer.compute_for_number(shifted_segment_length)
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

        add_vec = log_gamma_computer.compute_for_array(shifted_segment_count_vec, max_value=(self.alpha + self.cumsum[stop]))
        sub_vec = shifted_segment_count_vec * log_computer.compute_for_array(shifted_segment_length_vec, max_value=(self.beta + self.split_candidates[stop]))

        return add_vec - sub_vec

class SquareSplitter:
    def __init__(self, scorer_factory,
                length_regularization_multiplier=0,
                length_regularization_function=lambda x: x,
                split_number_regularization_multiplier=0,
                split_number_regularization_function=lambda x: x):
        self.scorer_factory = scorer_factory
        self.length_regularization_multiplier = length_regularization_multiplier
        self.split_number_regularization_multiplier = split_number_regularization_multiplier
        self.length_regularization_function = length_regularization_function
        self.split_number_regularization_function = split_number_regularization_function

    def scorer(self, counts, split_candidates):
        return self.scorer_factory(counts, split_candidates)

    def reduce_candidate_list(self, counts, split_candidates):
        score, splits = self.split(counts, split_candidates)
        return splits

    def split(self, counts, split_candidates):
        if self.split_number_regularization_multiplier == 0 and self.length_regularization_multiplier == 0:
            return self.split_without_normalizations(counts, split_candidates)
        else:
            return self.split_with_normalizations(counts, split_candidates)

    def split_with_normalizations(self, counts, split_candidates):
        score_computer = self.scorer(counts, split_candidates)
        num_split_candidates = len(split_candidates)

        prefix_scores = np.empty(num_split_candidates)
        prefix_scores[0] = 0

        previous_splits = np.empty(num_split_candidates, dtype=int)
        previous_splits[0] = 0

        num_splits = np.zeros(num_split_candidates)

        for prefix_end in range(1, num_split_candidates):
            score_if_last_split_at = score_computer.all_suffixes_self_score(prefix_end)
            score_if_last_split_at += prefix_scores[:prefix_end]

            if self.split_number_regularization_multiplier != 0:
                number_regularization = self.split_number_regularization_function(num_splits[:prefix_end] + 1)
                score_if_last_split_at -= self.split_number_regularization_multiplier * number_regularization
                score_if_last_split_at[0] += self.split_number_regularization_multiplier * self.split_number_regularization_function(1)

            if self.length_regularization_multiplier != 0:
                length_regulatization = self.length_regularization_function(split_candidates[prefix_end] - split_candidates[:prefix_end])
                last_segment_length_regularization = self.length_regularization_multiplier * length_regulatization
                score_if_last_split_at -= last_segment_length_regularization[:prefix_end]

            optimal_last_split = np.argmax(score_if_last_split_at)
            previous_splits[prefix_end] = optimal_last_split

            if optimal_last_split != 0:
                num_splits[prefix_end] = num_splits[optimal_last_split] + 1

            prefix_scores[prefix_end] = score_if_last_split_at[optimal_last_split] + score_computer.segment_creation_cost

        split_indices = SquareSplitter.collect_split_points(previous_splits)
        split_positions = split_candidates[split_indices]
        return prefix_scores[-1], split_positions

    def split_without_normalizations(self, counts, split_candidates):
        score_computer = self.scorer(counts, split_candidates)
        num_split_candidates = len(split_candidates)

        # prefix_scores[i] is the best score of prefix [0; i)
        prefix_scores = np.empty(num_split_candidates)
        prefix_scores[0] = 0

        # `previous_splits[i]` (later `ps`) is a position of the last split
        # in the best segmentation of prefix [0, i) which is not the end of the prefix
        # i.e. this prefix segmentation looks like `...... ps - 1) [ps ... i - 1)`
        previous_splits = np.empty(num_split_candidates, dtype=int)
        previous_splits[0] = 0

        # find the score and previous point of the best segmentation of the prefix [0, prefix_end)
        # such that the last split in segmentation is prefix_end (split is between `prefix_end - 1` and `prefix_end`)
        # (indexation runs over split candidates, not over all points)
        for prefix_end in range(1, num_split_candidates):
            # score consists of (a) score of the last segment
            score_if_last_split_at = score_computer.all_suffixes_self_score(prefix_end)
            #                   (b) score of the prefix before the last segment
            score_if_last_split_at += prefix_scores[:prefix_end]

            optimal_last_split = np.argmax(score_if_last_split_at)
            previous_splits[prefix_end] = optimal_last_split

            #                   (c) and constant penalty for segment creation
            prefix_scores[prefix_end] = score_if_last_split_at[optimal_last_split] + score_computer.segment_creation_cost

        # reminder: splits indexing is over split candidates, not contig positions
        split_indices = SquareSplitter.collect_split_points(previous_splits)
        # but we want to return contig positions of these splits
        split_positions = split_candidates[split_indices]
        return prefix_scores[-1], split_positions

    @staticmethod
    def collect_split_points(previous_splits):
        split_point = len(previous_splits) - 1
        split_points_collected = [split_point]
        while split_point != 0:
            split_point = previous_splits[split_point]
            split_points_collected.append(split_point)
        return split_points_collected[::-1]


class NotZeroReducer:
    def reduce_candidate_list(self, counts, split_candidates):
        if np.all(counts == 0):
            logger.info('Just zeros: %d --> 2 split points' % len(split_candidates))
            return np.array([0, len(counts)])
        else:
            logger.info('Not zeros. Interval not reduced.')
            return split_candidates


class NotConstantReducer:
    def reduce_candidate_list(self, counts, split_candidates):
        # ------left_to_border|right_to_border------
        (left_to_border_positions, ) = np.where( counts[:-1] != counts[1:] )
        points_of_count_change = 1 + left_to_border_positions
        nonconstant_split_positions = np.intersect1d(split_candidates, points_of_count_change, assume_unique=True)
        new_split_candidates = np.hstack([0, nonconstant_split_positions, len(counts)])
        logger.info('Constants reduced: %d --> %d split points' % (len(split_candidates), len(new_split_candidates)))
        return new_split_candidates

class SlidingWindow:
    def __init__(self, window_size, window_shift):
        self.window_size = window_size
        self.window_shift = window_shift

    def windows(self, arr):
        length = len(arr)
        for start in range(0, length, self.window_shift):
            stop = min(start + self.window_size + 1, length)
            completion = stop / length
            # start inclusive, stop exclusive
            yield (arr[start:stop], completion)

class SlidingWindowReducer:
    def __init__(self, sliding_window, base_reducer):
        self.sliding_window = sliding_window
        self.base_reducer = base_reducer

    # Single round of candidate list reduction
    def reduce_candidate_list(self, counts, split_candidates):
        new_split_candidates_set = set([0, len(counts)])
        for (split_candidates_in_window, completion) in self.sliding_window.windows(split_candidates):
            start = split_candidates_in_window[0]
            stop  = split_candidates_in_window[-1]
            logging_filter.put_to_context('window', '[%d, %d)' % (start, stop))

            splits_relative_pos = split_candidates_in_window - start
            reduced_splits_relative_pos = self.base_reducer.reduce_candidate_list(
                counts[start:stop], splits_relative_pos)
            new_split_candidates_set.update(reduced_splits_relative_pos + start)
            logger.info('Sliding (completion: %.2f %%): %d --> %d split-points' % (
                100 * completion, len(split_candidates_in_window), len(reduced_splits_relative_pos)))
        logging_filter.remove_from_context('window')
        return np.array(sorted(new_split_candidates_set))

class RoundReducer:
    def __init__(self, base_reducer, num_rounds=None):
        self.base_reducer = base_reducer
        self.num_rounds = num_rounds

    def reduce_candidate_list(self, counts, split_candidates):
        if self.num_rounds is None:
            num_rounds = len(counts)
        else:
            num_rounds = self.num_rounds

        for round_ in range(1, num_rounds + 1):
            logging_filter.put_to_context('round', round_)
            logger.info('Starting round, num_candidates %d' % len(split_candidates))
            new_split_candidates = self.base_reducer.reduce_candidate_list(counts, split_candidates)
            if np.array_equal(new_split_candidates, split_candidates):
                logger.info('No split points removed. Finishing round')
                logging_filter.remove_from_context('round')
                return new_split_candidates
            assert len(new_split_candidates) < len(split_candidates)
            logger.info('Finishing round, num_candidates %d' % len(new_split_candidates))
            split_candidates = new_split_candidates

        logging_filter.remove_from_context('round')
        logger.info('Splitting finished in %d rounds. Number of split points %d' % (round_, len(new_split_candidates)))
        return new_split_candidates

# Doesn't change list of split candidates. But for a given list of split candidates it can calculate score
# Designed to make it possible to convert reducer into splitter with `SpliiterCombiner(reducer, NopSplitter())`
class NopSplitter:
    def __init__(self, scorer_factory):
        self.scorer_factory = scorer_factory

    def scorer(self, counts, split_candidates):
        return self.scorer_factory(counts, split_candidates)

    def reduce_candidate_list(self, counts, split_candidates):
        return split_candidates

    def split(self, counts, split_candidates):
        scores = self.scorer(counts, split_candidates).scores()
        final_score = np.sum(scores)
        return (final_score, split_candidates)

class ReducerCombiner:
    def __init__(self, *reducers):
        self.reducers = reducers

    def reduce_candidate_list(self, counts, split_candidates):
        for reducer in self.reducers:
            split_candidates = reducer.reduce_candidate_list(counts, split_candidates)
        return split_candidates

    def split(self, counts, split_candidates):
        splitter = self.reducers[-1]
        if hasattr(splitter, 'split') and callable(splitter.split):
            for reducer in self.reducers[:-1]:
                split_candidates = reducer.reduce_candidate_list(counts, split_candidates)
            return splitter.split(counts, split_candidates)
        else:
            raise Exception('This ReducerCombiner has no splitter at the end of pipeline. Splitting no possible')

    def scorer(self, counts, split_candidates):
        splitter = self.reducers[-1]
        if hasattr(splitter, 'scorer') and callable(splitter.scorer):
            return splitter.scorer(counts, split_candidates)
        else:
            raise Exception('This ReducerCombiner has no splitter at the end of pipeline. Scoring not possible. Consider use of NopSplitter')


def parse_bedgraph(filename):
    '''
        yields pointwise profiles grouped by chromosome in form (chrom, profile, chromosome_start)
    '''
    chromosome_data = None
    previous_chrom = None
    with open(filename) as bedgraph_file:
        for line in bedgraph_file:
            if line.strip() == '':
                continue
            chrom, start, stop, coverage = line.strip().split()
            start = int(start)
            stop = int(stop)
            coverage = int(coverage)
            if chrom != previous_chrom:
                if previous_chrom is not None:
                    # overwrite chromosome_data not to retain both list and np.array in memory
                    # and let the former be garbage collected
                    chromosome_data = np.array(chromosome_data, dtype=int)
                    yield previous_chrom, chromosome_data, chromosome_start
                chromosome_data = []
                chromosome_start = start
            chromosome_data.extend([coverage]*(stop-start))
            previous_chrom = chrom
        chromosome_data = np.array(chromosome_data, dtype=int)
        yield chrom, chromosome_data, chromosome_start


def split_bedgraph(in_filename, out_filename, splitter):
    with open(out_filename, 'w') as outfile:
        logger.info('Reading input file %s' % (in_filename))
        for chrom, counts, chrom_start in parse_bedgraph(in_filename):
            logger.info('Starting chrom %s of length %d' % (chrom, len(counts)))
            split_candidates = np.arange(len(counts) + 1)
            score, splits = splitter.split(counts, split_candidates)
            scorer = splitter.scorer(counts, splits)
            sum_logfac = scorer.total_sum_logfac()
            log_likelyhood = score - sum_logfac
            logger.info('chrom %s finished, score %f, number of splits %d. '
                        'Log likelyhood: %f.'% (chrom, score, len(splits), log_likelyhood))
            logger.info('Starting output of chrom %s' % chrom)
            for i, (start, stop, mean_count, log_marginal_likelyhood) in enumerate(zip(splits, splits[1:], scorer.mean_counts(), scorer.log_marginal_likelyhoods())):
                outfile.write('%s\t%d\t%d\t%f\t%d\t%f\n' % (chrom,
                                                            start+chrom_start,
                                                            stop+chrom_start,
                                                            mean_count,
                                                            stop-start,
                                                            log_marginal_likelyhood))

def get_argparser():
    argparser = argparse.ArgumentParser(
        prog = "Pasio",
        description = '''
Example usage, simpliest for practical cases:
python pasio.py
      --bedgraph ~/<PATH TO INPUT bed.Graph FILE> -o ~/<PATH TO OUTPUT bedGraph FILE>
      --alpha 5 --beta 1 --algorithm rounds
      --window_shift 1250 --window_size 2500
''',
        formatter_class=argparse.RawTextHelpFormatter)
    argparser.add_argument('--algorithm',
                           choices=['slidingwindow', 'exact', 'rounds'],
                           required=True,
                           help="Algorithm to use")
    argparser.add_argument('--bedgraph', required=True,
                           help="Input bedgraph path")
    argparser.add_argument('-o', '--out_bedgraph', help="Output begraph path",
                           required=True)
    argparser.add_argument('--alpha', type=int, required=True,
                           help="alpha parameter of gamma distribution")
    argparser.add_argument('--beta', type=int, required=True,
                           help="beta parameter of gamma distribution")
    argparser.add_argument('--split_number_regularization', type=float, default=0,
                           help="Penalty multiplier for each split")
    argparser.add_argument('--length_regularization', type=float, default=0,
                           help="Penalty multiplier for length of each segment")
    argparser.add_argument('--length_regularization_function', type=str, default='none',
                           choices=['none', 'revlog', 'neglog'],
                           help='''Penalty function for length of segments.:
                           none: no length regulatization
                           revlog: 1/log(1+l)
                           ''')
    argparser.add_argument('--window_size', type=int,
                           help="Size of window fo split with exact algorithm")
    argparser.add_argument('--window_shift', type=int,
                           help = "Shift in one step")
    argparser.add_argument('--num_rounds', type=int,
                           help = '''Number of rounds for round algorithm.
                           If not set, run until no split points removed''')
    argparser.add_argument('--no_split_constant', action='store_true',
                           help = '''[experimental] If set, won't put splits between constant counts''')
    return argparser


if __name__ == '__main__':
    argparser = get_argparser()
    args = argparser.parse_args()
    logger.info("Pasio:"+ str(args))
    if args.algorithm in ['slidingwindow', 'rounds']:
        if args.window_shift is None:
            sys.exit('Argument --window_shift is required for algorithms slidingwingow and rounds')
        if args.window_size is None:
            sys.exit('Argument --window_size is required for algorithms slidingwingow and rounds')
    scorer_factory = lambda counts, split_candidates: LogMarginalLikelyhoodComputer(
        counts, args.alpha, args.beta, split_candidates)

    length_regularization_functions = {
        'none': lambda x: x,
        'revlog': lambda x: 1 / np.log(x + 1),
    }
    length_regularization_function = length_regularization_functions[args.length_regularization_function]
    length_regularization_multiplier = args.length_regularization
    split_number_regularization_multiplier = args.split_number_regularization

    if length_regularization_multiplier != 0:
        if args.length_regularization_function == 'none':
            sys.exit('Argument --length_regularization_function is required '
                     'for length regularization multiplier %s' %
                     args.length_regularization)

    if args.length_regularization_function != 'none':
        if length_regularization_multiplier == 0:
            sys.exit('Argument --length_regularization_multiplier is required '
                     'for length legularization function %s' %
                     args.length_regularization_function)

    square_splitter = SquareSplitter(scorer_factory,
        length_regularization_multiplier=length_regularization_multiplier,
        length_regularization_function=length_regularization_function,
        split_number_regularization_multiplier=split_number_regularization_multiplier,
        split_number_regularization_function=lambda x:x)

    if args.algorithm == 'exact':
        splitter = square_splitter
    else:
        if args.no_split_constant:
            square_splitter = ReducerCombiner(NotConstantReducer(), square_splitter)
        else:
            square_splitter = ReducerCombiner(NotZeroReducer(), square_splitter)

        if args.algorithm == 'slidingwindow':
            sliding_window = SlidingWindow(window_size = args.window_size, window_shift = args.window_shift)
            reducer = SlidingWindowReducer(sliding_window = sliding_window, base_reducer = square_splitter)
            splitter = ReducerCombiner(reducer, square_splitter)
        elif args.algorithm == 'rounds':
            sliding_window = SlidingWindow(window_size = args.window_size, window_shift = args.window_shift)
            base_reducer = SlidingWindowReducer(sliding_window = sliding_window, base_reducer = square_splitter)
            reducer = RoundReducer(base_reducer = base_reducer, num_rounds = args.num_rounds)
            splitter = ReducerCombiner(reducer, NopSplitter(scorer_factory))

    logger.info('Starting Pasio with args'+str(args))
    split_bedgraph(args.bedgraph, args.out_bedgraph, splitter)
