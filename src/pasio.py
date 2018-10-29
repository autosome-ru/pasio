# python2 src/pasio.py
#       --bedgraph ~/<PATH TO INPUT bed.Graph FILE> -o ~/<WAY TO OUTPUT bedGraph FILE>
#       --alpha 5 --beta 1 --algorithm rounds
#       --window_shift 1250 --window_size 2500
#       --tmpdir pasio_tmp --out_script pasio_parallel.sh --path_to_pasio 'python2 src/pasio.py'

import numpy as np
import argparse
import array
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

# Works only with non-negative integer values
class LogComputer:
    def __init__(self, cache_size = 1048576):
        self.cache_size = cache_size
        self.precomputed = np.log(np.arange(self.cache_size))

    # uses fast algorithm if maximal value of x is specified and doesn't exceed cache size
    def __call__(self, x, max_value = float('inf')):
        if type(x) is np.ndarray:
            if max_value < self.cache_size:
                return self.precomputed[x]
            else:
                result = np.zeros(x.shape)
                is_small = x < self.cache_size
                result[is_small] = self.precomputed[x[is_small]]
                result[~is_small] = np.log(x[~is_small])
                return result
        else:
            if x < self.cache_size:
                return self.precomputed[x]
            else:
                return np.log(x)

# Works only with non-negative integer values
class LogGammaComputer:
    def __init__(self, cache_size = 1048576):
        self.cache_size = cache_size
        self.precomputed = scipy.special.gammaln(np.arange(self.cache_size))

    # uses fast algorithm if maximal value of x is specified and doesn't exceed cache size
    def __call__(self, x, max_value = float('inf')):
        if type(x) is np.ndarray:
            if max_value < self.cache_size:
                return self.precomputed[x]
            else:
                result = np.zeros(x.shape)
                is_small = x < self.cache_size
                result[is_small] = self.precomputed[x[is_small]]
                result[~is_small] = scipy.special.gammaln(x[~is_small])
                return result
        else:
            if x < self.cache_size:
                return self.precomputed[x]
            else:
                return scipy.special.gammaln(x)

cached_log = LogComputer()
log_gamma = LogGammaComputer()

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
        self.logfac_cumsum = np.hstack([0, np.cumsum(log_gamma(counts + 1))])[self.split_candidates]

        self.constant = alpha*np.log(beta)-log_gamma(alpha)

    def segment_sum_logfac(self, start=None, stop=None):
        if start is None:
            start = 0
        if stop is None:
            stop = len(self.split_candidates)-1
        sum_logfac = self.logfac_cumsum[stop] - self.logfac_cumsum[start]
        return sum_logfac

    def log_marginal_likelyhood(self, start=None, stop=None):
        return self.score(start, stop) - self.segment_sum_logfac(start, stop)

    def score(self, start=None, stop=None):
        if start is None:
            start = 0
        if stop is None:
            stop = len(self.split_candidates)-1

        segment_count = self.cumsum[stop] - self.cumsum[start]
        shifted_segment_count = segment_count + self.alpha
        segment_length = self.split_candidates[stop] - self.split_candidates[start]
        shifted_segment_length = segment_length + self.beta
        add = log_gamma(shifted_segment_count)
        sub = shifted_segment_count * np.log(shifted_segment_length)
        return add - sub + self.constant

    def all_suffixes_log_marginal_likelyhoods(self, stop):
        segment_sum_logfac_vec = self.logfac_cumsum[stop] - self.logfac_cumsum[0:stop]
        return self.all_suffixes_score(self, stop) - segment_sum_logfac_vec

    # marginal likelihoods for segments [i, stop] for all i < stop
    def all_suffixes_score(self, stop):
        # segment_count + alpha
        shifted_segment_count_vec = (self.alpha + self.cumsum[stop]) - self.cumsum[0:stop]
        # it's more efficient to add up numbers, then add result to vector
        #   (alternative is to add numbers to a vector one-by-one)

        # segment_length + beta
        shifted_segment_length_vec = (self.beta + self.split_candidates[stop]) - self.split_candidates[:stop]

        add_vec = log_gamma(shifted_segment_count_vec, max_value=(self.alpha + self.cumsum[stop]))
        sub_vec = shifted_segment_count_vec * cached_log(shifted_segment_length_vec, max_value=(self.beta + self.split_candidates[stop]))

        return (add_vec - sub_vec) + self.constant

def compute_score_from_splits(counts, splits, scorer_factory):
    scorer = scorer_factory(counts)
    sum_scores = 0
    for start, stop in zip(splits, splits[1:]):
        sum_scores += scorer.score(start, stop)
    sum_scores += scorer.score(start = splits[-1])
    return sum_scores

def split_on_two_segments_or_not(counts, scorer_factory):
    scorer = scorer_factory(counts)
    best_score = scorer.score(0, len(counts))
    split_point = 0
    for i in range(len(counts)):
        current_score = scorer.score(stop=i)
        current_score += scorer.score(start=i)
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

class SquareSplitter:
    def __init__(self,
                length_regularization_multiplier=0,
                length_regularization_function=lambda x: x,
                split_number_regularization_multiplier=0,
                split_number_regularization_function=lambda x: x):
        self.length_regularization_multiplier = length_regularization_multiplier
        self.split_number_regularization_multiplier = split_number_regularization_multiplier
        self.length_regularization_function = length_regularization_function
        self.split_number_regularization_function = split_number_regularization_function

    def split(self, counts, score_computer_factory, split_candidates=None):
        if split_candidates is None:
            split_candidates = np.arange(len(counts)+1)
        else:
            if split_candidates[-1] == len(counts):
                split_candidates = np.array(split_candidates, dtype=int)
            else:
                split_candidates = np.append(np.array(split_candidates, dtype=int),
                                             [len(counts)])

        score_computer = score_computer_factory(counts,
                                                split_candidates=split_candidates)
        split_scores = np.zeros(len(split_candidates))
        right_borders = np.zeros(len(split_candidates), dtype=int)
        num_splits = np.zeros(len(split_candidates))
        split_scores[0] = 0
        split_scores[1] = score_computer.score(0, 1)

        for i, split in enumerate(split_candidates[1:], 1):
            score_if_split_at_ = score_computer.all_suffixes_score(i).astype('float64')
            score_if_split_at_ += split_scores[:i]

            if self.split_number_regularization_multiplier != 0:
                score_if_split_at_[:] -= self.split_number_regularization_multiplier*(
                    self.split_number_regularization_function(num_splits[:i]+1))
                score_if_split_at_[0] += self.split_number_regularization_multiplier*self.split_number_regularization_function(1)

            if self.length_regularization_multiplier != 0:
                last_segment_length_regularization = (self.length_regularization_multiplier*
                                                      self.length_regularization_function(
                                                          split - split_candidates[:i]))
                score_if_split_at_[:] -= last_segment_length_regularization[:i]

            right_borders[i] = np.argmax(score_if_split_at_)
            if right_borders[i] != 0:
                num_splits[i] = num_splits[right_borders[i]] + 1
            split_scores[i] = score_if_split_at_[right_borders[i]]

        return split_scores[-1], [split_candidates[i] for i in collect_split_points(right_borders[1:])]


class NotZeroSplitter:
    def __init__(self, base_splitter):
        self.base_splitter = base_splitter

    def split(self, counts, score_computer_factory, split_candidates=None):
        if np.all(counts == 0):
            logger.info('Window contains just zeros. Skipping.')
            scorer = score_computer_factory(counts, split_candidates)
            return scorer.score(), [0, len(counts)]
        logger.info('Not zeros. Spliting.')
        return self.base_splitter.split(counts, score_computer_factory, split_candidates=split_candidates)


class NotConstantSplitter:
    def __init__(self, base_splitter):
        self.base_splitter = base_splitter

    def get_non_constant_split_candidates(self, counts, split_candidates=None):
        if split_candidates is None:
            split_candidates = np.insert(np.where(counts[:-1] != counts[1:])[0] + 1, 0, 0)
        else:
            if split_candidates[0] == 0:
                split_candidates = split_candidates[1:]
            split_candidates = np.insert(
                                   split_candidates[counts[split_candidates] != counts[split_candidates-1]],
                                   0, 0)
        return split_candidates

    def split(self, counts, score_computer_factory, split_candidates=None):
        split_candidates = self.get_non_constant_split_candidates(counts, split_candidates)
        return self.base_splitter.split(counts, score_computer_factory, split_candidates=split_candidates)


class SlidingWindowSplitter:
    def __init__(self, window_size, window_shift, base_splitter):
        self.window_size = window_size
        self.window_shift = window_shift
        self.base_splitter = base_splitter

    def split(self, counts, score_computer_factory):
        split_points = set([0])
        for start in range(0, len(counts), self.window_shift):
            logger.info('Processing window at start:%d (%.2f %s of chrom)' % (start, 100*start/float(len(counts)), '%'))
            stop = min(start+self.window_size, len(counts))
            segment_score, segment_split_points = self.base_splitter.split(counts[start:stop], score_computer_factory)
            split_points.update([start+s for s in segment_split_points])
        logger.info('Final split of chromosome with %d split points' % (len(split_points)))
        return self.base_splitter.split(counts, score_computer_factory, split_candidates=sorted(split_points))


class RoundSplitter:
    def __init__(self, window_size, window_shift, base_splitter, num_rounds=None):
        self.window_size = window_size
        self.window_shift = window_shift
        self.num_rounds = num_rounds
        self.base_splitter = base_splitter

    def split(self, counts, score_computer_factory):
        possible_split_points = np.arange(len(counts)+1)
        if self.num_rounds is None:
            num_rounds = len(counts)
        else:
            num_rounds = self.num_rounds
        for round_ in range(num_rounds):
            new_split_points = set([0])
            logger.info('Starting split round %d, num_candidates %d' % (round_, len(possible_split_points)))
            for start_index in range(0, len(possible_split_points)-1, self.window_shift):
                stop_index = min(start_index+self.window_size, len(possible_split_points)-1)
                start = possible_split_points[start_index]
                stop = possible_split_points[stop_index]
                logger.info('Round:%d Splitting window [%d, %d], %d points, (%.2f %s of round complete)' % (
                    round_, start, stop, len(possible_split_points[start_index:stop_index]),
                    float(start_index)/len(possible_split_points)*100, '%'))
                segment_score, segment_split_points = self.base_splitter.split(
                    counts[start:stop], score_computer_factory,
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

def parse_bedgraph(filename):
    '''
        yields pointwise profiles grouped by chromosome in form (chrom, profile, chromosome_start)
    '''
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


def split_bedgraph(in_filename, out_filename, scorer_factory, splitter):
    with open(out_filename, 'w') as outfile:
        logger.info('Reading input file %s' % (in_filename))
        for chrom, counts, chrom_start in parse_bedgraph(in_filename):
            logger.info('Starting chrom %s of length %d' % (chrom, len(counts)))
            score, splits = splitter.split(counts, scorer_factory)
            sum_logfac = scorer_factory(counts, split_candidates=splits).segment_sum_logfac()
            log_likelyhood = compute_score_from_splits(counts, splits, scorer_factory) - sum_logfac
            logger.info('chrom %s finished, score %f, number of splits %d. '
                        'Log likelyhood: %f.'% (chrom, score, len(splits), log_likelyhood))
            scorer = scorer_factory(counts, splits+[len(counts)])
            logger.info('Starting output of chrom %s' % (chrom))
            for i, (start, stop) in enumerate(zip(splits, splits[1:])):
                outfile.write('%s\t%d\t%d\t%f\t%d\t%f\n' % (chrom,
                                                            start+chrom_start,
                                                            stop+chrom_start,
                                                            counts[start:stop].mean(),
                                                            stop-start,
                                                            scorer.log_marginal_likelyhood(i, i+1)))
            outfile.write('%s\t%d\t%d\t%f\t%d\t%f\n' % (chrom,
                                                        splits[-1]+chrom_start,
                                                        len(counts)+chrom_start,
                                                        counts[splits[-1]:].mean(),
                                                        len(counts)-splits[-1],
                                                        scorer.log_marginal_likelyhood(len(splits)-1)))

def get_argparser():
    argparser = argparse.ArgumentParser(
        prog = "Pasio",
        description = '''
Example usage, simpliest for practical cases:
python pasio.py
      --bedgraph ~/<PATH TO INPUT bed.Graph FILE> -o ~/<WAY TO OUTPUT bedGraph FILE>
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
    scorer_factory = lambda counts, split_candidates=None: LogMarginalLikelyhoodComputer(
        counts, args.alpha, args.beta, split_candidates = split_candidates)

    length_regularization_functions = {
         'none': lambda x:x
        ,'revlog': lambda x:1/np.log(x+1)
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

    square_splitter = SquareSplitter(
        length_regularization_multiplier=length_regularization_multiplier,
        length_regularization_function=length_regularization_function,
        split_number_regularization_multiplier=split_number_regularization_multiplier,
        split_number_regularization_function=lambda x:x)

    if args.no_split_constant:
        square_splitter = NotConstantSplitter(base_splitter = square_splitter)    

    if args.algorithm == 'slidingwindow':
        splitter = SlidingWindowSplitter(window_size=args.window_size,
                                         window_shift=args.window_shift,
                                         base_splitter=NotZeroSplitter(square_splitter))
    elif args.algorithm == 'exact':
        splitter = square_splitter
    elif args.algorithm == 'rounds':
        splitter = RoundSplitter(window_size=args.window_size,
                                 window_shift=args.window_shift,
                                 base_splitter=NotZeroSplitter(square_splitter),
                                 num_rounds=args.num_rounds)

    logger.info('Starting Pasio with args'+str(args))
    split_bedgraph(args.bedgraph, args.out_bedgraph, scorer_factory, splitter)

