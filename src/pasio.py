import numpy as np
import array

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
    def __init__(self, counts, alpha, beta):
        self.counts = counts
        self.alpha = alpha
        self.beta = beta
        self.cumsum = np.cumsum(counts)
        self.logfac_cumsum = np.cumsum(log_factorial(counts))
    def __call__(self, start=None, stop=None):
        if start is None:
            start = 0
        if stop is None:
            stop = len(self.counts)
        num_counts = stop-start
        if stop == 0:
            sum_counts = 0
            sum_log_fac = 0
        elif start == 0:
            sum_counts = self.cumsum[stop-1]
            sum_log_fac = self.logfac_cumsum[stop-1]
        else:
            sum_counts = self.cumsum[stop-1]-self.cumsum[start-1]
            sum_log_fac = self.logfac_cumsum[stop-1]-self.logfac_cumsum[start-1]
        add1 = log_factorial(sum_counts+self.alpha)
        sub1 = sum_log_fac
        sub2 = (sum_counts+self.alpha+1)*np.log(num_counts+self.beta)
        return add1-sub1-sub2

    def all_suffixes_score(self, stop):
        suffixes_score = np.zeros(stop, dtype='float64')

        counts_cumsum = np.zeros(stop)
        counts_cumsum[1:] += self.cumsum[stop-1]-self.cumsum[0:stop-1]
        counts_cumsum[0] = self.cumsum[stop-1]

        logfac_counts_cumsum = np.zeros(stop)
        logfac_counts_cumsum[1:] += (self.logfac_cumsum[stop-1]-
                                     self.logfac_cumsum[0:stop-1])
        logfac_counts_cumsum[0] = self.logfac_cumsum[stop-1]

        counts_cumsum = np.zeros(stop, dtype='int')
        counts_cumsum[1:] += self.cumsum[stop-1]-self.cumsum[0:stop-1]
        counts_cumsum[0] = self.cumsum[stop-1]

        add1_vec = log_factorial(counts_cumsum+self.alpha)

        sub2_vec = (counts_cumsum+self.alpha+1)*np.log(stop-np.arange(stop)+self.beta)

        suffixes_score = add1_vec - logfac_counts_cumsum - sub2_vec

        return suffixes_score


def split_on_two_segments_or_not(counts, scorer_factory):
    scorer = scorer_factory(counts)
    best_score = scorer(0, len(counts))
    split_point = None
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

def split_into_segments_square(counts, score_computer_factory, regularisation_multiplyer=0,
                               regularisation_function=None, split_candidates=None):
    if regularisation_function is None:
        regularisation_function = lambda x: x
    if split_candidates is None:
        split_candidates = range(1, len(counts)+1)
    score_computer = score_computer_factory(counts)
    split_scores = np.zeros((len(counts)+1,))
    right_borders = np.zeros((len(counts)+1,), dtype=int)
    num_splits = np.zeros((len(counts)+1,))
    split_scores[0] = 0
    split_scores[1] = score_computer(0, 1)
    for i, split in enumerate(split_candidates, 1):
        print '--------------------------------'
        score_if_split_at_ = score_computer.all_suffixes_score(i).astype('float64')
        print score_if_split_at_, split_scores[:i]
        score_if_split_at_ += split_scores[:i]

        print score_if_split_at_
        score_if_split_at_[:] -= regularisation_multiplyer*(regularisation_function(num_splits[:i]+1))
        score_if_split_at_[0] += regularisation_multiplyer*regularisation_function(1)
        print score_if_split_at_, num_splits[:i]
        right_borders[i] = np.argmax(score_if_split_at_)
        if right_borders[i] != 0:
            num_splits[i] = num_splits[right_borders[i]] + 1
        split_scores[i] = score_if_split_at_[right_borders[i]]
        print counts[0:i], num_splits[i], split_scores[i], right_borders[i]
        print num_splits[:]
        print split_scores[:]
    return split_scores[-1], collect_split_points(right_borders[1:])

def parse_bedgrah(filename):
    chromosomes = {}
    with open(filename) as bedgrpah_file:
        for line in bedgrpah_file:
            if line.strip() == '':
                continue
            chrom, start, stop, coverage = line.strip().split()
            start = int(start)
            stop = int(stop)
            coverage = int(coverage)
            if chrom not in chromosomes:
                chromosomes[chrom] = array.array('l')
            chromosomes[chrom].extend([coverage]*(stop-start))
    for chrom in chromosomes:
        chromosomes[chrom] = np.array(chromosomes[chrom])
    return chromosomes

if __name__ == '__main__':
    np.random.seed(1024)
    counts = np.concatenate([np.random.poisson(4096, 1000), np.random.poisson(20, 1000)])

    scorer_factory = lambda counts: LogMarginalLikelyhoodComputer(counts, 1, 1)
    points = split_into_segments_square(counts, scorer_factory)
    print points
