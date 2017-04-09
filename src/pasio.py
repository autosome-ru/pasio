import numpy as np
import array

class LogFactorialComputer:
    def approximate_log_factorial(self, x):
        return (x-1./2)*np.log(x) - x + (1./2)*np.log(2*np.pi) + 1./(12*x)
    def __init__(self):
        self.precomputed = np.zeros(4096)
        for i in range(4096):
            self.precomputed[i] = np.log(np.arange(1, i)).sum()
    def __call__(self, x):
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
        self.logfac_cumsum = np.zeros(len(counts))
        self.logfac_cumsum[counts>0] = log_factorial.approximate_log_factorial(
            np.cumsum(np.log(counts[counts>0])))
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

        add1_vec = np.zeros(stop)
        cumsum_large = counts_cumsum+self.alpha >= 4096
        add1_vec[~cumsum_large] = log_factorial.precomputed[counts_cumsum[~cumsum_large]+self.alpha]
        add1_vec[cumsum_large] = log_factorial.approximate_log_factorial(
                                     counts_cumsum[cumsum_large]+self.alpha)

        sub2_vec = (counts_cumsum+self.alpha+1)*np.log(stop-np.arange(stop)+self.beta)

        suffixes_score = add1_vec - logfac_counts_cumsum - sub2_vec

        return suffixes_score


def split_on_two_segments_or_not(counts, score_computer):
    best_score = score_computer(0, len(counts))
    split_point = None
    for i in range(len(counts)):
        current_score = score_computer(stop=i)
        current_score += score_computer(start=i)
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

def split_into_segments_square(counts, score_computer):
    split_scores = np.zeros((len(counts),))
    right_borders = np.zeros((len(counts),), dtype=int)
    split_scores[0] = score_computer(0, 1)
    for i in range(2, len(counts)+1):
        score_if_split_at_ = score_computer.all_suffixes_score(i).astype('float64')
        score_if_split_at_[1:] += split_scores[:i-1]
        right_borders[i-1] = np.argmax(score_if_split_at_)
        split_scores[i-1] = score_if_split_at_[right_borders[i-1]]
    return split_scores[-1], collect_split_points(right_borders)

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
    counts = np.concatenate([np.random.poisson(4096, 10000), np.random.poisson(20, 10000)])

    scorer = LogMarginalLikelyhoodComputer(counts, 1, 1)
    points = split_into_segments_square(counts, scorer)
    print points
