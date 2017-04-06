import numpy as np

class LogFactorialComputer:
    def approximate_log_factorial(self, x):
        return (x-1./2)*np.log(x)-x+(1./2)*np.log(2*np.pi)+\
            1./(12*x) - 1./(360*x**3)
    def __init__(self):
        self.precomputed = {}
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
        self.logcumsum = np.cumsum(np.log(counts[counts>0]))
    def __call__(self, start=None, stop=None):
        if start is None:
            start = 0
        if stop is None:
            stop = len(self.counts)
        num_counts = stop-start
        if stop == 0:
            sum_counts = 0
            sum_logs = 0
        elif start == 0:
            sum_counts = self.cumsum[stop-1]
            sum_logs = self.logcumsum[stop-1]
        else:
            sum_counts = self.cumsum[stop-1]-self.cumsum[start-1]
            sum_logs = self.logcumsum[stop-1]-self.logcumsum[start-1]
        add1 = log_factorial(sum_counts+self.alpha)
        sub1 = sum_logs
        sub2 = (sum_counts+self.alpha+1)*np.log(num_counts+self.beta)
        return add1-sub1-sub2


#def log_marginal_likelyhood(counts, alpha, beta):
#    computer = LogMarginalLikelyhoodComputer(counts, alpha, beta)
#    return computer.log_marginal_likelyhood(0, len(counts))

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
        best_i_score = score_computer(0, i)
        best_i_split = 0
        for j in range(1, i):
            score_if_split_at_j = score_computer(j, i)+split_scores[j-1]
            if score_if_split_at_j > best_i_score:
                best_i_score = score_if_split_at_j
                best_i_split = j
        split_scores[i-1] = best_i_score
        right_borders[i-1] = best_i_split
    return split_scores[-1], collect_split_points(right_borders)

if __name__ == '__main__':

    counts = np.concatenate([np.random.poisson(15, 1000), np.random.poisson(20, 1000)])

    scorer = LogMarginalLikelyhoodComputer(counts, 1, 1)
    points = split_into_segments_square(counts, scorer)
    print points
