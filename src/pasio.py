import numpy as np

class LogFactorialComputer:
    def approximate_log_factorial(self, x):
        return (x-1./2)*np.log(x)-x+(1./2)*np.log(2*np.pi)+\
            1./(12*x) - 1./(360*x**3)
    def __init__(self):
        self.precomputed = {}
        for i in range(256):
            self.precomputed[i] = np.log(np.arange(1, i)).sum()
    def __call__(self, x):
        if x < 256:
            return self.precomputed[x]
        else:
            return self.approximate_log_factorial(x)

log_factorial = LogFactorialComputer()

def log_marginal_likelyhood(counts, alpha, beta):
    num_counts = len(counts)
    counts = counts[counts > 0]
    sum_counts = counts.sum()
    add1 = log_factorial(sum_counts+alpha)
    sub1 = np.log(counts).sum()
    sub2 = (sum_counts+alpha+1)*np.log(num_counts+beta)
    return add1-sub1-sub2


def split_on_two_segments_or_not(counts, score_fn):
    best_score = score_fn(counts)
    split_point = None
    for i in range(len(counts)):
        current_score = score_fn(counts[:i])
        current_score += score_fn(counts[i:])
        if current_score > best_score:
            split_point = i
            best_score = current_score
    return best_score, split_point


def split_into_segments_square(counts, score_fn):
    split_scores = np.zeros((len(counts),))
    split_points = [[0]]
    split_scores[0] = score_fn(counts[0:1])
    for i in range(2, len(counts)+1):
        best_i_score = score_fn(counts[0:i])
        best_i_split = [0]
        for j in range(1, i):
            score_if_split_at_j = score_fn(counts[j:i])+split_scores[j-1]
            if score_if_split_at_j > best_i_score:
                best_i_score = score_if_split_at_j
                best_i_split = split_points[j-1]+[j]
        split_scores[i-1] = best_i_score
        split_points.append(best_i_split)
    return split_scores[-1], split_points[-1]
    #return split_scores, split_points

if __name__ == '__main__':
    score_fn = lambda c:1
    split_into_segments_square('abcd', score_fn)
    #exit(0)
    score_fn = lambda c: log_marginal_likelyhood(c, 1, 1)
    score_fn2 = lambda c: log_marginal_likelyhood(c, 100, 10)

    counts = np.concatenate([np.random.poisson(15, 100), np.random.poisson(20, 100)])

    two_splt = split_on_two_segments_or_not(counts, score_fn)
    print two_splt
    print split_on_two_segments_or_not(counts[two_splt[1]:], score_fn)
    print split_on_two_segments_or_not(counts[:two_splt[1]], score_fn)
    points = split_into_segments_square(counts, score_fn)
    #points2 = split_into_segments_square(counts, score_fn2)
    print points
    #print points2
    print '-------------'
    counts = np.concatenate([np.random.poisson(15, 100), np.random.poisson(1, 50), np.random.poisson(15, 100)])
    points = split_into_segments_square(counts, score_fn)
    print points
