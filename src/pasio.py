import numpy as np

def log_marginal_likelyhood(counts, alpha, beta):
    num_counts = len(counts)
    counts = counts[counts > 0]
    sum_counts = counts.sum()
    add1 = np.log(np.arange(1, sum_counts+alpha)).sum()
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

#def splits_from_matrix(matrix):
#    assert matrix.shape[0] == matrix.shape[1]
#    for i in range(matrix.shape[0]):
#        pass
#
#def get_splits_from_split_points(splits_at_points, start_point):
#    current_point = start_point
#    split_points = [len(splits_at_points)]
#    for i in range(len(splits_at_points))[::-1]:
#        if splits_at_points[current_point, i]:
#            current_point-=1
#            split_points.append(i)
#    assert current_point == 0
#    split_points.append(0)
#    return np.array(split_points[::-1])
#
#def split_into_segments_cubic(counts, score_fn):
#    matrix = np.zeros((len(counts), len(counts)))
#    splits_at_points = np.zeros((len(counts), len(counts)))
#    for i in range(0, len(counts)):
#        matrix[0,i] = score_fn(counts[:(i+1)])
#        for j in range(1, i+1):
#            score_if_no_split = counts[j:i+1]
#            score_if_split = counts[j-1:i+1]
#            if score_if_split > score_if_no_split:
#                matrix[j,i] = score_if_split
#                splits_at_points[i,j] = 1
#            else:
#                matrix[j,i] = score_if_no_split
#                splits_at_points[i,j] = 0
#    best_number_of_splits = np.argmin(matrix[-1,:])
#    splits = get_splits_from_split_points(split_at_point, best_number_of_splits)
#    return best_split_indexes

def split_into_segments_square(counts, score_fn):
    split_scores = np.zeros((len(counts),))
    split_points = [[0]]
    split_scores[0] = score_fn(counts[0:1])
    for i in range(2, len(counts)+1):
        best_i_score = score_fn(counts[0:i])
        best_i_split = [0]
        #print 'i=',i
        for j in range(1, i):
            #print 'j=',j,counts[j:i]
            score_if_split_at_j = score_fn(counts[j:i])+split_scores[j-1]
            if score_if_split_at_j > best_i_score:
                best_i_score = score_if_split_at_j
                best_i_split = split_points[j-1]+[j]
        #print '---------------'
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
    counts = np.concatenate([np.random.poisson(15, 100), np.random.poisson(20, 100), np.random.poisson(15, 100)])
    points = split_into_segments_square(counts, score_fn)
    print points
