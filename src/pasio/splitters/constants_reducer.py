import numpy as np
from ..logging import logger

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
