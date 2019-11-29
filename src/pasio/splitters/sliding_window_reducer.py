from future.builtins import range
import numpy as np
from ..logging import logger, logging_filter

class SlidingWindowReducer:
    def __init__(self, sliding_window, base_reducer):
        self.sliding_window = sliding_window
        self.base_reducer = base_reducer

    def reduce_candidates_in_window(self, counts, candidates_in_window):
        start = candidates_in_window[0]
        stop  = candidates_in_window[-1]
        logging_filter.put_to_context('window', '[%d, %d)' % (start, stop))

        candidates_relative_pos = candidates_in_window - start
        reduced_splits_relative_pos = self.base_reducer.reduce_candidate_list(
            counts[start:stop], candidates_relative_pos)
        return reduced_splits_relative_pos + start

    # Single round of candidate list reduction
    def reduce_candidate_list(self, counts, split_candidates):
        new_split_candidates_set = set([0, len(counts)])
        for (candidates_in_window, completion) in self.sliding_window.windows(split_candidates):
            reduced_splits = self.reduce_candidates_in_window(counts, candidates_in_window)
            new_split_candidates_set.update(reduced_splits)
            logger.info('Sliding (completion: %.2f %%): %d --> %d split-points' % (
                100 * completion, len(candidates_in_window), len(reduced_splits)))
        logging_filter.remove_from_context('window')
        return np.array(sorted(new_split_candidates_set))
