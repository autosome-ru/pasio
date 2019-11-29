from future.builtins import range
import numpy as np
from ..logging import logger, logging_filter

class RoundReducer:
    def __init__(self, base_reducer, num_rounds=None):
        self.base_reducer = base_reducer
        self.num_rounds = num_rounds

    def reduce_candidate_list(self, counts, split_candidates):
        if self.num_rounds is None:
            num_rounds = len(counts)
        else:
            num_rounds = self.num_rounds
        num_rounds = max(1, num_rounds)

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
