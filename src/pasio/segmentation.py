import numpy as np
from .logging import logger
from .dto.intervals import ScoredInterval

def segments_with_scores(profile, splitter):
    logger.info('Starting splitting profile of length %d' % len(profile))
    counts = np.array(profile)
    split_candidates = np.arange(len(counts) + 1)
    score, splits = splitter.split(counts, split_candidates)
    scorer = splitter.scorer(counts, splits)
    sum_logfac = scorer.total_sum_logfac()
    log_likelyhood = score - sum_logfac
    logger.info('Splitting finished, score %f, number of splits %d. '
                'Log likelyhood: %f.'% (score, len(splits), log_likelyhood))
    segment_iterator  = zip(splits[:-1], splits[1:],
                            scorer.mean_counts(),
                            scorer.log_marginal_likelyhoods())
    logger.info('Scores calculated')
    for (start, stop, mean_count, log_marginal_likelyhood) in segment_iterator:
        yield ScoredInterval(start, stop, mean_count, log_marginal_likelyhood)
