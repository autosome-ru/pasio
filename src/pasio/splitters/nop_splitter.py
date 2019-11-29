import numpy as np

# Doesn't change list of split candidates. But for a given list of split candidates it can calculate score
# Designed to make it possible to convert reducer into splitter with `SpliiterCombiner(reducer, NopSplitter())`
class NopSplitter:
    def __init__(self, scorer_factory):
        self.scorer_factory = scorer_factory

    def scorer(self, counts, split_candidates):
        return self.scorer_factory(counts, split_candidates)

    def reduce_candidate_list(self, counts, split_candidates):
        return split_candidates

    def split(self, counts, split_candidates):
        scores = self.scorer(counts, split_candidates).scores()
        final_score = np.sum(scores)
        return (final_score, split_candidates)
