class ReducerCombiner:
    def __init__(self, *reducers):
        self.reducers = reducers

    def reduce_candidate_list(self, counts, split_candidates):
        for reducer in self.reducers:
            split_candidates = reducer.reduce_candidate_list(counts, split_candidates)
        return split_candidates

    def split(self, counts, split_candidates):
        splitter = self.reducers[-1]
        if hasattr(splitter, 'split') and callable(splitter.split):
            for reducer in self.reducers[:-1]:
                split_candidates = reducer.reduce_candidate_list(counts, split_candidates)
            return splitter.split(counts, split_candidates)
        else:
            raise Exception('This ReducerCombiner has no splitter at the end of pipeline. Splitting no possible')

    def scorer(self, counts, split_candidates):
        splitter = self.reducers[-1]
        if hasattr(splitter, 'scorer') and callable(splitter.scorer):
            return splitter.scorer(counts, split_candidates)
        else:
            raise Exception('This ReducerCombiner has no splitter at the end of pipeline. Scoring not possible. Consider use of NopSplitter')
