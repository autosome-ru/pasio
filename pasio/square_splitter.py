from future.builtins import range
import numpy as np

class SquareSplitter:
    def __init__(self, scorer_factory,
                length_regularization_multiplier=0,
                length_regularization_function=lambda x: x,
                split_number_regularization_multiplier=0,
                split_number_regularization_function=lambda x: x):
        self.scorer_factory = scorer_factory
        self.length_regularization_multiplier = length_regularization_multiplier
        self.split_number_regularization_multiplier = split_number_regularization_multiplier
        self.length_regularization_function = length_regularization_function
        self.split_number_regularization_function = split_number_regularization_function

    def scorer(self, counts, split_candidates):
        return self.scorer_factory(counts, split_candidates)

    def reduce_candidate_list(self, counts, split_candidates):
        score, splits = self.split(counts, split_candidates)
        return splits

    def split(self, counts, split_candidates):
        if self.split_number_regularization_multiplier == 0 and self.length_regularization_multiplier == 0:
            return self.split_without_normalizations(counts, split_candidates)
        else:
            return self.split_with_normalizations(counts, split_candidates)

    def split_with_normalizations(self, counts, split_candidates):
        score_computer = self.scorer(counts, split_candidates)
        num_split_candidates = len(split_candidates)

        prefix_scores = np.empty(num_split_candidates)
        prefix_scores[0] = 0

        previous_splits = np.empty(num_split_candidates, dtype=int)
        previous_splits[0] = 0

        num_splits = np.zeros(num_split_candidates)

        for prefix_end in range(1, num_split_candidates):
            score_if_last_split_at = score_computer.all_suffixes_self_score(prefix_end)
            score_if_last_split_at += prefix_scores[:prefix_end]

            if self.split_number_regularization_multiplier != 0:
                number_regularization = self.split_number_regularization_function(num_splits[:prefix_end] + 1)
                score_if_last_split_at -= self.split_number_regularization_multiplier * number_regularization
                score_if_last_split_at[0] += self.split_number_regularization_multiplier * self.split_number_regularization_function(1)

            if self.length_regularization_multiplier != 0:
                length_regulatization = self.length_regularization_function(split_candidates[prefix_end] - split_candidates[:prefix_end])
                last_segment_length_regularization = self.length_regularization_multiplier * length_regulatization
                score_if_last_split_at -= last_segment_length_regularization[:prefix_end]

            optimal_last_split = np.argmax(score_if_last_split_at)
            previous_splits[prefix_end] = optimal_last_split

            if optimal_last_split != 0:
                num_splits[prefix_end] = num_splits[optimal_last_split] + 1

            prefix_scores[prefix_end] = score_if_last_split_at[optimal_last_split] + score_computer.segment_creation_cost

        split_indices = SquareSplitter.collect_split_points(previous_splits)
        split_positions = split_candidates[split_indices]
        return prefix_scores[-1], split_positions

    def split_without_normalizations(self, counts, split_candidates):
        score_computer = self.scorer(counts, split_candidates)
        num_split_candidates = len(split_candidates)

        # prefix_scores[i] is the best score of prefix [0; i)
        prefix_scores = np.empty(num_split_candidates)
        prefix_scores[0] = 0

        # `previous_splits[i]` (later `ps`) is a position of the last split
        # in the best segmentation of prefix [0, i) which is not the end of the prefix
        # i.e. this prefix segmentation looks like `...... ps - 1) [ps ... i - 1)`
        previous_splits = np.empty(num_split_candidates, dtype=int)
        previous_splits[0] = 0

        # find the score and previous point of the best segmentation of the prefix [0, prefix_end)
        # such that the last split in segmentation is prefix_end (split is between `prefix_end - 1` and `prefix_end`)
        # (indexation runs over split candidates, not over all points)
        for prefix_end in range(1, num_split_candidates):
            # score consists of (a) score of the last segment
            score_if_last_split_at = score_computer.all_suffixes_self_score(prefix_end)
            #                   (b) score of the prefix before the last segment
            score_if_last_split_at += prefix_scores[:prefix_end]

            optimal_last_split = np.argmax(score_if_last_split_at)
            previous_splits[prefix_end] = optimal_last_split

            #                   (c) and constant penalty for segment creation
            prefix_scores[prefix_end] = score_if_last_split_at[optimal_last_split] + score_computer.segment_creation_cost

        # reminder: splits indexing is over split candidates, not contig positions
        split_indices = SquareSplitter.collect_split_points(previous_splits)
        # but we want to return contig positions of these splits
        split_positions = split_candidates[split_indices]
        return prefix_scores[-1], split_positions

    @staticmethod
    def collect_split_points(previous_splits):
        split_point = len(previous_splits) - 1
        split_points_collected = [split_point]
        while split_point != 0:
            split_point = previous_splits[split_point]
            split_points_collected.append(split_point)
        return split_points_collected[::-1]
