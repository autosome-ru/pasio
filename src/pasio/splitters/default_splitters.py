from .square_splitter import SquareSplitter
from .nop_splitter import NopSplitter
from .constants_reducer import NotConstantReducer, NotZeroReducer
from .sliding_window_reducer import SlidingWindowReducer
from .round_reducer import RoundReducer
from .reducer_combiner import ReducerCombiner

from ..dto.sliding_window import SlidingWindow
from ..log_marginal_likelyhood import ScorerFactory

# all unknown arguments (kwargs) will be ignored
def configure_splitter(alpha=1, beta=1, algorithm='rounds',
                       window_size=2500, window_shift=1250, num_rounds=None,
                       split_constraints='constants',
                       length_regularization_function='none', length_regularization=0, split_number_regularization=0,
                       **kwargs):
    if algorithm not in ['exact', 'slidingwindow', 'rounds']:
        raise ValueError('Algorithm should be one of exact/slidingwindow/rounds')

    if algorithm in ['slidingwindow', 'rounds']:
        if window_shift is None:
            raise ValueError('Argument window_shift is required for algorithms slidingwingow and rounds')
        if window_size is None:
            raise ValueError('Argument window_size is required for algorithms slidingwingow and rounds')
        sliding_window = SlidingWindow(window_size=window_size, window_shift=window_shift)

    if (length_regularization != 0) and (length_regularization_function == 'none'):
        raise ValueError('Argument --length_regularization_function is required '
                         'for length regularization multiplier %s'
                         % length_regularization)
    if (length_regularization_function != 'none') and (length_regularization == 0):
        raise ValueError('Argument --length_regularization_multiplier is required '
                         'for length legularization function %s' %
                         length_regularization_function)

    REGULARIZATION_FUNCTIONS = {
        'none': lambda x: x,
        'revlog': lambda x: 1 / np.log(x + 1),
    }

    scorer_factory = ScorerFactory(alpha, beta)
    square_splitter = SquareSplitter(scorer_factory,
        length_regularization_multiplier=length_regularization,
        length_regularization_function=REGULARIZATION_FUNCTIONS[length_regularization_function],
        split_number_regularization_multiplier=split_number_regularization,
        split_number_regularization_function=REGULARIZATION_FUNCTIONS['none'])

    if algorithm == 'exact':
        return square_splitter

    # no_split_constant is a more aggressive optimization strategy than no_split_zeros
    if split_constraints == 'constants':
        base_splitter = ReducerCombiner(NotConstantReducer(), square_splitter)
    elif split_constraints == 'zeros':
        base_splitter = ReducerCombiner(NotZeroReducer(), square_splitter)
    elif split_constraints == 'none':
        base_splitter = square_splitter
    else:
        raise ValueError('Unknown split_constraints option `%s`' % split_constraints)

    sliding_window_reducer = SlidingWindowReducer(sliding_window=sliding_window, base_reducer=base_splitter)
    if algorithm == 'slidingwindow':
        return ReducerCombiner(sliding_window_reducer, splitter)
    elif algorithm == 'rounds':
        reducer = RoundReducer(base_reducer=sliding_window_reducer, num_rounds=num_rounds)
        return ReducerCombiner(reducer, NopSplitter(scorer_factory))
