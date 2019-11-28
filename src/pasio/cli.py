import argparse
import sys
import numpy as np

from .log_marginal_likelyhood import LogMarginalLikelyhoodComputer, ScorerFactory
from .square_splitter import SquareSplitter
from .sliding_window_reducer import SlidingWindow, SlidingWindowReducer
from .round_reducer import RoundReducer
from .constants_reducer import NotZeroReducer, NotConstantReducer
from .reducer_combiner import ReducerCombiner
from .nop_splitter import NopSplitter
from .process_bedgraph import split_bedgraph
from .logging import logger

def get_argparser():
    argparser = argparse.ArgumentParser(
        prog = "Pasio",
        description = '''
Example usage, simpliest for practical cases:
python pasio.py
      --bedgraph ~/<PATH TO INPUT bed.Graph FILE> -o ~/<PATH TO OUTPUT bedGraph FILE>
      --alpha 5 --beta 1 --algorithm rounds
      --window_shift 1250 --window_size 2500
''',
        formatter_class=argparse.RawTextHelpFormatter)
    argparser.add_argument('--algorithm',
                           choices=['slidingwindow', 'exact', 'rounds'],
                           required=True,
                           help="Algorithm to use")
    argparser.add_argument('--bedgraph', required=True,
                           help="Input bedgraph path")
    argparser.add_argument('-o', '--out_bedgraph', help="Output begraph path",
                           required=True)
    argparser.add_argument('--alpha', type=float, required=True,
                           help="alpha parameter of gamma distribution")
    argparser.add_argument('--beta', type=float, required=True,
                           help="beta parameter of gamma distribution")
    argparser.add_argument('--split_number_regularization', type=float, default=0,
                           help="Penalty multiplier for each split")
    argparser.add_argument('--length_regularization', type=float, default=0,
                           help="Penalty multiplier for length of each segment")
    argparser.add_argument('--length_regularization_function', type=str, default='none',
                           choices=['none', 'revlog', 'neglog'],
                           help='''Penalty function for length of segments.:
                           none: no length regulatization
                           revlog: 1/log(1+l)
                           ''')
    argparser.add_argument('--window_size', type=int,
                           help="Size of window fo split with exact algorithm")
    argparser.add_argument('--window_shift', type=int,
                           help = "Shift in one step")
    argparser.add_argument('--num_rounds', type=int,
                           help = '''Number of rounds for round algorithm.
                           If not set, run until no split points removed''')
    argparser.add_argument('--no_split_constant', action='store_true',
                           help = '''[experimental] If set, won't put splits between constant counts''')
    argparser.add_argument('--split_at_gaps', action='store_true',
                           help = 'By default gaps between intervals are filled with zeros.\n' +
                                  'Split at gaps overrides this behavior so that\n' +
                                  'non-adjacent intervals are segmented independently.')
    return argparser

def configure_splitter(args):
    if args.algorithm in ['slidingwindow', 'rounds']:
        if args.window_shift is None:
            sys.exit('Argument --window_shift is required for algorithms slidingwingow and rounds')
        if args.window_size is None:
            sys.exit('Argument --window_size is required for algorithms slidingwingow and rounds')
    scorer_factory = ScorerFactory(args.alpha, args.beta)

    length_regularization_functions = {
        'none': lambda x: x,
        'revlog': lambda x: 1 / np.log(x + 1),
    }
    length_regularization_function = length_regularization_functions[args.length_regularization_function]
    length_regularization_multiplier = args.length_regularization
    split_number_regularization_multiplier = args.split_number_regularization

    if length_regularization_multiplier != 0:
        if args.length_regularization_function == 'none':
            sys.exit('Argument --length_regularization_function is required '
                     'for length regularization multiplier %s' %
                     args.length_regularization)

    if args.length_regularization_function != 'none':
        if length_regularization_multiplier == 0:
            sys.exit('Argument --length_regularization_multiplier is required '
                     'for length legularization function %s' %
                     args.length_regularization_function)

    square_splitter = SquareSplitter(scorer_factory,
        length_regularization_multiplier=length_regularization_multiplier,
        length_regularization_function=length_regularization_function,
        split_number_regularization_multiplier=split_number_regularization_multiplier,
        split_number_regularization_function=lambda x:x)

    if args.algorithm == 'exact':
        splitter = square_splitter
    else:
        if args.no_split_constant:
            square_splitter = ReducerCombiner(NotConstantReducer(), square_splitter)
        else:
            square_splitter = ReducerCombiner(NotZeroReducer(), square_splitter)

        if args.algorithm == 'slidingwindow':
            sliding_window = SlidingWindow(window_size = args.window_size, window_shift = args.window_shift)
            reducer = SlidingWindowReducer(sliding_window = sliding_window, base_reducer = square_splitter)
            splitter = ReducerCombiner(reducer, square_splitter)
        elif args.algorithm == 'rounds':
            sliding_window = SlidingWindow(window_size = args.window_size, window_shift = args.window_shift)
            base_reducer = SlidingWindowReducer(sliding_window = sliding_window, base_reducer = square_splitter)
            reducer = RoundReducer(base_reducer = base_reducer, num_rounds = args.num_rounds)
            splitter = ReducerCombiner(reducer, NopSplitter(scorer_factory))
    return splitter

def main():
    argparser = get_argparser()
    args = argparser.parse_args()
    logger.info("Pasio:"+ str(args))
    splitter = configure_splitter(args)
    logger.info('Starting Pasio with args'+str(args))
    split_bedgraph(args.bedgraph, args.out_bedgraph, splitter, split_at_gaps=args.split_at_gaps)
