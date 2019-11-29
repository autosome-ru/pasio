import argparse
from .process_bedgraph import split_bedgraph
from .logging import logger
from .splitters.default_splitters import configure_splitter

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
    argparser.add_argument('--split-number-regularization', type=float, default=0,
                           help="Penalty multiplier for each split")
    argparser.add_argument('--length-regularization', type=float, default=0,
                           help="Penalty multiplier for length of each segment")
    argparser.add_argument('--length-regularization-function', type=str, default='none',
                           choices=['none', 'revlog', 'neglog'],
                           help='''Penalty function for length of segments.:
                           none: no length regulatization
                           revlog: 1/log(1+l)
                           ''')
    argparser.add_argument('--window-size', type=int,
                           help="Size of window fo split with exact algorithm")
    argparser.add_argument('--window-shift', type=int,
                           help = "Shift in one step")
    argparser.add_argument('--num-rounds', type=int,
                           help = '''Number of rounds for round algorithm.
                           If not set, run until no split points removed''')
    argparser.add_argument('--no-split-constant', action='store_true',
                           help = '''[experimental] If set, won't put splits between constant counts''')
    argparser.add_argument('--no-split-zeros', action='store_true',
                           help = '''If set, won't put splits at non-covered intervals''')
    argparser.add_argument('--split-at-gaps', action='store_true',
                           help = 'By default gaps between intervals are filled with zeros.\n' +
                                  'Split at gaps overrides this behavior so that\n' +
                                  'non-adjacent intervals are segmented independently.')
    return argparser

def main():
    argparser = get_argparser()
    args = argparser.parse_args()
    logger.info("Pasio:"+ str(args))
    splitter = configure_splitter(**vars(args))
    logger.info('Starting Pasio with args'+str(args))
    split_bedgraph(args.bedgraph, args.out_bedgraph, splitter, split_at_gaps=args.split_at_gaps)
