import sys
import argparse
from .process_bedgraph import split_bedgraph
from .logging import logger
import logging
from .splitters.default_splitters import configure_splitter
from .version import __version__

def get_argparser():
    argparser = argparse.ArgumentParser(
        prog = "pasio",
        description = 'PASIO produces segmentation of coverage profile into regions with uniform coverage\n',
        usage='pasio input.bedgraph[.gz] [options]',
        formatter_class=argparse.RawTextHelpFormatter)
    argparser.add_argument('bedgraph', metavar='input_bedgraph',
                            help="Input file in bedgraph format\n"
                                 "(it can be gzipped)")
    argparser.add_argument('--alpha', '-a', type=float, default=1.0, metavar='VAL',
                           help="alpha parameter of gamma distribution (default: %(default)s)")
    argparser.add_argument('--beta', '-b', type=float, default=1.0, metavar='VAL',
                           help="beta parameter of gamma distribution (default: %(default)s)")
    argparser.add_argument('--output-file', '-o',  metavar='FILE',
                          help="Output file. It will be in bedgraph/bed/tsv format\n"
                               "(can be gzipped)",
                          dest='output_file')
    argparser.add_argument('--output-mode', metavar='MODE', default='bedgraph',
                           choices=['bedgraph', 'bedgraph+length+LMM', 'bed'],
                           help="Formatting of output. Default: %(default)s.\n"
                                "Possible options: %(choices)s")
    argparser.add_argument('--algorithm',
                           choices=['slidingwindow', 'exact', 'rounds'],
                           default='rounds', metavar='ALGO',
                           help="Algorithm to use (default: %(default)s)\n"
                                "Possible options: %(choices)s")
    argparser.add_argument('--split-constraints', metavar='STRATEGY',
                          choices=('none', 'zeros', 'constants'),
                          default='constants',
                          help="Specify types of intervals which shouldn't be splitted.\n"
                               "Default: %(default)s\n"
                               "Options: %(choices)s")
    argparser.add_argument('--split-number-regularization', type=float, default=0,
                           metavar='VALUE',
                           help="Penalty multiplier for each split")
    argparser.add_argument('--length-regularization', type=float, default=0,
                           metavar='VALUE',
                           help="Penalty multiplier for length of each segment")
    argparser.add_argument('--length-regularization-function', type=str, default='none',
                           metavar='FUNC',
                           choices=['none', 'revlog'],
                           help='Penalty function for length of segments:\n'
                                'Default: %(default)s. Possible options:\n'
                                '* none -- no length regulatization\n'
                                '* revlog -- 1/log(1+l)\n')
    argparser.add_argument('--window-size', type=int, default=2500, metavar='SIZE',
                           help="Size of window for slidingwindow/rounds algorithms\n"
                           "(default: %(default)s)")
    argparser.add_argument('--window-shift', type=int, default=1250, metavar='SHIFT',
                           help = "Shift in one step (default: %(default)s)")
    argparser.add_argument('--num-rounds', type=int, metavar='N',
                           help = 'Number of rounds for round algorithm.\n'
                                   'If not set, run until no split points removed')
    argparser.add_argument('--split-at-gaps', action='store_true',
                           help = 'By default gaps between intervals are filled with zeros.\n' +
                                  'Split at gaps overrides this behavior so that\n' +
                                  'non-adjacent intervals are segmented independently.')
    argparser.add_argument('--verbosity', metavar='LEVEL', default='WARNING',
                          help='Set logging level (default: %(default)s)\n'
                               'Use `INFO` to show work progress')
    argparser.add_argument('--version', action='version', version='%(prog)s ' + __version__)
    return argparser

# main part of runner
def process():
    argparser = get_argparser()
    args = argparser.parse_args()
    logger.setLevel(getattr(logging, args.verbosity.upper()))
    logger.info("Pasio:"+ str(args))
    splitter = configure_splitter(**vars(args))
    logger.info('Starting Pasio with args'+str(args))
    split_bedgraph(in_filename=args.bedgraph,
                  out_filename=args.output_file,
                  splitter=splitter,
                  split_at_gaps=args.split_at_gaps,
                  output_mode=args.output_mode)

# Wrapper for setuptools
def main():
    try:
        process()
    except KeyboardInterrupt:
        logger.error('Program was interrupted')
        sys.exit(1)
