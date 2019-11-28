import argparse
import os
import logging
import tempfile

import pasio

logger = logging.getLogger(__name__)
stderr = logging.StreamHandler()
stderr.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stderr.setFormatter(formatter)
logger.addHandler(stderr)
logger.setLevel(logging.INFO)

def write_bedgraph(outfilename, chromosome_label, data):
    with open(outfilename, 'w') as outfile:
        for line in data:
            outfile.write('\t'.join((chromosome_label, line[0], line[1], line[2])) + '\n')
        

def parse_bedgraph_compact(bedgraph_filename):
    with open(bedgraph_filename)  as infile:
        for line in infile:
            chrom, start, stop, coverage = line.strip().split()
            yield chrom, start, stop, coverage
    

def split_by_chromosomes_and_get_sizes(source_filename, tmppath):
    logger.info('Parsing {}'.format(source_filename) )
    previous_chrom = None
    chromosome_index = 0
    chromosome_metadata  = {}
    chromosome_data = []
    for chromosome_label, start, stop, coverage in parse_bedgraph_compact(source_filename):
        if previous_chrom is not None and previous_chrom != chromosome_label:
            outfile_path = os.path.join(tmppath, '{}_{}'.format(chromosome_label,
                                                                os.path.basename(source_filename)))
            write_bedgraph(outfile_path,
                           chromosome_label,
                           chromosome_data)
            chromosome_length = int(chromosome_data[-1][1])
            chromosome_metadata[previous_chrom] = {'length':chromosome_length,
                                                   'index': chromosome_index, 
                                                   'label': previous_chrom,
                                                   'outfile': outfile_path}
            logger.info('Written {} of length {}'.format(previous_chrom,
                                                         chromosome_length) )
            chromosome_data = []
            chromosome_index += 1

        chromosome_data.append((start, stop, coverage))
        previous_chrom = chromosome_label
    outfile_path = os.path.join(tmppath, '{}_{}'.format(chromosome_label,
                                                        os.path.basename(source_filename)))
    write_bedgraph(outfile_path, 
                   chromosome_label,
                   chromosome_data)
    chromosome_length = int(chromosome_data[-1][1])
    chromosome_metadata[chromosome_label] = {'length':chromosome_length,
                                             'index': chromosome_index, 
                                             'label': chromosome_label,
                                             'outfile': outfile_path}
    logger.info('Written {} of length {}'.format(chromosome_label, chromosome_length) )
    return chromosome_metadata


def generate_commandlines_for_each_chrom(executable_path, chrom_data, arguments):
    chrom_data = sorted(chrom_data, key = lambda x:x['length'], reverse=True)
    commandlines = []
    for chrom in chrom_data:
        commandline = [executable_path] + arguments + ['--out_bedgraph', chrom['outfile']]
        commandlines.append(commandline)
    return commandlines


def add_parallel_arguments_to_parser(argparser, include_outfile = False):
    parallel_arguments = argparser.add_argument_group('Pasio parallel wrapper arguments')
    parallel_arguments.add_argument('--tmpdir')
    parallel_arguments.add_argument('--out_script', help="File to write bash script to",
                                    required=True)
    parallel_arguments.add_argument('--path_to_pasio', help="Path for pasio script",
                                    required=True)
    if include_outfile:
        parallel_arguments.add_argument('-o', '--out_bedgraph', help="Output begraph path",
                                        required=True)
    return argparser
    

def get_parallel_parsers():
    joined_argparser = pasio.get_argparser()
    joined_argparser = add_parallel_arguments_to_parser(joined_argparser)
    parallel_parser = argparse.ArgumentParser()
    parallel_parser = add_parallel_arguments_to_parser(parallel_parser,
                                                       include_outfile=True)
    return joined_argparser, parallel_parser


def get_args():
    joined_parser, parallel_parser = get_parallel_parsers()
    args = joined_parser.parse_args()
    parallel_args, pasio_cmdline = parallel_parser.parse_known_args()
    return args, pasio_cmdline 


def generate_sequential(cmdlines, outfilename):
    with open(outfilename, 'w') as outfile:
        for line in cmdlines:
            outfile.write(' '.join(line))
            outfile.write('\n')


def generate_parallel(cmdlines, outfilename):
    with open(outfilename, 'w') as outfile:
        for line in cmdlines:
            outfile.write(' '.join(line))
            outfile.write('\n')


def main():
    args, pasio_args_list = get_args()
    tmpdir = args.tmpdir
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp(prefix = 'pasio_' + args.bedgraph, dir='.')
    else:
        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)
    logger.info('Pasio temporary directory is chosen to be %s' % tmpdir)
    chrom_data = split_by_chromosomes_and_get_sizes(args.bedgraph, tmpdir)
    commandlines =  generate_commandlines_for_each_chrom(args.path_to_pasio, chrom_data.values(), pasio_args_list)
    generate_sequential(commandlines, args.out_script)


if __name__=='__main__':
    main()
