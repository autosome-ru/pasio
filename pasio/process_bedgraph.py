import numpy as np
from .logging import logger
import itertools

def bedgraph_intervals(filename):
    with open(filename) as bedgraph_file:
        for line in bedgraph_file:
            line = line.strip()
            if line == '':
                continue
            chrom, start, stop, coverage = line.split()[0:4]
            start = int(start)
            stop = int(stop)
            coverage = int(coverage)
            yield (chrom, start, stop, coverage)

def group_by_chromosome(intervals):
    return itertools.groupby(intervals, key=lambda (chrom, start, stop, coverage): chrom)

def parse_bedgraph(filename):
    '''
        yields pointwise profiles grouped by chromosome in form (chrom, profile, chromosome_start)
    '''
    for chromosome, intervals in group_by_chromosome(bedgraph_intervals(filename)):
        chromosome_start = None
        chromosome_data = []
        for (_, start, stop, coverage) in intervals:
            if chromosome_start is None:
                chromosome_start = start
            chromosome_data.extend([coverage]*(stop-start))
        # overwrite chromosome_data not to retain both list and np.array in memory
        # and let the former be garbage collected
        chromosome_data = np.array(chromosome_data, dtype=int)
        yield chromosome, chromosome_data, chromosome_start

def split_bedgraph(in_filename, out_filename, splitter):
    with open(out_filename, 'w') as outfile:
        logger.info('Reading input file %s' % (in_filename))
        for chrom, counts, chrom_start in parse_bedgraph(in_filename):
            logger.info('Starting chrom %s of length %d' % (chrom, len(counts)))
            split_candidates = np.arange(len(counts) + 1)
            score, splits = splitter.split(counts, split_candidates)
            scorer = splitter.scorer(counts, splits)
            sum_logfac = scorer.total_sum_logfac()
            log_likelyhood = score - sum_logfac
            logger.info('chrom %s finished, score %f, number of splits %d. '
                        'Log likelyhood: %f.'% (chrom, score, len(splits), log_likelyhood))
            logger.info('Starting output of chrom %s' % chrom)
            for i, (start, stop, mean_count, log_marginal_likelyhood) in enumerate(zip(splits, splits[1:], scorer.mean_counts(), scorer.log_marginal_likelyhoods())):
                outfile.write('%s\t%d\t%d\t%f\t%d\t%f\n' % (chrom,
                                                            start+chrom_start,
                                                            stop+chrom_start,
                                                            mean_count,
                                                            stop-start,
                                                            log_marginal_likelyhood))
