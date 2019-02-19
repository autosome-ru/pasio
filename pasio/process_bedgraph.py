import numpy as np
from .logging import logger

def parse_bedgraph(filename):
    '''
        yields pointwise profiles grouped by chromosome in form (chrom, profile, chromosome_start)
    '''
    chromosome_data = None
    previous_chrom = None
    with open(filename) as bedgraph_file:
        for line in bedgraph_file:
            line = line.strip()
            if line == '':
                continue
            chrom, start, stop, coverage = line.split()[0:4]
            start = int(start)
            stop = int(stop)
            coverage = int(coverage)
            if chrom != previous_chrom:
                if previous_chrom is not None:
                    # overwrite chromosome_data not to retain both list and np.array in memory
                    # and let the former be garbage collected
                    chromosome_data = np.array(chromosome_data, dtype=int)
                    yield previous_chrom, chromosome_data, chromosome_start
                chromosome_data = []
                chromosome_start = start
            chromosome_data.extend([coverage]*(stop-start))
            previous_chrom = chrom
        chromosome_data = np.array(chromosome_data, dtype=int)
        yield chrom, chromosome_data, chromosome_start


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
