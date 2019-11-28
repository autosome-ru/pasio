import numpy as np
from .logging import logger
import itertools
from .slice_when import slice_when

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

def get_interval_chromosome(interval):
    chrom, start, stop, coverage = interval
    return chrom

def group_by_chromosome(intervals):
    return itertools.groupby(intervals, key=get_interval_chromosome)

def fill_interval_gaps(intervals):
    previous_stop = None
    for interval in intervals:
        chrom, start, stop, coverage = interval
        if previous_stop and (previous_stop != start):
            yield (chrom, previous_stop, start, 0)
        yield interval
        previous_stop = stop

def intervals_not_adjacent(interval_1, interval_2):
    chrom_1, start_1, stop_1, coverage_1 = interval_1
    chrom_2, start_2, stop_2, coverage_2 = interval_2
    return stop_1 != start_2

def interval_groups(intervals, split_at_gaps):
    '''
    Yield groups of adjacent intervals in a fashion similar to itertools.groupby.
    If `split_at_gaps` is False, uncovered chromosome positions (gaps) are filled with 0-s
    If `split_at_gaps` is True, positions missing in bedgraph are treated as separators
     and divide chromosome into independent parts
    Flanking regions (i.e. chromosome ends) are not filled with gaps because
      we don't know chromosome length (thus cannot replace right end with zeros)
      and symmetrically doesn't fill left end for consistency reasons
    '''
    for chromosome, chromosome_intervals in group_by_chromosome(intervals):
        if split_at_gaps:
            for intervals_group in slice_when(chromosome_intervals, condition=intervals_not_adjacent):
                yield intervals_group
        else:
            yield fill_interval_gaps(chromosome_intervals)

def parse_bedgraph(filename, split_at_gaps=False):
    '''
        yields pointwise profiles grouped by chromosome in form (chrom, profile, chromosome_start)
    '''
    for intervals_group in interval_groups(bedgraph_intervals(filename), split_at_gaps=split_at_gaps):
        chromosome_data = []
        chromosome_start = None
        chromosome = None
        for (chrom, start, stop, coverage) in intervals_group:
            if chromosome_start is None:
                chromosome_start = start
                chromosome = chrom
            chromosome_data.extend([coverage]*(stop-start))
        # overwrite chromosome_data not to retain both list and np.array in memory
        # and let the former be garbage collected
        chromosome_data = np.array(chromosome_data, dtype=int)
        yield chromosome, chromosome_data, chromosome_start

def split_bedgraph(in_filename, out_filename, splitter, split_at_gaps):
    with open(out_filename, 'w') as outfile:
        logger.info('Reading input file %s' % (in_filename))
        for chrom, counts, chrom_start in parse_bedgraph(in_filename, split_at_gaps=split_at_gaps):
            logger.info('Starting chrom %s of length %d' % (chrom, len(counts)))
            split_candidates = np.arange(len(counts) + 1)
            score, splits = splitter.split(counts, split_candidates)
            scorer = splitter.scorer(counts, splits)
            sum_logfac = scorer.total_sum_logfac()
            log_likelyhood = score - sum_logfac
            logger.info('chrom %s finished, score %f, number of splits %d. '
                        'Log likelyhood: %f.'% (chrom, score, len(splits), log_likelyhood))
            logger.info('Starting output of chrom %s' % chrom)
            for (start, stop, mean_count, log_marginal_likelyhood) in zip(splits, splits[1:], scorer.mean_counts(), scorer.log_marginal_likelyhoods()):
                outfile.write('%s\t%d\t%d\t%f\t%d\t%f\n' % (chrom,
                                                            start+chrom_start,
                                                            stop+chrom_start,
                                                            mean_count,
                                                            stop-start,
                                                            log_marginal_likelyhood))
