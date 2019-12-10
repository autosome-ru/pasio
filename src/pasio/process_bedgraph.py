import numpy as np
from .logging import logger
import itertools
from .utils.slice_when import slice_when
from .segmentation import segments_with_scores
from .dto.intervals import BedgraphInterval
from .utils.gzip_utils import open_for_read, open_for_write

def fill_interval_gaps(intervals):
    previous_stop = None
    for interval in intervals:
        chrom, start, stop, coverage = interval
        if previous_stop and (previous_stop != start):
            yield (chrom, previous_stop, start, 0)
        yield interval
        previous_stop = stop

def intervals_not_adjacent(interval_1, interval_2):
    return interval_1.stop != interval_2.start

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
    for chromosome, chromosome_intervals in itertools.groupby(intervals, key=lambda interval: interval.chrom):
        if split_at_gaps:
            for intervals_group in slice_when(chromosome_intervals, condition=intervals_not_adjacent):
                yield intervals_group
        else:
            yield fill_interval_gaps(chromosome_intervals)

def parse_bedgraph(filename, split_at_gaps=False):
    '''
        yields pointwise profiles grouped by chromosome in form (chrom, profile, chromosome_start)
    '''
    with open_for_read(filename) as stream:
        for interval in parse_bedgraph_stream(stream):
            yield interval

def parse_bedgraph_stream(input_stream, split_at_gaps=False):
    intervals_stream = BedgraphInterval.each_in_stream(input_stream)
    for intervals_group in interval_groups(intervals_stream, split_at_gaps=split_at_gaps):
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

def split_bedgraph(in_filename, out_filename, splitter, split_at_gaps=False, output_mode='bedgraph'):
    with open_for_write(out_filename) as output_stream:
        with open_for_read(in_filename) as input_stream:
            split_bedgraph_stream(input_stream, output_stream, splitter, split_at_gaps=split_at_gaps, output_mode=output_mode)

def split_bedgraph_stream(input_stream, output_stream, splitter, split_at_gaps=False, output_mode='bedgraph'):
    logger.info('Reading input file')
    for chrom, counts, chrom_start in parse_bedgraph_stream(input_stream, split_at_gaps=split_at_gaps):
        logger.info('Starting chrom %s of length %d' % (chrom, len(counts)))
        if output_mode == 'bedgraph':
            for scored_interval in segments_with_scores(counts, splitter):
                output_stream.write('%s\t%d\t%d\t%f\n' % (chrom,
                                                          scored_interval.start + chrom_start,
                                                          scored_interval.stop + chrom_start,
                                                          scored_interval.mean_count))
        elif output_mode == 'bed':
            for scored_interval in segments_with_scores(counts, splitter):
                output_stream.write('%s\t%d\t%d\n' % (chrom,
                                                      scored_interval.start + chrom_start,
                                                      scored_interval.stop + chrom_start))
        elif output_mode == 'bedgraph+length+LMM':
            for scored_interval in segments_with_scores(counts, splitter):
                output_stream.write('%s\t%d\t%d\t%f\t%d\t%f\n' % (chrom,
                                                                  scored_interval.start + chrom_start,
                                                                  scored_interval.stop + chrom_start,
                                                                  scored_interval.mean_count,
                                                                  scored_interval.length,
                                                                  scored_interval.log_marginal_likelyhood))
        else:
            raise ValueError('Unknown output mode `%s`' % output_mode)
        logger.info('Output of chromosome %s finished' % chrom)
