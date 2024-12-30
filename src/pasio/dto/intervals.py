from collections import namedtuple
from ..utils.gzip_utils import open_for_read
from ..logging import logger

class ScoredInterval( namedtuple('ScoredInterval', ['start', 'stop', 'mean_count', 'log_marginal_likelyhood']) ):
    @property
    def length(self):
        return self.stop - self.start

class BedgraphInterval(namedtuple('BedgraphInterval', ['chrom', 'start', 'stop', 'count'])):
    @property
    def length(self):
        return self.stop - self.start

    @classmethod
    def from_string(cls, line):
        chrom, start, stop, count_str = line.split()[0:4]
        start = int(start)
        stop = int(stop)
        try:
            count = int(count_str)
        except:
            count = int(float(count_str))
            logger.warn(f"Pasio cannot be used with floating point counts. `{count_str}` was automatically converted to an integer `{count}` as an approximation. Make sure this value was designed to actually be an integer count.")
        return cls(chrom, start, stop, count)

    @classmethod
    def each_in_file(cls, filename):
        with open_for_read(filename) as stream:
            for interval in cls.each_in_stream(stream):
                yield interval

    @classmethod
    def each_in_stream(cls, stream):
        for line in stream:
            line = line.strip()
            if line == '':
                continue
            yield cls.from_string(line)
