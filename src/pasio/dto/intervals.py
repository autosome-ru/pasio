from collections import namedtuple
from ..utils.gzip_utils import open_for_read

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
        chrom, start, stop, count = line.split()[0:4]
        start = int(start)
        stop = int(stop)
        count = int(count)
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
