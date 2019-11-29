from collections import namedtuple
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
    def iter_from_bedgraph(cls, filename):
        with open(filename) as bedgraph_file:
            for line in bedgraph_file:
                line = line.strip()
                if line == '':
                    continue
                yield cls.from_string(line)
