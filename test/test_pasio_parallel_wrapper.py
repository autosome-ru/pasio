from pasio_parallel_wrapper import split_by_chromosomes_and_get_sizes

import tempfile
import os

class TestParallelWrapperSplit:
    def setup(self):
        self.chr1 = '''\
chr1   0   16050041    0
chr1   16050041    16050042    1
chr1   16050042    16050043    1
chr1   16050043    16050053    10
'''
        self.chr2 = '''\
chr2   0    2    10
chr2   2    12    10
chr2   12    13    0
chr2   13    100    100
'''
        self.chr3 = '''\
chr3   0    2    10
chr3   2    12    10
chr3   12    13    0
chr3   13    101    100
'''

    def test_split_one_chromosome(self):
        temp_file, filename = tempfile.mkstemp()
        temp_file = os.fdopen(temp_file, 'w')
        temp_file.write(self.chr1)
        temp_file.close()
        temp_dir = tempfile.mkdtemp()
        sizes = split_by_chromosomes_and_get_sizes(filename, temp_dir)
        assert len(sizes) == 1
        assert sizes['chr1']['index'] == 0
        assert sizes['chr1']['label'] == 'chr1'
        assert sizes['chr1']['length'] == 16050053

    def test_split_three_chromosomes(self):
        temp_file, filename = tempfile.mkstemp()
        temp_file = os.fdopen(temp_file, 'w')
        temp_file.write(self.chr1)
        temp_file.write(self.chr2)
        temp_file.write(self.chr3)
        temp_file.close()
        temp_dir = tempfile.mkdtemp()
        sizes = split_by_chromosomes_and_get_sizes(filename, temp_dir)
        print(sizes)
        assert len(sizes) == 3
        assert sizes['chr1']['index'] == 0
        assert sizes['chr1']['label'] == 'chr1'
        assert sizes['chr1']['length'] == 16050053
        assert sizes['chr2']['index'] == 1
        assert sizes['chr2']['label'] == 'chr2'
        assert sizes['chr2']['length'] == 100
        assert sizes['chr3']['index'] == 2
        assert sizes['chr3']['label'] == 'chr3'
        assert sizes['chr3']['length'] == 101
