from __future__ import division
from future.builtins import range

class SlidingWindow:
    def __init__(self, window_size, window_shift):
        self.window_size = window_size
        self.window_shift = window_shift

    def windows(self, arr):
        length = len(arr)
        for start in range(0, length - 1, self.window_shift):
            stop = min(start + self.window_size + 1, length)
            completion = stop / length
            # start inclusive, stop exclusive
            yield (arr[start:stop], completion)
