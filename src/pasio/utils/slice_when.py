# Analog for ruby method Enumerable#slice_when.
# Implemented similarly to groupby
class slice_when:
    def __init__(self, iterable, condition):
        self.it = iter(iterable)
        self.slice_condition = condition
        self.first_element = True
        self.finished = False
        self.cur_group = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.first_element:
            # For empty list it can raise StopIteration immediately, that's expected behavior
            self.cur_value = next(self.it)
            self.first_element = False
        if self.finished:
            raise StopIteration
        if self.cur_group:
            for _ in self.cur_group: # exhaust group
                pass
        self.cur_group = self._group_iterator()
        return self.cur_group
    
    next = __next__

    def _group_iterator(self):
        while True:
            yield self.cur_value
            try:
                self.prev_value = self.cur_value
                self.cur_value  = next(self.it)
                if self.slice_condition(self.prev_value, self.cur_value):
                    return
            except StopIteration:
                self.finished = True
                return
