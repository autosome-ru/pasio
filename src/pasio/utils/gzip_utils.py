import gzip
import sys
import contextlib
if hasattr(contextlib, 'nullcontext'):
    nullcontext = contextlib.nullcontext
else:
    class nullcontext:
        def __init__(self, enter_result=None):
            self.enter_result = enter_result
        def __enter__(self):
            return self.enter_result
        def __exit__(self, *excinfo):
            pass

def choose_open_function(filename, force_gzip=None):
    '''
    If `force_gzip` is True or False, use corresponding open function.
    If `force_gzip` is None, it's guessed based on filename extension
    '''
    if (force_gzip is True) or ((force_gzip is None) and filename.endswith('.gz')):
        return gzip.open
    elif (force_gzip is False) or ((force_gzip is None) and not filename.endswith('.gz')):
        return open
    else:
        raise ValueError("`force_gzip` should be one of True/False/None")

def open_for_write(filename, force_gzip=None, mode='wt'):
    if filename and (filename != '-'):
        open_func = choose_open_function(filename=filename, force_gzip=force_gzip)
        return open_func(filename, mode)
    else:
        return nullcontext(sys.stdout)

def open_for_read(filename, force_gzip=None, mode='rt'):
    if filename and (filename != '-'):
        open_func = choose_open_function(filename=filename, force_gzip=force_gzip)
        return open_func(filename, mode)
    else:
        return nullcontext(sys.stdin)
