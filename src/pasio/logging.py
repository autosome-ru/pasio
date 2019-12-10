from __future__ import absolute_import
import logging

class LoggingContextFilter:
    def __init__(self):
        self.context = {}

    def put_to_context(self, key, value):
        self.context[key] = value

    def remove_from_context(self, key):
        if key in self.context:
            del self.context[key]

    def filter(self, record):
        if 'window' in self.context:
            record.msg = 'Window %s: %s' % (self.context['window'], record.msg)
        if 'round' in self.context:
            record.msg = 'Round %d: %s' % (self.context['round'], record.msg)
        return True

logger = logging.getLogger('pasio')
stderr = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stderr.setFormatter(formatter)
logger.addHandler(stderr)
logger.setLevel(logging.WARNING)

logging_filter = LoggingContextFilter()
logger.addFilter(logging_filter)
