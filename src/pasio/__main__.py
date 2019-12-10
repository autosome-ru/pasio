import sys
from .logging import logger
from .cli import main

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.error('Program was interrupted')
        sys.exit(1)
