import logging


def setup(level=logging.DEBUG):
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s %(message)s')
