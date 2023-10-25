import logging


def setup(level=logging.DEBUG):
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s %(message)s')
