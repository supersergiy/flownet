import logging
import sys

logger = logging.getLogger('flownet-train')

def configure_logger(debug=True):
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format='%(message)s',
                        level=log_level,
                        stream=sys.stdout)
