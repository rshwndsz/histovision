import os
import logging
from logging.config import dictConfig
# import coloredlogs


def setup_logger(name, level='INFO'):
    # Constants
    # Path to current directory `pwd`
    _HERE = os.path.dirname(__file__)
    # Logging config as a Dict
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'loggers': {
            '': {
                'level': level,
                'handlers': ['console']
            },
        },
        'formatters': {
            'colored_console': {
                '()': 'coloredlogs.ColoredFormatter',
                'format': "%(asctime)s - %(levelname)-8s"
                          " - %(message)s",
                'datefmt': '%H:%M:%S'},
            'format_for_file': {
                'format': "%(asctime)s :: %(levelname)s :: %(funcName)s in "
                          "%(filename)s (l:%(lineno)d) :: %(message)s",
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
        },
        'handlers': {
            'console': {
                'level': level,
                'class': 'logging.StreamHandler',
                'formatter': 'colored_console',
                'stream': 'ext://sys.stdout'
            },
        },
    }
    # Load logging configuration
    dictConfig(logging_config)
    # Create logger
    logger = logging.getLogger(name)

    # Force add color to logs
    # coloredlogs.install(fmt=LOGGING_CONFIG['formatters']['colored_console']['format'],
    #                     # stream=sys.stdout,
    #                     level=level, logger=logger)

    return logger
