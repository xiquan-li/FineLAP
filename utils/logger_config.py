from loguru import logger
import sys
import logging
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("h5py").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def setup_logger(exp_dir=None, exp_name=None, log_file=None):
    """
    Setup a beautiful logger with colored output
    - Time: Blue
    - Level: Colored by level
    - Message: White/Default
    """
    logger.remove()

    # Console output with colors
    logger.add(
        sys.stdout, 
        format='<blue>[{time:YYYY-MM-DD HH:mm:ss}]</blue> <level>{level: <8}</level> | {file}:{line} | <level>{message}</level>', 
        level='INFO',
        colorize=True,
        filter=lambda record: record['extra'].get('indent', 0) == 1
    )
    
    # File output without colors
    if log_file:
        logger.add(
            log_file, 
            format='[{time:YYYY-MM-DD HH:mm:ss}] {level: <8} | {file}:{line} | {message}', 
            level='INFO',
            colorize=False,
            filter=lambda record: record['extra'].get('indent', 0) == 1
        )
    
    return logger.bind(indent=1)

main_logger = logger.bind(indent=1)
