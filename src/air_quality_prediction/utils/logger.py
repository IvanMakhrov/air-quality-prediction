import logging
import sys

_CONSOLE_FORMAT = "[%(asctime)s] %(levelname)-7s | %(name)s — %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def setup_logger(
    name: str,
    level: int = logging.INFO,
    force_reinit: bool = False,
) -> logging.Logger:
    """
    Настройка логгера с консольным выводом.
    Используйте в каждом модуле: `logger = setup_logger(__name__)`
    """
    logger = logging.getLogger(name)

    if logger.handlers and not force_reinit:
        return logger

    logger.handlers.clear()
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    class ColoredFormatter(logging.Formatter):
        GREY = "\x1b[38;20m"
        GREEN = "\x1b[32;20m"
        YELLOW = "\x1b[33;20m"
        RED = "\x1b[31;20m"
        BOLD_RED = "\x1b[31;1m"
        RESET = "\x1b[0m"

        LEVEL_COLORS = {
            logging.DEBUG: GREY,
            logging.INFO: GREEN,
            logging.WARNING: YELLOW,
            logging.ERROR: RED,
            logging.CRITICAL: BOLD_RED,
        }

        def format(self, record):
            log_fmt = (
                self.LEVEL_COLORS.get(record.levelno, self.GREY) + _CONSOLE_FORMAT + self.RESET
            )
            formatter = logging.Formatter(log_fmt, datefmt=_DATE_FORMAT)
            return formatter.format(record)

    handler.setFormatter(ColoredFormatter())
    logger.addHandler(handler)

    logger.propagate = False

    return logger
