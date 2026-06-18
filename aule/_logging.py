"""
    Internal logging setup for aule (private module, not part of the public API).

    Provides a single shared logger ("aule") with colored, level-aware
    console formatting, off by default (WARNING level shows nothing in
    practice since aule never logs at WARNING during normal operation
    unless something is actually worth flagging).

    Activate it globally with `aule.set_log_level("DEBUG")` (or any of
    "INFO", "WARNING", "ERROR"), or by setting the `AULE_LOG_LEVEL`
    environment variable before importing aule. Both apply for the rest
    of the process/session - there is no per-call logging parameter,
    by design, so existing function signatures are untouched.
"""

import logging
import os
import sys

_LOGGER_NAME = "aule"

# ANSI color codes, used only when the output stream is a real terminal
# (no escape codes leak into redirected output / log files / notebooks
# that capture stdout as plain text without a tty).
_LEVEL_COLORS = {
    logging.DEBUG: "\033[36m",      # cyan
    logging.INFO: "\033[32m",       # green
    logging.WARNING: "\033[33m",    # yellow
    logging.ERROR: "\033[31m",      # red
    logging.CRITICAL: "\033[1;31m",  # bold red
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    '''
        Formatter that prefixes each record with a level-colored tag and a
        short timestamp, falling back to plain (uncolored) text when the
        destination stream is not a terminal.
    '''

    def __init__(self, use_color: bool):
        super().__init__(datefmt="%H:%M:%S")
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, self.datefmt)
        level = record.levelname
        message = record.getMessage()

        if self.use_color:
            color = _LEVEL_COLORS.get(record.levelno, "")
            return f"{color}[aule] {timestamp} {level:<8}{_RESET} {message}"
        return f"[aule] {timestamp} {level:<8} {message}"


def _build_logger() -> logging.Logger:
    '''
        Builds and configures the shared "aule" logger: a single stream
        handler to stderr with the color formatter, level taken from the
        `AULE_LOG_LEVEL` environment variable if set (default: WARNING).

        Returns:
        --------
        - logger : logging.Logger
            The configured logger instance.
    '''

    logger = logging.getLogger(_LOGGER_NAME)

    handler = logging.StreamHandler(stream=sys.stderr)
    use_color = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    handler.setFormatter(_ColorFormatter(use_color=use_color))

    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False

    env_level = os.environ.get("AULE_LOG_LEVEL", "WARNING").upper()
    logger.setLevel(getattr(logging, env_level, logging.WARNING))

    return logger


logger = _build_logger()


def set_log_level(level: str) -> None:
    '''
        Sets the global aule logging level for the rest of the session.
        Affects every aule function call from this point on; there is no
        per-call logging parameter.

        Parameters:
        -----------
        - level : str
            One of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
            (case-insensitive).

        Usage:
        ------

        ```python
        import aule

        aule.set_log_level("DEBUG")  # verbose: see internal steps, shape
                                      # normalization decisions, force_shape
                                      # reshapes, report generation progress, ...
        aule.set_log_level("WARNING")  # back to quiet (default)
        ```
    '''

    level_upper = level.upper()
    if not hasattr(logging, level_upper):
        raise ValueError(
            f"Unknown log level '{level}'. Expected one of: "
            "DEBUG, INFO, WARNING, ERROR, CRITICAL."
        )
    logger.setLevel(getattr(logging, level_upper))
    logger.debug("Log level set to %s", level_upper)
