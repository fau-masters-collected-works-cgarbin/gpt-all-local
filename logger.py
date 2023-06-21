"""A simple logger to share logging configuration across the project.

- Info and debug messages go to stdout.
- Error and warning messages go to stderr.

Note that use a module-level logger is not the best practice. It's enough for hobbyst projects. Use the logging module
configuration file for more complex projects (https://docs.python.org/3/howto/logging.html#configuring-logging).
"""

import logging
import sys

logger = None  # pylint: disable=invalid-name
VERBOSE = False


def get_logger():
    """Return the logger, creating it if necessary."""
    class InfoAndBelowHandler(logging.StreamHandler):
        """A handler that only emits INFO and below to suppress ERROR messages going to stdout."""
        def emit(self, record):
            if record.levelno < logging.ERROR:
                super().emit(record)

    global logger  # pylint: disable=global-statement,invalid-name

    if logger is None:
        logger = logging.getLogger('my_app')
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt="%H:%M:%S")

        # Log to stdout
        stdout_handler = InfoAndBelowHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)

        # Log to stderr
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.ERROR)
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)

        # Prevents multiple log messages in some cases, e.g. with Streamlit
        logger.propagate = False

    return logger


def set_verbose(on: bool):
    """Set the logger to verbose mode."""
    global VERBOSE  # pylint: disable=global-statement,invalid-name
    VERBOSE = on

    level = logging.DEBUG if on else logging.INFO
    l = get_logger()  # noqa
    l.setLevel(level)  # Must set at the logger and handler level
    for handler in l.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            handler.setLevel(level)
