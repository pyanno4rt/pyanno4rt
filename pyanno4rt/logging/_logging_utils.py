"""Logging utilities."""

# %% External package import

from contextlib import contextmanager
from contextvars import ContextVar
from logging import getLogger

# %% Variable/Function definitions


# Initialize the context variable for the logger name
current_logger_name = ContextVar('current_logger_name', default='root')

def get_logger():
    """
    Get the current active logger.

    Returns
    -------
    object of class :class:`~logging.Logger`
        The object used to print and store logging messages.
    """

    return getLogger(current_logger_name.get())

@contextmanager
def set_logger_name(name):
    """
    Set the logger name.

    Parameters
    ----------
    name : str
        Name of the logger.
    """

    # Set the current active logger name
    token = current_logger_name.set(name)

    try:

        # Transfer control back to the caller
        yield

    finally:

        # Reset the name
        current_logger_name.reset(token)
