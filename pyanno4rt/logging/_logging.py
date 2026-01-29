"""Logging."""

# Author: Tim Ortkamp

# %% External package import

from importlib.metadata import PackageNotFoundError, version
from io import StringIO
from logging import Formatter, getLogger, StreamHandler
from pkg_resources import get_distribution
from platform import python_version

# %% Class definition


class Logging():
    """
    Logging class.

    This class provides methods to configure a logging instance, including \
    multiple stream handlers and formatters to print log messages.

    Parameters
    ----------
    label : str
        Label of the treatment plan instance.

    min_log_level : {'debug', 'info', 'warning', 'error', 'critical'}
        Minimum logging level for broadcasting messages to the console \
        and the object streams.

    Attributes
    ----------
    label : str
        See 'Parameters'.

    min_log_level : {'debug', 'info', 'warning', 'error', 'critical'}
        See 'Parameters'.

    logger : object of class :class:`~logging.Logger`
        The object used to interface the logging methods.
    """

    def __init__(
            self,
            label,
            min_log_level):

        # Get the input attributes
        self.label = label
        self.min_log_level = min_log_level

        # Initialize the logger
        self.logger = self.initialize(
            label=f'pyanno4rt - {label}', min_log_level=min_log_level)

        try:

            # Get the package version from importlib
            __version__ = version("pyanno4rt")

        except PackageNotFoundError:

            try:

                # Get the package version from pkg_resources
                __version__ = get_distribution("pyanno4rt").version

            except Exception:

                # Get the default package version
                __version__ = "1.x.x"

        # Log a message about the software versions used
        self.info(
            'Running pyanno4rt "Amadeus" v%s with Python %s ...',
            __version__, python_version())

        # Log a message about the warranty clause
        self.warning(
            'pyanno4rt is an open-source package and NOT a medical product. '
            'It is provided "as-is", without warranty of any kind, and '
            'intended for research and education only ...')

        # Log a message about the initialization of the class
        self.info("Initializing logger ...")

    def initialize(
            self,
            label,
            min_log_level):
        """
        Initialize a pyanno4rt logger by specifying the channel name, \
        handlers, and formatters.

        Parameters
        ----------
        label : str
            Name of the logger.

        min_log_level : {'debug', 'info', 'warning', 'error', 'critical'}
            Minimum logging level for broadcasting messages to the console \
            and the object streams.

        Returns
        -------
        object of class :class:`~logging.Logger`
            The object used to interface the logging methods.
        """

        # Get the logger
        logger = getLogger(name=label)

        # Set the basic logging level
        logger.setLevel(level=min_log_level.upper())

        # Clear the handlers
        logger.handlers.clear()

        # Initialize the console stream handler
        console_stream_handler = StreamHandler()

        # Set the logging level for the console stream handler
        console_stream_handler.setLevel(level=min_log_level.upper())

        # Initialize the text IO object
        text_stream = StringIO()

        # Initialize the text IO stream handler
        text_stream_handler = StreamHandler(stream=text_stream)

        # Set the logging level for the text IO stream handler
        text_stream_handler.setLevel(level=min_log_level.upper())

        # Initialize the output formatter
        formatter = Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')

        # Add the output formatter to the stream handlers
        console_stream_handler.setFormatter(fmt=formatter)
        text_stream_handler.setFormatter(fmt=formatter)

        # Add the stream handlers to the logger
        logger.addHandler(hdlr=console_stream_handler)
        logger.addHandler(hdlr=text_stream_handler)

        # Suppress message passing to the handlers of ancestor loggers
        logger.propagate = False

        return logger

    def change_log_levels(
            self,
            min_log_level):
        """
        Change the logging level for all handlers.

        Parameters
        ----------
        min_log_level : {'debug', 'info', 'warning', 'error', 'critical'}
            Minimum logging level for broadcasting messages to the console \
            and the object streams.
        """

        # Overwrite the attribute
        self.min_log_level = min_log_level

        # Loop over the handlers
        for handler in self.logger.handlers:

            # Set the logging level
            handler.setLevel(level=min_log_level.upper())

    def to_console(
            self,
            level,
            formatted_string,
            *args):
        """
        Call the display function specified by `level` for the message \
        given by `formatted_string`.

        Parameters
        ----------
        level : {'debug', 'info', 'warning', 'error', 'critical'}
            Level of the logging message.

        formatted_string : str
            Formatted string to be displayed.

        *args : tuple
            Optional display parameters.
        """

        # Map the values of 'level' to the logging methods
        logging_methods = {
            'debug': self.logger.debug,
            'info': self.logger.info,
            'warning': self.logger.warning,
            'error': self.logger.error,
            'critical': self.logger.critical}

        # Run the selected logging method
        logging_methods[level](formatted_string, *args)

    def debug(
            self,
            formatted_string,
            *args):
        """
        Display a logging message for the level 'debug'.

        Parameters
        ----------
        formatted_string : str
            Formatted string to be displayed.

        *args : tuple
            Optional display parameters.
        """

        self.to_console('debug', formatted_string, *args)

    def info(
            self,
            formatted_string,
            *args):
        """
        Display a logging message for the level 'info'.

        Parameters
        ----------
        formatted_string : str
            Formatted string to be displayed.

        *args : tuple
            Optional display parameters.
        """

        self.to_console('info', formatted_string, *args)

    def warning(
            self,
            formatted_string,
            *args):
        """
        Display a logging message for the level 'warning'.

        Parameters
        ----------
        formatted_string : str
            Formatted string to be displayed.

        *args : tuple
            Optional display parameters.
        """

        self.to_console('warning', formatted_string, *args)

    def error(
            self,
            formatted_string,
            *args):
        """
        Display a logging message for the level 'error'.

        Parameters
        ----------
        formatted_string : str
            Formatted string to be displayed.

        *args : tuple
            Optional display parameters.
        """

        self.to_console('error', formatted_string, *args)

    def critical(
            self,
            formatted_string,
            *args):
        """
        Display a logging message for the level 'critical'.

        Parameters
        ----------
        formatted_string : str
            Formatted string to be displayed.

        *args : tuple
            Optional display parameters.
        """

        self.to_console('critical', formatted_string, *args)
