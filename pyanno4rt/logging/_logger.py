"""Logging."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from io import StringIO
from logging import (CRITICAL, DEBUG, ERROR, Formatter, getLogger, INFO,
                     StreamHandler, WARNING)
from platform import python_version

# %% Class definition


class Logger():
    """
    Logging class.

    This class provides methods to configure an instance of the logger, \
    including multiple stream handlers and formatters to print messages at \
    different levels.

    Parameters
    ----------
    *args : tuple
        Tuple with optional (non-keyworded) logging parameters. The value \
        args[0] refers to the label of the treatment plan, while args[1] \
        specifies the minimum logging level.

    Attributes
    ----------
    logger : object of class :class:`~logging.Logger`
        The external object used to interface the logging methods.
    """

    def __init__(
            self,
            *args):

        # Check if arguments are passed
        if args:

            # Initialize the logger
            self.logger = self.initialize_logger(
                label=args[0], min_log_level=args[1])

            # Log a message about the initialization of the class
            self.display_info("Initializing logger ...")

            # Log a message about the python version used
            self.display_info(
                f"You are running Python version {python_version()} ...")

    def initialize_logger(
            self,
            label,
            min_log_level):
        """
        Initialize the logger by specifying the channel name, handlers, and \
        formatters.

        Parameters
        ----------
        label : str
            Label of the treatment plan instance.

        min_log_level : {'debug', 'info', 'warning', 'error', 'critical'}
            Minimum logging level for broadcasting messages to the console \
            and the object streams.
        """

        # Map the values of 'min_log_level' to the logging levels of the module
        levels = {'debug': DEBUG,
                  'info': INFO,
                  'warning': WARNING,
                  'error': ERROR,
                  'critical': CRITICAL}

        # Get the logger by the label
        logger = getLogger(name=f'pyanno4rt - {label}')

        # Set the basic logging level
        logger.setLevel(level=levels[min_log_level])

        # Clear the handlers
        logger.handlers.clear()

        # Initialize the console stream handler
        console_stream_handler = StreamHandler()

        # Set the logging level for the console stream handler
        console_stream_handler.setLevel(level=levels[min_log_level])

        # Initialize the string IO object
        object_stream = StringIO()

        # Initialize the string IO stream handler
        object_stream_handler = StreamHandler(stream=object_stream)

        # Set the logging level for the string IO stream handler
        object_stream_handler.setLevel(level=levels[min_log_level])

        # Initialize the output formatter
        formatter = Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')

        # Add the output formatter to the stream handlers
        console_stream_handler.setFormatter(fmt=formatter)
        object_stream_handler.setFormatter(fmt=formatter)

        # Add the stream handlers to the logger instance
        logger.addHandler(hdlr=console_stream_handler)
        logger.addHandler(hdlr=object_stream_handler)

        # Suppress message passing to the handlers of ancestor loggers
        logger.propagate = False

        return logger

    def display_to_console(
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

        # Map the values of 'level' to the logging methods of the attribute
        logging_methods = {'debug': self.logger.debug,
                           'info': self.logger.info,
                           'warning': self.logger.warning,
                           'error': self.logger.error,
                           'critical': self.logger.critical}

        # Run the selected logging method
        logging_methods[level](formatted_string, *args)

    def display_debug(
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

        self.display_to_console('debug', formatted_string, *args)

    def display_info(
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

        self.display_to_console('info', formatted_string, *args)

    def display_warning(
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

        self.display_to_console('warning', formatted_string, *args)

    def display_error(
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

        self.display_to_console('error', formatted_string, *args)

    def display_critical(
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

        self.display_to_console('critical', formatted_string, *args)
