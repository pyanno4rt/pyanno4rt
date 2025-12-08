"""CSV file handler."""

# Author: Tim Ortkamp

# %% External package import

from pandas import read_csv

# %% Class definition


class CSVHandler():
    """
    CSV file handler class.

    This class provides methods to handle outcome model data from CSV files.
    """

    def __init__(self):

        pass

    def load(
            self,
            path):
        """
        Load the outcome model data.

        Parameters
        ----------
        path : str
            Path to the outcome model data.

        Returns
        -------
        object of class :class:`~pandas.core.frame.DataFrame`
            A pandas dataframe for the outcome dataset.
        """

        return read_csv(path)

    def save(
            self,
            dataframe,
            path):
        """
        Save the outcome model data.

        Parameters
        ----------
        dataframe : object of class :class:`~pandas.core.frame.DataFrame`
            A pandas dataframe for the outcome dataset.

        path : str
            Path for storing the outcome data.
        """

        # Save the data to a CSV file
        dataframe.to_csv(path, index=False)
