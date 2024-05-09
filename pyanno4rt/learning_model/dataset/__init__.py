"""
Dataset module.

==================================================================

The module aims to provide methods and classes to import and restructure \
different types of learning model datasets (tabular, image-based, ...).
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._tabular_data_generator import TabularDataGenerator

__all__ = ['TabularDataGenerator']
