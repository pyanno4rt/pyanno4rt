"""
Dataset module.

==================================================================

The module aims to provide methods and classes to import and restructure \
different types of learning model datasets (empty, image-based, tabular, ...).
"""

# Author: Tim Ortkamp

from ._empty_data_generator import EmptyDataGenerator
from ._image_data_generator import ImageDataGenerator
from ._tabular_data_generator import TabularDataGenerator

__all__ = [
    'EmptyDataGenerator',
    'ImageDataGenerator',
    'TabularDataGenerator']
