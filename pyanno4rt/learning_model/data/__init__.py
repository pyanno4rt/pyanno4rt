"""
Data module.

==================================================================

The module aims to provide methods and classes to import the data set used \
for training, deconstruct it into features and labels, and modulate the \
data set according to the label viewpoint, i.e., which temporal kind of \
tissue reaction should be modeled.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._dataset import Dataset

__all__ = ['Dataset']
