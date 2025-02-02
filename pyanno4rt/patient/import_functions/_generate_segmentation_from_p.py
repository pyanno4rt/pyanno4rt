"""Python file-based segmentation dictionary generation."""

# Author: Tim Ortkamp

# %% Function definition


def generate_segmentation_from_p(data):
    """
    Generate the segmentation dictionary from a Python binary (.p) file.

    Parameters
    ----------
    data : dict
        Dictionary with information on the segmented structures.

    Returns
    -------
    dict
        Dictionary with information on the segmented structures.
    """

    return dict(sorted(data.items()))
