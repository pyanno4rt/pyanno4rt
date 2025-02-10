"""MATLAB file-based segmentation dictionary generation."""

# Author: Tim Ortkamp

# %% Function definition


def generate_segmentation_from_mat(data):
    """
    Generate the segmentation dictionary from a MATLAB (.mat) file.

    Parameters
    ----------
    data : ndarray
        Array with information on the segmented structures.

    Returns
    -------
    dict
        Dictionary with information on the segmented structures.
    """

    # Build a multi-layer tuple with the values for the dictionary
    raw_segment_values = (
        (segment_values[1], (
            segment_values[0],
            segment_values[2],
            segment_values[3].astype(int)-1,
            segment_values[3].astype(int)-1,
            segment_values[3].astype(int)-1,
            {f'{parameter[0].lower()}{parameter[1:]}':
             segment_values[4].__dict__[parameter]
             for parameter in segment_values[4].__dict__
             if parameter in ('Priority', 'alphaX', 'betaX', 'visibleColor')},
            None,
            None))
        for segment_values in data)

    # Set the dictionary keys
    segment_keys = (
        'index', 'type', 'raw_indices', 'prioritized_indices',
        'resized_indices', 'parameters', 'objective', 'constraint')

    # Merge the keys and the values into the segmentation dictionary
    segmentation = {
        values[0]: dict(zip(segment_keys, values[1]))
        for values in raw_segment_values}

    return dict(sorted(segmentation.items()))
