"""Model display options check map."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.input_check.check_functions import (
    check_type, check_value_in_set)

# %% Map definition


model_display_map = {
    'graphs': (
        partial(check_type, types=list),
        partial(check_value_in_set, options=('AUC-ROC', 'AUC-PR', 'F1'))
        ),
    'kpis': (
        partial(check_type, types=list),
        partial(check_value_in_set, options=(
            'Logloss', 'Brier score', 'Subset accuracy', 'Cohen Kappa',
            'Hamming loss', 'Jaccard score', 'Precision', 'Recall',
            'F1 score', 'MCC', 'AUC'))
        )
    }
