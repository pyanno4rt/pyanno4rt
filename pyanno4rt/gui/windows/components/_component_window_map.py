"""Component window map."""

# Author: Tim Ortkamp

# %% Internal package import

from pyanno4rt.gui.windows.components import (
    DecisionTreeNTCPWindow, DoseUniformityWindow, EquivalentUniformDoseWindow,
    KNeighborsNTCPWindow, LogisticRegressionNTCPWindow, LQPoissonTCPWindow,
    LKBNTCPWindow, MaximumDVHWindow, MeanDoseWindow, MinimumDVHWindow,
    NaiveBayesNTCPWindow, NeuralNetworkNTCPWindow, RandomForestNTCPWindow,
    SquaredDeviationWindow, SquaredOverdosingWindow, SquaredUnderdosingWindow,
    SupportVectorMachineNTCPWindow)

# %% Map definition


component_window_map = {
    'Decision Tree NTCP': DecisionTreeNTCPWindow,
    'Dose Uniformity': DoseUniformityWindow,
    'Equivalent Uniform Dose': EquivalentUniformDoseWindow,
    'K-Nearest Neighbors NTCP': KNeighborsNTCPWindow,
    'Logistic Regression NTCP': LogisticRegressionNTCPWindow,
    'LQ Poisson TCP': LQPoissonTCPWindow,
    'Lyman-Kutcher-Burman NTCP': LKBNTCPWindow,
    'Maximum DVH': MaximumDVHWindow,
    'Mean Dose': MeanDoseWindow,
    'Minimum DVH': MinimumDVHWindow,
    'Naive Bayes NTCP': NaiveBayesNTCPWindow,
    'Neural Network NTCP': NeuralNetworkNTCPWindow,
    'Random Forest NTCP': RandomForestNTCPWindow,
    'Squared Deviation': SquaredDeviationWindow,
    'Squared Overdosing': SquaredOverdosingWindow,
    'Squared Underdosing': SquaredUnderdosingWindow,
    'Support Vector Machine NTCP': SupportVectorMachineNTCPWindow}
