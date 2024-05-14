"""Component window map."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.gui.windows.components import (
    DecisionTreeNTCPWindow, DecisionTreeTCPWindow, DoseUniformityWindow,
    EquivalentUniformDoseWindow, KNeighborsNTCPWindow, KNeighborsTCPWindow,
    LogisticRegressionNTCPWindow, LogisticRegressionTCPWindow,
    LQPoissonTCPWindow, LKBNTCPWindow, MaximumDVHWindow, MeanDoseWindow,
    MinimumDVHWindow, NaiveBayesNTCPWindow, NaiveBayesTCPWindow,
    NeuralNetworkNTCPWindow, NeuralNetworkTCPWindow, RandomForestNTCPWindow,
    RandomForestTCPWindow, SquaredDeviationWindow, SquaredOverdosingWindow,
    SquaredUnderdosingWindow, SupportVectorMachineNTCPWindow,
    SupportVectorMachineTCPWindow)

# %% Map definition


component_window_map = {
    'Decision Tree NTCP': DecisionTreeNTCPWindow,
    'Decision Tree TCP': DecisionTreeTCPWindow,
    'Dose Uniformity': DoseUniformityWindow,
    'Equivalent Uniform Dose': EquivalentUniformDoseWindow,
    'K-Nearest Neighbors NTCP': KNeighborsNTCPWindow,
    'K-Nearest Neighbors TCP': KNeighborsTCPWindow,
    'Logistic Regression NTCP': LogisticRegressionNTCPWindow,
    'Logistic Regression TCP': LogisticRegressionTCPWindow,
    'LQ Poisson TCP': LQPoissonTCPWindow,
    'Lyman-Kutcher-Burman NTCP': LKBNTCPWindow,
    'Maximum DVH': MaximumDVHWindow,
    'Mean Dose': MeanDoseWindow,
    'Minimum DVH': MinimumDVHWindow,
    'Naive Bayes NTCP': NaiveBayesNTCPWindow,
    'Naive Bayes TCP': NaiveBayesTCPWindow,
    'Neural Network NTCP': NeuralNetworkNTCPWindow,
    'Neural Network TCP': NeuralNetworkTCPWindow,
    'Random Forest NTCP': RandomForestNTCPWindow,
    'Random Forest TCP': RandomForestTCPWindow,
    'Squared Deviation': SquaredDeviationWindow,
    'Squared Overdosing': SquaredOverdosingWindow,
    'Squared Underdosing': SquaredUnderdosingWindow,
    'Support Vector Machine NTCP': SupportVectorMachineNTCPWindow,
    'Support Vector Machine TCP': SupportVectorMachineTCPWindow}
