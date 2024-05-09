"""Components map."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.optimization.components import (
    DecisionTreeNTCP, DecisionTreeTCP, DoseUniformity, EquivalentUniformDose,
    KNeighborsNTCP, KNeighborsTCP, LogisticRegressionNTCP,
    LogisticRegressionTCP, LQPoissonTCP, LymanKutcherBurmanNTCP,
    MaximumDVH, MeanDose, MinimumDVH, NaiveBayesNTCP, NaiveBayesTCP,
    NeuralNetworkNTCP, NeuralNetworkTCP, RandomForestNTCP, RandomForestTCP,
    SquaredDeviation, SquaredOverdosing, SquaredUnderdosing,
    SupportVectorMachineNTCP, SupportVectorMachineTCP)

# %% Map definition


component_map = {'Decision Tree NTCP': DecisionTreeNTCP,
                 'Decision Tree TCP': DecisionTreeTCP,
                 'Dose Uniformity': DoseUniformity,
                 'Equivalent Uniform Dose': EquivalentUniformDose,
                 'K-Nearest Neighbors NTCP': KNeighborsNTCP,
                 'K-Nearest Neighbors TCP': KNeighborsTCP,
                 'Logistic Regression NTCP': LogisticRegressionNTCP,
                 'Logistic Regression TCP': LogisticRegressionTCP,
                 'LQ Poisson TCP': LQPoissonTCP,
                 'Lyman-Kutcher-Burman NTCP': LymanKutcherBurmanNTCP,
                 'Maximum DVH': MaximumDVH,
                 'Mean Dose': MeanDose,
                 'Minimum DVH': MinimumDVH,
                 'Naive Bayes NTCP': NaiveBayesNTCP,
                 'Naive Bayes TCP': NaiveBayesTCP,
                 'Neural Network NTCP': NeuralNetworkNTCP,
                 'Neural Network TCP': NeuralNetworkTCP,
                 'Random Forest NTCP': RandomForestNTCP,
                 'Random Forest TCP': RandomForestTCP,
                 'Squared Deviation': SquaredDeviation,
                 'Squared Overdosing': SquaredOverdosing,
                 'Squared Underdosing': SquaredUnderdosing,
                 'Support Vector Machine NTCP': SupportVectorMachineNTCP,
                 'Support Vector Machine TCP': SupportVectorMachineTCP}
