"""Objectives map."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.optimization.components.objectives import (
    DecisionTreeNTCP, DoseUniformity, EquivalentUniformDose,
    ExtremeGradientBoostingNTCP, KNeighborsNTCP, LogisticRegressionNTCP,
    LogisticRegressionTCP, LymanKutcherBurmanNTCP, MaximumDVH, MeanDose,
    MinimumDVH, Moments, NaiveBayesNTCP, NeuralNetworkNTCP, NeuralNetworkTCP,
    RandomForestNTCP, SquaredDeviation, SquaredOverdosing, SquaredUnderdosing,
    SupportVectorMachineNTCP, SupportVectorMachineTCP)

# %% Map definition


objectives_map = {'Decision Tree NTCP': DecisionTreeNTCP,
                  'Dose Uniformity': DoseUniformity,
                  'Equivalent Uniform Dose': EquivalentUniformDose,
                  'Extreme Gradient Boosting NTCP': (
                      ExtremeGradientBoostingNTCP),
                  'K-Nearest Neighbors NTCP': KNeighborsNTCP,
                  'Logistic Regression NTCP': LogisticRegressionNTCP,
                  'Logistic Regression TCP': LogisticRegressionTCP,
                  'Lyman-Kutcher-Burman NTCP': LymanKutcherBurmanNTCP,
                  'Maximum DVH': MaximumDVH,
                  'Mean Dose': MeanDose,
                  'Minimum DVH': MinimumDVH,
                  'Moments': Moments,
                  'Naive Bayes NTCP': NaiveBayesNTCP,
                  'Neural Network NTCP': NeuralNetworkNTCP,
                  'Neural Network TCP': NeuralNetworkTCP,
                  'Random Forest NTCP': RandomForestNTCP,
                  'Squared Deviation': SquaredDeviation,
                  'Squared Overdosing': SquaredOverdosing,
                  'Squared Underdosing': SquaredUnderdosing,
                  'Support Vector Machine NTCP': SupportVectorMachineNTCP,
                  'Support Vector Machine TCP': SupportVectorMachineTCP
                  }
