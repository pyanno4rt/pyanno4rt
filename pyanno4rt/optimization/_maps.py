"""Optimization maps."""

# Author: Tim Ortkamp

# %% Internal package import

from pyanno4rt.optimization.components import (
    DecisionTreeNTCP, DecisionTreeTCP, DoseUniformity, EquivalentUniformDose,
    KNeighborsNTCP, KNeighborsTCP, LogisticRegressionNTCP,
    LogisticRegressionTCP, LQPoissonTCP, LymanKutcherBurmanNTCP,
    MaximumDVH, MeanDose, MinimumDVH, NaiveBayesNTCP, NaiveBayesTCP,
    NeuralNetworkNTCP, NeuralNetworkTCP, RandomForestNTCP, RandomForestTCP,
    SquaredDeviation, SquaredOverdosing, SquaredUnderdosing,
    SupportVectorMachineNTCP, SupportVectorMachineTCP)
from pyanno4rt.optimization.methods import (
    LexicographicOptimization, ParetoOptimization, WeightedSumOptimization)
from pyanno4rt.optimization.projections import (
    ConstantRBEProjection, DoseProjection)
from pyanno4rt.optimization.solvers import (
    IpyoptSolver, ProxminSolver, PymooSolver, PyPop7Solver, SciPySolver)

# %% Map definitions


COMPONENTS = {
    'Decision Tree NTCP': DecisionTreeNTCP,
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

METHODS = {
    'lexicographic': LexicographicOptimization,
    'pareto': ParetoOptimization,
    'weighted-sum': WeightedSumOptimization}

PROJECTIONS = {
    'photon': DoseProjection,
    'proton': ConstantRBEProjection}

SOLVERS = {
    'ipyopt': IpyoptSolver,
    'proxmin': ProxminSolver,
    'pymoo': PymooSolver,
    'pypop7': PyPop7Solver,
    'scipy': SciPySolver}
