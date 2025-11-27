"""Optimization maps."""

# Author: Tim Ortkamp

# %% Internal package import

from pyanno4rt.optimization.components import (
    DecisionTreeOutcome, DoseUniformity, EquivalentUniformDose,
    KNeighborsOutcome, LogisticRegressionOutcome, LQPoissonTCP,
    LymanKutcherBurmanNTCP, MaximumDVH, MeanDose, MinimumDVH,
    NaiveBayesOutcome, NeuralNetworkOutcome, RandomForestOutcome,
    SquaredDeviation, SquaredOverdosing, SquaredUnderdosing,
    SupportVectorMachineOutcome)
from pyanno4rt.optimization.initializers import (
    DataMedoidInitializer, TargetCoverageInitializer, WarmStartInitializer)
from pyanno4rt.optimization.problems.lexicographic import LexicographicProblem
from pyanno4rt.optimization.problems.pareto import ParetoProblem
from pyanno4rt.optimization.problems.weighted import WeightedSumProblem
from pyanno4rt.optimization.projections import (
    ConstantRBEProjection, DoseProjection)
from pyanno4rt.optimization.solvers import (
    IpyoptSolver, Pyanno4rtSolver, PymooSolver, PyPop7Solver, SciPySolver)

# %% Map definitions


COMPONENTS = {
    'Decision Tree Outcome': DecisionTreeOutcome,
    'Dose Uniformity': DoseUniformity,
    'Equivalent Uniform Dose': EquivalentUniformDose,
    'K-Nearest Neighbors Outcome': KNeighborsOutcome,
    'Logistic Regression Outcome': LogisticRegressionOutcome,
    'LQ Poisson TCP': LQPoissonTCP,
    'Lyman-Kutcher-Burman NTCP': LymanKutcherBurmanNTCP,
    'Maximum DVH': MaximumDVH,
    'Mean Dose': MeanDose,
    'Minimum DVH': MinimumDVH,
    'Naive Bayes Outcome': NaiveBayesOutcome,
    'Neural Network Outcome': NeuralNetworkOutcome,
    'Random Forest Outcome': RandomForestOutcome,
    'Squared Deviation': SquaredDeviation,
    'Squared Overdosing': SquaredOverdosing,
    'Squared Underdosing': SquaredUnderdosing,
    'Support Vector Machine Outcome': SupportVectorMachineOutcome}

INITIALIZERS = {
    'data-medoid': DataMedoidInitializer,
    'target-coverage': TargetCoverageInitializer,
    'warm-start': WarmStartInitializer}

PROBLEMS = {
    'lexicographic': LexicographicProblem,
    'pareto': ParetoProblem,
    'weighted-sum': WeightedSumProblem}

PROJECTIONS = {
    'photon': DoseProjection,
    'proton': ConstantRBEProjection}

SOLVERS = {
    'ipyopt': IpyoptSolver,
    'pyanno4rt': Pyanno4rtSolver,
    'pymoo': PymooSolver,
    'pypop7': PyPop7Solver,
    'scipy': SciPySolver}
