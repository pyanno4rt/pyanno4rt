# pyanno4rt: Python-based Advanced Numerical Nonlinear Optimization for Radiotherapy

[![CI/CD](https://github.com/pyanno4rt/pyanno4rt/actions/workflows/ci-cd.yml/badge.svg?branch=master)](https://github.com/pyanno4rt/pyanno4rt/actions/workflows/ci-cd.yml)
[![Read the Docs](https://img.shields.io/readthedocs/pyanno4rt)](https://pyanno4rt.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/badge/PyPI-pyanno4rt-orange.svg)](https://pypi.org/project/pyanno4rt/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyanno4rt)
[![Coverage Status](https://coveralls.io/repos/github/pyanno4rt/pyanno4rt/badge.svg)](https://coveralls.io/github/pyanno4rt/pyanno4rt)
![GitHub Repo stars](https://img.shields.io/github/stars/pyanno4rt/pyanno4rt)
![GitHub forks](https://img.shields.io/github/forks/pyanno4rt/pyanno4rt)
[![GitHub Downloads](https://img.shields.io/github/downloads/pyanno4rt/pyanno4rt/total)](https://github.com/pyanno4rt/pyanno4rt/releases)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=pyanno4rt.pyanno4rt)
[![GitHub Release](https://img.shields.io/github/v/release/pyanno4rt/pyanno4rt)](https://github.com/pyanno4rt/pyanno4rt/releases)
[![GitHub Discussions](https://img.shields.io/github/discussions/pyanno4rt/pyanno4rt)](https://github.com/pyanno4rt/pyanno4rt/discussions)
[![GitHub Issues](https://img.shields.io/github/issues/pyanno4rt/pyanno4rt)](https://github.com/pyanno4rt/pyanno4rt/issues)
[![GitHub Contributors](https://img.shields.io/github/contributors/pyanno4rt/pyanno4rt)](https://github.com/pyanno4rt/pyanno4rt/graphs/contributors)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

---

# General :earth_americas:

*pyanno4rt* is a Python package for conventional and outcome prediction model-based inverse photon and proton treatment plan optimization, including radiobiological and machine learning (ML) models for normal tissue complication probability (NTCP) and tumor control probability (TCP). It leverages state-of-the-art local and global solution methods to handle both single- and multi-criteria (un)constrained optimization problems in radiotherapy treatment planning.

# Highlight Features :telescope:

<details>
<summary><b>📤 Data interoperability for patient/outcome data and dose-influence matrices</b></summary>
<br>

* Patient data: DICOM (.dcm), MATLAB (.mat) or Python files (.npy)
* Outcome data: CSV files (.csv)
* Dose-influence matrices: MATLAB (.mat) or Python files (.npy, .npz)
<br>
</details>

<details>
<summary><b>🛠️ Streamlined configuration and handling of treatment plans</b></summary>
<br>

* Class-based plan generation
* Automatic input validation to preserve the integrity
* Dedicated logging channels
* Load/save functionality to foster reproducibility
<br>
</details>

<details>
<summary><b>🎯 Multi-criteria inverse planning and optimization</b></summary>
<br>

* Physical & RBE-weighted dose-fluence projections
* Classical & data-driven fluence initialization strategies
    - Data medoid initialization
    - Tumor coverage initialization
    - Warm-start initialization
* Scalarization & multi-criteria optimization methods
    - Lexicographic method
    - Pareto method
    - Weighted-sum method
* 17-type dose-volume & outcome model-based optimization component catalogue
* Local & global solvers
    - Interior-point algorithms provided by [ipyopt](https://pypi.org/project/ipyopt/)
    - Internal custom algorithms provided by [pyanno4rt](https://github.com/pyanno4rt/pyanno4rt)
    - Multi-objective algorithms provided by [pymoo](https://pypi.org/project/pymoo/)
    - Population-based algorithms provided by [pypop7](https://pypi.org/project/pypop7/)
    - Local algorithms provided by [scipy](https://pypi.org/project/scipy/)
<br>
</details>

<details>
<summary><b>🔮 Data-driven outcome modeling</b></summary>
<br>

* 7 machine learning models with customizable building blocks for dataset handling, preprocessing, hyperparameter tuning, inspection & evaluation
    - Decision tree
    - K-nearest neighbors
    - Logistic regression
    - Naive Bayes
    - (Feed-forward) neural network
    - Random forest
    - Support vector machine
* Tabular dataset handler
    - Data loading, decomposition and engineering
    - Fold assignment for holdout or (repeated) stratified cross-validation
    - Holdout test set partitioning
    - Automatic or user-defined feature-to-function mapping
* Tabular preprocessor with 6-type preprocessing step catalogue
* Hyperparameter optimization strategies
    - Bayesian SMBO
    - Grid search 
    - Randomized search
* Model interpretation & XAI
    - Feature sensitivity
    - Permutation importance
* Model evaluation & validation
    - Curves: AUC-PR, AUC-ROC, F1
    - KPIs: log loss, Brier score, precision, recall, ...
* 24-type dosiomic & radiomic feature catalogue for input (re)calculation
* Model serialization and external loading from local snapshots
<br>
</details>

<details>
<summary><b>✅ Plan validation and quality analytics</b></summary>
<br>

* Cumulative & differential DVHs
* Dosimetrics & clinical quality measures
<br>
</details>

<details>
<summary><b>🖼️ Graphical user interface</b></summary>
<br>

* Feature-rich PyQt5 desktop application
    - Treatment plan editor
    - Workflow controls & plan comparison
    - CT/Dose preview
* (Standalone) PyQt5/Matplotlib visualizer suite
<br>
</details>
