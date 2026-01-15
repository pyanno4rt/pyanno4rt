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

<p align="center">
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/pyanno4rt/pyanno4rt/blob/develop/logo/logo_white.png?raw=true">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/pyanno4rt/pyanno4rt/blob/develop/logo/logo_black.png?raw=true">
    <img alt="logo" src="https://github.com/pyanno4rt/pyanno4rt/blob/develop/logo/logo_white.png?raw=true" width="600">
  </picture>
</p>

<h3 align='center'>Python-based Advanced Numerical Nonlinear Optimization for Radiotherapy</h3>

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

# Installation :computer:

### Python distribution

You can install the latest distribution via:

```bash
pip install pyanno4rt
```

### Source code

You can check the latest source code via:

```bash
git clone https://github.com/pyanno4rt/pyanno4rt.git
```

### Usage

*pyanno4rt* has two main classes which provide a code-based and a UI-based interface:

###### Base class import for CLI/IDE

```python
from pyanno4rt.base import TreatmentPlan
```

###### GUI import

```python
from pyanno4rt.gui import GraphicalUserInterface
```

### Dependencies

| Name                           | Version                               |
| -----------------------------: | :------------------------------------ |
| `python`                       | <font size="3"> >=3.11, <4.0 </font>  |
| `numpy`                        | <font size="3"> >=2.3.5 </font>       |
| `ipyopt`                       | <font size="3"> >=0.12.10 </font>     |
| `absl-py`                      | <font size="3"> >=2.3.1 </font>       |
| `pydicom`                      | <font size="3"> >=3.0.1 </font>       |
| `scikit-image`                 | <font size="3"> >=0.26.0 </font>      |
| `h5py`                         | <font size="3"> >=3.15.1 </font>      |
| `pandas`                       | <font size="3"> >=2.3.3 </font>       |
| `jax`                          | <font size="3"> >=0.8.2 </font>       |
| `jaxlib`                       | <font size="3"> >=0.8.2 </font>       |
| `numba`                        | <font size="3"> >=0.63.1 </font>      |
| `scikit-learn`                 | <font size="3"> >=1.8.0 </font>       |
| `tensorflow`                   | <font size="3"> >=2.20.0 </font>      |
| `hyperopt`                     | <font size="3"> >=0.2.7 </font>       |
| `pymoo`                        | <font size="3"> >=0.6.1.6 </font>     |
| `pyqt5-qt5`                    | <font size="3"> ==5.15.2 </font>      |
| `pyqt5`                        | <font size="3"> ==5.15.10 </font>     |
| `pyqtgraph`                    | <font size="3"> >=0.14.0 </font>      |
| `matplotlib`                   | <font size="3"> >=3.10.8 </font>      |
| `seaborn`                      | <font size="3"> >=0.13.2 </font>      |
| `pypop7`                       | <font size="3"> >=0.0.82 </font>      |

Moreover, we are using **Python v3.11.11** and **Spyder IDE v6.1.2** for development.

# Development :rocket:

### Important links

* [Github](https://github.com/pyanno4rt/pyanno4rt)
* [PyPI](https://pypi.org/project/pyanno4rt/)
* [Coveralls](https://coveralls.io/github/pyanno4rt/pyanno4rt)
* [Issue tracker](https://github.com/pyanno4rt/pyanno4rt/issues)

### Contributing

*pyanno4rt* is open for contributors of all experience levels. Please refer to our [contribution guidelines](CONTRIBUTING.md) or get in contact with us (see [Help and Support](#help-and-support)) to discuss the format of your contribution.

> Note: the [docs](https://github.com/pyanno4rt/pyanno4rt/tree/develop/docs) folder includes example files with CT/segmentation data, photon dose-influence matrix and a synthetic outcome dataset for the C-shape benchmark case from the AAPM TG-119, which can be used for development. You will find more realistic patient data e.g. in the CORT<sup>1</sup> or the TROTS<sup>2</sup> dataset.
\
\
><sub><sup>1</sup>D. Craft, M. Bangert, T. Long, et al. "Shared Data for Intensity Modulated Radiation Therapy (IMRT) Optimization Research: The CORT Dataset". *GigaScience* 3.1 (2014).
\
><sup>2</sup>S. Breedveld, B. Heijmen. "Data for TROTS - The Radiotherapy Optimisation Test Set". *Data in Brief* (2017).</sub>

# Help and Support :busts_in_silhouette:

### Resources

* [Documentation](https://pyanno4rt.readthedocs.io/en/latest/)
* [Github Discussions](https://github.com/pyanno4rt/pyanno4rt/discussions)
* [Github Issues](https://github.com/pyanno4rt/pyanno4rt/issues)

### Contact

* [Github Page](https://tortka.github.io)
* [Mail](mailto:tim.ortkamp@gmx.de?subject=Request (pyanno4rt))
* [LinkedIn](https://www.linkedin.com/in/tim-ortkamp)

### Citation

To cite *pyanno4rt*, either use the link in the right sidebar of the Github landing page labeled "Cite this repository" or copy the short-form bib-style paragraph below:

```tex
@software{pyanno4rt,
  title = {{pyanno4rt}: python-based advanced numerical nonlinear optimization for radiotherapy},
  author = {Ortkamp, Tim and Jäkel, Oliver and Frank, Martin and Wahl, Niklas},
  version = {1.0.0},
  license = {GPL-3.0},
  year = {2026},
  howpublished = {\url{http://github.com/pyanno4rt/pyanno4rt}}
}
```

