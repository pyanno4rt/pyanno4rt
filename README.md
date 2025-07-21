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
<img src="https://github.com/pyanno4rt/pyanno4rt/blob/develop/logo/logo_white.png?raw=true" alt="logo" width="600">
</p>

<h3 align='center'>Python-based Advanced Numerical Nonlinear Optimization for Radiotherapy</h3>

---

# General :thought_balloon:

*pyanno4rt* is a Python package for conventional and outcome prediction model-based inverse photon and proton treatment plan optimization, including radiobiological and machine learning (ML) models for normal tissue complication probability (NTCP) and tumor control probability (TCP). It leverages state-of-the-art local and global solution methods to handle both single- and multi-objective (un)constrained optimization problems in radiotherapy treatment planning.

# Highlight Features :telescope:

<h3>Import of patient data and dose influence matrices from different sources</h3>
<ul>
	<li> DICOM files (.dcm) </li>
	<li> MATLAB files (.mat) </li>
	<li> Python files (.npy, .npz, .p) </li>
</ul>

<h3>Easy configuration and management of treatment plans</h3>
<ul>
	<li> Class-based plan generation </li>
	<li> Automatic input checks to preserve the integrity </li>
	<li> Dedicated logging channels & singleton datahubs </li>
	<li> Snapshot/copycat functionality for storage/retrieval </li>
</ul>

<h3>Multi-objective treatment plan optimization</h3>
<ul>
	<li> Physical & RBE-weighted dose-fluence projections </li>
	<li> Classical & data-driven fluence initialization strategies
		<ul>
			<li> Data medoid initialization </li>
			<li> Tumor coverage initialization </li>
			<li> Warm start initialization </li>
		</ul>
	</li>
	<li> Scalarization & multi-objective optimization methods
		<ul> 
			<li> Lexicographic method </li> 
			<li> Pareto analysis </li> 
			<li> Weighted-sum method
		</ul>
	</li>
	<li> 24-type dose-volume & outcome prediction model-based optimization component catalogue
	</li>
	<li> Local & global solvers
		<ul>
			<li> Interior-point algorithms provided by <a href="https://pypi.org/project/ipyopt/">ipyopt</a> </li>
			<li> Proximal algorithms provided by <a href="https://pypi.org/project/proxmin/">proxmin</a> </li>
			<li> Multi-objective algorithms provided by <a href="https://pypi.org/project/pymoo/">pymoo</a> </li>
			<li> Population-based algorithms provided by <a href="https://pypi.org/project/pypop7/">pypop7</a> </li>
			<li> Local algorithms provided by <a href="https://pypi.org/project/scipy/">scipy</a> </li>
		</ul>
	</li>
</ul>

<h3>Data-driven outcome prediction model handling</h3>
<ul> 
	<li> Dataset import, handling & preprocessing </li>
	<li> 24-type dosiomic & radiomic feature catalogue </li>
	<li> 7 internal ML models (decision tree, KNN, logistic regression, naive Bayes, neural network, random forest, SVM) with individual preprocessing, inspection & evaluation units + Bayesian hyperparameter tuning </li>
	<li> External model loading via folder paths </li>
</ul>

<h3>Plan evaluation tools</h3>
<ul>
	<li> Cumulative & differential DVHs </li>
	<li> Dose statistics & clinical quality measures </li>
</ul>

<h3>Graphical user interface</h3>
<ul>
	<li> Easy-to-use & powerful PyQt5 main window
		<ul>
			<li> Treatment plan editor </li>
			<li> Workflow controls & plan comparison </li>
			<li> CT/Dose preview </li>
		</ul>
	</li>
	<li> (Standalone) PyQt/Matplotlib visualization window </li>
</ul>

# Installation :computer:

<h3>Python distribution</h3>

You can install the latest distribution via:

```bash
pip install pyanno4rt
```

<h3>Source code</h3>

You can check the latest source code via:

```bash
git clone https://github.com/pyanno4rt/pyanno4rt.git
```

<h3>Usage</h3>

*pyanno4rt* has two main classes which provide a code-based and a UI-based interface:

<h6>Base class import for CLI/IDE</h6>

```python
from pyanno4rt.base import TreatmentPlan
```

<h6>GUI import</h6>

```python
from pyanno4rt.gui import GraphicalUserInterface
```

<h3>Dependencies</h3>

| Name                           | Version                               |
| -----------------------------: | :------------------------------------ |
| `python`                       | <font size="3"> >=3.10, <3.12 </font> |
| `numpy`                        | <font size="3"> >=2.1.3 </font>       |
| `ipyopt`                       | <font size="3"> >=0.12.10 </font>     |
| `proxmin`                      | <font size="3"> >=0.6.12 </font>      |
| `absl-py`                      | <font size="3"> >=2.3.1 </font>       |
| `pydicom`                      | <font size="3"> >=3.0.1 </font>       |
| `scikit-image`                 | <font size="3"> >=0.25.2 </font>      |
| `h5py`                         | <font size="3"> >=3.14.0 </font>      |
| `pandas`                       | <font size="3"> >=2.3.0 </font>       |
| `jax`                          | <font size="3"> >=0.6.2 </font>       |
| `jaxlib`                       | <font size="3"> >=0.6.2 </font>       |
| `numba`                        | <font size="3"> >=0.61.2 </font>      |
| `scikit-learn`                 | <font size="3"> >=1.7.1 </font>       |
| `tensorflow`                   | <font size="3"> >=2.19.0 </font>      |
| `tensorflow-io-gcs-filesystem` | <font size="3"> ==0.31.0 </font>      |
| `hyperopt`                     | <font size="3"> >=0.2.7 </font>       |
| `pymoo`                        | <font size="3"> >=0.6.1.5 </font>     |
| `pyqt5-qt5`                    | <font size="3"> ==5.15.2 </font>      |
| `pyqt5`                        | <font size="3"> >=5.15.10 </font>     |
| `pyqtgraph`                    | <font size="3"> >=0.13.7 </font>      |
| `ipython`                      | <font size="3"> >=8.37.0 </font>      |
| `matplotlib`                   | <font size="3"> >=3.10.3 </font>      |
| `seaborn`                      | <font size="3"> >=0.13.2 </font>      |
| `pypop7`                       | <font size="3"> >=0.0.82 </font>      |

Moreover, we are using **Python v3.11.11** and **Spyder IDE v6.0.7** for development.

# Development :rocket:

<h3>Important links</h3>
<ul>
	<li> <a href="https://github.com/pyanno4rt/pyanno4rt">Github Repo</a> </li>
	<li> <a href="https://pypi.org/project/pyanno4rt/">PyPI</a> </li>
	<li> <a href="https://coveralls.io/github/pyanno4rt/pyanno4rt">Coveralls</a> </li>
	<li> <a href="https://github.com/pyanno4rt/pyanno4rt/issues">Issue tracker</a> </li>
</ul>

<h3>Contributing</h3>

*pyanno4rt* is open for contributors of all experience levels. Please refer to our [contribution guidelines](CONTRIBUTING.md) or get in contact with us (see [Help and Support](#help-and-support)) to discuss the format of your contribution.

> Note: the [docs](https://github.com/pyanno4rt/pyanno4rt/tree/develop/docs) folder includes example files with CT/segmentation data and the photon dose-influence matrix for the TG-119 case, a standard test phantom which can be used for development. You will find more realistic patient data e.g. in the CORT<sup>1</sup> or the TROTS<sup>2</sup> dataset.<br><br>
><sub><sup>1</sup>D. Craft, M. Bangert, T. Long, et al. "Shared Data for Intensity Modulated Radiation Therapy (IMRT) Optimization Research: The CORT Dataset". <i>GigaScience</i> 3.1 (2014).<br><sup>2</sup>S. Breedveld, B. Heijmen. "Data for TROTS - The Radiotherapy Optimisation Test Set". <i>Data in Brief</i> (2017).</sub>

# Help and Support :busts_in_silhouette:

<h3>Resources</h3>

<ul>
	<li> <a href="https://pyanno4rt.readthedocs.io/en/latest/">Documentation</a> </li>
	<li> <a href="https://github.com/pyanno4rt/pyanno4rt/discussions">Github Discussions</a> </li>
<li> <a href="https://github.com/pyanno4rt/pyanno4rt/issues">Github Issues</a> </li>
</ul>

<h3>Contact</h3>
<ul>
	<li> <a href="mailto:tim.ortkamp@gmx.de?subject=Request (pyanno4rt)">Mail</a> </li>
	<li> <a href="https://www.linkedin.com/in/tim-ortkamp">Linkedin</a>
</ul>

<h3>Citation</h3>

If you use *pyanno4rt* in your work, let us now and cite this repository:

```tex
@misc{pyanno4rt2024,
  title = {{pyanno4rt}: python-based advanced numerical nonlinear optimization for radiotherapy},
  author = {Ortkamp, Tim and Jäkel, Oliver and Frank, Martin and Wahl, Niklas},
  year = {2024},
  howpublished = {\url{http://github.com/pyanno4rt/pyanno4rt}}
}
```

