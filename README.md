[![CI/CD](https://github.com/pyanno4rt/pyanno4rt/actions/workflows/ci-cd.yml/badge.svg?branch=master)](https://github.com/pyanno4rt/pyanno4rt/actions/workflows/ci-cd.yml)
![Read the Docs](https://img.shields.io/readthedocs/pyanno4rt)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyanno4rt)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pyanno4rt)
[![Coverage Status](https://coveralls.io/repos/github/pyanno4rt/pyanno4rt/badge.svg)](https://coveralls.io/github/pyanno4rt/pyanno4rt)
[![GitHub Release](https://img.shields.io/github/v/release/pyanno4rt/pyanno4rt)](https://github.com/pyanno4rt/pyanno4rt/releases)
[![GitHub Downloads](https://img.shields.io/github/downloads/pyanno4rt/pyanno4rt/total)](https://github.com/pyanno4rt/pyanno4rt/releases) 
![GitHub Repo stars](https://img.shields.io/github/stars/pyanno4rt/pyanno4rt)
[![GitHub Discussions](https://img.shields.io/github/discussions/pyanno4rt/pyanno4rt)](https://github.com/pyanno4rt/pyanno4rt/discussions)
[![GitHub Issues](https://img.shields.io/github/issues/pyanno4rt/pyanno4rt)](https://github.com/pyanno4rt/pyanno4rt/issues)
[![GitHub Contributors](https://img.shields.io/github/contributors/pyanno4rt/pyanno4rt)](https://github.com/pyanno4rt/pyanno4rt/graphs/contributors)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

<p align="center">
<img src="https://github.com/pyanno4rt/pyanno4rt/blob/develop/logo/logo_white.png?raw=true" alt="logo" width="600">
</p>

<h3 align='center'>Python-based Advanced Numerical Nonlinear Optimization for Radiotherapy</h3>

---

# General information

*pyanno4rt* is a Python package for conventional and outcome prediction model-based inverse photon and proton treatment plan optimization, including radiobiological and machine learning (ML) models for tumor control probability (TCP) and normal tissue complication probability (NTCP). It leverages state-of-the-art local and global solution methods to handle both single- and multi-objective (un)constrained optimization problems in radiotherapy treatment planning.

---

# Highlight features

<h3>Import of patient data and dose influence matrices from different sources</h3>
<ul>
	<li> DICOM files (.dcm) </li>
	<li> MATLAB files (.mat) </li>
	<li> Python files (.npy, .npz, .p) </li>
</ul>
<br>

<h3>Easy configuration and management of treatment plans</h3>
<ul>
	<li> Class-based plan generation </li>
	<li> Automatic input checks to preserve the integrity </li>
	<li> Dedicated logging channels & singleton datahubs </li>
	<li> Snapshot/copycat functionality for storage/retrieval </li>
</ul>
<br>

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
			<li> Interior-point algorithms provided by Ipyopt </li>
			<li> Proximal algorithms provided by Proxmin </li>
			<li> Multi-objective algorithms provided by Pymoo </li>
			<li> Population-based algorithms provided by PyPop7 </li>
			<li> Local algorithms provided by SciPy </li>
		</ul>
	</li>
</ul>
<br>

<h3>Data-driven outcome prediction model handling</h3>
<ul> 
	<li> Dataset import, handling & preprocessing </li>
	<li> 24-type dosiomic & radiomic feature catalogue </li>
	<li> 7 internal ML models (decision tree, KNN, logistic regression, naive Bayes, neural network, random forest, SVM) with individual preprocessing, inspection & evaluation units + Bayesian hyperparameter tuning </li>
	<li> External model loading via folder paths </li>
</ul>
<br>

<h3>Plan evaluation tools</h3>
<ul>
	<li> Cumulative & differential DVHs </li>
	<li> Dose statistics & clinical quality measures </li>
</ul>
<br>

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
<br>

# Installation

<h3>Python distribution</h3>

You can install the latest distribution via:

```bash
pip install pyanno4rt
```
<br>

<h3>Source code</h3>

You can check the latest source code via:

```bash
git clone https://github.com/pyanno4rt/pyanno4rt.git
```
<br>

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
<br>

<h3>Dependencies</h3>

<ul>
	<li> python (>=3.10, <3.12)
	<li> numpy (>=2.1.3) </li>
	<li> ipyopt (>=0.12.9) </li>
	<li> proxmin (>=0.6.12) </li>
	<li> absl-py (>=2.2.0) </li>
	<li> pydicom (>=3.0.1) </li>
	<li> scikit-image (>=0.25.2) </li>
	<li> h5py (>=3.13.0) </li>
	<li> pandas (>=2.2.3) </li>
	<li> jax (>=0.5.3) </li>
	<li> jaxlib (>=0.5.3) </li>
	<li> numba (>=0.61.0) </li>
	<li> scikit-learn (>=1.6.1) </li>
	<li> tensorflow (>=2.19.0) </li>
	<li> tensorflow-io-gcs-filesystem (==0.31.0) </li>
	<li> hyperopt (>=0.2.7) </li>
	<li> pymoo (>=0.6.1.3) </li>
	<li> pyqt5-qt5 (==5.15.2) </li>
	<li> pyqt5 (==5.15.10) </li>
	<li> pyqtgraph (>=0.13.7) </li>
	<li> ipython (>=8.34.0) </li>
	<li> matplotlib (>=3.10.1) </li>
	<li> seaborn (>=0.13.2) </li>
	<li> pypop7 (>=0.0.82) </li>
</ul>
Moreover, we are using Python v3.11.11 and Spyder IDE v6.0.5 for development.<br><br>

# Development

<h3>Important links</h3>

<ul>
	<li> Official source code repo: <a href="https://github.com/pyanno4rt/pyanno4rt">https://github.com/pyanno4rt/pyanno4rt</a> </li>
	<li> Download releases: <a href="https://pypi.org/project/pyanno4rt/">https://pypi.org/project/pyanno4rt/</a> </li>
	<li> Issue tracker: <a href="https://github.com/pyanno4rt/pyanno4rt/issues">https://github.com/pyanno4rt/pyanno4rt/issues</a> </li>
</ul>
<br>

<h3>Contributing</h3>

pyanno4rt is open for contributors of all experience levels. Please refer to our contribution guidelines or get in contact with us (see "Help and support") to discuss the format of your contribution.
<br><br>
Note: the "docs" folder on Github includes example files with CT/segmentation data and the photon dose-influence matrix for the TG-119 case, a standard test phantom which can be used for development. You will find more realistic patient data e.g. in the CORT<sup>1</sup> or the TROTS<sup>2</sup> dataset.
<sub>
<br><br>
<sup>1</sup>D. Craft, M. Bangert, T. Long, et al. "Shared Data for Intensity Modulated Radiation Therapy (IMRT) Optimization Research: The CORT Dataset". <i>GigaScience</i> 3.1 (2014). <br>
<sup>2</sup>S. Breedveld, B. Heijmen. "Data for TROTS - The Radiotherapy Optimisation Test Set". <i>Data in Brief</i> (2017).
</sub>
<br><br>

# Help and support

<h3>Contact</h3>

<ul>
	<li> Mail: <a href="mailto:tim.ortkamp@gmx.de?subject=Request on pyanno4rt">tim.ortkamp(at)gmx.de</a> </li>
	<li> Github Discussions: <a href="https://github.com/pyanno4rt/pyanno4rt/discussions">https://github.com/pyanno4rt/pyanno4rt/discussions</a> </li>
	<li> LinkedIn: <a href="https://www.linkedin.com/in/tim-ortkamp">https://www.linkedin.com/in/tim-ortkamp</a>
	
</ul>

<h3>Citation</h3>

To cite this repository:

```
@misc{pyanno4rt2024,
  title = {{pyanno4rt}: python-based advanced numerical nonlinear optimization for radiotherapy},
  author = {Ortkamp, Tim and Jäkel, Oliver and Frank, Martin and Wahl, Niklas},
  year = {2024},
  howpublished = {\url{http://github.com/pyanno4rt/pyanno4rt}}
}
```
