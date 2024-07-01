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

<b>pyanno4rt</b> is a Python package for conventional and outcome prediction model-based inverse photon and proton treatment plan optimization, including radiobiological and machine learning (ML) models for tumor control probability (TCP) and normal tissue complication probability (NTCP). It leverages state-of-the-art local and global solution methods to handle both single- and multi-objective (un)constrained optimization problems, thereby covering a number of different problem designs. To summarize roughly, the following functionality is provided:
<ul>
	<li> <b>Import of patient data and dose information from different sources </b>
		<br>
		<ul> 
			<li> DICOM files (.dcm) </li>
			<li> MATLAB files (.mat) </li>
			<li> Python files (.npy, .p) </li>
		</ul>
	</li>
	<br>
	<li> <b>Individual configuration and management of treatment plan instances</b>
		<br>
		<ul> 
			<li> Dictionary-based plan generation </li>
			<li> Dedicated logging channels and singleton datahubs </li>
			<li> Automatic input checks to preserve the integrity </li>
			<li> Snapshot/copycat functionality for storage and retrieval </li>
		</ul>
	</li>
	<br>
	<li> <b>Multi-objective treatment plan optimization</b>
		<br>
		<ul>
			<li> Dose-fluence projections
				<ul>
					<li> Constant RBE projection </li>
					<li> Dose projection </li>
				</ul>
			</li>
			<li> Fluence initialization strategies
				<ul>
					<li> Data medoid initialization </li>
					<li> Tumor coverage initialization </li>
					<li> Warm start initialization </li>
				</ul>
			</li>
			<li> Optimization methods
				<ul> 
					<li> Lexicographic method </li> 
					<li> Weighted-sum method </li> 
					<li> Pareto analysis
				</ul>
			</li>
			<li> 24-type dose-volume and outcome prediction model-based optimization component catalogue
			</li>
			<li> Local and global solvers
				<ul> 
					<li> Proximal algorithms provided by Proxmin </li>
					<li> Multi-objective algorithms provided by Pymoo </li>
					<li> Population-based algorithms provided by PyPop7 </li>
					<li> Local algorithms provided by SciPy </li>
				</ul>
			</li>
		</ul>
	</li>
	<br>
	<li> <b>Data-driven outcome prediction model handling</b>
		<br>
		<ul> 
			<li> Dataset import and preprocessing </li>
			<li> Automatic feature map generation </li>
			<li> 27-type feature catalogue for iterative (re)calculation to support model integration into optimization </li>
			<li> 7 customizable internal model classes (decision tree, k-nearest neighbors, logistic regression, naive Bayes, neural network, random forest, support vector machine)
				<ul> 
					<li> Individual preprocessing, inspection and evaluation units </li>  
					<li> Adjustable hyperparameter tuning via sequential model-based optimization (SMBO) with robust k-fold cross-validation </li> 
					<li> Out-of-folds prediction for generalization assessment </li>
				</ul>
			</li>
			<li> External model loading via user-definable model folder paths </li>
		</ul>
	</li>
	<br>
	<li> <b>Evaluation tools</b>
		<br>
		<ul>
			<li> Cumulative and differential dose volume histograms (DVH) </li>
			<li> Dose statistics and clinical quality measures </li>
		</ul>
	</li>
	<br>
	<li> <b>Graphical user interface</b>
		<br>
		<ul>
			<li> Responsive PyQt5 design with easy-to-use and clear surface
				<ul> 
					<li> Treatment plan editor </li>
					<li> Workflow controls </li>
					<li> CT/Dose preview </li>
				</ul>
			</li>
			<li> Extendable visualization suite using Matplotlib and PyQt5
				<ul> 
					<li> Optimization problem analysis </li> 
					<li> Data-driven model review </li>
					<li> Treatment plan evaluation </li> 
				</ul>
			</li>
		</ul>
	</li>
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

pyanno4rt has two main classes which provide a code-based and a UI-based interface:  <br><br>

<i>Base class import for CLI/IDE</i>

```python
from pyanno4rt.base import TreatmentPlan
```

<i>GUI import</i>

```python
from pyanno4rt.gui import GraphicalUserInterface
```

<br>
<h3>Dependencies</h3>

<ul>
	<li> python (>=3.10, <3.11)
	<li> numpy (==1.26.4) </li>
	<li> proxmin (>=0.6.12) </li>
	<li> absl-py (>=2.1.0) </li>
	<li> pydicom (>=2.4.4) </li>
	<li> scikit-image (>=0.24.0) </li>
	<li> h5py (>=3.11.0) </li>
	<li> pandas (>=2.2.2) </li>
	<li> fuzzywuzzy (>=0.18.0) </li>
	<li> jax (>=0.4.30) </li>
	<li> jaxlib (>=0.4.30) </li>
	<li> numba (>=0.60.0) </li>
	<li> python-levenshtein (>=0.25.1) </li>
	<li> scikit-learn (>=1.5.1) </li>
	<li> tensorflow (==2.11.1) </li>
	<li> tensorflow-io-gcs-filesystem (==0.31.0) </li>
	<li> hyperopt (>=0.2.7) </li>
	<li> pymoo (>=0.6.1.1) </li>
	<li> pyqt5-qt5 (==5.15.2) </li>
	<li> pyqt5 (==5.15.10) </li>
	<li> pyqtgraph (>=0.13.7) </li>
	<li> ipython (>=8.26.0) </li>
	<li> matplotlib (==3.8.3) </li>
	<li> seaborn (>=0.13.2) </li>
	<li> pypop7 (>=0.0.80) </li>
</ul>
We are using Python version 3.10.14 with the Spyder IDE version 5.4.5 for development. For optimization, the package integrates external local and global solvers, where the L-BFGS-B algorithm from SciPy acts as default. <br><br>

# Development

<h3>Important links</h3>

<ul>
	<li> Official source code repo: https://github.com/pyanno4rt/pyanno4rt </li>
	<li> Download releases: https://pypi.org/project/pyanno4rt/
	<li> Issue tracker: https://github.com/pyanno4rt/pyanno4rt/issues </li>
	
</ul>

<br>
<h3>Contributing</h3>

pyanno4rt is open for new contributors of all experience levels. Please get in contact with us (see "Help and support") to discuss the format of your contribution.

Note: the "docs" folder on Github includes example files with CT/segmentation data and the photon dose-influence matrix for the TG-119 case, a standard test phantom which can be used for development. You will find more realistic patient data e.g. in the CORT dataset<sup>1</sup> or the TROTS dataset<sup>2</sup>. <br><br>

<sub>
<sup>1</sup>D. Craft, M. Bangert, T. Long, et al. "Shared Data for Intensity Modulated Radiation Therapy (IMRT) Optimization Research: The CORT Dataset". <i>GigaScience</i> 3.1 (2014). <br>
<sup>2</sup>S. Breedveld, B. Heijmen. "Data for TROTS - The Radiotherapy Optimisation Test Set". <i>Data in Brief</i> (2017).
</sub>
<br><br>

# Help and support

<h3>Contact</h3>

<ul>
	<li> Mail: <a href="mailto:tim.ortkamp@kit.edu?subject=Request on pyanno4rt">tim.ortkamp@kit.edu</a> </li>
	<li> Github Discussions: https://github.com/pyanno4rt/pyanno4rt/discussions </li>
	<li> LinkedIn: https://www.linkedin.com/in/tim-ortkamp/
	
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
