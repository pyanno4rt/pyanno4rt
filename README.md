[![pyanno4rt CI/CD](https://github.com/pyanno4rt/pyanno4rt/actions/workflows/ci-cd.yml/badge.svg?branch=master)](https://github.com/pyanno4rt/pyanno4rt/actions/workflows/ci-cd.yml)
![Read the Docs](https://img.shields.io/readthedocs/pyanno4rt)
[![Coverage Status](https://coveralls.io/repos/github/pyanno4rt/pyanno4rt/badge.svg)](https://coveralls.io/github/pyanno4rt/pyanno4rt)
[![GitHub Release](https://img.shields.io/github/v/release/pyanno4rt/pyanno4rt)](https://github.com/pyanno4rt/pyanno4rt/releases)
[![GitHub Downloads](https://img.shields.io/github/downloads/pyanno4rt/pyanno4rt/total)](https://github.com/pyanno4rt/pyanno4rt/releases) 
![GitHub Repo stars](https://img.shields.io/github/stars/pyanno4rt/pyanno4rt)
![GitHub Repo Size](https://img.shields.io/github/repo-size/pyanno4rt/pyanno4rt)
[![GitHub Discussions](https://img.shields.io/github/discussions/pyanno4rt/pyanno4rt)](https://github.com/pyanno4rt/pyanno4rt/discussions)
[![GitHub Issues](https://img.shields.io/github/issues/pyanno4rt/pyanno4rt)](https://github.com/pyanno4rt/pyanno4rt/issues)
[![GitHub Contributors](https://img.shields.io/github/contributors/pyanno4rt/pyanno4rt)](https://github.com/pyanno4rt/pyanno4rt/graphs/contributors)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

<p align="center">
<img src="./logo/logo_white.png" alt="logo">
</p>

<h3 align='center'>Python-based Advanced Numerical Nonlinear Optimization for Radiotherapy</h3>

---

# General information

<b>pyanno4rt</b> is a Python package for conventional and outcome prediction model-based inverse photon and proton treatment plan optimization, including radiobiological and machine learning (ML) models for tumor control probability (TCP) and normal tissue complication probability (NTCP). It leverages state-of-the-art local and global solution methods to handle both single- and multi-objective (un)constrained optimization problems, thereby covering a number of different problem designs. To summarize roughly, the following functionality is provided:
<ul>
	<br>
	<li> <b>Import of patient data and dose information from different sources </b> </li>
		<br>
		<ul> 
			<li> DICOM files (.dcm) </li>
			<li> MATLAB files (.mat) </li>
			<li> Python files (.npy, .p) </li>
		</ul>
		<br>
	<li> <b>Individual configuration and management of treatment plan instances</b> </li>
		<br>
		<ul> 
			<li> Dictionary-based plan generation </li>
			<li> Dedicated logging channels and singleton datahubs </li>
			<li> Automatic input checks to preserve the integrity </li>
			<li> Snapshot/copycat functionality for storage and retrieval </li>
		</ul>
		<br>
	<li> <b>Fluence vector initialization strategies</b> </li>
		<br>
		<ul> 
			<li> Data medoid initialization </li>
			<li> Tumor coverage initialization </li>
			<li> Warm start initialization </li>
		</ul>
		<br>
	<li> <b>Multi-objective treatment plan optimization</b> </li>
		<br>
		<ul>
			<li> Dose-volume and outcome prediction model-based optimization functions </li>
				<br>
				<ul>
					<li> 9 dose-volume objectives, e.g. squared deviation and squared overdosing </li>
					<li> 4 dose-volume constraints, e.g. minimum DVH and maximum DVH </li>
					<li> 9 outcome prediction model-based objectives, e.g. logistic regression and artificial neural networks </li>
					<li> 2 outcome prediction model-based constraints, minimum TCP and maximum NTCP </li>
				</ul>
				<br>
			<li> Dose-fluence projections </li>
				<br>
				<ul>
					<li> Constant RBE projection </li>
					<li> Dose projection </li>
				</ul>
				<br>
			<li> Optimization methods </li>
				<br>
				<ul> 
					<li> Lexicographic method </li> 
					<li> Weighted-sum method </li> 
					<li> Pareto analysis
				</ul>
				<br>
			<li> Local and global solvers </li>
				<br>
				<ul> 
					<li> Interior-point algorithms provided by COIN-OR, wrapped by cyipopt </li> 
					<li> Proximal algorithms provided by Proxmin </li>
					<li> Multi-objective algorithms provided by Pymoo </li>
					<li> Population-based algorithms provided by PyPop7 </li>
					<li> Local algorithms provided by SciPy </li>
				</ul>
				<br> 
		</ul>
	<li> <b>Data-driven outcome prediction model handling</b> </li>
		<br>
		<ul> 
			<li> Dataset import and preprocessing </li>
			<li> Automatic feature map generation </li>
			<li> 27-type feature catalog for iterative (re)calculation to support model integration into optimization </li>
			<li> 8 highly customizable internal model classes </li>
				<br>
				<ul> 
					<li> Individual preprocessing, inspection and evaluation units </li>  
					<li> Adjustable hyperparameter tuning via sequential model-based optimization (SMBO) with k-fold cross-validation </li> 
					<li> Out-of-folds prediction for generalization assessment </li>
				</ul>
				<br>
			<li> External model loading via user-definable model folder paths
		</ul>
		<br>
	<li> <b>Evaluation tools</b> </li>
		<br>
		<ul>
			<li> Cumulative and differential dose volume histograms (DVH) </li>
			<li> Dose statistics and clinical quality measures </li>
		</ul>
		<br>
	<li> <b>Graphical user interface</b> </li>
		<br>
		<ul>
			<li> Responsive PyQt5 design with easy-to-use and clear surface </li>
				<br>
				<ul> 
					<li> Treatment plan editor </li>
					<li> Workflow controls </li>
					<li> CT/Dose preview </li>
				</ul>
				<br> 
			<li> Extendable visualization suite using Matplotlib and PyQt5 </li>
				<br>
				<ul> 
					<li> Optimization problem analysis </li> 
					<li> Data-driven model review </li>
					<li> Treatment plan evaluation </li> 
				</ul>
				<br>
		</ul>
</ul>
This package integrates external local and global solvers to perform the optimization, where the L-BFGS-B algorithm from SciPy acts as default and fallback if IPOPT is not available to import. You will find comprehensive instructions on how to (optionally) install and configure IPOPT for Linux in the corresponding folder.
<br><br>
The project has been started in April 2022 by Tim Ortkamp<sup>1,</sup><sup>2,</sup><sup>3</sup> and since then has been enriched by the collaboration of
<br><br>
<ul>
	<li> Moritz Müller </li>
</ul>
<br>

<sub>
<sup>1</sup>Scientific Computing Center (SCC), Karlsruhe Institute of Technology (KIT), Karlsruhe, Germany <br>
<sup>2</sup>Medical Physics in Radiation Oncology (E040), German Cancer Research Center (DKFZ), Heidelberg, Germany <br>
<sup>3</sup>HIDSS4Health - Helmholtz Information and Data Science School for Health, Karlsruhe/Heidelberg, Germany
</sub>
<br><br>

# Installation

<h3>User installation</h3>

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

Base class import for CLI/IDE:

```python
from pyanno4rt.base import TreatmentPlan
```

GUI import:

```python
from pyanno4rt.gui import GraphicalUserInterface
```

<br>
<h3>Dependencies</h3>

<ul>
	<li> python (>=3.10, <3.12)
	<li> proxmin (>=0.6.12) </li>
	<li> absl-py (>=2.0.0) </li>
	<li> pydicom (>=2.4.4) </li>
	<li> scikit-image (>=0.22.0) </li>
	<li> h5py (>=3.10.0) </li>
	<li> pandas (>=2.1.4) </li>
	<li> fuzzywuzzy (>=0.18.0) </li>
	<li> jax (>=0.4.23) </li>
	<li> jaxlib (>=0.4.23) </li>
	<li> numba (>=0.58.1) </li>
	<li> python-levenshtein (>=0.23.0) </li>
	<li> scikit-learn (>=1.3.2) </li>
	<li> tensorflow-io-gcs-filesystem (==0.34.0) </li>
	<li> tensorflow (==2.14.0) </li>
	<li> hyperopt (>=0.2.7) </li>
	<li> xgboost (>=2.0.3) </li>
	<li> pypop7 (>=0.0.78) </li>
	<li> pymoo (>=0.6.1.1) </li>
	<li> pyqt5-qt5 (==5.15.2) </li>
	<li> pyqt5 (==5.15.10) </li>
	<li> pyqtgraph (>=0.13.3) </li>
	<li> ipython (>=8.19.0) </li>
</ul>
We are using Python version 3.11.6 with the Spyder IDE version 5.4.5 for development. IPOPT is the only (optional) third-party dependency (you can find the respective tarballs for installation under pyanno4rt/optimization/solvers).
<br><br>

# Development

<h3>Important links</h3>

<ul>
	<li> Official source code repo: https://github.com/pyanno4rt/pyanno4rt </li>
	<li> Issue tracker: https://github.com/pyanno4rt/pyanno4rt/issues </li>
	
</ul>

Note: our package includes .mat files with CT, segmentation and dose information for an example head-and-neck cancer patient, stored with Git Large File Storage (LFS). You need to install LFS before checking out the source code to be able to access these files.
<br><br>

# Help and support

<h3>Contact</h3>

<ul>
	<li> Mail: <a href="mailto:tim.ortkamp@kit.edu">tim.ortkamp@kit.edu</a> </li>
	<li> Github Discussions: https://github.com/pyanno4rt/pyanno4rt/discussions </li>
	<li> LinkedIn: https://www.linkedin.com/in/tim-ortkamp/
	
</ul>

<h3>Citation</h3>

To cite this repository:

```
@software{pyanno4rt2024,
  author = {Tim Ortkamp},
  title = {{pyanno4rt}: python-based advanced numerical nonlinear optimization for radiotherapy},
  url = {http://github.com/pyanno4rt/pyanno4rt},
  year = {2024},
}
```
