# pyanno4rt: Python-based Advanced Numerical Nonlinear Optimization for Radiotherapy

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
