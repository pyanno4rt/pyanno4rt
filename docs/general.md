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
