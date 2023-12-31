# pyanno4rt: Python-based Advanced Numerical Nonlinear Optimization for Radiotherapy

<b>pyanno4rt</b> is a Python package for conventional and outcome prediction model-based inverse photon and proton treatment plan optimization, including radiobiological and machine learning (ML) models for tumor control probability (TCP) and normal tissue complication probability (NTCP). It leverages state-of-the-art local and global solution methods to handle both single- and multi-objective (un)constrained optimization problems, thereby covering a number of different problem designs. To summarize roughly, the following functionality is provided:<br/><br/>

<b>Import of patient data and dose information from different sources </b>
		<br>
		<ul> 
			<li> DICOM files (.dcm) </li>
			<li> MATLAB files (.mat) </li>
			<li> Python files (.npy, .p) </li>
		</ul>

<b>Individual configuration and management of treatment plan instances</b> 
		<br>
		<ul> 
			<li> Dictionary-based plan generation </li>
			<li> Dedicated logging channels and singleton datahubs </li>
			<li> Automatic input checks to preserve the integrity </li>
			<li> Snapshot/copycat functionality for storage and retrieval </li>
		</ul>

<b>Fluence vector initialization strategies</b>
		<br>
		<ul> 
			<li> Data medoid initialization </li>
			<li> Tumor coverage initialization </li>
			<li> Warm start initialization </li>
		</ul>

<b>Multi-objective treatment plan optimization</b>
		<br>
		<ul>
			<li> Dose-volume and outcome prediction model-based optimization functions (18 objectives + 6 constraints) </li>
			<li> Dose-fluence projections (dose + constant RBE) </li>
			<li> Optimization methods (lexicographic, weighted-sum, Pareto) </li>
			<li> Local and global solvers (interior-point/proximal/multi-objective/population-based/local) </li>

<b>Data-driven outcome prediction model handling</b>
		<br>
		<ul> 
			<li> Dataset import and preprocessing </li>
			<li> Automatic feature map generation </li>
			<li> 27-type feature catalog for iterative (re)calculation to support model integration into optimization </li>
			<li> 8 highly customizable internal model classes (individual preprocessing/inspection/evaluation units, SMBO hyperparameter tuning, OOF prediction) </li>
			<li> External model loading via user-definable model folder paths </li>

<b>Evaluation tools</b>
		<br>
		<ul>
			<li> Cumulative and differential dose volume histograms (DVH) </li>
			<li> Dose statistics and clinical quality measures </li>
		</ul>

<b>Graphical user interface</b>
		<br>
		<ul>
			<li> Responsive PyQt5 design with easy-to-use and clear surface </li>
			<li> Extendable visualization suite using Matplotlib and PyQt5 </li>
		</ul>
</ul>
This package integrates external local and global solvers to perform the optimization, where the L-BFGS-B algorithm from SciPy acts as default and fallback if IPOPT is not available to import. You will find comprehensive instructions on how to (optionally) install and configure IPOPT for Linux in the corresponding folder.
