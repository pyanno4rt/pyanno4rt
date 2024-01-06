# pyanno4rt: Python-based Advanced Numerical Nonlinear Optimization for Radiotherapy

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
					<li> 16-type objective catalogue, e.g. squared deviation, squared overdosing, logistic regression NTCP, ... </li>
					<li> 6-type constraint catalogue, e.g. minimum DVH, maximum DVH, maximum NTCP, ... </li>
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
			<li> 3 highly customizable internal model classes </li>
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
