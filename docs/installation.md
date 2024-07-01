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
