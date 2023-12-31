<h3>How to install IPOPT with HSL and cyipopt for Python</h3>

Here is a very rough guide for installing the version of **cyipopt** with custom compiled **IPOPT** and **HLS** solvers on a Ubuntu system (see also [https://cyipopt.readthedocs.io/en/latest/install.html#conda-forge-binaries-with-hsl](https://cyipopt.readthedocs.io/en/latest/install.html#conda-forge-binaries-with-hsl)):

1) Install system wide dependencies:
    `sudo apt install pkg-config python-dev wget`
    `sudo apt build-dep coinor-libipopt1v5`
note: it might be necessary to enable additional **deb-src** to build the **coinor** dependency. 

2) Compile **IPOPT**:
    `cd pyanno4rt/optimization/solvers`
    `wget https://www.coin-or.org/download/source/Ipopt/Ipopt-3.11.8.tgz`
    `tar -xvf Ipopt-3.11.8.tgz`
    `export IPOPTDIR=.../solvers/Ipopt-3.11.8.`
note: when opening the terminal within the solvers folder, you can just use \`pwd\` to get the current working directory (i.e. \`pwd\`/Ipopt-3.11.8 is equivalent to the above).

3) To make use of the linear solvers of **HSL**, you need to download it from [https://www.hsl.rl.ac.uk/ipopt/](https://www.hsl.rl.ac.uk/ipopt/). This requires registering for an academic version of the software, which you are not allowed to distribute! Extract the **HSL** source code using the `tar ..` command (see above), rename the extracted folder to **coinhsl** and move it into the folder **Ipopt-3.11.8/ThirdParty/HSL**. 
	 
4) Build **IPOPT**:
    `mkdir $IPOPTDIR/build`
    `cd $IPOPTDIR/build`
    `../configure`
    `make`
    `make test`
    `make install`
note: `configure`, `make` and `make test` should run without errors (warnings don't matter too much).

5) Set environment variables:
    `export IPOPT_PATH=.../solvers/Ipopt-3.11.8/build`
    `export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$IPOPT_PATH/lib/pkgconfig`
    `export PATH=$PATH:$IPOPT_PATH/bin`
    Additionally, we need to set `LD_LIBRARY_PATH`. This can be done temporarily via
    `export LD_LIBRARY_PATH=.../solvers/Ipopt-3.11.8/build/lib`
    or permanently via
    `echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.../solvers/Ipopt-3.11.8/build/lib' >> ~/.bashrc`
    Setting these variables should enable **cyipopt** to find all shared files from **IPOPT** and **HSL** during compilation and installation.

6) Compile **cyipopt**:
    `cd .../solvers`
    `wget https://files.pythonhosted.org/packages/05/57/a7c5a86a8f899c5c109f30b8cdb278b64c43bd2ea04172cbfed721a98fac/ipopt-0.1.9.tar.gz`
    `tar -xvf ipopt-0.1.9.tar.gz`
    `cd ipopt-0.1.9`
    `python setup.py build`
    `ldd build/lib.linux-x86_64-3.7/cyipopt.cpython-37m-x86_64-linux-gnu.so` # 3.7 is the Python version used; this command should output a list of .so files with their sources (libipopt.so.1 and libcoinhsl.so.1 should have a custom url)
    `python setup.py install`
    note: when installing within a virtual environment, the latter installation has to be adapted, i.e., run
    `/path/to/env/bin/python setup.py install`
    from the cyipopt folder, where /path/to/env is the path to your virtual environment folder.
    
7) Check the installation:
    `cd .../solvers/ipopt-0.1.9/test`
    `python -c "import cyipopt"`
    `python examplehs071.py`
    This should return some descriptive optimization output when being successful.

8) Import and use **cyipopt** in Python:
    `import cyipopt`
    functions: `problem, addOption, solve, ...`
    important: `addOption('linear_solver': 'ma57')` # or `ma27`, ...
		
		
		
		
