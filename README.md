# Multiscale Approximation
## Introduction
This is an infrastructure for scattered data multiscale approximation.
This readme describes the code and how to run it. 
Numerical examples from our paper and how to run them can be found in `NumericalExamples`.

The code is modular, and designed for customization. It is possible to choose and add new methods:
- Manifold - The range of the approximated function.
- Approximation method - Currently implemented quasi-interpolation, multiscale approach, naive kernel interpolation,...
- Data structure - The algorithm of storing and querying the data sites.
- Data sites generation - The method of choosing the sampling sites.

## How to run?
Install Requirements
```bash
pip install -r requirements.txt
```
For flexible options, run `runner.py`:
```bash
# Approximate 3 scales both multiscale and single scale
python runner.py -m rotations -f ExampleFunctions.euler -s -n 3

# For help:
python runner.py -h
```

To run experiments from the paper, run:
```bash
python NumericalExamples/{{EXPERIMENT_NAME}}
```

## Design
### Main experiment
The main multiscale logic is in `Experiment`, in the `multiscale_approximation()`. 
One can run an experiment from the `runner`, 
which is flexible, or run the examples from the paper in `NumericalExamples`.

### Config
The module `Config` contains the `config` object that 
holds the configurations for the current experiment. 
`config` is an instance of the `Config` class. It allows to `set_base_config`,
and `renew` to the base config. It loads its default values from `defaults`. 
In order to update the config by a differences `dict`, use the method `update_config_with_diff`.
### Approximation methods
- `Quasi` performs averaging using RBF coefficients.
    - $Q^Mf(x):=av_M(\Phi(x),f(\Xi))$
- `Moving` is a moving least squares that promises polynomial reproduction. It is based on the `PolynomialReproduction` module.
### Manifolds
- `AbstractManifold.py` is the base class for the manifolds. It implements naively some required APIs for a manifold.
- `Circle.py` is the $S^1$ single dimensional sphere manifold. 
	- The suggested average is geodesic.
	- Exp-Log pair is
		- $exp(x,y)={{x+y}\over{|x+y|}}$
		- $log(x,y)={y\over{|<x,y>|}}-x$
-   `SymmetricPositiveDefinite.py` is the $SPD$ manifold.
	- Averages using `KarcherMean.py`.
	- Visualizes using `Visualization.py`
	- Exp-Log pair is 
		- $exp(x,y)=x^{1/2}EXP(x^{-1/2}yx^{-1/2})x^{1/2}$
		- $log(x,y)=x^{1/2}LOG(x^{-1/2}yx^{-1/2})x^{1/2}$
### Tools
- `Utils.py` has some util functions:
	- caching
	- operations on functions
	- grid operations
	- plotting
	- RBF functions
	- output directory management  
- `KarcherMean.py` - calculates weighted Karcher mean.
- `Visualization.py` - Ellipsoid visualization of SPD matrices.
