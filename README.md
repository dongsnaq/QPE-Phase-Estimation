# QSP Phase Estimation

A Python implementation of optimally-designed programmable signal for quantum phase estimation (QPE) via quantum signal processing (QSP), together with an iterative refinement scheme.

## Reference

Zikang Jia, Suying Liu, Yulong Dong. Programmable Signal Design for Quantum Phase Estimation via Quantum Signal Processing.


## Files

`solving_optimally_designed_signal.py`: Design an optimal cosine polynomial signal for QPE. Given a degree `d`, a prior center `θ̂`, and a prior radius `r`, it maximizes the proxy `2α·β` subject to amplitude and derivative constraints on the prior interval, then returns the scaled Fourier cosine coefficients and the minimum derivative lower bound `L` over the prior.

`iterative_refinement_scheme.py`: Implements the iterative refinement loop. At each stage it calls `solve_optimally_designed_signal` to design a signal adapted to the current prior, simulates a noisy measurement outcome, applies the inverse mapping via bounded scalar minimization, and updates the estimate and prior radius.

`test.ipynb`: Test notebook containing two experiments:
- **Experiment 1** – sweeps over `d` and plots the resulting signals, zoom-in views with the sensitivity parameter κ, and Fourier coefficient spectra in a figure.
- **Experiment 2** – runs a single step of the iterative refinement scheme and prints the refined estimate, estimation error, updated radius, `L`, and κ.

## Requirements

### Python packages

```
python >= 3.8
numpy
scipy
matplotlib
seaborn
tqdm
gurobipy
```

Install the Python dependencies (except Gurobi) with:

```bash
pip install numpy scipy matplotlib seaborn tqdm
```

## Running the tests

Once the requirements are installed, open and run `test.ipynb` from top to bottom. Test 1 will print progress via `tqdm` and generate the figure. Test 2 will print output of the form:

```
Refined estimate: 0.4981...
Error:            0.0018...
New radius:       0.0049...
Lk:               7.37...
```
