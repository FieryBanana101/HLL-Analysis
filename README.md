
# Analysis of the Harrow–Hassidim–Lloyd (HHL) Algorithm Through Comparison with Classical Linear System Solvers


## Description
This repository contains an experiment to analyze the Harrow-Hassidim-Lloyd quantum linear system solver and compare it to existing and general use classical linear solver such as the Gauss-Jordan elimination method. The error between ratio of the output will be compared, and time efficiency will also be a benchmark. The experiment is carried out using the Qiskit quantum circuit simulator and Qiskit Aer quantum computer runner.

## How to Use
```
git clone https://github.com/FieryBanana101/HLL-Analysis
python3 -m venv .venv
pip3 install -r requirements.txt
python3 HHL.py
```

## Directory Structure
.
├── HHL.py
├── README.md
└── requirements.txt