# Sparse-control
Code for control design algorithms and experiments in paper "Pointwise-Sparse Actuator Scheduling for Linear Systems with Controllability Guarantee" by L. Ballotta, G. Joseph, and I. R. Thete.

## How to run
There are two runnable scripts:
- _main.py_ performs the control design for given LTI system (_A_,_B_), sparsity constraint _s_, and time horizon _h_ comparing different algorithms;
- _main_s.py_ performs the control design for given LTI system (_A_,_B_) and time horizon _h_ with _s_-sparse greedy and _s_-sparse MCMC for a given range of sparsity constraints [_s_min_, _s_max_].

The design algorithms are defined in _control_design/control_design.py_.
The cost functions are defined in _control_design/cost_function.py_.
For more details, please see the documentation in the scripts.

## Link to paper
Arxiv: https://arxiv.org/abs/2407.12125

Please cite as: L. Ballotta, G. Joseph, and I. R. Thete, "Pointwise-Sparse Actuator Scheduling for Linear Systems with Controllability Guarantee," _arXiv e-prints_, page arxiv:2407.12125, 2024.
