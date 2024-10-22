# Sparse-control
Code for control design algorithms and experiments in paper "Pointwise-Sparse Actuator Scheduling for Linear Systems with Controllability Guarantee" by L. Ballotta, G. Joseph, and I. R. Thete.
For a given LTI system of the form

$$ x(t+1) = Ax(t) + B(t)u(t) \qquad t = 0,1,\dots, h-1$$

the algorithms compute an $s$-sparse actuator schedule $S = (S_0, S_1, \dots, S_{h-1})$ where each $S_k$ contains the indices of the actuators (i.e., columns of $B$/elements of $u$) that are active at time $k$, under the point-wise sparsity constraint $|S_k| \le s \ \forall k$.
Specifically, given dimension $n$ of the state $x$, the algorithms attempt to heuristicaly solve the optimization problem

$$ \min_S \qquad \rho(W_S)$$

$$ \mbox{subject to} \quad |S_k| \le s \ \forall k $$

$$ \hspace{2cm}      \mathrm{rk}(W_S) = n$$

where $W_S$ is the controllability Gramian under schedule $S$, $\rho(W_S)$ is a performance metric related to the control energy, and the rank constraint enforces controllability with the scheduled channels.
The $s$-sparse greedy algorithm _s_greedy_ in _control_design/control_design.py_ provides formal controllability guarantees under the sparsity constraint.

## How to run
There are two runnable scripts:
- _main.py_ performs the control design for given LTI system ($A,B$), sparsity constraint $s$, and time horizon $h$ comparing different algorithms;
- _main_s.py_ performs the control design for given LTI system ($A,B$) and time horizon $h$ with $s$-sparse greedy and $s$-sparse MCMC for a given range of sparsity constraints $[s_\text{min},s_\text{max}]$;
- _plotter_s.py_ plots the two cost curves obtained with _main_s.py_ analogously to Fig. 1 in the paper.

The design algorithms are defined in _control_design/control_design.py_.
The cost functions are defined in _control_design/cost_function.py_.
For more details, please see the documentation in the scripts.

## Link to papers
IEEEXplore: https://ieeexplore.ieee.org/document/10706838

Arxiv: https://arxiv.org/abs/2407.12125

Citation:
```
@ARTICLE{10706838,
  author={Ballotta, Luca and Joseph, Geethu and Thete, Irawati Rahul},
  journal={IEEE Control Systems Letters}, 
  title={Pointwise-Sparse Actuator Scheduling for Linear Systems With Controllability Guarantee}, 
  year={2024},
  volume={8},
  pages={2361 - 2366},
  doi={10.1109/LCSYS.2024.3475886}
}
```
