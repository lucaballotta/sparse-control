import matplotlib.pyplot as plt
import matplotlib.axes as ax
import pickle
import tikzplotlib

from matplotlib.lines import Line2D
from matplotlib.legend import Legend

# fix for tikzplotlib
Line2D._us_dashSeq    = property(lambda self: self._dash_pattern[1])
Line2D._us_dashOffset = property(lambda self: self._dash_pattern[0])
Legend._ncol = property(lambda self: self._ncols)

# import plot data
file_name = 'ex5_s_logdet_202407040140'
with open('exp/' + file_name + '.pickle', 'rb') as f:
    vars = pickle.load(f) # type: dict

cost_s_greedy = vars['cost_s_greedy'] # type: dict
cost_s_greedy_mcmc = vars['cost_s_greedy_mcmc'] # type: dict

s_greedy_plt = sorted(cost_s_greedy.items())
s_greedy_mcmc_plt = sorted(cost_s_greedy_mcmc.items())
s, cost_s_greedy = zip(*s_greedy_plt)
_, cost_s_greedy_mcmc = zip(*s_greedy_mcmc_plt)

# plot data
plt.figure
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_xlabel(r'Sparsity $s$', fontsize=18)
ax.set_ylabel(r'Cost', fontsize=18)
plt.plot(s, cost_s_greedy,
         marker='o',
         linewidth=2,
         markersize=18,
         label=r'$s$-sparse greedy'
)
plt.plot(s, cost_s_greedy_mcmc,
         marker='s',
         linestyle='dashed',
         linewidth=2,
         markersize=18,
         label=r'$s$-sparse greedy + MCMC'
)
plt.grid(which='major')
ax.legend(loc='best', fontsize='x-large')
tikzplotlib.save("exp.tex")
plt.close()