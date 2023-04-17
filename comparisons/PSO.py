import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from IPython.display import Image

import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx 
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
from pyswarms.utils.plotters.formatters import Mesher
from pyswarms.utils.plotters.formatters import Designer
# in : (n,d)  out: (n)
sigma = 2
func_4 = lambda x: np.array([-20*np.exp(-0.2*np.sqrt(0.5*x[i][0]*x[i][0]+0.5*x[i][1]*x[i][1])) - np.exp(0.5*np.cos(2*math.pi*x[i][0]) + 0.5*np.cos(2*math.pi*x[i][1])) + 20 + math.e  for i in range(len(x))])
# func_3 = lambda x: [ - 1/(2 * np.pi * (sigma**2)) * np.exp(-((xi[0]+5)**2+(xi[1]+5)**2)/(2 * sigma**2) - 0.7*1/(2 * np.pi * (sigma**2)) * np.exp(-((xi[0]-5)**2+(xi[1]-5)**2)/(2 * sigma**2)) ) for xi in x]
x_max = 10 * np.ones(2)
x_min = -1 * x_max
bounds = (x_min, x_max)
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=2, options=options, bounds=bounds)
cost, pos = optimizer.optimize(func_4, iters=100)

# plot_cost_history(cost_history=optimizer.cost_history)
# plt.show()

m = Mesher(func=func_4)
# animation = plot_contour(pos_history=optimizer.pos_history,
#                          mesher=m,
#                          mark=(0,0))
# animation.save('plot0.gif', writer='imagemagick', fps=10)
# Image(url='plot0.gif')

pos_history_3d = m.compute_history_3d(optimizer.pos_history)
d = Designer(limits=[(-1,1), (-1,1), (-0.1,1)], label=['x-axis', 'y-axis', 'z-axis'])
animation3d = plot_surface(pos_history=pos_history_3d, # Use the cost_history we computed
                           mesher=m, designer=d,       # Customizations
                           mark=(0,0,0))  
animation3d.save('plot1.gif', writer='imagemagick', fps=10)
Image(url='plot1.gif')