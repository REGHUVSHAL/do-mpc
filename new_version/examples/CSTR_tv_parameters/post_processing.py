#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc
import scipy.io as sio
import matplotlib.pyplot as plt
import pickle


with open('./results/results.pkl', 'rb') as f:
    results = pickle.load(f)

graphics = do_mpc.backend_graphics()


fig, ax = plt.subplots(3, sharex=True)
graphics.add_line(var_type='_x', var_name='C_a', axis=ax[0])
graphics.add_line(var_type='_x', var_name='C_b', axis=ax[0])
graphics.add_line(var_type='_u', var_name='Q_dot', axis=ax[1])
graphics.add_line(var_type='_u', var_name='F', axis=ax[2])
ax[0].set_ylabel('c [mol/l]')
ax[1].set_ylabel('Q_heat [kW]')
ax[2].set_ylabel('Flow [l/h]')

"""
Static plot Example
"""
opti_lines = graphics.plot_results(results['optimizer'])
simu_lines = graphics.plot_results(results['optimizer'])

plt.sca(ax[0])
ax[0].add_artist(plt.legend(opti_lines[:2], ['Ca', 'Cb'], title='optimizer', loc=1))
plt.sca(ax[0])
ax[0].add_artist(plt.legend(simu_lines[:2], ['Ca', 'Cb'], title='Simulator', loc=2))
plt.show()
