import os
from dataclasses import dataclass

import scipy
import numpy as np
import matplotlib.pyplot as plt

from cellparams import load_cell_params
from util import sim_cc

plt.style.use(['tableau-colorblind10', './thesisformat-lg.mplstyle'])


if __name__ == '__main__':
    params = load_cell_params(os.path.join('XLSX_CELLDEFS','cellNMC30.xlsx'))
    Q = params.const.Q()

    # Define constants.
    ts = 1          # sampling interval [s]
    TdegC = 25      # cell temperature [degC]
    C_rate = 1
    soc0_pct = 60
    socf_pct = 58
    t_rest = 1000
    xneg = np.linspace(0, 1, 20)
    xpos = np.linspace(2, 3, 20)

    # Run PyBaMM simulation.
    sim = sim_cc(params, C_rate*Q, soc0_pct, socf_pct, TdegC=TdegC, t_rest=t_rest, ts=ts)
    time = sim.solution['time [s]']()
    iapp = sim.solution['iapp [A]']()
    tend = np.argwhere(iapp==0)[0][0]
    thetass0 = sim.solution['thetass'](x=0)
    thetass3 = sim.solution['thetass'](x=3)
    neg_thetass = sim.solution['neg_thetass'](x=xneg)
    pos_thetass = sim.solution['pos_thetass'](x=xpos)
    vcell_pybamm = sim.solution['vcell [V]']()

    # Plot vcell
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_box_aspect(1 / scipy.constants.golden)
    plt.plot(time, vcell_pybamm, label='Newman')
    plt.xlabel(r'Time, $t$ [sec]')
    plt.ylabel(r'Cell Voltage, $v_\mathrm{cell}(t)$')
    plt.title(r'Cell Voltage')
    plt.legend()

    # Plot Thetass
    # Vs time:
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_box_aspect(1/scipy.constants.golden)
    plt.plot(time, thetass0, label='x=0')
    plt.plot(time, thetass3, label='x=3')
    plt.xlabel(r'Time, $t$ [sec]')
    plt.ylabel(r'$\theta_\mathrm{ss}(\tilde{x}=3)$')
    plt.title(r'Surf. Stoich. at $\tilde{x}=3$ vs. Time')
    plt.legend()
    # Vs electrode thickness:
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_box_aspect(1 / scipy.constants.golden)
    plt.plot(xneg, neg_thetass[:,tend])
    plt.plot(xpos, pos_thetass[:,tend])
    plt.xlabel(r'Linear Position, $\tilde{x}$')
    plt.ylabel(r'$\theta_\mathrm{ss}$')
    plt.title(r'Surf. Stoich. at $t=380\,\mathrm{s}$ vs. Position')
    plt.legend()

    plt.show()

