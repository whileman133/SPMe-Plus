import os
import numpy as np
import matplotlib.pyplot as plt
import util
from spme import SPMe

plt.style.use(['tableau-colorblind10', './thesisformat-lg.mplstyle'])
plt.rcParams["text.usetex"] = True


if __name__ == '__main__':
    cell = util.load_mat_cell_model(os.path.join('MAT_CELLDEFS', 'cellGraphite.mat'))

    # Define constants.
    ts = 1          # sampling interval [s]
    TdegC = 25      # cell temperature [degC]
    C_rate = 2.0
    soc0_pct = 100
    socf_pct = 0
    t_rest = 1000
    xpos = np.linspace(1, 2, 20)
    spme = SPMe(cell, ns=10, ne_eff=5, ne_pos=10, ts=ts, TdegC=TdegC)
    r = spme.solid_fvs.edges
    xThetaeSPM = spme.electrolyte_fvs.edges

    # Run PyBaMM simulation.
    sim = util.sim_cc(
        cell,
        i_galv=C_rate * cell.cst.QAh,
        soc0=soc0_pct, socf=socf_pct,
        TdegC=TdegC, ts=ts,
    )
    time = sim.solution['time [s]']()
    iapp = sim.solution['iapp [A]']()
    thetass2 = sim.solution['thetass'](x=2)
    thetass = sim.solution['pos_thetass'](x=xpos)
    thetas2 = sim.solution['pos_thetas'](x=2, r=r)
    vcell_pybamm = sim.solution['vcell [V]']()
    sim.plot()