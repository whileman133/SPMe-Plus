"""
runCC.py

Simulate constant-current discharge.
"""
import os

from cellparams import load_cell_params
from util import sim_cc

if __name__ == '__main__':
    params = load_cell_params(os.path.join('XLSX_CELLDEFS','cellNMC30.xlsx'))
    Q = params.const.Q()
    sim = sim_cc(params, 1*Q, 100, 80, t_rest=10*60)
    sim.plot()