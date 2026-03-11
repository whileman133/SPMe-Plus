"""
gen_spm_datasets.py

Augment PyBaMM training/testing data with SPM simulation results.

2024.10.28 | Created | Wesley Hileman <whileman@uccs.edu>
"""

import os
import pickle
from spme import SPMe

if __name__ == '__main__':
    # Load PyBaMM simulation data from file.
    cell_name = 'cellLMO'
    with open(os.path.join('datasets', f'PDAE_{cell_name}_25degC.pickle'), "rb") as f:
        data = pickle.load(f)

    # Simulate SPMe.
    cell = data['cell']
    spme = SPMe(
        cell,
        ns=10, ne_eff=5, ne_pos=10,
        ts=data['ts'], TdegC=data['TdegC'],
        use_constant_Ds=True
    )
    for dataset_key, dataset in data['datasets'].items():  # train, test
        for series_key, series in dataset.items():    # cc, gitt, drive
            for sim_data in series:
                soc0Pct = sim_data['soc0']
                iapp = sim_data['pybamm']['iapp']
                # Place simulation results in another key in the data dictionary.
                sim_data['spme'] = spme.run(iapp, soc0Pct)


    # Store SPMe instance in data dictionary.
    data['spme'] = spme

    # Save data dictionary to file.
    with open(os.path.join('datasets', f'PDAE_SPMe_{cell_name}_25degC.pickle'), "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
