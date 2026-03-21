import os
import pickle

import numpy as np
import pandas as pd
import scipy
import matplotlib
from matplotlib import pyplot as plt

import util
from spme import SPMePlus, SPMe, SSCTimeConstantTheoretical_0p, SolidStoichiometryCorrection, SSCGainTheoretical

matplotlib.set_loglevel("error")
plt.style.use(['tableau-colorblind10', './thesisformat-lg.mplstyle'])

if __name__ == '__main__':
    # Load PyBaMM simulation data from file.
    with open(os.path.join('datasets', 'PDAE_cellGraphite_25degC.pickle'), "rb") as f:
        sim_data = pickle.load(f)

    # Load trained IIR filters.
    with open(os.path.join('datasets', 'IIR_cellGraphite_25degC.pickle'), "rb") as f:
        iir_data = pickle.load(f)

    gain1 = SSCGainTheoretical(sim_data['cell'], 1, TdegC=sim_data['TdegC'])
    gain2 = SSCGainTheoretical(sim_data['cell'], 2, TdegC=sim_data['TdegC'])
    tau1 = SSCTimeConstantTheoretical_0p(sim_data['cell'], 1, TdegC=sim_data['TdegC'])
    tau2 = SSCTimeConstantTheoretical_0p(sim_data['cell'], 2, TdegC=sim_data['TdegC'])
    c1 = SolidStoichiometryCorrection(
        sim_data['cell'], gain1, tau1,
        ts=sim_data['ts'],
        TdegC=sim_data['TdegC'],
        gain_placement='cc', )
    c2 = SolidStoichiometryCorrection(
        sim_data['cell'], gain2, tau2,
        ts=sim_data['ts'],
        TdegC=sim_data['TdegC'],
        gain_placement='cc', )

    # c1 = iir_data['correction1_3p']
    # c2 = iir_data['correction2_3p']

    c1.gain.plot(title=r"Gain at $\tilde{x}=1$")
    plt.savefig(os.path.join('plots-graphite', 'iir_params', f'gain1.png'), bbox_inches='tight')
    plt.savefig(os.path.join('plots-graphite', 'iir_params', f'gain1.eps'), bbox_inches='tight')
    c2.gain.plot(title=r"Gain at $\tilde{x}=2$")
    plt.savefig(os.path.join('plots-graphite', 'iir_params', f'gain2.png'), bbox_inches='tight')
    plt.savefig(os.path.join('plots-graphite', 'iir_params', f'gain2.eps'), bbox_inches='tight')

    # ax1 = c1bb.tau.plot(title=r"Time Constant at $\tilde{x}=1$", label='BB')
    # c1cb.tau.plot(ax=ax1, label='CB', color='C1')
    # c1cc.tau.plot(ax=ax1, label='CC', color='C2')
    # c1bc.tau.plot(ax=ax1, label='BC', color='C3')
    # c1blend.tau.plot(ax=ax1, label='Blend', color='C4')
    ax1 = c1.tau.plot(title=r"Time Constant at $\tilde{x}=1$", label='0p', color='C1')
    plt.savefig(os.path.join('plots-graphite', 'iir_params', f'tau1.png'), bbox_inches='tight')
    plt.savefig(os.path.join('plots-graphite', 'iir_params', f'tau1.eps'), bbox_inches='tight')

    # ax2 = c2bb.tau.plot(title=r"Time Constant at $\tilde{x}=2$", label='BB')
    # c2cb.tau.plot(ax=ax2, label='CB', color='C1')
    # c2cc.tau.plot(ax=ax2, label='CC', color='C2')
    # c2bc.tau.plot(ax=ax2, label='BC', color='C3')
    # c2blend.tau.plot(ax=ax2, label='Blend', color='C4')
    ax2 = c2.tau.plot(title=r"Time Constant at $\tilde{x}=2$", label='0p', color='C1')
    plt.savefig(os.path.join('plots-graphite', 'iir_params', f'tau2.png'), bbox_inches='tight')
    plt.savefig(os.path.join('plots-graphite', 'iir_params', f'tau2.eps'), bbox_inches='tight')

    plt.close()

    # Construct SPMe, SPMe+.
    spme = SPMe(
        sim_data['cell'],
        ns=10, ne_eff=5, ne_pos=10,
        ts=sim_data['ts'], TdegC=sim_data['TdegC'],
        use_constant_Ds=False,
    )
    spme_plus = SPMePlus(
        sim_data['cell'],
        ns=10, ne_eff=5, ne_pos=10,
        ts=sim_data['ts'], TdegC=sim_data['TdegC'],
        correction1=c1,
        correction2=c2,
        use_constant_Ds=False,
        use_ssc_feedback=True,
    )

    vars = [
        dict(name='thetas_bar1', pdae='pos_thetas_bar1', spme=None, spme_p='thetas_bar1', label=r'$\theta_\mathrm{s,avg,r}(\tilde{x}=1) - \theta_\mathrm{s,avg}$'),
        dict(name='thetas_bar2', pdae='pos_thetas_bar2', spme=None, spme_p='thetas_bar2', label=r'$\theta_\mathrm{s,avg,r}(\tilde{x}=2) - \theta_\mathrm{s,avg}$'),
        dict(name='thetass1', pdae='thetass1', spme='thetass', spme_p='thetass1', label=r'$\theta_\mathrm{ss}(\tilde{x}=1)$'),
        dict(name='thetass2', pdae='thetass2', spme='thetass', spme_p='thetass2', label=r'$\theta_\mathrm{ss}(\tilde{x}=2)$'),
        dict(name='if1', pdae='if1', spme='pos_if', spme_p='pos_if1', label=r'$i_\mathrm{f}(\tilde{x}=1)$ [A]'),
        dict(name='if2', pdae='if2', spme='pos_if', spme_p='pos_if2', label=r'$i_\mathrm{f}(\tilde{x}=2)$ [A]'),
        dict(name='phie2', pdae='phie2', spme='phie2', spme_p='phie2', label=r'$\phi_\mathrm{e}(\tilde{x}=2)$ [V]'),
        dict(name='etas1', pdae='etas1', spme='pos_etas', spme_p='pos_etas1', label=r'$\eta_\mathrm{s}(\tilde{x}=1)$ [V]'),
        dict(name='etas2', pdae='etas2', spme='pos_etas', spme_p='pos_etas2', label=r'$\eta_\mathrm{s}(\tilde{x}=2)$ [V]'),
        dict(name='vcell', pdae='vcell', spme='vcell', spme_p='vcell', label=r'Cell Voltage, $v_\mathrm{cell}$ [V]'),
    ]

    # Define restricted plotted time domain for drive-cycle plots for improved readability.
    time_ind_dict = {
        'cc': lambda t: np.ones_like(t, dtype=bool),
        'gitt': lambda t: np.ones_like(t, dtype=bool),
        'drive': [
            lambda t: np.logical_and(0 <= t, t <= 1600),
            lambda t: np.logical_and(3200 <= t, t <= 4200),
            lambda t: np.logical_and(0 <= t, t <= 100),
        ],
    }

    for dataset_key, dataset in sim_data['datasets'].items():  # train, test

        series_labels = []
        spec_labels = []
        rmse_vcell_spme = []
        rmse_vcell_spme_plus = []

        for series_key, series in dataset.items():  # cc, gitt, drive
            for sim_data in series:
                spec_string = util.sim_specs(series_key, sim_data)

                soc0Pct = sim_data['soc0']
                fom_sim = sim_data['pybamm']
                spme_sim = spme.run(fom_sim['iapp'], soc0Pct)
                spme_plus_sim = spme_plus.run(fom_sim['iapp'], soc0Pct)
                soc_pct = sim_data['pybamm']['soc_pct']
                time = fom_sim['time']
                ind_rmse = soc_pct >= 3.0

                # Make plot directory.
                plot_dir = os.path.join('plots-graphite', dataset_key, series_key, spec_string)
                if not os.path.isdir(plot_dir):
                    os.makedirs(plot_dir)

                time_ind_arr = time_ind_dict[series_key]
                if not isinstance(time_ind_arr, list):
                    time_ind_arr = [time_ind_arr]

                for kt, time_ind_fn in enumerate(time_ind_arr):
                    time_ind = time_ind_fn(time)
                    for var in vars:
                        name = var['name']
                        label = var['label']
                        qty_true = fom_sim[var['pdae']]
                        if var['spme'] is None:
                            qty_spme = None
                        else:
                            qty_spme = spme_sim[var['spme']]
                        qty_spme_plus = spme_plus_sim[var['spme_p']]
                        if qty_spme is None:
                            rmse_spme = None
                        else:
                            rmse_spme = 1000 * np.sqrt(np.mean((qty_true[ind_rmse] - qty_spme[ind_rmse]) ** 2, axis=0))
                        rmse_spme_plus = 1000 * np.sqrt(np.mean((qty_true[ind_rmse] - qty_spme_plus[ind_rmse]) ** 2, axis=0))

                        if var['name'] == 'vcell' and kt == 0:
                            series_labels.append(series_key)
                            spec_labels.append(spec_string)
                            rmse_vcell_spme.append(rmse_spme)
                            rmse_vcell_spme_plus.append(rmse_spme_plus)
                            # print(f"{dataset_key} {series_key} {spec_string}: {rmse_spme:.4f}mV -> "
                            #       f" {rmse_spme_plus_bb:.4f} / {rmse_spme_plus_cb:.4f} / "
                            #       f"{rmse_spme_plus_cc:.4f} / {rmse_spme_plus_bc:.4f} / "
                            #       f"{rmse_spme_plus_blend:.4f} / {rmse_spme_plus_1p:.4f} / "
                            #       f"{rmse_spme_plus_3p:.4f}mV RMSE")
                            print(f"{dataset_key} {series_key} {spec_string}: "
                                  f"{rmse_spme:.4f}mV -> "
                                  f"{rmse_spme_plus:.4f}mV RMSE")

                        _, ax = plt.subplots(constrained_layout=True)
                        ax.set_box_aspect(1 / scipy.constants.golden)
                        plt.plot(time[time_ind]/60, qty_true[time_ind], label='PDAE')
                        plt.plot(time[time_ind] / 60, qty_spme_plus[time_ind], label='SPMe+ (0pCC)', linestyle='--')
                        if qty_spme is not None:
                            plt.plot(time[time_ind]/60, qty_spme[time_ind], label='SPMe', linestyle=':')
                        if name.startswith('etas'):
                            ind_plt = soc_pct[time_ind] >= 5.0
                            plt.ylim(np.min(qty_true[time_ind][ind_plt]), np.max(qty_true[time_ind][ind_plt]))
                        plt.legend()
                        plt.xlabel(r'Time, $t$ [min]')
                        plt.ylabel(label)
                        plt.title(f'{name}: {dataset_key} {series_key} {spec_string}')
                        plt.savefig(os.path.join(plot_dir, f'{name}-t{kt}.png'), bbox_inches='tight')
                        plt.savefig(os.path.join(plot_dir, f'{name}-t{kt}.eps'), bbox_inches='tight')
                        plt.close()

        # Make a data directory.
        data_dir = os.path.join('performance_data_graphite', dataset_key)
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        metrics = pd.DataFrame({
            'series': series_labels,
            'spec': spec_labels,
            'rmse_vcell_spme': rmse_vcell_spme,
            'rmse_vcell_spme_plus': rmse_vcell_spme_plus,
        })
        metrics.to_excel(os.path.join(data_dir, 'metrics.xlsx'))