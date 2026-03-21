import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy
import matplotlib
from matplotlib import pyplot as plt

import util
from spme import SPMePlus, SPMe, SSCTimeConstantTheoretical_0p, SolidStoichiometryCorrection, SSCGainTheoretical, SSCTimeConstantTF

matplotlib.set_loglevel("error")
plt.style.use(['tableau-colorblind10', './thesisformat-lg.mplstyle'])

if __name__ == '__main__':
    # Load PyBaMM simulation data from file.
    with open(os.path.join('datasets', 'PDAE_cellLMO_25degC.pickle'), "rb") as f:
        sim_data = pickle.load(f)

    # Load trained IIR filters.
    with open(os.path.join('datasets', 'IIR_25degC.pickle'), "rb") as f:
        iir_data = pickle.load(f)

    tau1_tf, theta1_tf = util.load_mat_tau(os.path.join('MATLAB','CELLPARAMS','tau1.mat'))
    tau2_tf, theta2_tf = util.load_mat_tau(os.path.join('MATLAB','CELLPARAMS','tau2.mat'))

    # c1bb = iir_data['correction1bb']
    # c2bb = iir_data['correction2bb']
    # c1cb = iir_data['correction1cb']
    # c2cb = iir_data['correction2cb']
    # c1cc = iir_data['correction1cc']
    # c2cc = iir_data['correction2cc']
    # c1bc = iir_data['correction1bc']
    # c2bc = iir_data['correction2bc']
    # c1_1p = iir_data['correction1_1p']
    # c2_1p = iir_data['correction2_1p']
    # c1_3p = iir_data['correction1_3p']
    # c2_3p = iir_data['correction2_3p']
    c1blend = iir_data['correction1blend']
    c2blend = iir_data['correction2blend']

    variants = ['bb', 'cb', 'cc', 'bc']

    gain1 = SSCGainTheoretical(sim_data['cell'], 1, TdegC=sim_data['TdegC'])
    gain2 = SSCGainTheoretical(sim_data['cell'], 2, TdegC=sim_data['TdegC'])
    #tau1 = SSCTimeConstantTheoretical_0p(sim_data['cell'], 1, TdegC=sim_data['TdegC'])
    #tau2 = SSCTimeConstantTheoretical_0p(sim_data['cell'], 2, TdegC=sim_data['TdegC'])
    tau1 = SSCTimeConstantTF(theta1_tf, tau1_tf)
    tau2 = SSCTimeConstantTF(theta2_tf, tau2_tf)
    c1 = dict()
    c2 = dict()
    for k, v in enumerate(variants):
        c1[v] = SolidStoichiometryCorrection(
            sim_data['cell'], gain1, tau1,
            ts=sim_data['ts'],
            TdegC=sim_data['TdegC'],
            gain_placement=v, )
        c2[v] = SolidStoichiometryCorrection(
            sim_data['cell'], gain2, tau2,
            ts=sim_data['ts'],
            TdegC=sim_data['TdegC'],
            gain_placement=v,
        )
    variants.append('blend')
    c1['blend'] = c1blend
    c2['blend'] = c2blend
    # c1cc = SolidStoichiometryCorrection(
    #     sim_data['cell'], gain1, tau1,
    #     ts=sim_data['ts'],
    #     TdegC=sim_data['TdegC'],
    #     gain_placement='cc', )
    # c2cc = SolidStoichiometryCorrection(
    #     sim_data['cell'], gain2, tau2,
    #     ts=sim_data['ts'],
    #     TdegC=sim_data['TdegC'],
    #     gain_placement='cc', )

    gain1.plot(title=r"Gain at $\tilde{x}=1$")
    plt.savefig(os.path.join('plots', 'iir_params', f'gain1.png'), bbox_inches='tight')
    plt.savefig(os.path.join('plots', 'iir_params', f'gain1.eps'), bbox_inches='tight')
    gain2.plot(title=r"Gain at $\tilde{x}=2$")
    plt.savefig(os.path.join('plots', 'iir_params', f'gain2.png'), bbox_inches='tight')
    plt.savefig(os.path.join('plots', 'iir_params', f'gain2.eps'), bbox_inches='tight')

    ax1 = gain1.plot(title=r"Gain at $\tilde{x}=1,2$", label=r"$\tilde{x}=1$")
    gain2.plot(ax=ax1, label=r"$\tilde{x}=2$", color='C1')
    plt.legend()
    plt.savefig(os.path.join('plots', 'iir_params', f'gain_combined.png'), bbox_inches='tight')
    plt.savefig(os.path.join('plots', 'iir_params', f'gain_combined.eps'), bbox_inches='tight')

    # ax1 = c1bb.tau.plot(title=r"Time Constant at $\tilde{x}=1$", label='BB')
    # c1cb.tau.plot(ax=ax1, label='CB', color='C1')
    # c1cc.tau.plot(ax=ax1, label='CC', color='C2')
    # c1bc.tau.plot(ax=ax1, label='BC', color='C3')
    # c1blend.tau.plot(ax=ax1, label='Blend', color='C4')
    # ax1 = c1_0p.tau.plot(title=r"Time Constant at $\tilde{x}=1$", label='0p', color='C1')
    # c1bb.tau.plot(ax=ax1, label='8pBB', color='C2')
    # c1cc.tau.plot(ax=ax1, label='8pCC', color='C3')
    _, ax = plt.subplots(constrained_layout=True)
    ax.set_box_aspect(1 / scipy.constants.golden)
    plt.plot(theta1_tf, tau1_tf)
    plt.xlabel(r"Average Solid Stoichiometry, $\theta_\mathrm{s,avg}$")
    plt.ylabel(r"Time Constant, $\tau$ [s]")
    plt.title(r"Time Constant at $\tilde{x}=1$")
    plt.ylim(0, 600)
    plt.savefig(os.path.join('plots', 'iir_params', f'tau1.png'), bbox_inches='tight')
    plt.savefig(os.path.join('plots', 'iir_params', f'tau1.eps'), bbox_inches='tight')

    # ax2 = c2bb.tau.plot(title=r"Time Constant at $\tilde{x}=2$", label='BB')
    # c2cb.tau.plot(ax=ax2, label='CB', color='C1')
    # c2cc.tau.plot(ax=ax2, label='CC', color='C2')
    # c2bc.tau.plot(ax=ax2, label='BC', color='C3')
    # c2blend.tau.plot(ax=ax2, label='Blend', color='C4')
    #ax2 = c2cc.tau.plot(title=r"Time Constant at $\tilde{x}=2$", label='Fitted 8pCC (Cubic Spline)', color='C1', zorder=100)
    #ax2 = c2_0p.tau.plot(title=r"Time Constant at $\tilde{x}=2$", label='0p', color='C1')
    # c2bb.tau.plot(ax=ax2, label='8pBB', color='C3')
    _, ax = plt.subplots(constrained_layout=True)
    ax.set_box_aspect(1 / scipy.constants.golden)
    plt.plot(theta2_tf, tau2_tf)
    plt.xlabel(r"Average Solid Stoichiometry, $\theta_\mathrm{s,avg}$")
    plt.ylabel(r"Time Constant, $\tau$ [s]")
    plt.title(r"Time Constant at $\tilde{x}=2$")
    plt.ylim(0, 600)
    plt.savefig(os.path.join('plots', 'iir_params', f'tau2.png'), bbox_inches='tight')
    plt.savefig(os.path.join('plots', 'iir_params', f'tau2.eps'), bbox_inches='tight')

    _, ax = plt.subplots(constrained_layout=True)
    ax.set_box_aspect(1 / scipy.constants.golden)
    plt.plot(theta1_tf, tau1_tf, label=r'$\tilde{x}=1$')
    plt.plot(theta2_tf, tau2_tf, label=r'$\tilde{x}=2$')
    plt.xlabel(r"Average Solid Stoichiometry, $\theta_\mathrm{s,avg}$")
    plt.ylabel(r"Time Constant, $\tau$ [s]")
    plt.title(r"Time Constant at $\tilde{x}=1,2$")
    plt.legend()
    plt.ylim(0, 600)
    plt.savefig(os.path.join('plots', 'iir_params', f'tau_combined.png'), bbox_inches='tight')
    plt.savefig(os.path.join('plots', 'iir_params', f'tau_combined.eps'), bbox_inches='tight')

    plt.close()

    # Construct SPMe, SPMe+.
    ns = 5
    ne_eff = 3
    ne_pos = 5
    spme = SPMe(
        sim_data['cell'],
        ns=ns, ne_eff=ne_eff, ne_pos=ne_pos,
        ts=sim_data['ts'], TdegC=sim_data['TdegC'],
    )
    spme_plus = dict()
    for v in variants:
        spme_plus[v] = SPMePlus(
            sim_data['cell'],
            ns=ns, ne_eff=ne_eff, ne_pos=ne_pos,
            ts=sim_data['ts'], TdegC=sim_data['TdegC'],
            correction1=c1[v],
            correction2=c2[v],
        )

    vars = [
        dict(name='thetas_bar1', title=r'Stoichiometry Offset at $\tilde{x}=1$', pdae='pos_thetas_bar1', spme=None, spme_p='thetas_bar1', label=r'$\theta_\mathrm{s,avg,r}(\tilde{x}=1) - \theta_\mathrm{s,avg}$'),
        dict(name='thetas_bar2', title=r'Stoichiometry Offset at $\tilde{x}=2$', pdae='pos_thetas_bar2', spme=None, spme_p='thetas_bar2', label=r'$\theta_\mathrm{s,avg,r}(\tilde{x}=2) - \theta_\mathrm{s,avg}$'),
        dict(name='thetass1', title=r'Surf. Stoich. at $\tilde{x}=1$', pdae='thetass1', spme='thetass', spme_p='thetass1', label=r'$\theta_\mathrm{ss}(\tilde{x}=1)$'),
        dict(name='thetass2', title=r'Surf. Stoich. at $\tilde{x}=2$', pdae='thetass2', spme='thetass', spme_p='thetass2', label=r'$\theta_\mathrm{ss}(\tilde{x}=2)$'),
        dict(name='thetas_avg', title=r'Avg. Stoich', pdae='pos_thetas_avg', spme='thetas_avg', spme_p='thetas_avg', label=r'$\theta_\mathrm{s,avg}$'),
        dict(name='if1', pdae='if1', title=r'Rxn. Current at $\tilde{x}=1$', spme='pos_if', spme_p='pos_if1', label=r'$i_\mathrm{f}(\tilde{x}=1)$ [A]'),
        dict(name='if2', pdae='if2', title=r'Rxn. Current at $\tilde{x}=2$', spme='pos_if', spme_p='pos_if2', label=r'$i_\mathrm{f}(\tilde{x}=2)$ [A]'),
        dict(name='phie2', title=r'Elec. Potential at $\tilde{x}=2$', pdae='phie2', spme='phie2', spme_p='phie2', label=r'$\phi_\mathrm{e}(\tilde{x}=2)$ [V]'),
        dict(name='etas1', title=r'Overpot. at $\tilde{x}=1$', pdae='etas1', spme='pos_etas', spme_p='pos_etas1', label=r'$\eta_\mathrm{s}(\tilde{x}=1)$ [V]'),
        dict(name='etas2', title=r'Overpot. at $\tilde{x}=2$', pdae='etas2', spme='pos_etas', spme_p='pos_etas2', label=r'$\eta_\mathrm{s}(\tilde{x}=2)$ [V]'),
        dict(name='thetae1', title=r'Salt Conc. at $\tilde{x}=1$', pdae='thetae1', spme='thetae1', spme_p='thetae1', label=r'$\theta_\mathrm{e}(\tilde{x}=1)$'),
        dict(name='thetae2', title=r'Salt Conc. at $\tilde{x}=2$', pdae='thetae2', spme='thetae2', spme_p='thetae2', label=r'$\theta_\mathrm{e}(\tilde{x}=2)$'),
        dict(name='vcell', title=r'Cell Voltage', pdae='vcell', spme='vcell', spme_p='vcell', label=r'Cell Voltage, $v_\mathrm{cell}$ [V]'),
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
        rmse_vcell_spme_plus = defaultdict(list)

        for series_key, series in dataset.items():  # cc, gitt, drive
            for sim_data in series:
                spec_string = util.sim_specs(series_key, sim_data)

                soc0Pct = sim_data['soc0']
                fom_sim = sim_data['pybamm']
                spme_sim = spme.run(fom_sim['iapp'], soc0Pct)
                spme_plus_sim = dict()
                for v in variants:
                    spme_plus_sim[v] = spme_plus[v].run(fom_sim['iapp'], soc0Pct)
                soc_pct = sim_data['pybamm']['soc_pct']
                time = fom_sim['time']
                ind_rmse = soc_pct >= 3.0

                # Make plot directory.
                plot_dir = os.path.join('plots', dataset_key, series_key, spec_string)
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
                        qty_spme_plus = dict()
                        for v in variants:
                            qty_spme_plus[v] = spme_plus_sim[v][var['spme_p']]
                        if qty_spme is None:
                            rmse_spme = None
                        else:
                            rmse_spme = 1000 * np.sqrt(np.mean((qty_true[ind_rmse] - qty_spme[ind_rmse]) ** 2, axis=0))
                        rmse_spme_plus = dict()
                        for v in variants:
                            rmse_spme_plus[v] = 1000 * np.sqrt(np.mean((qty_true[ind_rmse] - qty_spme_plus[v][ind_rmse]) ** 2, axis=0))

                        if var['name'] == 'vcell' and kt == 0:
                            series_labels.append(series_key)
                            spec_labels.append(spec_string)
                            rmse_vcell_spme.append(rmse_spme)
                            for v in variants:
                                rmse_vcell_spme_plus[v].append(rmse_spme_plus[v])
                            # print(f"{dataset_key} {series_key} {spec_string}: {rmse_spme:.4f}mV -> "
                            #       f" {rmse_spme_plus_bb:.4f} / {rmse_spme_plus_cb:.4f} / "
                            #       f"{rmse_spme_plus_cc:.4f} / {rmse_spme_plus_bc:.4f} / "
                            #       f"{rmse_spme_plus_blend:.4f} / {rmse_spme_plus_1p:.4f} / "
                            #       f"{rmse_spme_plus_3p:.4f}mV RMSE")

                            print(f"{dataset_key} {series_key} {spec_string}: {rmse_spme:.4f}mV -> ", end='')
                            for v in variants:
                                print(f"{rmse_spme_plus[v]:.4f}mV", end='   ')
                            print()

                        _, ax = plt.subplots(constrained_layout=True)
                        ax.set_box_aspect(1 / scipy.constants.golden)
                        plt.plot(time[time_ind]/60, qty_true[time_ind], label='PDAE')
                        plt.plot(time[time_ind] / 60, qty_spme_plus['cc'][time_ind], label='SPMe+', linestyle='--')
                        if qty_spme is not None:
                            plt.plot(time[time_ind]/60, qty_spme[time_ind], label='SPMe', linestyle=':')
                        if name.startswith('etas'):
                            ind_plt = soc_pct[time_ind] >= 5.0
                            plt.ylim(bottom=np.min(qty_true[time_ind][ind_plt]))
                        if var['name'] == 'thetass1':
                            plt.legend()
                        plt.xlabel(r'Time, $t$ [min]')
                        plt.ylabel(label)
                        plt.title(f'{var['title']} ({series_key.upper()} {spec_string})')
                        plt.savefig(os.path.join(plot_dir, f'{name}-t{kt}.png'), bbox_inches='tight')
                        plt.savefig(os.path.join(plot_dir, f'{name}-t{kt}.eps'), bbox_inches='tight')
                        plt.close()

        # Make a data directory.
        data_dir = os.path.join('performance_data', dataset_key)
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        data = {
            'series': series_labels,
            'spec': spec_labels,
            'rmse_vcell_spme': rmse_vcell_spme,
        }
        for v in variants:
            data['rmse_vcell_spme_plus_' + v] = rmse_vcell_spme_plus[v]
        metrics = pd.DataFrame(data)
        metrics.to_excel(os.path.join(data_dir, 'metrics.xlsx'))