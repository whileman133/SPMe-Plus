import os
import pickle
from abc import ABC, abstractmethod

import numpy as np
import scipy
from sko.PSO import PSO
import pyswarms as ps
from matplotlib import pyplot as plt

import util
from spme import SSCGainTheoretical, SSCTimeConstantSpline, SolidStoichiometryCorrection, SSCGainSpline, SSCMultiplicativeSplineTimeConstant, SSCMultiplierSpline, SSCTimeConstantPropdUocp, SSCBlend, SSCTimeConstantTheoretical, SSCTimeConstantTheoretical_3p, SSCTimeConstantTF


class IIRRegressor(ABC):
    def __init__(
        self,
        cell: util.LMBCell,
        x: float,
        dataset: dict,
        TdegC: float = 25,
        ts: float = 1.0,
     ):
        self.cell = cell
        self.x = x
        self.dataset = dataset
        self.ts = ts
        self.TdegC = TdegC

    @abstractmethod
    def get_solid_stoichiometry_correction(
        self,
        sim_data: dict,
        param_vect: np.ndarray
    ):
        pass

    def cost(self, param_vect: np.ndarray):
        """
        Cost function for minimization.
        Total toot-mean-square (RMS) error accumulated over all profiles.
        """
        ndim = param_vect.ndim
        shape = param_vect.shape

        if ndim > 1:
            cost = np.zeros(shape[0])
        else:
            cost = 0

        for series_key, series in self.dataset.items():  # cc, gitt, drive
            for sim_data in series:
                fom_sim = sim_data['pybamm']
                if self.x == 2:
                    thetas_bar_true = fom_sim['pos_thetas_bar2']
                elif self.x == 1:
                    thetas_bar_true = fom_sim['pos_thetas_bar1']
                else:
                    raise ValueError(f"Unsupported value for x: {self.x}")
                if ndim > 1:
                    cost1 = np.zeros(shape[0])
                    for k in range(shape[0]):
                        thetas_bar_est = self.get_solid_stoichiometry_correction(
                            sim_data, param_vect[k,:])
                        cost1[k] = np.sqrt(np.mean(np.abs(thetas_bar_est - thetas_bar_true) ** 2))
                else:
                    thetas_bar_est = self.get_solid_stoichiometry_correction(
                        sim_data, param_vect)
                    cost1 = np.sqrt(np.mean(np.abs(thetas_bar_est - thetas_bar_true)**2))
                cost += cost1
        return cost


class IIRRegressor1Param(IIRRegressor):
    def __init__(
        self,
        cell: util.LMBCell,
        x: float,
        dataset: dict,
        TdegC: float = 25,
        ts: float = 1.0,
        m_lb: float = 0.1,
        m_ub: float = 10,
        gain_placement: str = 'b',
     ):
        super().__init__(cell, x, dataset, TdegC, ts)
        self.lb = m_lb
        self.ub = m_ub
        self.gp = gain_placement

        # Precompute DC gain for the solid stoichiometry correction.
        self.gain = SSCGainTheoretical(cell, x, TdegC=TdegC)

        self.tau = None
        self.correction = None

    def get_solid_stoichiometry_correction(
        self,
        sim_data: dict,
        m: float,
    ):
        fom_sim = sim_data['pybamm']
        spme_sim = sim_data['spme']
        iapp = fom_sim['iapp']
        thetas_avg = spme_sim.thetas_avg
        thetass_bar = spme_sim.thetass - spme_sim.thetas_avg
        tau = SSCTimeConstantTheoretical(self.cell, m, TdegC=self.TdegC)
        return SolidStoichiometryCorrection(
            self.cell, self.gain, tau, ts=self.ts, TdegC=self.TdegC,
            gain_placement=self.gp,
        ).run(iapp, thetas_avg, thetass_bar)

    def run(self):
        lb_vect = self.lb
        ub_vect = self.ub

        # Next, run a conventional optimizer to refine the solution.
        def cb(intermediate_result):
            if not hasattr(cb, "nit"):
                cb.nit = 0
            cb.nit += 1
            print(f"Iteration: {cb.nit:4d}, "
                  f"Cost: {intermediate_result.fun:.4f}, "
                  f"x: {np.array2string(intermediate_result.x, precision=2)}")
        best_cost = np.inf
        best_param = None
        for m0 in [1, 0.5, 1.5]:
            res = scipy.optimize.minimize(
                self.cost,
                m0,
                bounds=[(lb_vect, ub_vect)],
                callback=cb
            )
            if res.fun < best_cost:
                best_cost = res.fun
                best_param = res.x
        m = best_param

        self.tau = SSCTimeConstantTheoretical(self.cell, m, TdegC=self.TdegC)
        self.correction = SolidStoichiometryCorrection(
            self.cell, self.gain, self.tau, ts=self.ts, TdegC=self.TdegC,
            gain_placement=self.gp,)


class IIRRegressor3Param(IIRRegressor):
    def __init__(
        self,
        cell: util.LMBCell,
        x: float,
        dataset: dict,
        TdegC: float = 25,
        ts: float = 1.0,
        m_lb: float = 0.1,
        m_ub: float = 10,
        gain_placement: str = 'b',
     ):
        super().__init__(cell, x, dataset, TdegC, ts)
        self.lb = m_lb
        self.ub = m_ub
        self.gp = gain_placement

        # Precompute DC gain for the solid stoichiometry correction.
        self.gain = SSCGainTheoretical(cell, x, TdegC=TdegC)

        self.tau = None
        self.correction = None

    def get_solid_stoichiometry_correction(
        self,
        sim_data: dict,
        m: np.ndarray,
    ):
        fom_sim = sim_data['pybamm']
        spme_sim = sim_data['spme']
        iapp = fom_sim['iapp']
        thetas_avg = spme_sim.thetas_avg
        thetass_bar = spme_sim.thetass - spme_sim.thetas_avg
        tau = SSCTimeConstantTheoretical_3p(self.cell, m, TdegC=self.TdegC)
        return SolidStoichiometryCorrection(
            self.cell, self.gain, tau, ts=self.ts, TdegC=self.TdegC,
            gain_placement=self.gp,
        ).run(iapp, thetas_avg, thetass_bar)

    def run(self):
        lb_vect = self.lb*np.ones(3)
        ub_vect = self.ub*np.ones(3)

        # Next, run a conventional optimizer to refine the solution.
        def cb(intermediate_result):
            if not hasattr(cb, "nit"):
                cb.nit = 0
            cb.nit += 1
            print(f"Iteration: {cb.nit:4d}, "
                  f"Cost: {intermediate_result.fun:.4f}, "
                  f"x: {np.array2string(intermediate_result.x, precision=2)}")
        best_cost = np.inf
        best_param = None
        for k in range(10):
            m0 = np.ones(3) + np.random.uniform(-1.0, 1.0, 3)
            res = scipy.optimize.minimize(
                self.cost,
                m0,
                bounds=[(lb, ub) for lb, ub in zip(lb_vect, ub_vect)],
                callback=cb
            )
            if res.fun < best_cost:
                best_cost = res.fun
                best_param = res.x
        m = best_param

        self.tau = SSCTimeConstantTheoretical_3p(self.cell, m, TdegC=self.TdegC)
        self.correction = SolidStoichiometryCorrection(
            self.cell, self.gain, self.tau, ts=self.ts, TdegC=self.TdegC,
            gain_placement=self.gp,)


class IIRRegressorTheoreticalGain(IIRRegressor):
    def __init__(
        self,
        cell: util.LMBCell,
        x: float,
        dataset: dict,
        TdegC: float = 25,
        ts: float = 1.0,
        tau_lb: float = 0.001,
        tau_ub: float = 1000,
        n_control_pts: int = 8,
        gain_placement: str = 'b',
     ):
        super().__init__(cell, x, dataset, TdegC, ts)
        self.lb = tau_lb
        self.ub = tau_ub
        self.gp = gain_placement

        # Define control points for the interpolation of tau over thetas_avg.
        self.thetas_avg_ctl = np.linspace(
            cell.pos.theta100,  # =theta_min
            cell.pos.theta0,    # =theta_max
            n_control_pts
        )

        # Precompute DC gain for the solid stoichiometry correction.
        self.gain = SSCGainTheoretical(cell, x, TdegC=TdegC)

        self.cost0 = None
        self.tau = None
        self.correction = None

    def get_solid_stoichiometry_correction(
        self,
        sim_data: dict,
        tau_vect_log10: np.ndarray
    ):
        fom_sim = sim_data['pybamm']
        spme_sim = sim_data['spme']
        iapp = fom_sim['iapp']
        thetas_avg = spme_sim.thetas_avg
        thetass_bar = spme_sim.thetass - spme_sim.thetas_avg
        tau = SSCTimeConstantSpline(
            self.thetas_avg_ctl, tau_vect_log10, lb=self.lb, ub=self.ub)
        return SolidStoichiometryCorrection(
            self.cell, self.gain, tau, ts=self.ts, TdegC=self.TdegC,
            gain_placement=self.gp,
        ).run(iapp, thetas_avg, thetass_bar)

    def run(self):
        # First, run PSO to find an initial point.
        lb_vect = np.log10(self.lb * np.ones(len(self.thetas_avg_ctl)))
        ub_vect = np.log10(self.ub * np.ones(len(self.thetas_avg_ctl)))
        pso = ps.single.LocalBestPSO(
            n_particles=30,
            dimensions=len(lb_vect),
            options={'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 3, 'p': 2},
        )
        best_cost, best_pos = pso.optimize(self.cost, iters=100)

        # Next, run a conventional optimizer to refine the solution.
        def cb(intermediate_result):
            if not hasattr(cb, "nit"):
                cb.nit = 0
            cb.nit += 1
            print(f"Iteration: {cb.nit:4d}, Cost: {intermediate_result.fun}")
        res = scipy.optimize.minimize(
            self.cost,
            best_pos,
            bounds=[(lb, ub) for lb, ub in zip(lb_vect, ub_vect)],
            callback=cb
        )
        tau_vec_log10 = res.x
        self.cost0 = res.fun

        self.tau = SSCTimeConstantSpline(
            self.thetas_avg_ctl, tau_vec_log10, lb=self.lb, ub=self.ub)
        self.correction = SolidStoichiometryCorrection(
            self.cell, self.gain, self.tau, ts=self.ts, TdegC=self.TdegC,
            gain_placement=self.gp,)


class IIRRegressorTheoreticalGainBlend(IIRRegressor):
    def __init__(
        self,
        cell: util.LMBCell,
        x: float,
        dataset: dict,
        tau: SSCTimeConstantTF,
        TdegC: float = 25,
        ts: float = 1.0,
        n_control_pts: int = 8,
     ):
        super().__init__(cell, x, dataset, TdegC, ts)
        self.gp = 'blend'

        # Define control points for the interpolation of tau over thetas_avg.
        self.thetas_avg_ctl = np.linspace(
            cell.pos.theta100,  # =theta_min
            cell.pos.theta0,    # =theta_max
            n_control_pts
        )

        # Precompute DC gain for the solid stoichiometry correction.
        self.gain = SSCGainTheoretical(cell, x, TdegC=TdegC)
        self.tau = tau

        self.cost0 = None
        self.blend = None
        self.correction = None

    def get_solid_stoichiometry_correction(
        self,
        sim_data: dict,
        param_vect: np.ndarray
    ):
        ba, bd = param_vect
        fom_sim = sim_data['pybamm']
        spme_sim = sim_data['spme']
        iapp = fom_sim['iapp']
        thetas_avg = spme_sim.thetas_avg
        thetass_bar = spme_sim.thetass - spme_sim.thetas_avg
        blend = SSCBlend(ba, bd)
        return SolidStoichiometryCorrection(
            self.cell, self.gain, self.tau, blend, self.ts, self.TdegC,
            gain_placement=self.gp,
        ).run(iapp, thetas_avg, thetass_bar)

    def run(self):
        lb_vect = np.zeros(2)
        ub_vect = np.ones(2)

        pop_size = 10
        # Run PSO to find an initial point.
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k': 3, 'p': 2}
        pso = ps.single.LocalBestPSO(
            n_particles=pop_size,
            dimensions=len(lb_vect),
            options=options,
            bounds=(lb_vect, ub_vect),
        )
        best_cost, best_pos = pso.optimize(self.cost, iters=50)

        # Next, run a conventional optimizer to refine the solution.
        def cb(intermediate_result):
            if not hasattr(cb, "nit"):
                cb.nit = 0
            cb.nit += 1
            print(f"Iteration: {cb.nit:4d}, "
                  f"Cost: {intermediate_result.fun:.4f}, "
                  f"x: {np.array2string(intermediate_result.x, precision=2)}")
        res = scipy.optimize.minimize(
            self.cost,
            best_pos,
            bounds=[(lb, ub) for lb, ub in zip(lb_vect, ub_vect)],
            callback=cb
        )
        cost = res.fun
        param_vect = res.x
        ba, bd = param_vect

        self.cost0 = cost
        self.blend = SSCBlend(ba, bd)
        self.correction = SolidStoichiometryCorrection(
            self.cell, self.gain, self.tau, self.blend, self.ts, self.TdegC,
            gain_placement=self.gp,)


if __name__ == '__main__':
    # Load PyBaMM simulation data from file.
    with open(os.path.join('datasets', 'PDAE_SPMe_25degC.pickle'), "rb") as f:
        data = pickle.load(f)

    enable = {
        #'bb',
        #'cb',
        #'cc',
        #'bc',
        #'1p',
        #'3p',
        'blend',
    }

    cell = data['cell']
    TdegC = data['TdegC']
    ts = data['ts']
    theta0p = cell.pos.theta0
    theta100p = cell.pos.theta100

    filename = os.path.join("datasets", f"IIR_{TdegC:d}degC.pickle")
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            iir_data = pickle.load(f)
    else:
        iir_data = {
            'cell': cell,
            'TdegC': TdegC,
            'ts': ts,
        }

    # BB
    if 'bb' in enable:
        regressor1bb = IIRRegressorTheoreticalGain(cell, x=1, TdegC=TdegC, ts=ts, dataset=data['datasets']['train'], gain_placement='bb')
        regressor1bb.run()
        regressor2bb = IIRRegressorTheoreticalGain(cell, x=2, TdegC=TdegC, ts=ts, dataset=data['datasets']['train'], gain_placement='bb')
        regressor2bb.run()
        iir_data['correction1bb'] = regressor1bb.correction
        iir_data['correction2bb'] = regressor2bb.correction

    # CB
    if 'cb' in enable:
        regressor1cb = IIRRegressorTheoreticalGain(cell, x=1, TdegC=TdegC, ts=ts, dataset=data['datasets']['train'], gain_placement='cb')
        regressor1cb.run()
        regressor2cb = IIRRegressorTheoreticalGain(cell, x=2, TdegC=TdegC, ts=ts, dataset=data['datasets']['train'], gain_placement='cb')
        regressor2cb.run()
        iir_data['correction1cb'] = regressor1cb.correction
        iir_data['correction2cb'] = regressor2cb.correction

    # CC
    if 'cc' in enable:
        regressor1cc = IIRRegressorTheoreticalGain(cell, x=1, TdegC=TdegC, ts=ts, dataset=data['datasets']['train'], gain_placement='cc')
        regressor1cc.run()
        regressor2cc = IIRRegressorTheoreticalGain(cell, x=2, TdegC=TdegC, ts=ts, dataset=data['datasets']['train'], gain_placement='cc')
        regressor2cc.run()
        iir_data['correction1cc'] = regressor1cc.correction
        iir_data['correction2cc'] = regressor2cc.correction

    # BC
    if 'bc' in enable:
        regressor1bc = IIRRegressorTheoreticalGain(cell, x=1, TdegC=TdegC, ts=ts, dataset=data['datasets']['train'], gain_placement='bc')
        regressor1bc.run()
        regressor2bc = IIRRegressorTheoreticalGain(cell, x=2, TdegC=TdegC, ts=ts, dataset=data['datasets']['train'], gain_placement='bc')
        regressor2bc.run()
        iir_data['correction1bc'] = regressor1bc.correction
        iir_data['correction2bc'] = regressor2bc.correction

    # 1 parameter
    if '1p' in enable:
        regressor1_1p = IIRRegressor1Param(cell, x=1, TdegC=TdegC, ts=ts, dataset=data['datasets']['train'], gain_placement='cc')
        regressor1_1p.run()
        regressor2_1p = IIRRegressor1Param(cell, x=2, TdegC=TdegC, ts=ts, dataset=data['datasets']['train'], gain_placement='cc')
        regressor2_1p.run()
        iir_data['correction1_1p'] = regressor1_1p.correction
        iir_data['correction2_1p'] = regressor2_1p.correction

    # 3 parameter
    if '3p' in enable:
        regressor1_3p = IIRRegressor3Param(cell, x=1, TdegC=TdegC, ts=ts, dataset=data['datasets']['train'], gain_placement='cc')
        regressor1_3p.run()
        regressor2_3p = IIRRegressor3Param(cell, x=2, TdegC=TdegC, ts=ts, dataset=data['datasets']['train'], gain_placement='cc')
        regressor2_3p.run()
        iir_data['correction1_3p'] = regressor1_3p.correction
        iir_data['correction2_3p'] = regressor2_3p.correction

    # Full blend
    if 'blend' in enable:
        tau1_tf, theta1_tf = util.load_mat_tau(os.path.join('MATLAB', 'CELLPARAMS', 'tau1.mat'))
        tau2_tf, theta2_tf = util.load_mat_tau(os.path.join('MATLAB', 'CELLPARAMS', 'tau2.mat'))
        tau1 = SSCTimeConstantTF(theta1_tf, tau1_tf)
        tau2 = SSCTimeConstantTF(theta2_tf, tau2_tf)
        regressor1blend = IIRRegressorTheoreticalGainBlend(
            cell, x=1, tau=tau1, TdegC=TdegC, ts=ts, dataset=data['datasets']['train'])
        regressor1blend.run()
        regressor2blend = IIRRegressorTheoreticalGainBlend(
            cell, x=2, tau=tau2, TdegC=TdegC, ts=ts, dataset=data['datasets']['train'])
        regressor2blend.run()
        iir_data['correction1blend'] = regressor1blend.correction
        iir_data['correction2blend'] = regressor2blend.correction

    # Save results to file.
    with open(filename, "wb") as f:
        pickle.dump(iir_data, f, pickle.HIGHEST_PROTOCOL)
