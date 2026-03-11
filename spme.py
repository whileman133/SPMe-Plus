"""
spme.py

Enhanced single-particle model for LMB cell.

2026.02.21 | Add solid stoichiometry correction and SPMePlus | WH
2024.09.15 | Created | Wesley Hileman <whileman@uccs.edu>
"""
import math
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
import numpy as np
import pybamm
import scipy.constants
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline, PchipInterpolator

import util
from util import LMBCell


@dataclass
class FVSData:
    """
    Data output from finite-volume simulation.
    """

    edges: np.ndarray   # vector of volume edge locations
    X: np.ndarray       # Nsim x Nstates matrix of states
    ts: float           # Sampling interval [s]
    iapp: np.ndarray    # applied current vector [A]


class BaseFVS(ABC):
    """
    Abstract base class for finite-volume simulations (FVS).
    """

    N: int      # total number of states (volumes)
    ts: float   # sampling interval [s]
    edges: np.ndarray    # vector of volume edge locations (x or r)
    centers: np.ndarray  # vector of volume center points (x or r)

    # Continuous-time state-space matrices.
    A: np.ndarray
    B: np.ndarray

    # Discrete-time state-space matrices.
    Ad: np.ndarray
    Bd: np.ndarray

    # Stateful simulation properties.
    x: np.ndarray   # current state vector

    def __init__(self):
        self.x = np.array([])

    @abstractmethod
    def get_initial(self, **kwargs):
        pass

    def get_ss_matrices(self, x: np.ndarray):
        """
        Fetch state-space matrices given the present state.
        """

        # Default: time invariant.
        return self.Ad, self.Bd

    def run(self, iapp: np.ndarray, **kwargs) -> FVSData:
        """
        Run the finite-volume simulation given the applied current vector.
        Any additional keyword arguments will be passed to get_initial() to determine the
        initial state.
        """

        N = self.N        # number of states
        kmax = len(iapp)  # number of simulation steps

        # Initialize storage.
        X = np.zeros(shape=(kmax, N))

        # Initialize state vector.
        x = self.get_initial(**kwargs)

        X[0, :] = x.T
        for k in range(1, kmax):
            Ad, Bd = self.get_ss_matrices(x)   # ss matrices for this iteration
            x = Ad @ x + Bd * iapp[k - 1]
            X[k, :] = x.T

        return FVSData(self.edges, X, self.ts, iapp)

    def initialize(self, **kwargs):
        """
        Initialize the finite volume simulation.
        Any additional keyword arguments will be passed to get_initial() to determine the
        initial state.
        Returns the initial state vector.
        """

        # Initialize state vector.
        self.x = self.get_initial(**kwargs)
        self.Ad, self.Bd = self.get_ss_matrices(self.x)

        return self.x

    def step(self, iappk: float):
        """
        Run a single time-step of the FVS.
        Returns the new state vector.
        """
        self.x = self.Ad @ self.x + self.Bd * iappk
        self.Ad, self.Bd = self.get_ss_matrices(self.x)  # ss matrices for next iteration
        return self.x


class ElectrolyteFVS(BaseFVS):
    """
    Finite-volume simulation for electrolyte in an LMB cell.
    """

    def __init__(self, cell: LMBCell, Ne: int, Np: int, ts: float = 1.0, TdegC: float = 25):
        super().__init__()
        self.cell = cell
        self.Ne = Ne
        self.Np = Np
        self.N = Ne + Np
        self.ts = ts
        self.TdegC = TdegC

        # Fetch relevant parameters.
        psi = cell.cst.psi
        ke = cell.eff.kappa
        te = cell.eff.tau
        kp = cell.pos.kappa
        tp = cell.pos.tau
        T = TdegC+273.15  # to Kelvin
        dxe = 1/Ne
        dxp = 1/Np
        dxie = (kp*dxe + ke*dxp)/2/kp
        dxip = (kp*dxe + ke*dxp)/2/ke

        # Build state-space matrices (time invariant) ----------------------------------------------
        A = np.zeros(shape=(Ne+Np, Ne+Np), dtype=np.float64)
        B = np.zeros(shape=(Ne+Np, 1), dtype=np.float64)

        # Eff layer contribution.
        for ssi, i in enumerate(range(1, Ne+1)):
            if i == 1:
                A[ssi, [ssi+1, ssi]] = np.array(
                    [1, -1]
                )/te/dxe/dxe
                B[ssi] = 1/te/ke/psi/T/dxe
            elif i == Ne:
                A[ssi, [ssi+1, ssi, ssi-1]] = np.array(
                    [1/dxie/dxe, -(1/dxie/dxe + 1/dxe/dxe), 1/dxe/dxe]
                )/te
            else:
                A[ssi, [ssi+1, ssi, ssi-1]] = np.array(
                    [1, -2, 1]
                )/te/dxe/dxe

        # Pos layer contribution.
        for ssi, i in enumerate(range(1, Np+1), start=Ne):
            B[ssi] = -1/tp/kp/psi/T
            if i == 1:
                A[ssi, [ssi+1, ssi, ssi-1]] = np.array(
                    [1/dxp/dxp, -(1/dxp/dxp + 1/dxip/dxp), 1/dxip/dxp]
                )/tp
            elif i == Np:
                A[ssi, [ssi, ssi-1]] = np.array(
                    [-1, 1]
                )/tp/dxp/dxp
            else:
                A[ssi, [ssi+1, ssi, ssi-1]] = np.array(
                    [1, -2, 1]
                )/tp/dxp/dxp

        # Convert to discrete time using backward Euler method.
        Ad = np.linalg.inv(np.identity(Ne+Np) - ts*A)
        Bd = Ad@(ts*B)

        # Determine edges of the volumes.
        xeff = np.linspace(0, 1, Ne+1)
        xpos = np.linspace(1, 2, Np+1)
        edges = np.concatenate((xeff[:-1], xpos))

        # Determine center points of the volumes.
        centers = edges[:-1] + np.diff(edges) / 2

        self.A = A
        self.B = B
        self.Ad = Ad
        self.Bd = Bd
        self.edges = edges
        self.centers = centers

    def get_initial(self, **kwargs):
        # Begin the simulation fully equilibrated.
        return np.ones(shape=(self.N, 1))

    def thetae(self, x, Xe=None):
        """
        Evaluate the electrolyte stoichiometry. Cubic spline interpolation over radial position.

        :param x: Linear position(s) at which to evaluate thetae. Float or numpy array.
        :param Xe: State matrix for the FVS. If None, uses the current state. Default None.
        :return: thetae
        """
        if Xe is None:
            Xe = self.x.T

        if isinstance(x, np.ndarray):
            # Force row vector.
            x = x.reshape(-1)

        return PchipInterpolator(self.centers, Xe, extrapolate=True, axis=1)(x)


class SolidFVS(BaseFVS):
    """
    Finite-volume simulation in spherical solid particle for LMB cell.
    """

    def __init__(self, cell: LMBCell, N: int, ts: float = 1.0, TdegC: float = 25, use_constant_Ds=True):
        """

        :param cell: Cell model containing parameter values.
        :param N: Number of shells to employ in FVS.
        :param ts: Sampling interval [s].
        :param TdegC: Temperature for computing log-average diffusivity.
        """

        super().__init__()
        self.cell = cell
        self.N = N
        self.ts = ts
        self.use_constant_Ds = use_constant_Ds

        theta_vect, Ds_vect = util.get_ds_curve(cell, TdegC)
        ind = np.argsort(theta_vect)
        theta_vect = theta_vect[ind]
        Ds_vect = Ds_vect[ind]
        self.theta = theta_vect
        self.Ds = Ds_vect

        # Determine edges of the volumes.
        self.edges = np.linspace(0, 1, N + 1)

        # Determine center points of the volumes.
        self.centers = self.edges[:-1] + np.diff(self.edges) / 2

        # Precompute SS matrices if we're using constant diffusivity.
        if use_constant_Ds:
            # Use log-average diffusivity.
            Ds = 10 ** np.mean(np.log10(Ds_vect))
            QAh = cell.cst.QAh
            theta0 = cell.pos.theta0
            theta100 = cell.pos.theta100

            # Compute constant state-space matricies.
            A = np.zeros(shape=(N, N), dtype=np.float64)
            B = np.zeros(shape=(N, 1), dtype=np.float64)
            for ssi, i in enumerate(range(1, N + 1)):
                if i == 1:
                    A[ssi, [ssi + 1, ssi]] = np.array(
                        [1, -1]
                    ) * 3 * Ds * N ** 2
                elif i == N:
                    A[ssi, [ssi, ssi - 1]] = np.array(
                        [-1, +1]
                    ) * 3 * Ds * N ** 2 * (N - 1) ** 2 / (N ** 3 - (N - 1) ** 3)
                    B[ssi] = np.abs(theta100 - theta0) * N ** 3 / (N ** 3 - (N - 1) ** 3) / 3600 / QAh
                else:
                    A[ssi, [ssi + 1, ssi, ssi - 1]] = np.array(
                        [Ds * i ** 2, -(Ds * i ** 2 + Ds * (i - 1) ** 2), Ds * (i - 1) ** 2]
                    ) * 3 * N ** 2 / (i ** 3 - (i - 1) ** 3)

            # Convert to discrete time using backward Euler method.
            Ad = np.linalg.inv(np.identity(N) - ts * A)
            Bd = Ad @ (ts * B)

            self.Ad = Ad
            self.Bd = Bd
            self.Ds_constant = Ds

    def get_initial(self, **kwargs):
        # Begin the simulation fully equilibrated at the given SOC.
        socPct0 = kwargs['soc0_pct']
        theta0 = self.cell.pos.theta0
        theta100 = self.cell.pos.theta100
        thetas0 = theta0 + (socPct0 / 100) * (theta100 - theta0)
        return np.ones(shape=(self.N, 1))*thetas0

    def get_ss_matrices(self, x: np.ndarray):
        if self.use_constant_Ds:
            return self.Ad, self.Bd

        # Otherwise, build time-variant matrices:

        cell = self.cell
        N = self.N
        ts = self.ts
        QAh = cell.cst.QAh
        theta0 = cell.pos.theta0
        theta100 = cell.pos.theta100

        # Fetch solid diffusivity at the edges of the shells.
        thetas_edges = self.thetas(self.edges, x.reshape(-1))  # force x row vector
        Ds_edges = 10**PchipInterpolator(self.theta, np.log10(self.Ds), extrapolate=True)(thetas_edges)

        # Built continuous time matrices.
        A = np.zeros(shape=(N, N), dtype=np.float64)
        B = np.zeros(shape=(N, 1), dtype=np.float64)
        for ssi, i in enumerate(range(1, N + 1)):
            if i == 1:
                Ds = Ds_edges[1]
                A[ssi, [ssi + 1, ssi]] = np.array(
                    [1, -1]
                ) * 3 * Ds * N ** 2
            elif i == N:
                Ds = Ds_edges[N - 1]
                A[ssi, [ssi, ssi - 1]] = np.array(
                    [-1, +1]
                ) * 3 * Ds * N ** 2 * (N - 1) ** 2 / (N ** 3 - (N - 1) ** 3)
                B[ssi] = np.abs(theta100 - theta0) * N ** 3 / (N ** 3 - (N - 1) ** 3) / 3600 / QAh
            else:
                Ds_i = Ds_edges[i]
                Ds_i_1 = Ds_edges[i - 1]
                A[ssi, [ssi + 1, ssi, ssi - 1]] = np.array(
                    [Ds_i * i ** 2, -(Ds_i * i ** 2 + Ds_i_1 * (i - 1) ** 2), Ds_i_1 * (i - 1) ** 2]
                ) * 3 * N ** 2 / (i ** 3 - (i - 1) ** 3)

        # Convert to discrete time using backward Euler method.
        Ad = np.linalg.inv(np.identity(N) - ts * A)
        Bd = Ad @ (ts * B)

        return Ad, Bd

    def thetas(self, r, Xs=None):
        """
        Evaluate the solid stoichiometry. Pchip interpolation over radial position.

        :param r: Radial position(s) at which to evaluate thetas. Float or numpy array.
        :param Xs: State matrix for the FVS. If None, uses the current state. Default None.
        :return: thetas
        """
        if Xs is None:
            Xs = self.x.T

        if isinstance(r, np.ndarray):
            # Force row vector.
            r = r.reshape(-1)

        return PchipInterpolator(self.centers, Xs, extrapolate=True, axis=1)(r)

    def thetas_avg(self, Xs=None):
        """
        Calculate the average solid stoichiometry in the spherical particle.

        :param Xs: State matrix for the FVS. If None, uses the current state. Default None.
        :return: thetas_avg
        """

        if Xs is None:
            Xs = self.x.T

        # Compute sum(theta_i * V_i) / V_total where:
        #   V_i = volume of ith shell = (4/3)*pi*(r[i]^3 - r[i-1]^3)
        #   V_total = volume of unit sphere = (4/3)*pi*(1^3)
        return np.sum(Xs * np.diff(self.edges**3), axis=1)


@dataclass
class SPMeOutput:
    Xe: np.ndarray
    Xs: np.ndarray
    iapp: np.ndarray
    vcell: np.ndarray
    thetae0: np.ndarray
    thetae1: np.ndarray
    thetae2: np.ndarray
    thetass: np.ndarray
    thetas_avg: np.ndarray
    phie2: np.ndarray
    neg_phis: np.ndarray
    pos_phise: np.ndarray
    pos_etas: np.ndarray
    pos_Uss: np.ndarray
    pos_if: np.ndarray

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class SPMe:
    """
    Implements single-particle model with electrolyte for an LMB cell.
    """

    def __init__(self, cell: LMBCell, ns: int, ne_eff: int, ne_pos: int, ts: float = 1.0, TdegC: float = 25, use_constant_Ds=True):
        self.cell = cell
        self.ts = ts
        self.TdegC = TdegC
        self.electrolyte_fvs = ElectrolyteFVS(cell, Ne=ne_eff, Np=ne_pos, ts=ts, TdegC=TdegC)
        self.solid_fvs = SolidFVS(cell, N=ns, ts=ts, TdegC=TdegC, use_constant_Ds=use_constant_Ds)

    def run(self, iapp: np.ndarray, soc0_pct: float, **kwargs) -> SPMeOutput:
        """
        Simulate the SPMe given the applied current vector.
        """

        Ne = self.electrolyte_fvs.N  # number of electrolyte states
        Ns = self.solid_fvs.N        # number of solid states
        kmax = len(iapp)             # number of simulation steps

        # Initialize storage.
        Xe = np.zeros(shape=(kmax, Ne))  # electrolyte state
        Xs = np.zeros(shape=(kmax, Ns))  # solid state

        # Initialize state vectors.
        xe = self.electrolyte_fvs.initialize()
        xs = self.solid_fvs.initialize(soc0_pct=soc0_pct)

        # Run simulation.
        Xe[0, :] = xe.T   # initial electrolyte state
        Xs[0, :] = xs.T   # initial solid state
        for k in range(1, kmax):
            xe = self.electrolyte_fvs.step(iapp[k - 1])
            xs = self.solid_fvs.step(iapp[k - 1])
            Xe[k, :] = xe.T
            Xs[k, :] = xs.T

        return self.get_output_variables(Xe, Xs, iapp, **kwargs)

    def get_output_variables(self, Xe, Xs, iapp) -> SPMeOutput:
        """
        Fetch output variables given state matrices.
        """

        # Collect cell parameter values.
        W = self.cell.cst.W
        psi = self.cell.cst.psi
        T = self.TdegC + 273.15
        ke = self.cell.eff.kappa
        kp = self.cell.pos.kappa
        k0n = self.cell.neg.k0
        betan = self.cell.neg.beta
        Rfp = self.cell.pos.Rf
        f = pybamm.constants.F.value / pybamm.constants.R.value / T

        # Fetch solid and electrolyte concentration variables
        # (matrices with dim0=time and dim1=position).
        thetae = self.electrolyte_fvs.thetae(x=np.array([0, 1, 2]), Xe=Xe)
        thetae0 = thetae[:, 0]
        thetae1 = thetae[:, 1]
        thetae2 = thetae[:, 2]
        thetass = self.solid_fvs.thetas(r=1, Xs=Xs)
        thetas_avg = self.solid_fvs.thetas_avg(Xs=Xs)
        # Enforce physical limits.
        thetass = np.clip(thetass, 0, 1)

        # Calculate phie2.
        phie2 = -iapp*(1/ke + 1/kp/2) + W*psi*T*(thetae2 - thetae0)

        # Assign Faradaic currents.
        ifn = iapp
        ifp = -iapp

        def get_vcell(thetass1, ifn1, ifp1, phie2_1, thetae0_1, thetae2_1):
            # Calculate exchange currents and solid-surface potential at pos.
            i0n = k0n * thetae0_1 ** (1 - betan)
            i0p, Uss = util.get_exchange_current(self.cell, thetass1, thetae2_1, self.TdegC)

            # Calculate eta_s(neg)=phis(neg)
            # (assumes beta_n=0.5).
            phis_n = 2 / f * np.arcsinh(ifn1 / i0n / 2)

            # Calculate eta_s(pos) and phise(pos)
            # (assumes all beta_p=0.5).
            eta_p = 2 / f * np.arcsinh(ifp1 / i0p / 2)
            phise_p = Uss + eta_p + Rfp * ifp1

            # Calculate cell voltage.
            vcell = phise_p + phie2_1 - phis_n

            return vcell, phis_n, phise_p, eta_p, Uss

        vcell, phis_n, phise_p, eta_p, Uss = get_vcell(thetass, ifn, ifp, phie2, thetae0, thetae2)

        # Collect output.
        return SPMeOutput(
            Xe=Xe,
            Xs=Xs,
            iapp=iapp,
            vcell=vcell,
            thetae0=thetae0,
            thetae1=thetae1,
            thetae2=thetae2,
            thetass=thetass,
            thetas_avg=thetas_avg,
            phie2=phie2,
            neg_phis=phis_n,
            pos_phise=phise_p,
            pos_etas=eta_p,
            pos_Uss=Uss,
            pos_if=ifp,
        )


class SSCParameter(ABC):
    """
    Parameter for IIR filter that varies with the average
    solid stoichiometry of the porous electrode.

    SSC = Solid Stoichiometry Correction
    """

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class SSCSplineParameter(SSCParameter):
    def __init__(self, x_vec: np.ndarray, param_vec_log10: np.ndarray, lb: float, ub: float):
        super().__init__()
        self.thetas_vec = x_vec
        self.param_vec_log10 = param_vec_log10
        self.lb = lb
        self.ub = ub
        self.interp_log10 = CubicSpline(
            x_vec,
            param_vec_log10,
            extrapolate=True
        )

    def __call__(self, x_vec: np.ndarray, *args, **kwargs):
        param_vec = 10**self.interp_log10(x_vec)
        return np.clip(param_vec, self.lb, self.ub)

    @property
    @abstractmethod
    def param_name(self):
        pass

    @property
    @abstractmethod
    def param_symbol(self):
        pass

    @property
    def x_label(self):
        return r"Average Solid Stoichiometry, $\theta_\mathrm{s,avg}$"

    def plot(self, title=None, ax=None, label="Cubic Spline (log10)", color='C0'):
        if title is None:
            title = f"{self.param_name} vs. Average Solid Stoichiometry"
        thetas_vec = np.linspace(np.min(self.thetas_vec), np.max(self.thetas_vec), 1000)
        param_vec = self(thetas_vec)
        noax = False
        if ax is None:
            noax = True
            _, ax = plt.subplots(constrained_layout=True)
            ax.set_box_aspect(1 / scipy.constants.golden)
        plt.plot(thetas_vec, param_vec, label=label, color=color)
        plt.plot(self.thetas_vec, self(self.thetas_vec), "o", markersize=5, label=None, color=color)
        if noax:
            plt.xlabel(self.x_label)
            plt.ylabel(f"{self.param_name}, {self.param_symbol}")
            plt.title(title)
        return ax


class SSCGainTheoretical(SSCParameter):
    def __init__(self, cell: LMBCell, x: float, TdegC: float = 25):
        """
        :param cell: Cell model containing parameter values.
        :param x: Linear position in porous electrode for computing dc gain,
            1 <= x <= 2 [unitless].
        :param TdegC: Temperature in degrees Celsius.
        """

        super().__init__()

        # Cache the dc gain of thetas_bar(x) as a function of the
        # average solid stoichiometry of the electrode.
        ocp_data = util.get_ocp_curve(cell, TdegC)
        theta_vec = ocp_data.theta
        dUocp_vec = ocp_data.dUocp
        w = cell.cst.W
        k = cell.pos.kappa
        s = cell.pos.sigma
        dcgain0 = 1 / 6 / s - (1 + w) / 3 / k + (x - 1) * (
            2 / s + 3 / k - (1 / s + 1 / k) * (x + 3) / 2 - w * (x - 3) / 2 / k
        )
        dcgain_vec = (1 / dUocp_vec) * dcgain0
        self.theta_vec = theta_vec
        self.dcgain_vec = dcgain_vec
        self.interp = PchipInterpolator(
            np.flip(theta_vec),  # flip for ascending order
            np.flip(dcgain_vec),
            extrapolate=True
        )

    def __call__(self, thetas_avg_vec: np.ndarray, *args, **kwargs):
        return self.interp(thetas_avg_vec)

    def plot(self, title=None):
        if title is None:
            title = f"Gain vs. Average Solid Stoichiometry"
        _, ax = plt.subplots(constrained_layout=True)
        ax.set_box_aspect(1 / scipy.constants.golden)
        plt.plot(self.theta_vec, self.dcgain_vec)
        plt.xlabel(r"Average Solid Stoichiometry, $\theta_\mathrm{s,avg}$")
        plt.ylabel(r"DC Gain, $\bar{\theta}_\mathrm{s,dc}$")
        plt.title(title)
        return ax


class SSCTimeConstantPropdUocp(SSCParameter):
    def __init__(self, cell: LMBCell, tau0: float, TdegC: float = 25):
        """
        :param cell: Cell model containing parameter values.
        :param TdegC: Temperature in degrees Celsius.
        """

        super().__init__()

        # Cache the dc gain of thetas_bar(x) as a function of the
        # average solid stoichiometry of the electrode.
        ocp_data = util.get_ocp_curve(cell, TdegC)
        theta_vec = ocp_data.theta
        dUocp_vec = ocp_data.dUocp
        tau_vec = -(1 / dUocp_vec)
        tau_vec /= np.mean(tau_vec)
        tau_vec *= tau0
        self.theta_vec = theta_vec
        self.tau_vec = tau_vec
        self.interp = PchipInterpolator(
            np.flip(theta_vec),  # flip for ascending order
            np.flip(tau_vec),
            extrapolate=True
        )

    def __call__(self, thetas_avg_vec: np.ndarray, *args, **kwargs):
        return self.interp(thetas_avg_vec)

    def plot(self, title=None, ax=None, label=None, color='C0'):
        if title is None:
            title = f"Gain vs. Average Solid Stoichiometry"
        noax = False
        if ax is None:
            noax = True
            _, ax = plt.subplots(constrained_layout=True)
            ax.set_box_aspect(1 / scipy.constants.golden)
        plt.plot(self.theta_vec, self.tau_vec, label=label, color=color)
        if noax:
            plt.xlabel(r"Average Solid Stoichiometry, $\theta_\mathrm{s,avg}$")
            plt.ylabel(r"Time Constant [s]")
            plt.title(title)
        return ax


class SSCTimeConstantTheoretical_0p(SSCParameter):
    def __init__(self, cell: LMBCell, x: float, TdegC: float = 25):
        """
        :param cell: Cell model containing parameter values.
        :param TdegC: Temperature in degrees Celsius.
        """

        super().__init__()

        # Cache the time constant of thetas_bar(x) as a function of the
        # average solid stoichiometry of the electrode.
        Dsref = cell.pos.Ds_ref
        Rfp = cell.pos.Rf
        sp = cell.pos.sigma
        kp = cell.pos.kappa
        W = cell.cst.W
        QAh = cell.cst.QAh
        theta0p = cell.pos.theta0
        theta100p = cell.pos.theta100
        ocp = util.get_ocp_curve(cell, TdegC)
        Cs = -3600 * QAh / abs(theta100p - theta0p) / ocp.dUocp
        Ds = -ocp.f * Dsref * ocp.theta * (1 - ocp.theta) * ocp.dUocp
        i0, Uss = util.get_exchange_current(cell, ocp.theta, np.ones_like(ocp.theta), TdegC)
        Rct = 1 / i0 / ocp.f
        dcgain0 = 1 / 6 / sp - (1 + W) / 3 / kp + (x - 1) * (
                2 / sp + 3 / kp - (1 / sp + 1 / kp) * (x + 3) / 2 - W * (x - 3) / 2 / kp
        )
        # Rdc = Rfp + Rct + 1/15/Cs/Ds + (1+W)/3/kp + 1/3/sp
        # tau = Rdc*Cs
        tau1 = Rct * Cs
        tau2 = 1 / 15 / Ds
        tau3 = ((1+W*(x-1))/3/kp + 1/3/sp) * Cs
        tau = tau1 + tau2 + tau3

        self.theta_vec = ocp.theta
        self.tau_vec = tau
        self.interp = PchipInterpolator(
            np.flip(ocp.theta),  # flip for ascending order
            np.flip(tau),
            extrapolate=True
        )

    def __call__(self, thetas_avg_vec: np.ndarray, *args, **kwargs):
        return self.interp(thetas_avg_vec)

    def plot(self, title=None, ax=None, label=None, color='C0'):
        if title is None:
            title = f"Time Constant vs. Average Solid Stoichiometry"
        noax = False
        if ax is None:
            noax = True
            _, ax = plt.subplots(constrained_layout=True)
            ax.set_box_aspect(1 / scipy.constants.golden)
        plt.plot(self.theta_vec, self.tau_vec, label=label, color=color)
        if noax:
            plt.xlabel(r"Average Solid Stoichiometry, $\theta_\mathrm{s,avg}$")
            plt.ylabel(r"Time Constant [s]")
            plt.title(title)
        return ax


class SSCTimeConstantTheoretical(SSCParameter):
    def __init__(self, cell: LMBCell, m: float, TdegC: float = 25):
        """
        :param cell: Cell model containing parameter values.
        :param TdegC: Temperature in degrees Celsius.
        """

        super().__init__()

        # Cache the time constant of thetas_bar(x) as a function of the
        # average solid stoichiometry of the electrode.
        Dsref = cell.pos.Ds_ref
        Rfp = cell.pos.Rf
        sp = cell.pos.sigma
        kp = cell.pos.kappa
        W = cell.cst.W
        QAh = cell.cst.QAh
        theta0p = cell.pos.theta0
        theta100p = cell.pos.theta100
        ocp = util.get_ocp_curve(cell, TdegC)
        Cs = -3600 * QAh / abs(theta100p - theta0p) / ocp.dUocp
        Ds = -ocp.f * Dsref * ocp.theta * (1 - ocp.theta) * ocp.dUocp
        i0, Uss = util.get_exchange_current(cell, ocp.theta, np.ones_like(ocp.theta), TdegC)
        Rct = 1 / i0 / ocp.f
        # Rdc = Rfp + Rct + 1/15/Cs/Ds + (1+W)/3/kp + 1/3/sp
        # tau = Rdc*Cs
        tau1 = Rct * Cs
        tau2 = 1 / 15 / Ds
        tau3 = ((1 + W) / 3 / kp + 1 / 3 / sp + Rfp) * Cs
        tau = tau1 + tau2 + tau3*m
        self.theta_vec = ocp.theta
        self.tau_vec = tau
        self.interp = PchipInterpolator(
            np.flip(ocp.theta),  # flip for ascending order
            np.flip(tau),
            extrapolate=True
        )

    def __call__(self, thetas_avg_vec: np.ndarray, *args, **kwargs):
        return self.interp(thetas_avg_vec)

    def plot(self, title=None, ax=None, label=None, color='C0'):
        if title is None:
            title = f"Time Constant vs. Average Solid Stoichiometry"
        noax = False
        if ax is None:
            noax = True
            _, ax = plt.subplots(constrained_layout=True)
            ax.set_box_aspect(1 / scipy.constants.golden)
        plt.plot(self.theta_vec, self.tau_vec, label=label, color=color)
        if noax:
            plt.xlabel(r"Average Solid Stoichiometry, $\theta_\mathrm{s,avg}$")
            plt.ylabel(r"Time Constant [s]")
            plt.title(title)
        return ax


class SSCTimeConstantTheoretical_3p(SSCParameter):
    def __init__(self, cell: LMBCell, m: np.ndarray, TdegC: float = 25):
        """
        :param cell: Cell model containing parameter values.
        :param TdegC: Temperature in degrees Celsius.
        """

        super().__init__()

        # Cache the time constant of thetas_bar(x) as a function of the
        # average solid stoichiometry of the electrode.
        Dsref = cell.pos.Ds_ref
        Rfp = cell.pos.Rf
        sp = cell.pos.sigma
        kp = cell.pos.kappa
        W = cell.cst.W
        QAh = cell.cst.QAh
        theta0p = cell.pos.theta0
        theta100p = cell.pos.theta100
        ocp = util.get_ocp_curve(cell, TdegC)
        Cs = -3600 * QAh / abs(theta100p - theta0p) / ocp.dUocp
        Ds = -ocp.f * Dsref * ocp.theta * (1 - ocp.theta) * ocp.dUocp
        i0, Uss = util.get_exchange_current(cell, ocp.theta, np.ones_like(ocp.theta), TdegC)
        Rct = 1 / i0 / ocp.f
        # Rdc = Rfp + Rct + 1/15/Cs/Ds + (1+W)/3/kp + 1/3/sp
        # tau = Rdc*Cs
        tau1 = Rct * Cs
        tau2 = 1 / 15 / Ds
        tau3 = ((1 + W) / 3 / kp + 1 / 3 / sp + Rfp) * Cs
        tau = tau1*m[0] + tau2*m[1] + tau3*m[2]
        self.theta_vec = ocp.theta
        self.tau_vec = tau
        self.interp = PchipInterpolator(
            np.flip(ocp.theta),  # flip for ascending order
            np.flip(tau),
            extrapolate=True
        )

    def __call__(self, thetas_avg_vec: np.ndarray, *args, **kwargs):
        return self.interp(thetas_avg_vec)

    def plot(self, title=None, ax=None, label=None, color='C0'):
        if title is None:
            title = f"Time Constant vs. Average Solid Stoichiometry"
        noax = False
        if ax is None:
            noax = True
            _, ax = plt.subplots(constrained_layout=True)
            ax.set_box_aspect(1 / scipy.constants.golden)
        plt.plot(self.theta_vec, self.tau_vec, label=label, color=color)
        if noax:
            plt.xlabel(r"Average Solid Stoichiometry, $\theta_\mathrm{s,avg}$")
            plt.ylabel(r"Time Constant [s]")
            plt.title(title)
        return ax


class SSCGainSpline(SSCSplineParameter):
    def __init__(self, x_vec: np.ndarray, gain_vec_log10: np.ndarray, lb: float, ub: float, sign=+1):
        super().__init__(x_vec, gain_vec_log10, lb, ub)
        self.sign = sign

    @property
    def param_name(self):
        return "Gain"

    @property
    def param_symbol(self):
        return r"$\bar{\theta}_\mathrm{s,dc}$"

    def __call__(self, x_vec: np.ndarray, *args, **kwargs):
        gain = super().__call__(x_vec)
        return self.sign*gain


class SSCTimeConstantSpline(SSCSplineParameter):
    """
    Time constant for IIR filter that varies with the average
    solid stoichiometry of the porous electrode.
    """

    def __init__(self, x_vec: np.ndarray, tau_vec_log10: np.ndarray, lb: float, ub: float):
        super().__init__(x_vec, tau_vec_log10, lb, ub)

    @property
    def param_name(self):
        return "Time Constant"

    @property
    def param_symbol(self):
        return r"$\tau$ [s]"


class SSCMultiplierSpline(SSCSplineParameter):
    def __init__(self, x_vec: np.ndarray, m_vec_log10: np.ndarray, lb: float, ub: float):
        super().__init__(x_vec, m_vec_log10, lb, ub)

    @property
    def param_name(self):
        return "Multiplier"

    @property
    def param_symbol(self):
        return r"$m$"

    @property
    def x_label(self):
        return r"$\theta_\mathrm{ss} - \theta_\mathrm{s,avg}$"


class SSCMultiplicativeSplineTimeConstant(SSCParameter):
    def __init__(self, tau: SSCTimeConstantSpline, multiplier: SSCMultiplierSpline):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier

    def __call__(self, thetas_avg_vec: np.ndarray, thetass_bar_vec: np.ndarray, *args, **kwargs):
        return self.tau(thetas_avg_vec) * self.multiplier(thetass_bar_vec)

    def plot(self, title=None):
        ax1 = self.tau.plot(title=title)
        ax2 = self.multiplier.plot(title=title)
        return ax1, ax2


class SSCBlend(SSCParameter):
    def __init__(self, blend_a, blend_gain):
        super().__init__()
        self.blend_a = blend_a
        self.blend_gain = blend_gain

    def __call__(self):
        return self.blend_a, self.blend_gain


class SolidStoichiometryCorrection:
    """
    Time-variant 1-pole IIR filter that estimates the difference between the
    radial-average solid stoichiometry at linear position x and the average
    solid stoichiometry of the entire porous electrode.

    Define:
        thetas_bar(x) = thetas_avg_radial(x) - thetas_avg
        dcgain(thetas_avg) is the dc gain of the IIR filter.
        tau(thetas_avg) is the time constant of the IIR filter.
    """

    def __init__(
        self,
        cell: LMBCell,
        gain: SSCParameter,
        tau: SSCParameter,
        blend: SSCParameter = None,
        ts: float = 1.0,
        TdegC: float = 25,
        gain_placement: str = 'bb',
    ):
        """
        :param cell: Cell model containing parameter values.
        :param gain: SocVariantDCGain object for computing dc gain.
        :param tau: SocVariantTimeConstant object for computing time constant.
        :param ts: Sampling interval [s].
        """

        self.cell = cell
        self.ts = ts
        self.TdegC = TdegC
        self.gain = gain
        self.tau = tau
        self.blend = blend
        self.x = None  # internal state
        self.gp = gain_placement

    def get_initial(self, **kwargs):
        # Default to fully equilibrated state.
        return 0

    def get_dc_gain(self, thetas_avg, thetass_bar=None):
        """
        Get the dc gain of thetas_bar(x) for the given average stoichiometry of the
        porous electrode.
        """
        return self.gain(thetas_avg, thetass_bar)

    def get_tau(self, thetas_avg, thetass_bar=None):
        """
        Get the time constant of the IIR filter for the given average stoichiometry
        of the porous electrode.
        """
        return self.tau(thetas_avg, thetass_bar)

    def get_blend(self):
        if not hasattr(self, 'blend') or self.blend is None:
            return None
        else:
            return self.blend()

    def run(self, iapp: np.ndarray, thetas_avg: np.ndarray, thetass_bar: np.ndarray = None, **kwargs):
        """
        Run time-variant IIR filter to estimate thetas_bar(x).
        """

        # Fetch dc gain and time constant of the IIR filter as
        # a function of the average solid stoichiometry of the electrode.
        dcgain_vec = self.get_dc_gain(thetas_avg, thetass_bar)
        tau_vec = self.get_tau(thetas_avg, thetass_bar)
        blend = self.get_blend()
        a_vec = np.exp(-self.ts / tau_vec)  # pole location

        tb_vec = np.zeros_like(iapp)  # thetas_bar(x) at each time step
        x = self.get_initial(**kwargs)
        for k in range(1, len(iapp)):
            a = a_vec[k - 1]
            d = dcgain_vec[k - 1]
            if self.gp == 'bb':
                x = a*x + d*(1-a)*iapp[k - 1]
                tb_vec[k] = x
            elif self.gp == 'cb':
                x = a*x + (1-a)*iapp[k - 1]
                tb_vec[k] = x*d
            elif self.gp == 'cc':
                x = a*x + iapp[k - 1]
                tb_vec[k] = x*d*(1-a)
            elif self.gp == 'bc':
                x = a*x + d*iapp[k - 1]
                tb_vec[k] = x*(1-a)
            elif self.gp == 'blend':
                ba, bd = blend
                x = a*x + np.abs(d)**(1-ba) * (1-a)**(1-ba) * iapp[k - 1]
                tb_vec[k] = x * np.sign(d) * np.abs(d)**ba * (1-a)**ba
            else:
                raise ValueError(f'Unknown gain placement {self.gp}')
        return tb_vec

    def initialize(self, **kwargs):
        """
        Initialize the simulation.
        Any additional keyword arguments will be passed to get_initial() to determine the
        initial state.
        Returns the initial state.
        """

        # Initialize state.
        self.x = self.get_initial(**kwargs)
        return self.x

    def step(self, iappk: float, thetas_avgk: float, thetass_bark: float = None):
        """
        Run a single time-step of the solid stoichiometry correction.
        Returns the new state vector.
        """
        d = self.get_dc_gain(thetas_avgk, thetass_bark)
        tau = self.get_tau(thetas_avgk, thetass_bark)
        blend = self.get_blend()
        a = math.exp(-self.ts/tau)  # pole location
        if self.gp == 'bb':
            self.x = a*self.x + d*(1-a)*iappk
            return self.x
        elif self.gp == 'cb':
            self.x = a*self.x + (1-a)*iappk
            return self.x*d
        elif self.gp == 'cc':
            self.x = a*self.x + iappk
            return self.x*d*(1-a)
        elif self.gp == 'bc':
            self.x = a*self.x + d*iappk
            return self.x*(1-a)
        elif self.gp == 'blend':
            ba, bd = blend
            self.x = a*self.x + abs(d)**(1-bd) * (1-a)**(1-ba) * iappk
            return self.x * np.sign(d)*abs(d)**bd * (1-a)**ba
        else:
            raise ValueError(f'Unknown gain placement {self.gp}')


@dataclass
class SPMePlusOutput:
    Xe: np.ndarray
    Xs: np.ndarray
    Xc: np.ndarray
    iapp: np.ndarray
    vcell: np.ndarray
    thetae0: np.ndarray
    thetae1: np.ndarray
    thetae2: np.ndarray
    thetass: np.ndarray
    thetass1: np.ndarray
    thetass2: np.ndarray
    thetas_avg: np.ndarray
    radial_thetas_avg1: np.ndarray
    radial_thetas_avg2: np.ndarray
    thetas_bar1: np.ndarray
    thetas_bar2: np.ndarray
    phie2: np.ndarray
    neg_phis: np.ndarray
    pos_phise2: np.ndarray
    pos_etas1: np.ndarray
    pos_etas2: np.ndarray
    pos_Uss1: np.ndarray
    pos_Uss2: np.ndarray
    pos_if1: np.ndarray
    pos_if2: np.ndarray

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class SPMePlus:
    """
    SPMe with solid stoichiometry corrections at the edges of the porous electrode.
    """

    def __init__(self,
        cell: LMBCell,
        ns: int, ne_eff: int, ne_pos: int,
        correction1: SolidStoichiometryCorrection,  # at x=1
        correction2: SolidStoichiometryCorrection,  # at x=2
        ts: float = 1.0, TdegC: float = 25, use_constant_Ds=True,
     ):
        self.cell = cell
        self.ts = ts
        self.TdegC = TdegC
        self.electrolyte_fvs = ElectrolyteFVS(cell, Ne=ne_eff, Np=ne_pos, ts=ts, TdegC=TdegC)
        self.solid_fvs = SolidFVS(cell, N=ns, ts=ts, TdegC=TdegC, use_constant_Ds=use_constant_Ds)
        self.c1 = correction1
        self.c2 = correction2

    def run(self, iapp: np.ndarray, soc0_pct: float, **kwargs) -> SPMePlusOutput:
        """
        Simulate the SPMe+ given the applied current vector.
        """

        Ne = self.electrolyte_fvs.N  # number of electrolyte states
        Ns = self.solid_fvs.N        # number of solid states
        kmax = len(iapp)             # number of simulation steps

        # Initialize storage.
        Xe = np.zeros(shape=(kmax, Ne))  # electrolyte state
        Xs = np.zeros(shape=(kmax, Ns))  # solid state
        Xc = np.zeros(shape=(kmax, 2))   # solid stoichiometry corrections (x=1, x=2)

        # Initialize state vectors.
        xe = self.electrolyte_fvs.initialize()
        xs = self.solid_fvs.initialize(soc0_pct=soc0_pct)
        xc = np.array([self.c1.initialize(), self.c2.initialize()]).reshape(-1,1)

        # Run simulation.
        Xe[0, :] = xe.T   # initial electrolyte state
        Xs[0, :] = xs.T   # initial solid state
        Xc[0, :] = xc.T   # initial solid stoichiometry corrections
        for k in range(1, kmax):
            iappkm = float(iapp[k - 1])
            thetas_avgkm = self.solid_fvs.thetas_avg()
            thetass_km = self.solid_fvs.thetas(r=1)
            xe = self.electrolyte_fvs.step(iappkm)
            xs = self.solid_fvs.step(iappkm)
            xc = np.array([
                self.c1.step(iappkm, thetas_avgkm, thetass_km - thetas_avgkm),
                self.c2.step(iappkm, thetas_avgkm, thetass_km - thetas_avgkm)
            ]).reshape(-1,1)
            Xe[k, :] = xe.T
            Xs[k, :] = xs.T
            Xc[k, :] = xc.T

        return self.get_output_variables(Xe, Xs, Xc, iapp, **kwargs)

    def get_output_variables(self, Xe, Xs, Xc, iapp) -> SPMePlusOutput:
        """
        Fetch output variables given state matrices.
        """

        # Collect cell parameter values.
        W = self.cell.cst.W
        QAh = self.cell.cst.QAh
        psi = self.cell.cst.psi
        T = self.TdegC + 273.15
        ke = self.cell.eff.kappa
        kp = self.cell.pos.kappa
        k0n = self.cell.neg.k0
        betan = self.cell.neg.beta
        Rfp = self.cell.pos.Rf
        f = pybamm.constants.F.value / pybamm.constants.R.value / T
        theta0p = self.cell.pos.theta0
        theta100p = self.cell.pos.theta100

        # Fetch solid and electrolyte concentration variables
        # (matrices with dim0=time and dim1=position).
        thetae = self.electrolyte_fvs.thetae(x=np.array([0, 1, 2]), Xe=Xe)
        thetae0 = thetae[:, 0]
        thetae1 = thetae[:, 1]
        thetae2 = thetae[:, 2]
        thetass = self.solid_fvs.thetas(r=1, Xs=Xs)
        thetas_avg = self.solid_fvs.thetas_avg(Xs=Xs)
        thetas_bar1 = Xc[:, 0]  # correction to radial thetas_avg at x=1
        thetas_bar2 = Xc[:, 1]  # correction to radial thetas_avg at x=2
        radial_thetas_avg1 = np.clip(thetas_avg + thetas_bar1, 0, 1)
        radial_thetas_avg2 = np.clip(thetas_avg + thetas_bar2, 0, 1)
        thetass1 = np.clip(thetass + thetas_bar1, 0, 1)
        thetass2 = np.clip(thetass + thetas_bar2, 0, 1)

        # Assign Faradaic currents (vanilla SPMe).
        ifn = iapp
        ifp = -iapp

        # Calculate corrected Faradaic currents in porous electrode from
        # corrected radial average solid stoichiometry (SPMe+).
        # Note: these are time-delayed (k-1)!
        ifp1m = -3600 * QAh * np.diff(radial_thetas_avg1, prepend=thetas_avg[0]) / self.ts / abs(theta100p - theta0p)
        ifp2m = -3600 * QAh * np.diff(radial_thetas_avg2, prepend=thetas_avg[0]) / self.ts / abs(theta100p - theta0p)

        # Sync the Faradaic currents in time
        ifp1 = np.append(ifp1m[1:], ifp1m[-1])
        ifp2 = np.append(ifp2m[1:], ifp2m[-1])

        # Calculate phie2 (vanilla SPMe)
        phie2 = -iapp*(1/ke+1/kp/2) + W*psi*T*(thetae2 - thetae0)

        # Calculate phie2 (SPMe+)
        phie2_plus = phie2 + (ifp2 - ifp1)/12/kp

        # Calculate exchange currents and solid-surface potential at pos.
        i0n = k0n * thetae0 ** (1 - betan)
        i0p1, Uss1 = util.get_exchange_current(self.cell, thetass1, thetae1, self.TdegC)
        i0p2, Uss2 = util.get_exchange_current(self.cell, thetass2, thetae2, self.TdegC)

        # Calculate eta_s(neg)=phis(neg)
        # (assumes beta_n=0.5).
        phis_n = 2 / f * np.arcsinh(ifn / i0n / 2)

        # Calculate eta_s(pos) and phise(pos)
        # (assumes all beta_p=0.5).
        etas1 = 2 / f * np.arcsinh(ifp1 / i0p1 / 2)
        etas2 = 2 / f * np.arcsinh(ifp2 / i0p2 / 2)
        phise2 = Uss2 + etas2 + Rfp * ifp2

        # Calculate cell voltage.
        vcell = phise2 + phie2_plus - phis_n

        # Collect output.
        return SPMePlusOutput(
            Xe=Xe,
            Xs=Xs,
            Xc=Xc,
            iapp=iapp,
            vcell=vcell,
            thetae0=thetae0,
            thetae1=thetae1,
            thetae2=thetae2,
            thetass=thetass,
            thetass1=thetass1,
            thetass2=thetass2,
            thetas_bar1=thetas_bar1,
            thetas_bar2=thetas_bar2,
            thetas_avg=thetas_avg,
            radial_thetas_avg1=radial_thetas_avg1,
            radial_thetas_avg2=radial_thetas_avg2,
            phie2=phie2_plus,
            neg_phis=phis_n,
            pos_phise2=phise2,
            pos_etas1=etas1,
            pos_etas2=etas2,
            pos_Uss1=Uss1,
            pos_Uss2=Uss2,
            pos_if1=ifp1,
            pos_if2=ifp2,
        )


