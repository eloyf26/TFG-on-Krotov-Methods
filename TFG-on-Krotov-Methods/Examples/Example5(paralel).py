import sys
import os
import qutip
import numpy as np
import scipy
import matplotlib
import matplotlib.pylab as plt
import krotov
import copy
from functools import partial
from itertools import product

if sys.platform != 'linux':
    krotov.parallelization.set_parallelization(use_loky=True)
from krotov.parallelization import parallel_map

def two_qubit_transmon_liouvillian(
    ω1, ω2, ωd, δ1, δ2, J, q1T1, q2T1, q1T2, q2T2, T, Omega, n_qubit
):
    from qutip import tensor, identity, destroy

    b1 = tensor(identity(n_qubit), destroy(n_qubit))
    b2 = tensor(destroy(n_qubit), identity(n_qubit))

    H0 = (
        (ω1 - ωd - δ1 / 2) * b1.dag() * b1
        + (δ1 / 2) * b1.dag() * b1 * b1.dag() * b1
        + (ω2 - ωd - δ2 / 2) * b2.dag() * b2
        + (δ2 / 2) * b2.dag() * b2 * b2.dag() * b2
        + J * (b1.dag() * b2 + b1 * b2.dag())
    )

    H1_re = 0.5 * (b1 + b1.dag() + b2 + b2.dag())  # 0.5 is due to RWA
    H1_im = 0.5j * (b1.dag() - b1 + b2.dag() - b2)

    H = [H0, [H1_re, Omega], [H1_im, ZeroPulse]]

    A1 = np.sqrt(1 / q1T1) * b1  # decay of qubit 1
    A2 = np.sqrt(1 / q2T1) * b2  # decay of qubit 2
    A3 = np.sqrt(1 / q1T2) * b1.dag() * b1  # dephasing of qubit 1
    A4 = np.sqrt(1 / q2T2) * b2.dag() * b2  # dephasing of qubit 2

    L = krotov.objectives.liouvillian(H, c_ops=[A1, A2, A3, A4])
    return L

GHz = 2 * np.pi
MHz = 1e-3 * GHz
ns = 1
μs = 1000 * ns

ω1 = 4.3796 * GHz  # qubit frequency 1
ω2 = 4.6137 * GHz  # qubit frequency 2
ωd = 4.4985 * GHz  # drive frequency
δ1 = -239.3 * MHz  # anharmonicity 1
δ2 = -242.8 * MHz  # anharmonicity 2
J = -2.3 * MHz     # effective qubit-qubit coupling
q1T1 = 38.0 * μs   # decay time for qubit 1
q2T1 = 32.0 * μs   # decay time for qubit 2
q1T2 = 29.5 * μs   # dephasing time for qubit 1
q2T2 = 16.0 * μs   # dephasing time for qubit 2
T = 400 * ns       # gate duration

tlist = np.linspace(0, T, 2000)
n_qubit = 5        # number of transmon levels to consider

def Omega(t, args):
    E0 = 35.0 * MHz
    return E0 * krotov.shapes.flattop(t, 0, T, t_rise=(20 * ns), func='sinsq')

def ZeroPulse(t, args):
    return 0.0

L = two_qubit_transmon_liouvillian(
    ω1, ω2, ωd, δ1, δ2, J, q1T1, q2T1, q1T2, q2T2, T, Omega, n_qubit
)

def plot_pulse(pulse, tlist, xlimit=None):
    fig, ax = plt.subplots()
    if callable(pulse):
        pulse = np.array([pulse(t, None) for t in tlist])
    ax.plot(tlist, pulse/MHz)
    ax.set_xlabel('time (ns)')
    ax.set_ylabel('pulse amplitude (MHz)')
    if xlimit is not None:
        ax.set_xlim(xlimit)
    plt.show()

plot_pulse(L[1][1], tlist)

#Optimization Objectives
SQRTISWAP = qutip.Qobj(np.array(
    [[1, 0,               0,               0],
     [0, 1  / np.sqrt(2), 1j / np.sqrt(2), 0],
     [0, 1j / np.sqrt(2), 1  / np.sqrt(2), 0],
     [0, 0,               0,               1]]),
    dims=[[2, 2], [2, 2]]
)

ket00 = qutip.ket((0, 0), dim=(n_qubit, n_qubit))
ket01 = qutip.ket((0, 1), dim=(n_qubit, n_qubit))
ket10 = qutip.ket((1, 0), dim=(n_qubit, n_qubit))
ket11 = qutip.ket((1, 1), dim=(n_qubit, n_qubit))
basis = [ket00, ket01, ket10, ket11]

weights = np.array([20, 1, 1], dtype=np.float64)
weights *= len(weights) / np.sum(weights) # manual normalization
weights /= np.array([0.3, 1.0, 0.25]) # purities

objectives = krotov.gate_objectives(
    basis,
    SQRTISWAP,
    L,
    liouville_states_set='3states',
    weights=weights,
    normalize_weights=False,
)
print(objectives)

#Dynamics under the guess pulse

full_liouville_basis = [psi * phi.dag() for (psi, phi) in product(basis, basis)]
def propagate_guess(initial_state):
    return objectives[0].mesolve(
        tlist,
        rho0=initial_state,
    ).states[-1]

full_states_T = parallel_map(
    propagate_guess, values=full_liouville_basis,
)
print("F_avg = %.3f" % krotov.functionals.F_avg(full_states_T, basis, SQRTISWAP))

rho00, rho01, rho10, rho11 = [qutip.ket2dm(psi) for psi in basis]

def propagate_guess_for_expvals(initial_state):
    return objectives[0].propagate(
        tlist,
        propagator=krotov.propagators.DensityMatrixODEPropagator(),
        rho0=initial_state,
        e_ops=[rho00, rho01, rho10, rho11]
    )

def plot_population_dynamics(dyn00, dyn01, dyn10, dyn11):
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(16, 8))
    axs = np.ndarray.flatten(axs)
    labels = ['00', '01', '10', '11']
    dyns = [dyn00, dyn01, dyn10, dyn11]
    for (ax, dyn, title) in zip(axs, dyns, labels):
        for (i, label) in enumerate(labels):
            ax.plot(dyn.times, dyn.expect[i], label=label)
        ax.legend()
        ax.set_title(title)
    plt.show()

plot_population_dynamics(
    *parallel_map(
        propagate_guess_for_expvals,
        values=[rho00, rho01, rho10, rho11],
    )
)

#Optimization
pulse_options = {
    L[i][1]: dict(
        lambda_a=1.0,
        update_shape=partial(
            krotov.shapes.flattop, t_start=0, t_stop=T, t_rise=(20 * ns))
        )
    for i in [1, 2]
}
opt_result = krotov.optimize_pulses(
    objectives,
    pulse_options,
    tlist,
    propagator=krotov.propagators.DensityMatrixODEPropagator(reentrant=True),
    chi_constructor=krotov.functionals.chis_re,
    info_hook=krotov.info_hooks.print_table(J_T=krotov.functionals.J_T_re),
    iter_stop=3,
)

dumpfile = "./3states_opt_result.dump"
if os.path.isfile(dumpfile):
    opt_result = krotov.result.Result.load(dumpfile, objectives)
else:
    opt_result = krotov.optimize_pulses(
        objectives,
        pulse_options,
        tlist,
        propagator=krotov.propagators.DensityMatrixODEPropagator(reentrant=True),
        chi_constructor=krotov.functionals.chis_re,
        info_hook=krotov.info_hooks.print_table(J_T=krotov.functionals.J_T_re),
        iter_stop=3,
        continue_from=opt_result
    )
    opt_result.dump(dumpfile)

print(opt_result)

#Optimization Result

optimized_control = opt_result.optimized_controls[0] + 1j * opt_result.optimized_controls[1]

plot_pulse(np.abs(optimized_control), tlist)

def propagate_opt(initial_state):
    return opt_result.optimized_objectives[0].propagate(
        tlist,
        propagator=krotov.propagators.DensityMatrixODEPropagator(),
        rho0=initial_state,
    ).states[-1]

opt_full_states_T = parallel_map(
    propagate_opt, values=full_liouville_basis,
)

print("F_avg = %.3f" % krotov.functionals.F_avg(opt_full_states_T, basis, SQRTISWAP))

def propagate_opt_for_expvals(initial_state):
    return opt_result.optimized_objectives[0].propagate(
        tlist,
        propagator=krotov.propagators.DensityMatrixODEPropagator(),
        rho0=initial_state,
        e_ops=[rho00, rho01, rho10, rho11]
    )

plot_population_dynamics(
    *parallel_map(
        propagate_opt_for_expvals,
        values=[rho00, rho01, rho10, rho11],
    )
)

def plot_convergence(result):
    fig, ax = plt.subplots()
    ax.semilogy(result.iters, result.info_vals)
    ax.set_xlabel('OCT iteration')
    ax.set_ylabel(r'optimization error $J_{T, re}$')
    plt.show()

plot_convergence(opt_result)