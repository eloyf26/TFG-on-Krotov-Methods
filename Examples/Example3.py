import qutip
import numpy as np
import scipy
import matplotlib
import matplotlib.pylab as plt
import krotov
from IPython.display import display
import weylchamber as wc
from weylchamber.visualize import WeylChamber
from weylchamber.coordinates import from_magic


w1 = 1.1  # qubit 1 level splitting
w2 = 2.1  # qubit 2 level splitting
J = 0.2  # effective qubit coupling
u0 = 0.3  # initial driving strength
la = 1.1  # relative pulse coupling strength of second qubit
T = 25.0  # final time
nt = 250  # number of time steps

tlist = np.linspace(0, T, nt)

def eps0(t, args):
    return u0 * krotov.shapes.flattop(
        t, t_start=0, t_stop=T, t_rise=(T / 20), t_fall=(T / 20), func='sinsq'
    )

def plot_pulse(pulse, tlist):
    fig, ax = plt.subplots()
    if callable(pulse):
        pulse = np.array([pulse(t, args=None) for t in tlist])
    ax.plot(tlist, pulse)
    ax.set_xlabel('time')
    ax.set_ylabel('pulse amplitude')
    plt.show()

def hamiltonian(w1=w1, w2=w2, J=J, la=la, u0=u0):
    """Two qubit Hamiltonian

    Args:
        w1 (float): energy separation of the first qubit levels
        w2 (float): energy separation of the second qubit levels
        J (float): effective coupling between both qubits
        la (float): factor that pulse coupling strength differs for second qubit
        u0 (float): constant amplitude of the driving field
    """
    # local qubit Hamiltonians
    Hq1 = 0.5 * w1 * np.diag([-1, 1])
    Hq2 = 0.5 * w2 * np.diag([-1, 1])

    # lift Hamiltonians to joint system operators
    H0 = np.kron(Hq1, np.identity(2)) + np.kron(np.identity(2), Hq2)

    # define the interaction Hamiltonian
    sig_x = np.array([[0, 1], [1, 0]])
    sig_y = np.array([[0, -1j], [1j, 0]])
    Hint = 2 * J * (np.kron(sig_x, sig_x) + np.kron(sig_y, sig_y))
    H0 = H0 + Hint

    # define the drive Hamiltonian
    H1 = np.kron(np.array([[0, 1], [1, 0]]), np.identity(2)) + la * np.kron(
        np.identity(2), np.array([[0, 1], [1, 0]])
    )

    # convert Hamiltonians to QuTiP objects
    H0 = qutip.Qobj(H0)
    H1 = qutip.Qobj(H1)

    return [H0, [H1, eps0]]


H = hamiltonian(w1=w1, w2=w2, J=J, la=la, u0=u0)

class sigma(krotov.second_order.Sigma):
    def __init__(self, A, epsA=0):
        self.A = A
        self.epsA = epsA

    def __call__(self, t):
        ϵ, A = self.epsA, self.A
        return -max(ϵ, 2 * A + ϵ)

    def refresh(
        self,
        forward_states,
        forward_states0,
        chi_states,
        chi_norms,
        optimized_pulses,
        guess_pulses,
        objectives,
        result,
    ):
        try:
            Delta_J_T = result.info_vals[-1][0] - result.info_vals[-2][0]
        except IndexError:  # first iteration
            Delta_J_T = 0
        self.A = krotov.second_order.numerical_estimate_A(
            forward_states, forward_states0, chi_states, chi_norms, Delta_J_T
        )

def S(t):
    """Shape function for the field update"""
    return krotov.shapes.flattop(
        t, t_start=0, t_stop=T, t_rise=T / 20, t_fall=T / 20, func='sinsq'
    )

def print_fidelity(**args):
    basis = [objectives[i].initial_state for i in [0, 1, 2, 3]]
    states = [args['fw_states_T'][i] for i in [0, 1, 2, 3]]
    U = wc.gates.gate(basis, states)
    c1, c2, c3 = wc.coordinates.c1c2c3(from_magic(U))
    g1, g2, g3 = wc.local_invariants.g1g2g3_from_c1c2c3(c1, c2, c3)
    conc = wc.perfect_entanglers.concurrence(c1, c2, c3)
    F_PE = wc.perfect_entanglers.F_PE(g1, g2, g3)
    print("    F_PE: %f\n    gate conc.: %f" % (F_PE, conc))
    return F_PE, [c1, c2, c3]

def check_PE(result):
    # extract F_PE from (F_PE, [c1, c2, c3])
    F_PE = result.info_vals[-1][0]
    if F_PE <= 0:
        return "achieved perfect entangler"
    else:
        return None

#------------------------------------------------------------------------------------------    

plot_pulse(eps0, tlist)

psi_00 = qutip.Qobj(np.kron(np.array([1, 0]), np.array([1, 0])))
psi_01 = qutip.Qobj(np.kron(np.array([1, 0]), np.array([0, 1])))
psi_10 = qutip.Qobj(np.kron(np.array([0, 1]), np.array([1, 0])))
psi_11 = qutip.Qobj(np.kron(np.array([0, 1]), np.array([0, 1])))

proj_00 = qutip.ket2dm(psi_00)
proj_01 = qutip.ket2dm(psi_01)
proj_10 = qutip.ket2dm(psi_10)
proj_11 = qutip.ket2dm(psi_11)

objectives = krotov.gate_objectives(
    basis_states=[psi_00, psi_01, psi_10, psi_11], gate="PE", H=H
)

print(objectives)

# NBVAL_IGNORE_OUTPUT
for obj in objectives:
    display(obj.initial_state)

chi_constructor = wc.perfect_entanglers.make_PE_krotov_chi_constructor(
    [psi_00, psi_01, psi_10, psi_11]
)

pulse_options = {H[1][1]: dict(lambda_a=1.0e2, update_shape=S)}

opt_result = krotov.optimize_pulses(
    objectives,
    pulse_options=pulse_options,
    tlist=tlist,
    propagator=krotov.propagators.expm,
    chi_constructor=chi_constructor,
    info_hook=krotov.info_hooks.chain(
        krotov.info_hooks.print_debug_information, print_fidelity
    ),
    check_convergence=check_PE,
    sigma=sigma(A=0.0),
    iter_stop=20,
)

print(opt_result)

w = WeylChamber()
c1c2c3 = [opt_result.info_vals[i][1] for i in range(len(opt_result.iters))]
for i in range(len(opt_result.iters)):
    w.add_point(c1c2c3[i][0], c1c2c3[i][1], c1c2c3[i][2])
w.plot()

plot_pulse(opt_result.optimized_controls[0], tlist)