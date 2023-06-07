import pandas as pd
from scipy.interpolate import interp1d
import sys
sys.path.append("../../krotov/src")
import numpy as np
import qutip
import krotov
import matplotlib.pyplot as plt

# read excel file into a pandas DataFrame
df = pd.read_excel('..\\Controls\\control1.xlsx', engine='openpyxl')

# convert the DataFrame into a numpy array
controls = df.iloc[:, -1].values
print(len(controls))
# Assuming your time grid has 500 points and goes from 0 to 5
tlist = np.linspace(0, 5, 500)

# define the control function using interpolation
control_func = lambda t: interp1d(tlist[1:], controls, kind='cubic', fill_value="extrapolate")(t)

def guess_control(t, args):
    return control_func(t)

N = 10

def S(t):
    """Shape function for the field update"""
    #return 0.2 * np.sin(np.pi * t / 5)
    return krotov.shapes.one_shape(t)
    return krotov.shapes.flattop(
        t, t_start=0, t_stop=5, t_rise=0.05, t_fall=3, func='blackman'
    )

# diagonal Hamiltonian: simple energy level structure
H0 = qutip.qdiags(np.arange(N), 0)

# Control Hamiltonian for two-photon transitions
H1_elems = np.zeros((N, N))
for i in range(N-2):
    H1_elems[i, i+2] = H1_elems[i+2, i] = 1
H1 = qutip.Qobj(H1_elems)

# Total Hamiltonian
H = [H0, [H1, guess_control]]

# Define the states
psi0 = qutip.basis(N, 0)  # ground state
psi_target = qutip.basis(N, 2)  # second excited state

# Define the objective
objective = krotov.Objective(initial_state=psi0, target=psi_target, H=H)

# Define the pulse options
pulse_options = {H[1][1]: dict(lambda_a=5, update_shape=S)}

# Run the optimization
opt_result = krotov.optimize_pulses(
    [objective],
    pulse_options=pulse_options,
    tlist=np.linspace(0, 5, 500),
    propagator=krotov.propagators.expm,
    chi_constructor=krotov.functionals.chis_ss,
    info_hook=krotov.info_hooks.print_table(J_T=krotov.functionals.J_T_ss),
    check_convergence=krotov.convergence.Or(
        krotov.convergence.value_below(1e-3, name='J_T'),
        krotov.convergence.check_monotonic_error,
    ),
    store_all_pulses=True,
)

print(opt_result)
