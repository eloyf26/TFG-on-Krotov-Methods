import sys
sys.path.append("../../krotov/src")
import qutip
import numpy as np
import scipy
import matplotlib.cm as cm
import matplotlib.pylab as plt
import krotov
from scipy.interpolate import CubicSpline


def generate_arrays(d1 = [], d2 = []):
    # Create array with d1 on the diagonal
    arr1 = np.diag(d1)
    
    # Create array with d2 on the upper and lower diagonals
    arr2 =   np.diag(d2,1) + np.diag(d2,-1)# lower diagonal

    return arr1, arr2

def hamiltonian(guess_control,guess_control2, N = 3, d1 = [], d2 = []):
    """Two-level-system Hamiltonian

    Args:
        omega (float): energy separation of the qubit levels
        ampl0 (float): constant amplitude of the driving field
    """
    
    # diag = []
    # for i in range(N):
    #     diag.append((i+1)**2/4)

    arr1,arr2 = generate_arrays(d1,d2); 
    H0 = qutip.Qobj(arr2)
    H1 = qutip.Qobj(arr1)
    H2 = qutip.Qobj(arr2)

    return [H0, [H1, guess_control],[H2, guess_control2]]

def plot_pulse(pulse, tlist):
    fig, ax = plt.subplots()
    if callable(pulse):
        pulse = np.array([pulse(t, args=None) for t in tlist])
    ax.plot(tlist, pulse)
    ax.set_xlabel('time')
    ax.set_ylabel('pulse amplitude')
    plt.show()

def plot_population(result):
    fig, ax = plt.subplots()
    for i in range(0,Dim):
        ax.plot(result.times, result.expect[i], label=str(i))    

    ax.legend()
    ax.set_xlabel('time')
    ax.set_ylabel('population')
    plt.show()

def GetDimInfo(iteration,n_iters,Dim):
    

    if iteration == 0:
        ls = '--'  # dashed
        alpha = 1  # full opacity
        ctr_label = 'guess'
    elif iteration == opt_result.iters[-1]:
        ls = '-'  # solid
        alpha = 1  # full opacity
        ctr_label = 'optimized'
    else:
        ls = '-'  # solid
        alpha = 0.5 * float(iteration) / float(n_iters)  # max 50%
        ctr_label = None

    if ctr_label:
        pop_labels = ['{number} {tipo}'.format(number = i,tipo = ctr_label) for i in range(Dim)]
        ctr_labels = ['{i} {ctr_label}'.format(i = i + 1,ctr_label = ctr_label) for i in range(N_Controls)]
    else:
        pop_labels = [None for i in range(Dim)]
        ctr_labels = [None for i in range (N_Controls)]

    #Compute the list of colors
    color_map = cm.get_cmap('Set1', Dim)  # Get a colormap with N colors
    color_set = [color_map(i)[:3] for i in range(Dim)]  # Extract RGB values for each color

    # Convert RGB values to hexadecimal color codes
    colors = ['#%02x%02x%02x' % tuple(int(255 * c) for c in color) for color in color_set]

    return ls,alpha,ctr_labels,pop_labels,colors

def plot_iterations(opt_result):
    """Plot the control fields in population dynamics over all iterations.

    This depends on ``store_all_pulses=True`` in the call to
    `optimize_pulses`.
    """
    fig, [ax_ctr, ax_dyn] = plt.subplots(nrows=2, figsize=(8, 10))
    n_iters = len(opt_result.iters)
    for (iteration, pulses) in zip(opt_result.iters, opt_result.all_pulses):
        controls = [
            krotov.conversions.pulse_onto_tlist(pulse)
            for pulse in pulses
        ]
        objectives = opt_result.objectives_with_controls(controls)
        dynamics = objectives[0].mesolve(
            opt_result.tlist, e_ops=projs
        )

        ls,alpha,ctr_labels,pop_labels,colors = GetDimInfo(iteration,n_iters,Dim)
        
        if len(controls) > Dim:
            raise Exception("Dimension lower than controls, not enough colors")
            
            
        for i in range(len(controls)):    
            ax_ctr.plot(
                dynamics.times,
                controls[i],
                label=ctr_labels[i],
                color=colors[i],
                ls=ls,
                alpha=alpha,
            )

        for i in range(Dim):

            ax_dyn.plot(
                dynamics.times,
                dynamics.expect[i],
                label=pop_labels[i],
                color=colors[i],  
                ls=ls,
                alpha=alpha,
            )

    ax_dyn.legend()
    ax_dyn.set_xlabel('time')
    ax_dyn.set_ylabel('population')
    ax_ctr.legend()
    ax_ctr.set_xlabel('time')
    ax_ctr.set_ylabel('control amplitude')
    plt.show()


#-------------------------------------------------------------------------------------------------

#3 level Hamiltonian
Dim = 2
N_Controls = 2

diag = []
for i in range(Dim-1):
    diag.append(1)

d2 = diag

diag = []
for i in range(Dim):
    diag.append((i+1)**2/4)
d1 = diag

def S(t):
    """Shape function for the field update"""
    return krotov.shapes.flattop(
        t, t_start=0, t_stop=5, t_rise=0.3, t_fall=0.3, func='blackman'
    )
def guess_control(t, args, omega=1.0, ampl0=0.2):
    return ampl0 * krotov.shapes.blackman(
        t, t_start=0, t_stop=5
    )
def guess_control2(t, args):
    return krotov.shapes.flattop(
        t, t_start = 0, t_stop = 5, t_rise = 0.1, t_fall= 0.1
    )

H1 = hamiltonian(guess_control,guess_control2,N=Dim,d1=d1,d2=d2)
tlist = np.linspace(0, 5, 500)
plot_pulse(H1[1][1], tlist)
plot_pulse(H1[2][1], tlist)

kets = np.eye(Dim)
qKets  = []
for i in range(Dim):
    qKets.append ( qutip.Qobj(kets[i]) )

objectives = [
    krotov.Objective(
        initial_state= qKets[0], target=qKets[-1], H=H1
    )
]

pulse_options = {
    H1[1][1]: dict(lambda_a=5, update_shape=S),
    H1[2][1]: dict(lambda_a=5, update_shape=S)
}

projs = []
for i in range(0,Dim):
    projs.append(qutip.ket2dm(qKets[i]))


guess_dynamics = objectives[0].mesolve(tlist, e_ops=projs)
plot_population(guess_dynamics)

opt_result = krotov.optimize_pulses(
    objectives,
    pulse_options=pulse_options,
    tlist=tlist,
    propagator=krotov.propagators.expm,
    chi_constructor=krotov.functionals.chis_ss,
    info_hook=krotov.info_hooks.print_table(J_T=krotov.functionals.J_T_ss),
    check_convergence=krotov.convergence.Or(
        krotov.convergence.value_below('1e-3', name='J_T'),
        krotov.convergence.check_monotonic_error,
    ),
    store_all_pulses=True,
)
print(opt_result)

opt_dynamics = opt_result.optimized_objectives[0].mesolve(
    tlist, e_ops=projs)

plot_population(opt_dynamics)
plot_iterations(opt_result)

#------------------------------------------------------------------------------

#Interpolate prevoius result to obtain smooth control function

prev_opt_controls =opt_result.optimized_controls[0]
t_interp = opt_result.tlist
# Interpolate the optimized_controls array using a cubic spline
control_interp_func = lambda t: CubicSpline(t_interp, prev_opt_controls)(t)  # perform cubic spline interpolation

def guess_control(t, args):
    return control_interp_func(t)

prev_opt_controls2 =opt_result.optimized_controls[1]
# Interpolate the optimized_controls array using a cubic spline
control_interp_func2 = lambda t: CubicSpline(t_interp, prev_opt_controls2)(t)  # perform cubic spline interpolation

def guess_control2(t, args):
    return control_interp_func2(t)

#Dim level Hamiltonian with initial guess from prev result

Dim = 5
N_Controls = 2

diag = []
for i in range(Dim-1):
    diag.append(1)

d2 = diag

diag = []
for i in range(Dim):
    diag.append((i+1)**2/4)
d1 = diag

H2 = hamiltonian(guess_control,guess_control2,N=Dim,d1=d1,d2=d2)
#tlist = np.linspace(0, 5, 500)
plot_pulse(H2[1][1], tlist)

kets = np.eye(Dim)
qKets  = []
for i in range(Dim):
    qKets.append ( qutip.Qobj(kets[i]) )

objectives = [
    krotov.Objective(
        initial_state= qKets[0], target=qKets[2], H=H2
    )
]

pulse_options = {
    H2[1][1]: dict(lambda_a=5, update_shape=S),
    H2[2][1]: dict(lambda_a=5, update_shape=S)
}

projs = []
for i in range(0,Dim):
    projs.append(qutip.ket2dm(qKets[i]))


guess_dynamics = objectives[0].mesolve(tlist, e_ops=projs)
plot_population(guess_dynamics)

opt_result = krotov.optimize_pulses(
    objectives,
    pulse_options=pulse_options,
    tlist=tlist,
    propagator=krotov.propagators.expm,
    chi_constructor=krotov.functionals.chis_ss,
    info_hook=krotov.info_hooks.print_table(J_T=krotov.functionals.J_T_ss),
    check_convergence=krotov.convergence.Or(
        krotov.convergence.value_below('1e-3', name='J_T'),
        krotov.convergence.check_monotonic_error,
    ),
    store_all_pulses=True,
)
print(opt_result)

opt_dynamics = opt_result.optimized_objectives[0].mesolve(
    tlist, e_ops=projs)

plot_population(opt_dynamics)
plot_iterations(opt_result)