import sys
sys.path.append("../../krotov/src")
import qutip 
import numpy as np
import matplotlib.cm as cm
import matplotlib.pylab as plt
import krotov
import pandas as pd
from scipy.interpolate import interp1d
import csv


#3 level Hamiltonian
Dim = 100
N_Controls = 2
ToleranceDict = {1e-3:"1e-3Tolerance", 1e-2:"1e-2Tolerance"}
Tolerance = 1e-2
ToleranceString = ToleranceDict[Tolerance]

# read excel file into a pandas DataFrame
df1 = pd.read_csv(f'..\\Controls\\{ToleranceString}\\control1_dim_12from9_6_3_state0to1.csv')
df2 = pd.read_csv(f'..\\Controls\\{ToleranceString}\\control2_dim_12from9_6_3_state0to1.csv')

# convert the DataFrame into a numpy array
controls1 = df1.iloc[:, -1].values
print(len(controls1))
controls2 = df2.iloc[:, -1].values
print(len(controls2))
# Assuming your time grid has 500 points and goes from 0 to 5
tlist = np.linspace(0, 5, 500)

# define the control function using interpolation
control_func1 = lambda t: interp1d(tlist[1:], controls1, kind='cubic', fill_value="extrapolate")(t)
control_func2 = lambda t: interp1d(tlist[1:], controls2, kind='cubic', fill_value="extrapolate")(t)

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
        pop_labels = ['{number} {tipo}'.format(number = f"{i}",tipo = f"{ctr_label}") for i in range(Dim)]
        ctr_labels = ['{i} {ctr_label}'.format(i = i,ctr_label = ctr_label) for i in range(N_Controls)]
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

def get_J_T_prev(**kwargs):
        try:
            return kwargs['info_vals'][-1]
        except IndexError:
            return 0

def write_functional_values(**kwargs):
    """Write the current value of the objective function to a CSV file."""
    with open(f'..\\Analisis\\{ToleranceString}\\TESTTfunctional_valuesd_dim_100from12_9_6_3_0to1.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if kwargs['info_vals']:
            iteration = kwargs['iteration']
            J_T_val = krotov.functionals.J_T_ss(**kwargs)
            Σgₐdt = np.sum(kwargs['g_a_integrals'])
            J = J_T_val + Σgₐdt
            if iteration > 0:
                J_T_prev_val = get_J_T_prev(**kwargs)
                ΔJ_T = J_T_val - J_T_prev_val
                ΔJ = ΔJ_T + Σgₐdt
            secs = int(kwargs['stop_time'] - kwargs['start_time'])
            # J is the SUM of J_T_val and Σgₐdt
            # ΔJ is the change on J (the sum) and ΔJ_T is the change on J_T_val
            writer.writerow([iteration , J_T_val, Σgₐdt, J, ΔJ_T, ΔJ, secs ])

def custom_propagator(H, dt, rho, c_ops, args=None, options=None):
    """Propagate the density matrix over a single time step using qutip.propagator."""
    # Note: this function uses the 'adams' method
    if options is None:
        options = qutip.Options()
        options.method = 'adams'
    tlist = [0, dt]
    U = qutip.propagator(H, tlist, c_op_list=[], args=args, options=options)
    # calculate the propagated state by multiplying the propagator with the state
    propagated_rho = U[-1] * rho * U[-1].dag()
    return propagated_rho
#-------------------------------------------------------------------------------------------------

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
    return krotov.shapes.one_shape(t)
    return krotov.shapes.flattop(
        t, t_start=0, t_stop=5, t_rise=0.3, t_fall=0.3, func='blackman'
    )

def guess_control(t, args, omega=1.0, ampl0=0.2,):
    return control_func1(t)
    return krotov.shapes.flattop(
        t, t_start=0, t_stop=5, t_rise=0.3, t_fall=0.3, func='blackman'
    )
def guess_control2(t, args, omega=1.0, ampl0=0.4,):
    return control_func2(t)
    return krotov.shapes.flattop(
        t, t_start=0, t_stop=5, t_rise=0.3, t_fall=0.3, func='blackman'
    )

H1 = hamiltonian(guess_control,guess_control2, N=Dim,d1=d1,d2=d2)
tlist = np.linspace(0, 5, 500)
plot_pulse(H1[1][1], tlist)
plot_pulse(H1[2][1], tlist)

kets = np.eye(Dim)
qKets  = []
for i in range(Dim):
    qKets.append ( qutip.Qobj(kets[i]) )

objectives = [
    krotov.Objective(
        initial_state= qKets[0], target=qKets[1], H=H1
    )
]

print(objectives[0].summarize())

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
    info_hook=krotov.info_hooks.chain(
        krotov.info_hooks.print_table(J_T=krotov.functionals.J_T_ss),
        write_functional_values
    ),
    check_convergence=krotov.convergence.Or(
        krotov.convergence.value_below(str(Tolerance), name='J_T'),
        krotov.convergence.check_monotonic_error,
    ),
    store_all_pulses=True,
)
print(opt_result)

opt_dynamics = opt_result.optimized_objectives[0].mesolve(
    tlist, e_ops=projs)

plot_population(opt_dynamics)
plot_iterations(opt_result)

