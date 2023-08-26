import numpy as np
from time import time
from matplotlib import pyplot as plt

# Harmonic Oscillator Function
def hoscillator(q,p):
    q = np.array(q)
    p = np.array(p)

    q_out = p
    p_out = -q
    
    return q_out, p_out

# Base Integrator Class
class Integrator():
    
    def __init__(self,f, h):
        self.function = f
        self.h = h
        self.times = None
        self.qs = None
        self.ps = None
    
    def step(self, t, q, p):
        q,p = self.function(q,p)
        return t, q, p
    
    def integrate(self, interval, q, p):
        t = interval[0]
        t_fin = interval[1]
        
        self.times = [t]
        self.qs = [np.array(q)]
        self.ps = [np.array(p)]        

        while t<t_fin:
            t, q, p = self.step(t, q, p)
            self.times.append(t)
            self.qs.append(np.array(q))
            self.ps.append(np.array(p))

    def plot_orbit(self,n):
        plot_q = [element[n] for element in self.qs]
        plot_p = [element[n] for element in self.ps]           
        
        plt.plot(plot_p,plot_q)
        plt.gca().axis('equal')     

    def energy(self):
        return [0.5 * q**2 + 0.5 * p**2 for q, p in zip(self.qs, self.ps)]

# Symplectic Euler Method
class SymplecticEuler(Integrator):
    
    def step(self, t, q, p):
        
        q = np.array(q)
        p = np.array(p)
        
        Q, P = self.function(q,p)
        
        p = p + self.h*np.array(P)
        
        Q, P = self.function(q,p)
        
        q = q + self.h*np.array(Q)
        
        t = t+self.h
        
        return t, q, p

# Forward Euler Method
class ForwardEuler(Integrator):
    
    def step(self, t, q, p):
        
        q = np.array(q)
        p = np.array(p)
        
        Q, P = self.function(q,p)
        
        p = p + self.h*np.array(P)
                
        q = q + self.h*np.array(Q)
        
        t = t+self.h
        
        return t, q, p

# Runge Kutta 4 Method
class RungeKutta4(Integrator):
    
    def step(self, t, q, p):
        
        q = np.array(q)
        p = np.array(p)

        Q1, P1 = self.function(q,p)
        
        Q2, P2 = self.function(q+np.array(Q1)*self.h/2, p + np.array(P1)*self.h/2)
        
        Q3, P3 = self.function(q+np.array(Q2)*self.h/2, p + np.array(P2)*self.h/2)
        
        Q4, P4 = self.function(q+np.array(Q3)*self.h, p + np.array(P3)*self.h)
        
        q = q + 1/6*self.h*(np.array(Q1) + 2*np.array(Q2) + np.array(Q3) +np.array(Q4))

        p = p + 1/6*self.h*(np.array(P1) + 2*np.array(P2) + np.array(P3) + np.array(P4))

        t = t + self.h

        return t, q, p

# Adams Bashforth Method
class AdamsBashforth(Integrator):
    def __init__(self, f, h):
        super().__init__(f, h)
        self.prev_p = None

    def step(self, t, q, p):
        Q, P = self.function(q, p)
        
        if self.prev_p is None:
            self.prev_p = P

        p_new = p + self.h * (1.5 * P - 0.5 * self.prev_p)
        q_new = q + self.h * p_new
        
        self.prev_p = P
        t += self.h
        return t, q_new, p_new

# Symplectic Order 4 Method
class SymplecticOrder4(Integrator):
    
    def __init__(self, h, function):
        self.h = h
        self.function = function

        # precalculate constants
        self.c1 = 1/(2*(2-2**(1/3)))
        self.c4 = self.c1
        self.c2 = (1 - 2**(1/3))/(2*(2-2**(1/3)))
        self.c3 = self.c2
        self.d1 = 1/((2-2**(1/3)))
        self.d3 = self.d1
        self.d2 = ( - 2**(1/3))/(2-2**(1/3))
        
    def step(self, t, q, p):

        q = np.array(q)
        p = np.array(p)

        q = self.Q_step(self.c4, q, p)
        p = self.P_step(self.d3, q, p)
        q = self.Q_step(self.c3, q, p)
        p = self.P_step(self.d2, q, p)
        q = self.Q_step(self.c2, q, p)
        p = self.P_step(self.d1, q, p)
        q = self.Q_step(self.c1, q, p)

        t += self.h
        return t, q, p
    
    def Q_step(self, c, q, p):
        
        Q, _ = self.function(q, p)
        return q + [self.h*c*q_i for q_i in Q]
    
    def P_step(self, d, q, p):
        _, P = self.function(q, p)
        return p + [self.h*d*p_i for p_i in P]

# ... [your class definitions for Integrator, SymplecticEuler, ForwardEuler, etc.] ...

def compare_methods(methods, names, interval, q_init, p_init):
    # Lists to store computation times and energy differences for each method
    computation_times = []
    energy_diffs = []

    # Iterate over each method and compute performance and accuracy metrics
    for method in methods:
        # Measure computation time
        start_time = time()
        method.integrate(interval, q_init, p_init)
        end_time = time()
        computation_times.append(end_time - start_time)
        
        # Compute initial and final energies
        initial_energy = 0.5 * (q_init[0]**2 + p_init[0]**2)
        final_q = method.qs[-1]
        final_p = method.ps[-1]
        final_energy = 0.5 * (final_q[0]**2 + final_p[0]**2)
        
        # Compute energy difference
        energy_diffs.append(abs(final_energy - initial_energy))
        
        # Plot the orbit
        method.plot_orbit(0)
        plt.title(f"Orbit for {names[methods.index(method)]}")
        plt.show()

    # Display results
    print("Computation Times (in seconds):")
    for name, t in zip(names, computation_times):
        print(f"{name}: {t:.4f} seconds")

    print("\nEnergy Differences (initial vs. final):")
    for name, diff in zip(names, energy_diffs):
        print(f"{name}: {diff:.4f}")

# Define the function and initial conditions
interval = [0, 10 * np.pi]
q_init = np.ones(100)
p_init = np.zeros(100)

# List of methods and their names
methods = [SymplecticEuler(hoscillator, 0.1), ForwardEuler(hoscillator, 0.1),
           RungeKutta4(hoscillator, 0.1), SymplecticOrder4(0.1, hoscillator),
           AdamsBashforth(hoscillator, 0.1)]
names = ['Symplectic Euler', 'Forward Euler', 'Runge Kutta 4', 'Symplectic Order 4', 'Adams Bashforth 4']

# Call the comparison function
compare_methods(methods, names, interval, q_init, p_init)
