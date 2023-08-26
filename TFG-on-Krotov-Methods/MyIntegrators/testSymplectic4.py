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

def hoscillator(q,p):
    q = np.array(q)
    p = np.array(p)

    q_out = p
    p_out = -q
    
    return q_out, p_out


sympO4.h = 0.1

sympO4.integrate([0,5*np.pi],[1,1],[0,0])

sympO4.plot_orbit(1)