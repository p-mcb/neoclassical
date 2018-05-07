################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class NCG:
    
    """
    class to represent a basic neoclassical growth model.
    unless otherwise stated, variables are in per (effective) worker terms.
    cobb-doublas production w/ labor-augmenting technology, 
    inelastic labor supply, deterministic,  population and technology growth,
    CES preferences
    """
    
    def __init__(self, params = None, gridk = np.linspace(0.04 ,0.2, 100)):
        
        
        self.params = {'delta':1,
                       'alpha':0.3,
                       'beta':0.9, 
                       'n':0.01, 
                       'g':0.03, 
                       'theta':1}
        
        if params:
            self.params.update(params)
        
        self.gridk = gridk
        
    def output(self, k):
        """
        returns productive output given factor input k
        """
        return k**self.params['alpha'] 
    
    def f_prime(self, k):
        """
        returns derivative of production function
        """
        return self.params['alpha']*k**(self.params['alpha']-1)
    
    def f_prime_inv(self, x):
        """
        inverse of derivative of production function
        """
        return (x/self.params['alpha']) ** (1/(self.params['alpha']-1))
    
    def cons(self, k, kp):
        """
        k:  k(t) ... capital chosen yesterday
        kp: k(t+1) ... capital chosen today
        returns consumption c(t) as residual of resource constraint given k, kp
        """
        return (self.output(k) 
                + (1-self.params['delta'])*k 
                - (1+self.params['n'])*(1+self.params['g'])*kp )
    
    def prices(self, k):
        """
        returns dict with real interest rate and wages
        """
        out = {'r': self.f_prime(k), 
               'w': self.output(k) - self.f_prime(k)*k}
        
        return out
    
    def inst_util(self, c):
        """
        returns instantaneous utility given consumption level
        """
        if self.params['theta'] == 1:
            u = np.log(c)
        
        else:
            u = (c**(1-self.params['theta']) - 1) / (1-self.params['theta'])
        
        return u
        
    def vfi(self, iter_max=50):
        """
        proceeds with value function iteration 
        to approximate value and policy functions
        
        note that an iteration counter is used,
        as opposed to a tolerance
        """
        
        nk = len(self.gridk)
        
        def obj(i, j, vf):
            """
            arguments: i:  grid index for today's capital
                       j:  grid index for tomorrow's capital
                       vf: array such that vf[index] = V(gridk[index])
                           where V is value function
                       
            returns :  objective function {u(c) + beta*vf[j]}
                       where c is determined by state capital gridk[i]
                       and chosen capital gridk[j]
            """
            inst = self.inst_util( self.cons(self.gridk[i],self.gridk[j]) )
            cont = self.params['beta'] * vf[j]
            return inst + cont
        
        def max_obj(i, vf):
            """
            arguments: i:  grid index for today's capital
                       vf: array such that vf[index] = V(gridk[index])
                           where V is value function
            
            returns: grid index for arg max of value fcn V(gridk[i])
            
            takes brute force approach  ...
            more sophisticated methods exist
            """
            it_max = -np.inf        # record maximum value of V found so far
            for j in range(nk):     # step through capital grid seeking argmax
                it = obj(i,j,vf)    # calculate objective functtion at gridk[j]
                if it > it_max:     # check if best value so far
                    it_max = it     # update max value
                    out = j         # update arg max index
            return out
            
        
        ## run through the iteration
        
        vf = [0 for i in range(nk)]  # initial guess
        
        it = 0  # iteration counter
        
        while it < iter_max:
            
            # policy function: 
            #  pf[i]=j such that policy at state gridk[i] is choice gridk[j]
            pf = [max_obj(i, vf) for i in range(nk)]
            
            # value function:
            #  vf[i]=x such that value fcn at gridk[i] is x
            vf = [obj(i, pf[i], vf) for i in range(nk)]
            
            it += 1  # increment counter
    
        # save approximations as instance attributes ...
        self.vf =[vf[x] for x in range(nk)]
        
        # save & reformat policy function structure for easier handling ...
        #  pf[i]=k such that policy at state gridk[i] is choice k
        self.pf = [self.gridk[pf[x]] for x in range(nk)]
        
        return 0
    
    def k_steady(self):
        """
        calculate steady state capital
        """
        return self.f_prime_inv((1+self.params['n'])
                                *(1+self.params['g'])
                                /self.params['beta']
                                -(1-self.params['delta']))
    
    def solve(self, k):
        """
        given state variable k,
        calculate prices & quantities
        """
        
        out = {'k' : k,
               'kp': self.pf[np.where(self.gridk == k)[0][0]], 
               'y' : self.output(k)}
        
        out.update({'c' : self.cons(k, out['kp'])})
        
        out.update(self.prices(k))
        
        return out
        
    
    def simulate(self, k0, T=8, plots=True):
        """
        simulate model
        
        arguments: k0: initial capital
                   T: number of periods to solve
                   plots: show results
                   
        returns: DataFrame with simulated series
        """
        
        # pandas DataFrame to hold simulation results
        sim = pd.DataFrame(data=np.nan, index=np.arange(T), 
                           columns=['k','kp','y','c','r','w','L','A'])
        
        # t=0 ... prices and quantities from solve()
        #     ... normalize population and labor-augmenting capital to 1
        sim.loc[0,] = {** self.solve(k0), ** {'L':1, 'A':1}}
        
        # time index 
        t = 1
        
        # iterate for remaining periods
        while t < T:
            
            # calculate population and technology ...
            LA = {'L':(1+self.params['n'])**t, 
                  'A':(1+self.params['g'])**t}
            
            # ... and prices and quantities
            sim.loc[t,] = {** self.solve(sim.loc[t-1,'kp']), ** LA}
            
            t += 1
        
        # graph simulation resuts
        if plots:
            fig, ax = plt.subplots(4,2, figsize=(15,15))
        
            for i,k in enumerate(sim.columns):
                x = i % 4
                y = i // 4
                ax[x][y].plot(sim[k])
                ax[x][y].set_title(k)
          
            plt.show()
        
        return sim
        