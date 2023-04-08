import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



Up = 50
Low = -50

def draw_iter(fig, e_sparks, e_fits ):
    x,y = np.mgrid[Low:Up:200j,Low:Up:200j]
    sigma = 2
    z_1 = -1/(2 * np.pi * (sigma**2)) * np.exp(-(x**2+y**2)/(2 * sigma**2)) 
    z_2 = -1/(2 * np.pi * (sigma**2)) * np.exp(-((x+0)**2+(y+0)**2)/(2 * sigma**2)) - 1/(2 * np.pi * (sigma**2)) * np.exp(-((x-5)**2+(y-5)**2)/(2 * sigma**2))
    
    ax = Axes3D(fig)
    ax.plot_surface(x, y, z_2, rstride=1, cstride=1, cmap='rainbow',alpha = 0.9)

    e_sparks = np.array(e_sparks)
    e_fits = np.squeeze(np.array(e_fits))
    print(e_sparks[:,0].shape, e_sparks[:,1].shape, e_fits.shape)
    ax.scatter(e_sparks[:,0], e_sparks[:,1], e_fits, c='r', marker='o')


    #plt.show()
    plt.pause(2.5)
    plt.clf()





class BBFWA(object):

    def  __init__(self):
        # Parameters

        # params of method
        self.sp_size = None       # total spark size
        self.init_amp = None      # initial dynamic amplitude
        
        # params of problem
        self.evaluator = None
        self.dim = None
        self.upper_bound = None
        self.lower_bound = None

        self.max_iter = None
        self.max_eval = None


        # States

        # private states
        self._num_iter = None
        self._num_eval = None
        self._dyn_amp = None

        # public states
        self.best_idv = None    # best individual found
        self.best_fit = None    # best fitness found
        self.trace = None       # trace of best individual in each generation

        # for inspection
        self.time = None

    def load_prob(self, 
                  # params for prob
                  evaluator = None,
                  dim = 2,
                  upper_bound = Up,
                  lower_bound = Low,
                  max_iter = 10000,
                  max_eval = 20000,
                  # params for method
                  sp_size = 200,
                  init_amp = 200,
                  ):

        # load params
        self.evaluator = evaluator
        self.dim = dim
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        self.max_iter = max_iter
        self.max_eval = max_eval

        self.sp_size = sp_size
        self.init_amp = init_amp
        
        # init states
        self._num_iter = 0
        
        self._num_eval = 0
        self._dyn_amp = init_amp
        self.best_idv = None
        self.best_fit = None
        self.trace = []

        self.time = 0

    def run(self):
        begin_time = time.time()

        fireworks, fits = self._init_fireworks()
        fig = plt.figure()
        for idx in range(self.max_iter):
            
            if self._terminate():
                break

            fireworks, fits = self.iter(idx, fig, fireworks, fits)
            print("iter: ",idx, len(fireworks[0]),fireworks, fits, np.sum((np.array(fireworks[0])**2)))
        
        self.time = time.time() - begin_time

        return self.best_fit

    def iter(self, idx, fig, fireworks, fits):
        
        e_sparks, e_fits = self._explode(fireworks, fits)
        self._explode_v2(fireworks) 
        n_fireworks, n_fits = self._select(fireworks, fits, e_sparks, e_fits)    

        # update states
        if n_fits[0] < fits[0]:
            self._dyn_amp *= 1.2
        else:
            self._dyn_amp *= 0.9

        self._num_iter += 1
        self._num_eval += len(e_sparks)
            
        self.best_idv = n_fireworks[0]
        self.best_fit = n_fits[0]
        self.trace.append([n_fireworks[0], n_fits[0], self._dyn_amp])

        fireworks = n_fireworks
        fits = n_fits
        print("e_sparks, e_fits ",np.array(e_sparks).shape, np.array(e_fits).shape, np.array(fits).shape )
        
        if idx%10 == 0:
            draw_iter(fig, e_sparks, e_fits)


        return fireworks, fits

    def _init_fireworks(self):
    
        fireworks = np.random.uniform(self.lower_bound, 
                                      self.upper_bound, 
                                      [1, self.dim])
        fireworks = fireworks.tolist()
        fits = self.evaluator(fireworks)

        return fireworks, fits

    def _terminate(self):
        if self._num_iter >= self.max_iter:
            return True
        if self._num_eval >= self.max_eval:
            return True
        return False

    def _explode(self, fireworks, fits):
        
        bias = np.random.uniform(-self._dyn_amp, self._dyn_amp, [self.sp_size, self.dim])
        rand_samples = np.random.uniform(self.lower_bound, self.upper_bound, [self.sp_size, self.dim])
        e_sparks = fireworks + bias

        in_bound = (e_sparks > self.lower_bound) * (e_sparks < self.upper_bound)     #(sp_size, dim) Ture/False
        e_sparks = in_bound * e_sparks + (1 - in_bound) * rand_samples
        e_sparks = e_sparks.tolist()
        e_fits = self.evaluator(e_sparks)
        
        return e_sparks, e_fits    

    def _explode_v2(self, fireworks):

        delta = 1e-3
        
        e_fits = np.array(self.evaluator(fireworks))   #200
        e_sparks = np.array(fireworks)        #200,2
        #calculate amp
        amp = np.zeros(self.sp_size)
        sum = np.sum(e_fits - np.min(e_fits))
        amp = self._dyn_amp*(e_fits - np.min(e_fits)+delta)/(sum + delta)
        print("amp", amp, e_sparks.shape)






    def _select(self, fireworks, fits, e_sparks, e_fits):
        idvs = fireworks + e_sparks
        print("111",np.array(fireworks).shape, np.array(e_sparks).shape)
        fits = fits + e_fits
        idx = np.argmin(fits)
        return [idvs[idx]], [fits[idx]]



