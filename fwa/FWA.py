import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as la
import random
import math
import os


Up = 5
Low = -5
def draw_iter(fig,idx, all_sparks, all_fits, e_sparks, e_fits):
    x,y = np.mgrid[Low:Up:200j,Low:Up:200j]
    sigma = 2
    z_1 = -1/(2 * np.pi * (sigma**2)) * np.exp(-(x**2+y**2)/(2 * sigma**2)) 
    z_2 = -1/(2 * np.pi * (sigma**2)) * np.exp(-((x+5)**2+(y+5)**2)/(2 * sigma**2)) - 0.7*1/(2 * np.pi * (sigma**2)) * np.exp(-((x-5)**2+(y-5)**2)/(2 * sigma**2))
    z_3 = -20*np.exp(-0.2*np.sqrt(0.5*x*x+0.5*y*y))- np.exp(0.5*np.cos(2*math.pi*x) + 0.5*np.cos(2*math.pi*y)) + 20 + math.e
    ax = Axes3D(fig)
    print("min value: ", (e_fits[0]))
    ax.plot_surface(x, y, z_3, rstride=1, cstride=1, cmap='summer',alpha = 0.6)

    all_sparks = np.array(all_sparks)                         #所有火花
    all_fits = np.squeeze(np.array(all_fits))
    #print(all_sparks[:,0].shape, all_sparks[:,1].shape, all_fits.shape)
    ax.scatter(all_sparks[:,0], all_sparks[:,1], all_fits, c='b', marker='o', s = 25)

    e_sparks = np.array(e_sparks)                            #筛选后的火花
    e_fits = np.squeeze(np.array(e_fits))
    #print(e_sparks[:,0].shape, e_sparks[:,1].shape, e_fits.shape)
    ax.scatter(e_sparks[:,0], e_sparks[:,1], e_fits, c='r', marker='x', s = 45)
    
    os.makedirs("./results/fwa/50_degree/", exist_ok=True)
    os.makedirs("./results/fwa/0_degree/", exist_ok=True)

    ax.view_init(elev=50)
    plt.savefig("./results/fwa/50_degree/iter_"+str(idx)+".png", bbox_inches='tight', dpi = 300)      #减少空白边框，提高图片分辨率
    #plt.pause(2.5)
    

    ax.view_init(elev=0)
    plt.savefig("./results/fwa/0_degree/iter_"+str(idx)+".png", bbox_inches='tight', dpi = 300)
    #plt.pause(2.5)
    plt.clf()


    with open("./results/fwa/min_value.txt","a") as f:
        f.write("iter_"+str(idx)+": "+str(e_fits[0])+"\n")





class FWA(object):

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
                  sp_size = 30,
                  init_amp = 10,
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
            #print("iter: ",idx, len(fireworks[0]),fireworks, fits, np.sum((np.array(fireworks[0])**2)))
        
        self.time = time.time() - begin_time


        return self.best_fit, self.trace

    def iter(self, idx, fig, fireworks, fits):
        
        
        e_sparks, e_fits = self._explode_v2(fireworks) 
        n_fireworks, n_fits = self._select(e_sparks, e_fits)    #select the best sp_size
        print("\niter", idx)
        #print("fits",n_fireworks.shape , n_fits.shape)
     

        self._num_iter += 1
        self._num_eval += len(e_sparks)
            
        self.best_idv = n_fireworks[0]
        self.best_fit = n_fits[0]
        self.trace.append([n_fits[0]])

        fireworks = n_fireworks
        fits = n_fits
        print("e_sparks, e_fits ",np.array(e_sparks).shape, np.array(e_fits).shape, np.array(fits).shape )
          
        if idx%10 == 0:
            draw_iter(fig,idx, e_sparks, e_fits, n_fireworks, n_fits)

        if idx%15 == 0:
            self._dyn_amp*=0.95                 #幅值更新，可以改进

        return fireworks, fits


    def _init_fireworks(self):
    
        fireworks = np.random.uniform(self.lower_bound, 
                                      self.upper_bound, 
                                      [self.sp_size, self.dim])
        fireworks = fireworks.tolist()
        fits = self.evaluator(fireworks)

        return fireworks, fits

    def _terminate(self):
        if self._num_iter >= self.max_iter:
            return True
        if self._num_eval >= self.max_eval:
            return True
        return False

   

    def _explode_v2(self, fireworks):

        delta = 1e-5
        normal_num =  20
        gaussian_num = 10
        e_fits = np.array(self.evaluator(fireworks))   #sp_size
        e_sparks = np.array(fireworks)        #sp_size,2
        #calculate amp
        amp = np.zeros(self.sp_size)
        sum = np.sum(e_fits - np.min(e_fits))
        amp = self._dyn_amp*(e_fits - np.min(e_fits)+delta)/(sum + delta)
          
        index = np.argsort(e_fits)
        normal_spark = e_sparks[index][0:normal_num,:]
        normal_amp = amp[index][0:normal_num]
       
        
        gaussian_spark = e_sparks[index][0:gaussian_num,:]
        gaussian_amp = amp[index][0:gaussian_num]
       
        new_sparks = []

        
        for i in range(normal_num):
            for j in range(10):
                choice = np.array([np.random.choice([0,1]), np.random.choice([0,1])])
                tmp = normal_amp[i]*np.random.uniform(-1,1)*choice + normal_spark[i]
                tmp[0] = self.bound(tmp[0])
                tmp[1] = self.bound(tmp[1])
                new_sparks.append(tmp)

        for i in range(gaussian_num):
            for j in range(5):
                tmp = gaussian_amp[i]*np.random.normal(loc = 10,scale = 5) + gaussian_spark[i]
                tmp[0] = self.bound(tmp[0])
                tmp[1] = self.bound(tmp[1])
                new_sparks.append(tmp)

        e_sparks = np.array(new_sparks)
        e_fits = np.array(self.evaluator(e_sparks))

        return e_sparks, e_fits

    def bound(self, x):
        if x > Up:
            x = Up
        if x < Low:
            x = Low
        return x

    def distance(self, x, y):
        (rowx, colx) = x.shape
        (rowy, coly) = y.shape
        dis = np.zeros((rowx, rowy))
        for i in range(0, rowx):
            for j in range(0, rowy):
                dis[i, j] = la.norm(x[i] - y[j])**2
        return dis


    def _select(self, fireworks, fits):

        idx = np.argsort(fits)
        fireworks = fireworks[idx]
        fits = fits[idx]


        dis = self.distance(fireworks, fireworks)
        
        r = np.sum(dis, axis=0)
        p = r/np.sum(r)              #每个点被选择的概率
        index = self.roulette_selection(p, self.sp_size-1)
        index = [0] + index
        select_sp = fireworks[index]
        select_sp = np.array(select_sp)

        select_fits = fits[index]
        select_fits = np.array(select_fits)
        
        #print("select", fireworks.shape,fits.shape, dis.shape, select_sp.shape, select_fits.shape, index)
        return select_sp,select_fits


    #轮盘赌算法
    def roulette_selection(self, prob_list, num_select):
        selected = []
        for i in range(num_select): 
            r = random.random()
            for j, p in enumerate(prob_list):
                r -= p
                if r <= 0:                  #索引可重复
                    selected.append(j)
                    break
        return selected    #返回选择元素的索引

    
