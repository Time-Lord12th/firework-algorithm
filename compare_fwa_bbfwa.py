from fwa.BBFWA import *
from fwa.FWA import *
import numpy as np
import math
import matplotlib.pyplot as plt

def func_ackley(x): 
    bs = x.shape[0]
    result = np.zeros(bs)
    for i in range(len(x)):
        result[i] = 20 + np.e -20 * np.exp(-0.2 * np.sqrt(1/2 * np.sum(x[i]**2))) - np.exp(1/2 * np.sum(np.cos(2*np.pi*x[i]**2)))
    return result

def compare_fwa_bbfwa():
	func_1 = lambda x: [sum([_ * _ for _ in xi]) for xi in x]

	sigma = 2
	func_2 = lambda x: [ - 1/(2 * np.pi * (sigma**2)) * np.exp(-(xi[0]**2+xi[1]**2)/(2 * sigma**2))  for xi in x]
	func_3 = lambda x: [ - 1/(2 * np.pi * (sigma**2)) * np.exp(-((xi[0]+5)**2+(xi[1]+5)**2)/(2 * sigma**2) - 0.7*1/(2 * np.pi * (sigma**2)) * np.exp(-((xi[0]-5)**2+(xi[1]-5)**2)/(2 * sigma**2)) ) for xi in x]
	func_4 = lambda x: [-20*np.exp(-0.2*np.sqrt(0.5*xi[0]*xi[0]+0.5*xi[1]*xi[1])) - np.exp(0.5*np.cos(2*math.pi*xi[0]) + 0.5*np.cos(2*math.pi*xi[1])) + 20 + math.e for xi in x]

	algo = BBFWA()
	algo.load_prob(evaluator=func_ackley,
		dim=2,
		max_eval=10*10000,
		upper_bound = 5,
                lower_bound = -5,
		)
	_, result, trace_1 = algo.run()
	
	algo = FWA()
	algo.load_prob(evaluator=func_4,
		dim=2,
		max_eval=10*10000,
		)
	result, trace_2 = algo.run()
	
	plt.figure()
	iters  = range(10, 10*(len(trace_1)+1), 10)
	plt.plot(trace_1, c='r', label='BBFWA')  
	plt.plot(trace_2, c='g', label='FWA')  
	plt.xlabel('iter')  
	plt.ylabel('min value')  
	plt.legend()
	plt.savefig("./results/comparison.png")
	plt.show()





if __name__ == "__main__":
	
	compare_fwa_bbfwa()

	







