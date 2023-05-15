from fwa.FWA import *
import numpy as np
import math
import matplotlib.pyplot as plt

func_4 = lambda x: [-20*np.exp(-0.2*np.sqrt(0.5*xi[0]*xi[0]+0.5*xi[1]*xi[1])) - np.exp(0.5*np.cos(2*math.pi*xi[0]) + 0.5*np.cos(2*math.pi*xi[1])) + 20 + math.e for xi in x]


def ablation_spark_type():

	algo = FWA()

	algo.load_prob(evaluator=func_4,
		dim=2,
		max_iter=300,
		max_eval=1e10,
		normal_num = 20,
		gaussian_num = 10
		)
	result1, trace_1 = algo.run()
	
	algo.load_prob(evaluator=func_4,
		dim=2,
		max_iter=300,
		max_eval=1e10,
		normal_num = 30,
		gaussian_num = 0
		)
	result2, trace_2 = algo.run()
	
	algo.load_prob(evaluator=func_4,
		dim=2,
		max_iter=300,
		max_eval=1e10,
		normal_num = 00,
		gaussian_num = 30
		)
	result3, trace_3 = algo.run()

	print("normal + gaussuan :", result1)
	print("normal:", result2)
	print("gaussuan :", result3)

	plt.figure()
	plt.clf()
	plt.plot(trace_1, c='r', label='normal + Gaussuan')  
	plt.plot(trace_2, c='g', label='normal only')  
	plt.plot(trace_3, c='b', label='Gaussuan only')  
	plt.xlabel('iter')  
	plt.ylabel('min value')  
	plt.legend()
	plt.savefig("./results/ablation_spark_type.png")

def ablation_spark_num():

	algo = FWA()

	algo.load_prob(evaluator=func_4,
		dim=2,
		max_iter=1000,
		max_eval=1e10,
		normal_num = 10,
		gaussian_num = 5,
		sp_size = 15,
		)
	result1, trace_1 = algo.run()
	
	algo.load_prob(evaluator=func_4,
		dim=2,
		max_iter=1000,
		max_eval=1e10,
		normal_num = 20,
		gaussian_num = 10,
		sp_size = 30,
		)
	result2, trace_2 = algo.run()
	
	algo.load_prob(evaluator=func_4,
		dim=2,
		max_iter=1000,
		max_eval=1e10,
		normal_num = 40,
		gaussian_num = 20,
		sp_size = 60,
		)
	result3, trace_3 = algo.run()

	print("10+5:", result1)
	print("20+10:", result2)
	print("40+20 :", result3)

	plt.figure()
	plt.clf()
	plt.plot(trace_1, c='r', label='10 normal + 5  Gaussuan')  
	plt.plot(trace_2, c='g', label='20 normal + 10 Gaussuan')  
	plt.plot(trace_3, c='b', label='40 normal + 20 Gaussuan')  
	plt.xlabel('iter')  
	plt.ylabel('min value')  
	plt.legend()
	plt.savefig("./results/ablation_spark_num.png")



if __name__ == "__main__":
	
	ablation_spark_num()
	# ablation_spark_type()

	








