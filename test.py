from fwa.BBFWA import *
import numpy as np




if __name__ == "__main__":

	algo = BBFWA()
	func_1 = lambda x: [sum([_ * _ for _ in xi]) for xi in x]

	sigma = 2
	func_2 = lambda x: [ - 1/(2 * np.pi * (sigma**2)) * np.exp(-(xi[0]**2+xi[1]**2)/(2 * sigma**2))  for xi in x]
	func_3 = lambda x: [ - 1/(2 * np.pi * (sigma**2)) * np.exp(-(xi[0]**2+xi[1]**2)/(2 * sigma**2) - 1/(2 * np.pi * (sigma**2)) * np.exp(-((xi[0]-5)**2+(xi[1]-5)**2)/(2 * sigma**2)) ) for xi in x]


 
	algo.load_prob(evaluator=func_1,
		dim=2,
		max_eval=30*10000,
		)
	result = algo.run()
	print("min function value", result)




