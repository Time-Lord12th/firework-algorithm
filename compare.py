from comparisons.BBFWA import BBFWA
import pyswarms as ps
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse
import os
import geatpy as ea
### define dimension, lower bound, upper bound as glocal variant
d = 2
lb = np.array([-1, -1])
ub = np.array([1,1])

save_dir = './results'
os.makedirs(save_dir, exist_ok=True)

## define sets of evaluation function 
# input : numpy ndarray of size (n,d)
# return : numpy ndarray of size (n) 

def func_warpper(func):
    def inner_func(x):
        return func(x)[:,None]
    return inner_func
    

def func_1(x):  # Sphere function
    bs = x.shape[0]
    result = np.zeros(bs)
    for i in range(len(x)):
        result[i] = np.sum(x[i] ** 2)
    return result

def func_2(x): # Rosenbrock function
    bs = x.shape[0]
    result = np.zeros(bs)
    for i in range(len(x)):
        result[i] = np.sum(100 * (x[i,1:]-x[i,:-1]**2) **2 + (x[i,:-1]-1)**2)
    return result
    

def func_3(x):  # Rastrigrin function
    bs = x.shape[0]
    result = np.zeros(bs)
    for i in range(len(x)):
        result[i] = np.sum(x[i] ** 2 - 10 * np.cos(2 * np.pi * x[i]) + 10)
    return result

def func_4(x): # Griewank function
    bs = x.shape[0]
    result = np.zeros(bs)
    for i in range(len(x)):
        result[i] = 1 + np.sum(x[i] ** 2 / 4000) - np.prod(np.cos(x[i] / np.sqrt(np.arange(d) + 1)))
    return result

def func_5(x): # ellipse function
    bs = x.shape[0]
    result = np.zeros(bs)
    for i in range(len(x)):
        result[i] = np.sum(x[i] ** 2 * 10 ** (4 * np.arange(d) / (d-1)))
    return result
    
def func_6(x):  # cigar
    bs = x.shape[0]
    result = np.zeros(bs)
    for i in range(len(x)):
        result[i] = x[i,0]**2 + np.sum(x[i,1:] ** 2 * 10 ** 4)
    return result

def func_7(x):  # tablet
    bs = x.shape[0]
    result = np.zeros(bs)
    for i in range(len(x)):
        result[i] = (10 ** 4) * (x[i,0]**2) + np.sum(x[i,1:] ** 2)
    return result

def func_8(x):  # schwefel
    bs = x.shape[0]
    result = np.zeros(bs)
    for i in range(len(x)):
        result[i] = np.sum((x[i,0]-x[i,:]**2) ** 2 + (x[i,:]-1) ** 2)
    return result

def func_9(x): # ackley
    bs = x.shape[0]
    result = np.zeros(bs)
    for i in range(len(x)):
        result[i] = 20 + np.e -20 * np.exp(-0.2 * np.sqrt(1/d * np.sum(x[i]**2))) - np.exp(1/d * np.sum(np.cos(2*np.pi*x[i]**2)))
    return result

func = [func_1, func_2, func_3, func_4, func_5, func_6, func_7, func_8, func_9]

def evaluate_on_bbfwa(iters, func_to_eval):
    algo = BBFWA()
    algo.load_prob(evaluator=func_to_eval,
		dim=d,
        upper_bound = ub,
        lower_bound = lb,
        max_iter = iters,
		max_eval = 1e+10,
		)
    optimal_point, optimal_value, trajectory = algo.run()
    plt.figure()
    plt.plot(trajectory, "k:")
    plt.savefig(save_dir+'/bbfwa.png')
    print('save optimization curves to ----' + save_dir+'/bbfwa.png')
    return optimal_point, optimal_value    

def evaluate_on_pso(iters, func_to_eval):
    x_max = ub * np.ones(d)
    x_min = lb * np.ones(d)
    bounds = (x_min, x_max)
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=d, options=options, bounds=bounds)
    optimal_value, optimal_point = optimizer.optimize(func_to_eval, iters=iters)
    plt.figure()
    plt.plot(optimizer.cost_history, "k:")
    plt.savefig(save_dir+'/pso.png')
    return optimal_point.tolist(), optimal_value.tolist()

def evaluate_on_genetic(iters, func_to_eval):
    problem = ea.Problem(name='soea quick start demo',
                        M=1,  # 目标维数
                        maxormins=[1],  # 目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标
                        Dim=d,  # 决策变量维数
                        varTypes=[0] * d,  # 决策变量的类型列表，0：实数；1：整数
                        lb=[lb] * d,  # 决策变量下界
                        ub=[ub] * d,  # 决策变量上界
                        evalVars=func_to_eval)
# 构建算法
    algorithm = ea.soea_SEGA_templet(problem,
                                    ea.Population(Encoding='RI', NIND=20),
                                    MAXGEN=iters,  # 最大进化代数。
                                    logTras=1,  # 表示每隔多少代记录一次日志信息，0表示不记录。
                                    trappedValue=1e-6,  # 单目标优化陷入停滞的判断阈值。
                                    maxTrappedCount=1e+10)  # 进化停滞计数器最大上限值。
# 求解
    res = ea.optimize(algorithm, seed=1, verbose=True, drawing=0, outputMsg=False, drawLog=False, saveFlag=False, dirName='result')
    trace = np.array(algorithm.log['f_opt'])
    plt.figure()
    plt.plot(trace, "k:")
    plt.savefig(save_dir+'/gene.png')
    print('save optimization curves to ----' + save_dir+'/gene.png')
    return res['Vars'][0].tolist(),res['ObjV'][0][0]
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dimension', default=15, type=int, help='dimension for vector x')
    parser.add_argument('--lower_bound', default=-8, type=int, help='lower bound for x')
    parser.add_argument('--upper_bound', default=8, type=int, help='upper bound for x')
    parser.add_argument('--eval_func_idex', type=int, default=0, help='index for evaluation funcs')
    parser.add_argument('--iteration', type=int, default=1000, help='iterations')
    args = parser.parse_args()
    d = args.dimension
    lb = args.lower_bound
    ub = args.upper_bound
    func_to_eval = func[args.eval_func_idex]
    iters = args.iteration
    point_fwa, value_fwa = evaluate_on_bbfwa(iters, func_to_eval)
    point_pso, value_pso = evaluate_on_pso(iters, func_to_eval)
    point_gen, value_gen = evaluate_on_genetic(iters, func_warpper(func_to_eval))
    print('Optimal point for Firework algotithm:'+str(point_fwa))
    print('Optimal point for Particle Swarm Optimization:'+str(point_pso))
    print('Optimal point for Genetic algorithm:'+str(point_gen))
    print('Optimal value for Firework algotithm:'+str(value_fwa))
    print('Optimal value for Particle Swarm Optimization:'+str(value_pso))
    print('Optimal value for Genetic algorithm:'+str(value_gen))
    print('-----')
    

