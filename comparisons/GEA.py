import geatpy as ea
import numpy as np
import math
r = 1  # 模拟该案例问题计算目标函数时需要用到的额外数据
def evalVars(Vars):  # 定义目标函数（含约束）
    ObjV = np.sum((Vars - r) ** 2, 1, keepdims=True)  # 计算目标函数值
    return ObjV  # 返回目标函数值矩阵和违反约束程度矩阵
func_1 = lambda x: np.array([sum([_ * _ for _ in xi]) for xi in x])[:,None]
# func_4 = lambda x: np.array([-20*np.exp(-0.2*np.sqrt(0.5*x[i][0]*x[i][0]+0.5*x[i][1]*x[i][1])) - np.exp(0.5*np.cos(2*math.pi*x[i][0]) + 0.5*np.cos(2*math.pi*x[i][1])) + 20 + math.e for i in len(range(x))])[:,None]
problem = ea.Problem(name='soea quick start demo',
                        M=1,  # 目标维数
                        maxormins=[1],  # 目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标
                        Dim=5,  # 决策变量维数
                        varTypes=[0, 0, 1, 1, 1],  # 决策变量的类型列表，0：实数；1：整数
                        lb=[-1, 1, 2, 1, 0],  # 决策变量下界
                        ub=[1, 4, 5, 2, 1],  # 决策变量上界
                        evalVars=func_1)
# 构建算法
algorithm = ea.soea_SEGA_templet(problem,
                                    ea.Population(Encoding='RI', NIND=20),
                                    MAXGEN=60,  # 最大进化代数。
                                    logTras=1,  # 表示每隔多少代记录一次日志信息，0表示不记录。
                                    trappedValue=1e-6,  # 单目标优化陷入停滞的判断阈值。
                                    maxTrappedCount=100)  # 进化停滞计数器最大上限值。
# 求解
res = ea.optimize(algorithm, seed=1, verbose=True, drawing=1, outputMsg=True, drawLog=False, saveFlag=True, dirName='result')
