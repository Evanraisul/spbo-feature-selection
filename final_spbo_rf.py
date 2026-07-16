import numpy as np
from numpy.random import rand
from functionHO_RF import Fun
import random
import os
import math

def target_function():
    return


def initial_population(population_size, dim, min_values, max_values, target_function = target_function):
    
    X = np.zeros((population_size, dim + 1))
    #print( len(min_values) + 1 )
    
    for i in range(0, population_size):
        for j in range(0, dim):
            
            # Create random 2 Population in 2 Column
             X[i,j] = random.uniform(0, 1)
             
        # Evaluate those 2 values using target Esom and assign in 3rd Column
        #print(X[i,0:X.shape[1]-1])
        #print()
        X[i,-1] = target_function(X[i,0:X.shape[1]-1])
    return X

# Function: Update Best Student
def update_best_student(population, dim, best_student, min_values, max_values, target_function = target_function):
    
    idx  = list(range(0, population.shape[0]))
    idx  = random.choice(idx)
    
    k    = [1, 2]
    k    = random.choice(k)
    
    n_b  = np.copy(best_student)
    
    for j in range(0, dim):
        
        r1     = random.random()
        
        n_b[j] = np.clip(n_b[j] + ((-1)**k)*r1*(n_b[j] - population[idx, j]), min_values[0,j], max_values[0,j])
        
    n_b[-1] = target_function(n_b[:-1]) 
    
    if (n_b[-1] > best_student[-1]):
       n_b = np.copy(best_student)
       
    return n_b

# Function: Update Good Student Type A
def update_good_student_a(population, dim, best_student, type_idx, min_values, max_values, target_function = target_function):
    
    n_p = np.copy(population)
    
    for i in type_idx:
        
        for j in range(0, dim):
            
            r1        = random.random()
            n_p[i, j] = np.clip(population[i, j] + r1*(best_student[j] - population[i, j]), min_values[0,j], max_values[0,j])
        
        n_p[i,-1] = target_function(n_p[0,0:n_p.shape[1]-1])
        
    return n_p[type_idx,:]


# Function: Update Good Student Type B
def update_good_student_b(population, dim, best_student, type_idx, min_values, max_values, target_function = target_function):
    
    n_p = np.copy(population)
    
    for i in type_idx:
        
        for j in range(0, dim):
            
            r1        = random.random()
            r2        = random.random()
            n_p[i, j] = np.clip(population[i, j] + r1*(best_student[j] - population[i, j]) + (r2*(population[i, j] - population[type_idx, j].mean())), min_values[0,j], max_values[0,j])
        
        n_p[i,-1] = target_function(n_p[0,0:n_p.shape[1]-1])
        
    return n_p[type_idx,:]


# Function: Update Average Student 
def update_average_student(population, dim, type_idx, min_values, max_values, target_function = target_function):
    
    n_p = np.copy(population)
    
    for i in type_idx:
        
        for j in range(0, dim):
            
            r1        = random.random()
            n_p[i, j] = np.clip(population[i, j] + r1*(np.mean(population[type_idx, j]) - population[i, j]), min_values[0,j], max_values[0,j])
        
        n_p[i,-1] = target_function(n_p[0,0:n_p.shape[1]-1]) 
        
    return n_p[type_idx,:]


# Function: Update Random Student 
def update_random_student(population, dim, type_idx, min_values, max_values, target_function = target_function):
    
    n_p = np.copy(population)
    
    for i in type_idx:
        
        for j in range(0, dim):
            
            r1        = random.random()
            n_p[i, j] = np.clip(np.min(population[:, j]) + r1*(np.max(population[:, j]) - np.min(population[:, j])), min_values[0,j], max_values[0,j])
        
        n_p[i,-1] = target_function(n_p[0,0:n_p.shape[1]-1]) 
        
    return n_p[type_idx,:]


# Levy flight generator using Mantegna's algorithm
def levy_flight(dim, beta=1.5):
    # Mantegna's algorithm
    from math import gamma, sin, pi
    sigma_u = (gamma(1 + beta) * sin(pi * beta / 2) /
               (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma_u, size=dim)
    v = np.random.normal(0, 1, size=dim)
    step = u / (np.abs(v) ** (1.0 / beta))
    return step


# Epsilon-greedy + Levy-enhanced average student update
def update_average_student_epsilon(population, dim, type_idx, min_values, max_values, eps=0.1, levy_beta=1.5, A=1.0, Pthreshold=0.6, target_function = target_function):
    n_p = np.copy(population)

    for i in type_idx:
        for j in range(0, dim):
            if random.random() < eps:
                # exploration: with Levy flight sometimes, otherwise random spread
                Plevy = random.random()
                if Plevy >= Pthreshold:
                    step = levy_flight(1, levy_beta)[0]
                    n_p[i, j] = np.clip(n_p[i, j] + A * step, min_values[0,j], max_values[0,j])
                else:
                    r1 = random.random()
                    n_p[i, j] = np.clip(np.min(population[:, j]) + r1*(np.max(population[:, j]) - np.min(population[:, j])), min_values[0,j], max_values[0,j])
            else:
                # exploitation: move towards mean of group (existing behavior)
                r1 = random.random()
                n_p[i, j] = np.clip(population[i, j] + r1*(np.mean(population[type_idx, j]) - population[i, j]), min_values[0,j], max_values[0,j])

        n_p[i,-1] = target_function(n_p[0,0:n_p.shape[1]-1])

    return n_p[type_idx,:]


# Epsilon-greedy + Levy-enhanced random student update
def update_random_student_epsilon(population, dim, type_idx, min_values, max_values, eps=0.1, levy_beta=1.5, A=1.0, Pthreshold=0.6, target_function = target_function):
    n_p = np.copy(population)

    for i in type_idx:
        for j in range(0, dim):
            if random.random() < eps:
                Plevy = random.random()
                if Plevy >= Pthreshold:
                    step = levy_flight(1, levy_beta)[0]
                    n_p[i, j] = np.clip(n_p[i, j] + A * step, min_values[0,j], max_values[0,j])
                else:
                    r1 = random.random()
                    n_p[i, j] = np.clip(np.min(population[:, j]) + r1*(np.max(population[:, j]) - np.min(population[:, j])), min_values[0,j], max_values[0,j])
            else:
                r1 = random.random()
                n_p[i, j] = np.clip(np.min(population[:, j]) + r1*(np.max(population[:, j]) - np.min(population[:, j])), min_values[0,j], max_values[0,j])

        n_p[i,-1] = target_function(n_p[0,0:n_p.shape[1]-1])

    return n_p[type_idx,:]


# Function: Segments
def segments(population):
    
    n          = len(population) - 1
    x          = []
    a, b, c, d = 0, 0, 0, 0
    
    for i in range(0, n):
        a = i
        for j in range(i + 1, n):
            b = j
            for k in range(j + 1, n):
                c = k
                for L in range(k + 1, n + (i > 0)):
                    d = L
                    x.append((a, b, c, d))
    
    #print()
    #print(x)
    #print()
    
    return x

# Function: Classify Class
def classify_class(population, x):
    
    p_lst       = list(range(0, len(population)))
    random.shuffle(p_lst)
    y           = random.sample(x, 1)[0]
    
    i, j, k, L  = y
    
    ga          = p_lst[ :i]
    gb          = p_lst[i:j]
    av          = p_lst[j:L]
    rd          = p_lst[L: ]
    
    return ga, gb, av, rd


#//////////////////////////////////////////////------------------------------

def binary_conversion(X, thres, N, dim):
    
    Xbin = np.zeros([N, dim], dtype='int')
    
    for i in range(N):
        for d in range(dim):
            
            #print()
            #print("i = ", i ,"d = ", d , "X[i,d] =" ,X[i,d])
            #print()
            
            if X[i,d] > thres:
                Xbin[i,d] = 1
            else:
                Xbin[i,d] = 0
    
    return Xbin


def boundary(x, lb, ub):
    
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    
    return x
    
#------------------------------------------------------------------------------

def jfs(xtrain, ytrain, opts):
    
    # Parameters
    ub    = 0
    lb    = 1
    thres = 0.5
    
    #w     = 0.9    # inertia weight
    #c1    = 2      # acceleration factor
    #c2    = 2      # acceleration factor
    
    #Population Size
    N        = opts['N']
    
    #Iteration/Generation
    max_iter = opts['T']
    
    #Minimum Values
    if 'mn' in opts:
        mn    = opts['mn']
        
    #Max Values
    if 'mx' in opts:
        mx   = opts['mx']
        
    min_values=mn;
    max_values=mx;
        
    #if 'c2' in opts:
       # c2   = opts['c2'] 
    
    
    # Dimension
    dim = np.size(xtrain, 1)
    #print("Dim= ",dim)
    
    ub = 5 * np.ones([1, dim], dtype='float')
    max_values=ub
    
    lb = (-5) * np.ones([1, dim], dtype='float')
    min_values=lb
        
    # Initialize position & velocity
    #X             = init_position(lb, ub, N, dim)
    # X is the Population
    
    target_function = easom
    X             = initial_population(N,dim,mn, mx, target_function)
    
    population=X
    #print(X)
    
    #V, Vmax, Vmin = init_velocity(lb, ub, N, dim) 
    
    best_student  = np.copy(X[X[:,-1].argsort()][0,:])
    x             = segments(X)
    
    
    # Pre
    fit   = np.zeros([N, 1], dtype='float')
    Xgb   = np.zeros([1, dim], dtype='float')
    fitG  = float('inf')
    
    #Xpb   = np.zeros([N, dim], dtype='float')
    #fitP  = float('inf') * np.ones([N, 1], dtype='float')
    
    curve = np.zeros([1, max_iter], dtype='float') 
    t     = 0
    
    while t < max_iter:
        
        # Binary conversion
        Xbin = binary_conversion(X, thres, N, dim)
        
        # Fitness
        for i in range(N):
            
            #print("i =", i, fit[i,0])
            #print()
            
            fit[i,0] = Fun(xtrain, ytrain, Xbin[i,:], opts)
            
            if fit[i,0] < fitG:
                
                Xgb[0,:]  = X[i,0:41]
                fitG      = fit[i,0]
        
        # Store result
        curve[0,t] = fitG.copy()
        print("Iteration:", t + 1)
        print("Best (SPB0):", curve[0,t])
        t += 1
       
        
        ga, gb, av, rd = classify_class(population, x)
        
        
        best_student   = update_best_student(population, dim, best_student, min_values, max_values, target_function)
        population_1   = update_good_student_a(population, dim, best_student, ga, min_values, max_values, target_function)
        population_2   = update_good_student_b(population, dim, best_student, gb, min_values, max_values, target_function)
        # epsilon-greedy and Levy flight parameters (defaults when not provided in opts)
        eps = opts.get('eps', 0.1)
        levy_beta = opts.get('levy_beta', 1.5)
        levy_A = opts.get('levy_A', 1.0)
        Pthreshold = opts.get('Pthreshold', 0.6)

        population_3   = update_average_student_epsilon(population, dim, av, min_values, max_values, eps, levy_beta, levy_A, Pthreshold, target_function)
        population_4   = update_random_student_epsilon(population, dim, rd, min_values, max_values, eps, levy_beta, levy_A, Pthreshold, target_function)
        
        population     = np.vstack([population, population_1, population_2, population_3, population_4])
        population     = population[population[:, -1].argsort()]
        
        
        population     = population[:N,:]
        value          = np.copy(population[0,:])
        
        if (best_student[-1] > value[-1]):
            best_student = np.copy(value)
            
        for a in range (N):
            for b in range (dim):
                population[a,b] = boundary(population[a,b], lb[0,b], ub[0,b])
                
            
        X=population
        
        
    #Xgb=population     
    # Best feature subset
    #print()
    #print("Xgb= ",Xgb)
    #print()
    
    Gbin       = binary_conversion(Xgb, thres, 1, dim) 
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    # Create dictionary
    spbo_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
    
    return spbo_data    
    
    

def easom(variables_values):
    
    o = 0.0
    # o=sum([1:variable].*(x.^4));
    
    for i in range(len(variables_values)):
        o += ((i+1)*np.power(variables_values[i], 2))
    
    #print()
    #print("Easom = ", o)
    #print()
    
    return o




