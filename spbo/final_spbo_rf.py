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

            # Sample uniformly inside the actual search bounds
             X[i,j] = random.uniform(min_values[0,j], max_values[0,j])

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

        n_p[i,-1] = target_function(n_p[i,0:n_p.shape[1]-1])

    return n_p[type_idx,:]


# Function: Update Good Student Type B
def update_good_student_b(population, dim, best_student, type_idx, min_values, max_values, target_function = target_function):
    
    n_p = np.copy(population)
    
    for i in type_idx:
        
        for j in range(0, dim):
            
            r1        = random.random()
            r2        = random.random()
            n_p[i, j] = np.clip(population[i, j] + r1*(best_student[j] - population[i, j]) + (r2*(population[i, j] - population[type_idx, j].mean())), min_values[0,j], max_values[0,j])

        n_p[i,-1] = target_function(n_p[i,0:n_p.shape[1]-1])
        
    return n_p[type_idx,:]


# Function: Update Average Student 
def update_average_student(population, dim, type_idx, min_values, max_values, target_function = target_function):
    
    n_p = np.copy(population)
    
    for i in type_idx:
        
        for j in range(0, dim):
            
            r1        = random.random()
            n_p[i, j] = np.clip(population[i, j] + r1*(np.mean(population[type_idx, j]) - population[i, j]), min_values[0,j], max_values[0,j])

        n_p[i,-1] = target_function(n_p[i,0:n_p.shape[1]-1])
        
    return n_p[type_idx,:]


# Function: Update Random Student 
def update_random_student(population, dim, type_idx, min_values, max_values, target_function = target_function):
    
    n_p = np.copy(population)
    
    for i in type_idx:
        
        for j in range(0, dim):
            
            r1        = random.random()
            n_p[i, j] = np.clip(np.min(population[:, j]) + r1*(np.max(population[:, j]) - np.min(population[:, j])), min_values[0,j], max_values[0,j])

        n_p[i,-1] = target_function(n_p[i,0:n_p.shape[1]-1])
        
    return n_p[type_idx,:]


# Function: Segments
def segments(population):

    # This used to materialise every ordered 4-tuple of cut points, which is
    # O(N^4): at N=80 that is ~1.5M tuples (hundreds of MB) built once per run
    # just so classify_class could pick ONE of them at random. Drawing 4 sorted
    # distinct cut points with random.sample is uniform over the same set, so
    # we only need to carry the number of students.
    return len(population)

# Function: Classify Class
def classify_class(population, n_students):

    if n_students < 4:
        raise ValueError('SPBO needs a population of at least 4 students, got %d' % n_students)

    p_lst       = list(range(0, len(population)))
    random.shuffle(p_lst)

    # Note: as in the reference SPBO implementation, the third cut point k is
    # drawn but not used - the average group spans j:L.
    i, j, k, L  = sorted(random.sample(range(0, n_students), 4))

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


#------------------------------------------------------------------------------

def jfs(xtrain, ytrain, opts):
    
    # Parameters
    thres = 0.5

    # Reproducibility: the search itself was previously unseeded.
    if 'seed' in opts:
        random.seed(opts['seed'])
        np.random.seed(opts['seed'])

    #w     = 0.9    # inertia weight
    #c1    = 2      # acceleration factor
    #c2    = 2      # acceleration factor
    
    #Population Size
    N        = opts['N']
    
    #Iteration/Generation
    max_iter = opts['T']
    
    # Dimension
    dim = np.size(xtrain, 1)
    #print("Dim= ",dim)

    # Search space matched to the binarisation threshold. A wider range
    # (the old [-5, 5]) saturates: once a coordinate drifts far from thres
    # the clip-based updates can no longer flip that feature bit.
    # 'mn'/'mx' may be given as a scalar or as a per-dimension array.
    min_values = np.broadcast_to(np.asarray(opts.get('mn', 0.0), dtype='float'),
                                 (1, dim)).astype('float').copy()
    max_values = np.broadcast_to(np.asarray(opts.get('mx', 1.0), dtype='float'),
                                 (1, dim)).astype('float').copy()
    lb         = min_values
    ub         = max_values

    # Initialize position & velocity
    #X             = init_position(lb, ub, N, dim)
    # X is the Population
    
    # The objective the search actually optimises: the wrapper fitness
    # (RF error rate + selected-feature penalty) of the binarised position.
    def target_function(variables_values):
        v    = np.asarray(variables_values, dtype='float').reshape(1, dim)
        xbin = binary_conversion(v, thres, 1, dim)[0, :]
        return Fun(xtrain, ytrain, xbin, opts)

    X             = initial_population(N, dim, min_values, max_values, target_function)
    
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

    # Fitness is already computed by target_function whenever the population is
    # built/updated, so reuse it instead of re-fitting the RF.
    def track_best(pop, Xgb, fitG):
        for i in range(N):

            fit[i,0] = pop[i,-1]

            if fit[i,0] < fitG:

                Xgb[0,:]  = pop[i,0:dim]
                fitG      = fit[i,0]

        return fitG

    fitG = track_best(X, Xgb, fitG)

    while t < max_iter:

        ga, gb, av, rd = classify_class(population, x)
        
        
        best_student   = update_best_student(population, dim, best_student, min_values, max_values, target_function)
        population_1   = update_good_student_a(population, dim, best_student, ga, min_values, max_values, target_function)
        population_2   = update_good_student_b(population, dim, best_student, gb, min_values, max_values, target_function)
        population_3   = update_average_student(population, dim, av, min_values, max_values, target_function)
        population_4   = update_random_student(population, dim, rd, min_values, max_values, target_function)
        
        population     = np.vstack([population, population_1, population_2, population_3, population_4])
        population     = population[population[:, -1].argsort()]
        
        
        population     = population[:N,:]
        value          = np.copy(population[0,:])
        
        if (best_student[-1] > value[-1]):
            best_student = np.copy(value)

        X = population

        # Track the generation just produced, so the final one is not lost.
        fitG       = track_best(X, Xgb, fitG)
        curve[0,t] = fitG.copy()
        print("Iteration:", t + 1)
        print("Best (SPB0):", curve[0,t])
        t += 1


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
