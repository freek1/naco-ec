import numpy as np
import matplotlib.pyplot as plt
import os.path
from os import path
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import time

# Load file-tsp.txt
with open('data/file-tsp.txt', 'r') as f:
    lines = f.readlines()

lines = np.array(list(map(lambda x: x.strip(), lines)))
data = np.zeros([len(lines), 2])
for i, x in enumerate(lines):
    data[i, :] = x.split()

# data: [x_i, y_i] coordinates for each city i = 1...50

def fitness(data, c):
    ''' Returns fitness score of tour cs.
        Tour cs (list) contains c1, c2 , ..., cn: a list of indices of cities
        Fitness = 1/total distance travelled
    '''
    distance = 0
    for i in range(len(c)-1):
        # print(i)
        # print(euclidean(data[int(c[i])], data[int(c[i+1])]))
        distance += euclidean(data[int(c[i])], data[int(c[i+1])])
    return 1/distance

runs = 10

gens = 1500 # Generations
mu = 0.01 # Mutation probability
mu_cross = 0.1 # Crossover probability
pool = 500 # Size of pool
fit_scores_plot = np.zeros((gens))
cs = np.zeros((pool, len(data)))
prev_best = 0
fit_tot = np.zeros((runs, gens))
best_route = []

def crossover(p1, p2):
    '''Applies crossover permutation
    Input:
        p1: parent candidate list (length len(data))
        p2: parent candidate list (length len(data))
    Output:
        c1: children, mutated candidate lists (length len(data))
        c2: children, mutated candidate lists (length len(data))
    '''
    c1 = np.ones_like(p1)*-1
    c2 = np.ones_like(p2)*-1

    # Choose two cut points
    cut1 = 2 #np.random.randint(1, len(p1)-1)
    cut2 = 4 #np.random.randint(cut1, len(p1))

    # Keep middle piece
    c1[cut1:cut2] = p1[cut1:cut2]
    c2[cut1:cut2] = p2[cut1:cut2]

    # Take complement in other parent
    o1 = []
    o2 = []

    for i in range(len(p1)):
        if p1[i] not in c2[cut1:cut2]:
            o2.append(p1[i])
        if p2[i] not in c1[cut1:cut2]:
            o1.append(p2[i])
    
    # Fill gaps in order (starting after 2nd cut)
    for i in range(cut2, cut1+len(c1)):
        c1[i%len(c1)] = o1.pop(0)
    for i in range(cut2, cut1+len(c1)):
        c2[i%len(c1)] = o2.pop(0)

    return c1, c2

plt.figure()
for r in range(runs):
    # --- Init random candidates
    for p in range(pool):
        cs[p] = np.random.permutation(len(data)) # Random candidates

    # --- Computing fitness
    fit_scores = np.zeros((pool))
    for i, c in enumerate(cs):
        fit_scores[i] = fitness(data, c)

    for gen in tqdm(range(gens), desc=f'Run {r+1}'):
                
        # --- Selection
        prop_sel = np.zeros((len(cs)))
        for i in range(len(cs)):
            prop_sel[i] = fit_scores[i]/np.sum(fit_scores)

        parents = np.random.choice(len(prop_sel), size=2, p=prop_sel)
        while parents[0] == parents[1]:
            parents = np.random.choice(len(prop_sel), size=2, p=prop_sel)

        # --- Recombine and mutate
        # Crossover (mu_cross?????)
        c1, c2 = crossover(cs[parents[0]], cs[parents[1]])

        # Random mutation with mu
        if np.random.rand(1) < mu:
            switch = np.random.randint(0, len(c1), 2)
            while(switch[0] == switch[1]):
                switch = np.random.randint(0, len(c1), 2)
            c1[[switch[0], switch[1]]] = c1[[switch[1], switch[0]]]
        
        if np.random.rand(1) < mu:
            switch = np.random.randint(0, len(c1), 2)
            while(switch[0] == switch[1]):
                switch = np.random.randint(0, len(c1), 2)
            c2[[switch[0], switch[1]]] = c2[[switch[1], switch[0]]]
        

        # Compute fitness scores of new candidates
        fit_p1 = fitness(data, cs[parents[0]])
        fit_p2 = fitness(data, cs[parents[1]])
        fit_c1 = fitness(data, c1)
        fit_c2 = fitness(data, c2)
        # Select (not) best parent
        ps = [fit_p1, fit_p2]
        notbest_p_idx = np.argmin(ps)
        # Randomly switch one of the children with the worst parent
        if np.random.rand() > 0.5:
            # Switch candidate
            cs[parents[notbest_p_idx]] = c1
            # Replace fitness score of switched candidate
            fit_scores[parents[notbest_p_idx]] = fit_c1
        else:
            # Switch candidate
            cs[parents[notbest_p_idx]] = c2
            # Replace fitness score of switched candidate
            fit_scores[parents[notbest_p_idx]] = fit_c2

        # Store best fitness score 
        fit_scores_plot[gen] = max(fit_scores)
        
        
        # Store best route
        # best_idx = np.argmax(fit_scores)
        # print(cs[best_idx])
        # if max(fit_scores) < prev_best:
        #     print(f'max(fit_scores): {max(fit_scores)}')
        #     print(f'notbest_p_idx: {notbest_p_idx}')
        #     print(f'parents idx: {parents}')
        #     print(fit_p1, fit_p2)
        #     print(fit_c1, fit_c2)
        # prev_best = max(fit_scores)
    
    plt.plot(np.arange(gens), fit_scores_plot, '--', label=f'run {r+1}')
    fit_tot[r, :] = fit_scores_plot

plt.plot(np.arange(gens), np.mean(np.array(fit_tot), axis=0), 'k', label='Avg')
plt.title(f'Simple EA on TSP problem for {runs} runs')
plt.ylabel('Fitness score')
plt.xlabel('Generations')
plt.legend()
plt.savefig('imgs/ex3_EA.png')
plt.show()
