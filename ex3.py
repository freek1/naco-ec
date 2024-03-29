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
given = np.zeros([len(lines), 2])
for i, x in enumerate(lines):
    given[i, :] = x.split()

# Load bav28.txt
with open('data/bav28.txt', 'r') as f:
    lines = f.readlines()

lines = np.array(list(map(lambda x: x.strip(), lines)))
Bays29 = np.zeros([len(lines), 2])
for i, x in enumerate(lines):
    Bays29[i, :] = x.split()

# data: [x_i, y_i] coordinates for each city i = 1...50


########
# Params
########
runs = 10

gens = 100 # Generations
mu = 0.01 # Mutation probability
mu_cross = 0.1 # Crossover probability
pool = 25 # Size of pool
fit_scores_plot = np.zeros((gens)) # For plotting runs

prev_best = 0 
fit_tot = np.zeros((runs, gens)) # For comptuting mean of fitness over runs
best_route = [] # Best route per generation
best_run_nr = 0 # Keeping track which run has highest fitness
best_route_run = np.zeros((runs, len(given))) # Storing the best route of each run
prev_fittest = 0 # Keeping track which run has highest fitness

local_search = True


###############
# Main function
###############
def GA(local_search, prev_fittest, data, name):
    plt.figure(figsize=(8, 6))

    for r in range(runs):
        # Canidates
        cs = np.zeros((pool, len(data)))

        # --- Init random candidates
        for p in range(pool):
            cs[p] = np.random.permutation(len(data)) # Random candidates

        
        if local_search:
            print('Performing local search...')
            # --- Local search
            for i in range(len(cs)):
                cs[i] = run2Opt(data, cs[i])
            print('Done.')

        # --- Computing fitness
        fit_scores = np.zeros((pool))
        for i, c in enumerate(cs):
            fit_scores[i] = fitness(data, c)

        # Storing best route
        best_route = []
        for gen in tqdm(range(gens), desc=f'Run {r+1}'):
            # Storing
            best_route_gen = []
            # --- Selection
            prop_sel = np.zeros((len(cs)))
            for i in range(len(cs)):
                prop_sel[i] = fit_scores[i]/np.sum(fit_scores)

            # Next gen:
            cs_new = np.ones_like(cs)
            for N in range(len(cs)-1):
                # ###
                # Choosing parents from proportional selection:
                # ###
                
                parents = np.random.choice(len(prop_sel), size=2, p=prop_sel)
                while parents[0] == parents[1]:
                    parents = np.random.choice(len(prop_sel), size=2, p=prop_sel)

                # --- Recombine and mutate
                # Crossover
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
                

                if local_search:
                    # --- Local search
                    c1new = run2Opt(data, c1)
                    c2new = run2Opt(data, c2)
            
                    # Elitist approach
                    fitc1n = fitness(data, c1new)
                    fitc2n = fitness(data, c2new)
                    if fitc1n > fit_scores[N]:
                        # Adding the new children
                        cs_new[N] = c1new
                        fit_scores[N] = fitc1n
                    else: 
                        cs_new[N] = cs[N]
                    if fitc2n > fit_scores[N+1]:
                        # Adding the new children
                        cs_new[N+1] = c2new
                        fit_scores[N+1] = fitc2n
                    else: 
                        cs_new[N+1] = cs[N+1]
                else:
                    fitc1 = fitness(data, c1)
                    fitc2 = fitness(data, c2)
                    # Elitist approach
                    if fitc1 > fit_scores[N]:
                        # Adding the new children
                        cs_new[N] = c1
                        fit_scores[N] = fitc1
                    else: 
                        cs_new[N] = cs[N]
                    if fitc2 > fit_scores[N+1]:
                        # Adding the new children
                        cs_new[N+1] = c2
                        fit_scores[N+1] = fitc2
                    else: 
                        cs_new[N+1] = cs[N+1]

            cs = cs_new.copy()

            # Store best fitness score 
            fit_scores_plot[gen] = max(fit_scores)
            
            # Store best route of gen
            best_route_gen.append(cs[np.argmax(fit_scores)])

        # Best route of run
        best_route_run[r] = (best_route_gen[-1])
        if fit_scores_plot[-1] > prev_fittest:
            prev_fittest = fit_scores_plot[-1]
            best_run_nr = r

        plt.plot(np.arange(gens), fit_scores_plot, '--', label=f'run {r+1}')
        fit_tot[r, :] = fit_scores_plot

    print(f'Best route (run {best_run_nr+1}): \n{best_route_run[best_run_nr]}')


    plt.plot(np.arange(gens), np.mean(np.array(fit_tot), axis=0), 'k', label='Avg')
    plt.title(f'Simple MA on TSP problem for {runs} runs')
    plt.ylabel('Fitness score')
    plt.xlabel('Generations')
    plt.ylim(0.0005, 0.0089)
    plt.legend()
    plt.savefig(f'imgs/ex3_ma-{name}.png')
    plt.show()

    plt.figure()
    plt.plot(data[:, 0], data[:, 1], 'ro')
    for i in range(len(best_route_run[best_run_nr])-1):
        point1 = best_route_run[best_run_nr][i]
        point2 = best_route_run[best_run_nr][i+1]
        xp1 = data[int(point1), 0]
        yp1 = data[int(point1), 1]
        xp2 = data[int(point2), 0]
        yp2 = data[int(point2), 1]
        plt.plot([xp1, xp2], [yp1, yp2], 'k-')
    plt.title(f'TSP graph MA on {name} dataset (run {best_run_nr+1})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['Cities', 'Candidate solution'])  
    #if not path.exists('imgs/simple-ea-solution-bav.png'):
    plt.savefig(f'imgs/simple-ma-solution-{name}.png')
    plt.show()


###############
# Helper functions
###############
def fitness(data, c):
    ''' Returns fitness score of tour cs.
        Tour cs (list) contains c1, c2 , ..., cn: a list of indices of cities
        Fitness = 1/total distance travelled
    '''
    distance = 0
    for i in range(len(c)-1):
        distance += euclidean(data[int(c[i])], data[int(c[i+1])])
    return 1/distance

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
    cut1 = np.random.randint(1, len(p1)-1)
    cut2 = np.random.randint(cut1, len(p1))
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

def run2Opt(data, c):
    n_cities = len(c)
    best_neighbour = c
    foundImprovement = True
    while foundImprovement:
        foundImprovement = False
        for i in range(n_cities-1):
            for j in range(i+1, n_cities-1):
                old_distance = euclidean(data[int(best_neighbour[i-1])], data[int(best_neighbour[i])]) + euclidean(data[int(best_neighbour[j])], data[int(best_neighbour[j+1])])
                new_distance = euclidean(data[int(best_neighbour[i-1])], data[int(best_neighbour[j])]) + euclidean(data[int(best_neighbour[i])], data[int(best_neighbour[j+1])])

                if new_distance<old_distance:
                    best_neighbour = swap2Opt(best_neighbour, i, j, n_cities)
                    foundImprovement = True       
    return best_neighbour


def swap2Opt(c, i, j, n_cities):
    # begin = c[0:i]
    # middle = list(reversed(c[i:j+1]))
    # end = c[j+1:n_cities]
    c_new = np.concatenate((c[0:i], list(reversed(c[i:j+1])), c[j+1:n_cities]))
    return c_new

GA(local_search, prev_fittest, given, "given")
#GA(local_search, prev_fittest, given)
