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


# Load bav28.txt
with open('data/bav28.txt', 'r') as f:
    lines = f.readlines()

lines = np.array(list(map(lambda x: x.strip(), lines)))
data_bav = np.zeros([len(lines), 2])
for i, x in enumerate(lines):
    data_bav[i, :] = x.split()


def fitness(data, c):
    ''' Returns fitness score of tour cs.
        Tour cs (list) contains c1, c2 , ..., cn: a list of indices of cities
        Fitness = 1/total distance travelled
    '''
    distance = 0
    for i in range(len(c)-1):
        distance += euclidean(data[int(c[i])], data[int(c[i+1])])
    return 1/distance

runs = 10

gens = 1500 # Generations
mu = 0.01 # Mutation probability
mu_cross = 0.1 # Crossover probability
pool_size = 100 # Size of pool_size
fit_scores_plot = np.zeros((gens)) # For plotting runs
cs = np.zeros((pool_size, len(data))) # Canidates
prev_best = 0 
fit_tot = np.zeros((runs, gens)) # For comptuting mean of fitness over runs
best_route = [] # Best route per generation
best_run_nr = 0 # Keeping track which run has highest fitness
best_route_run = np.zeros((runs, len(data))) # Storing the best route of each run
prev_fittest = 0 # Keeping track which run has highest fitness

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
    n_cities = len(data)
    best_local_neighbour = c
    for i in range(n_cities-2):
        for j in range(i+2, n_cities):
            if i==j:
                continue
            c_new = c.copy()
            begin = c_new[0:i]
            middle = list(reversed(c_new[i:j+1]))
            end = c_new[j+1:n_cities]
            c_new = np.concatenate((begin, middle, end))
            # delta_fit = fitness(data, c_new) - fitness(data, best_local_neighbour)
            delta_fit = -
            if delta_fit>0:
                best_local_neighbour = c_new
    return best_local_neighbour

def GA(gens, mu, mu_cross, pool_size, cs, run_number, prev_fittest):

    # --- Init random candidates
    for p in range(pool_size):
        cs[p] = np.random.permutation(len(data)) # Random candidates
    
    tot_fit = 0
    for c in cs:
        tot_fit+=fitness(data, c)
    print(tot_fit)

    # --- Local search
    for i in range(len(cs)):
        cs[i] = run2Opt(data, c)

    tot_fit = 0
    for c in cs:
        tot_fit+=fitness(data, c)
    print(tot_fit)


    # --- Computing fitness
    fit_scores = np.zeros((pool_size))
    for i, c in enumerate(cs):
        fit_scores[i] = fitness(data, c)

    for gen in tqdm(range(gens), desc=f'Run {run_number+1}'):
        # Storing
        best_route_gen = []

        # --- Selection
        prop_sel = np.zeros((len(cs)))
        for i in range(len(cs)):
            prop_sel[i] = fit_scores[i]/np.sum(fit_scores)

        # ###
        # Choosing parents from proportional selection:
        # ###
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

        # best fitness score
        max_fit = max(fit_scores)
        
        # best route
        best_route = cs[np.argmax(fit_scores)]
    
    return max_fit, best_route


GA(gens, mu, mu_cross, pool_size, cs, 1, prev_fittest)


# plt.figure(figsize=(8, 6))

# for run_number in range(runs):
#     best_route = []
#     fit_tot, best_route_gen = GA(gens, mu, mu_cross, pool_size, cs, run_number, prev_fittest)

#     plt.plot(np.arange(gens), fit_scores_plot, '--', label=f'run {run_number+1}')
#     fit_tot[run_number, :] = fit_scores_plot

#     # Best route of run
#     best_route_run[r] = (best_route_gen[-1])
#     if fit_scores_plot[-1] > prev_fittest:
#         prev_fittest = fit_scores_plot[-1]
#         best_run_nr = r
    
#     plt.plot(np.arange(gens), fit_scores_plot, '--', label=f'run {r+1}')
#     fit_tot[r, :] = fit_scores_plot

# print(f'Best route (run {best_run_nr+1}): \n{best_route_run[best_run_nr]}')

# plt.plot(np.arange(gens), np.mean(np.array(fit_tot), axis=0), 'k', label='Avg')
# plt.title(f'Simple EA on TSP problem for {runs} runs')
# plt.ylabel('Fitness score')
# plt.xlabel('Generations')
# plt.legend()
# # if not path.exists('imgs/ex3_ea.png'):
# #     plt.savefig('imgs/ex3_ea.png')
# plt.show()

# plt.figure()
# plt.plot(data[:, 0], data[:, 1], 'ro')
# for i in range(len(best_route_run[best_run_nr])-1):
#     point1 = best_route_run[best_run_nr][i]
#     point2 = best_route_run[best_run_nr][i+1]
#     xp1 = data[int(point1), 0]
#     yp1 = data[int(point1), 1]
#     xp2 = data[int(point2), 0]
#     yp2 = data[int(point2), 1]
#     plt.plot([xp1, xp2], [yp1, yp2], 'k-')
# plt.title(f'TSP graph (run {best_run_nr+1})')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(['Cities', 'Candidate solution'])
# # if not path.exists('imgs/simple-ea-solution.png'):
# #     plt.savefig('imgs/simple-ea-solution.png')
# plt.show()



# for r in range(runs):
#     # --- Init random candidates
#     for p in range(pool_size):
#         cs[p] = np.random.permutation(len(data)) # Random candidates

#     # --- Computing fitness
#     fit_scores = np.zeros((pool_size))
#     for i, c in enumerate(cs):
#         fit_scores[i] = fitness(data, c)

#     # Storing best route
#     best_route = []
#     for gen in tqdm(range(gens), desc=f'Run {r+1}'):
#         # Storing
#         best_route_gen = []

#         # --- Selection
#         prop_sel = np.zeros((len(cs)))
#         for i in range(len(cs)):
#             prop_sel[i] = fit_scores[i]/np.sum(fit_scores)

#         # ###
#         # Choosing parents from proportional selection:
#         # ###
#         parents = np.random.choice(len(prop_sel), size=2, p=prop_sel)
#         while parents[0] == parents[1]:
#             parents = np.random.choice(len(prop_sel), size=2, p=prop_sel)

#         # ###
#         # Best parents choice: (not sure if it works)
#         # ###
#         # parents = [np.argsort(prop_sel)[-1], np.argsort(prop_sel, axis=0)[-2]]

#         # --- Recombine and mutate
#         # Crossover (mu_cross?????)
#         c1, c2 = crossover(cs[parents[0]], cs[parents[1]])

#         # Random mutation with mu
#         if np.random.rand(1) < mu:
#             switch = np.random.randint(0, len(c1), 2)
#             while(switch[0] == switch[1]):
#                 switch = np.random.randint(0, len(c1), 2)
#             c1[[switch[0], switch[1]]] = c1[[switch[1], switch[0]]]
        
#         if np.random.rand(1) < mu:
#             switch = np.random.randint(0, len(c1), 2)
#             while(switch[0] == switch[1]):
#                 switch = np.random.randint(0, len(c1), 2)
#             c2[[switch[0], switch[1]]] = c2[[switch[1], switch[0]]]
        

#         # Compute fitness scores of new candidates
#         fit_p1 = fitness(data, cs[parents[0]])
#         fit_p2 = fitness(data, cs[parents[1]])
#         fit_c1 = fitness(data, c1)
#         fit_c2 = fitness(data, c2)
#         # Select (not) best parent
#         ps = [fit_p1, fit_p2]
#         notbest_p_idx = np.argmin(ps)
#         # Randomly switch one of the children with the worst parent
#         if np.random.rand() > 0.5:
#             # Switch candidate
#             cs[parents[notbest_p_idx]] = c1
#             # Replace fitness score of switched candidate
#             fit_scores[parents[notbest_p_idx]] = fit_c1
#         else:
#             # Switch candidate
#             cs[parents[notbest_p_idx]] = c2
#             # Replace fitness score of switched candidate
#             fit_scores[parents[notbest_p_idx]] = fit_c2

#         # Store best fitness score 
#         fit_scores_plot[gen] = max(fit_scores)
        
#         # Store best route of gen
#         best_route_gen.append(cs[np.argmax(fit_scores)])

#     # Best route of run
#     best_route_run[r] = (best_route_gen[-1])
#     if fit_scores_plot[-1] > prev_fittest:
#         prev_fittest = fit_scores_plot[-1]
#         best_run_nr = r
    
#     plt.plot(np.arange(gens), fit_scores_plot, '--', label=f'run {r+1}')
#     fit_tot[r, :] = fit_scores_plot

