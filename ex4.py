import random
import numpy as np
import matplotlib.pyplot as plt
from os import path
from tqdm import tqdm

''' Assignment 4.4: Exploitation vs exploration '''

goal_string = 'group two cool'
K = 2 # Tournament parameter
mu = 0.01 # Mutation rate

l = 14 # Length of strings
alphabet = 'abcdefghijklmnopqrstuvwxyz '
N = 1000 # Population size
gens = 1500 # Generations
p_cross = 1 # Crossover prob

prev_fit = 0
fittest = np.zeros(gens)

# Defining fitness score (closeness to goal sequence)
def fitness(s, goal_string):
    '''
    Fraction of correct characters
    '''
    bools = [[*s][i] == [*goal_string][i] for i in range(len(goal_string))]
    return np.sum(bools*1)

def crossover(p1, p2):
    '''Applies crossover permutation
    Input:
        p1: parent candidate list (length len(string))
        p2: parent candidate list (length len(string))
    Output:
        c1: children, mutated candidate lists (length len(string))
        c2: children, mutated candidate lists (length len(string))
    '''
    p1 = [*p1]
    p2 = [*p2]
    
    c1 = np.empty_like(p1)
    c2 = np.empty_like(p2)
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
    if len(o1) == (cut1+len(c1)-cut2) and len(o2) > (cut1+len(c1)-cut2):
        for i in range(cut2, cut1+len(c1)):
            c1[i%len(c1)] = o1.pop(0)
        for i in range(cut2, cut1+len(c1)):
            c2[i%len(c1)] = o2.pop(0)
    else:
        for i in range(cut2, cut1+len(c1)):
            c1[i%len(c1)] = ' '
        for i in range(cut2, cut1+len(c1)):
            c2[i%len(c1)] = ' '
    return c1, c2

def select_tournament(group):
    ''' Tournament selection from tournament 'group' '''
    fs = 0
    parent = -1
    for p in group:
        if fit_scores[p] > fs:
            parent = p
    return parent

# Create N random strings
strings = [''.join(random.choices(alphabet, k=l)) for s in range(N)]

for gen in tqdm(range(gens)):
    # Compute fitness
    fit_scores = np.zeros((N))
    for i, s in enumerate(strings):
        fit_scores[i] = fitness(s, goal_string)

    # Select K indiv.
    parents = np.random.choice(len(fit_scores), size=K)
    # Select fittest 2
    p1 = select_tournament(parents)

    # Select K indiv.
    parents = np.random.choice(len(fit_scores), size=K)
    # Select fittest 2
    p2 = select_tournament(parents)

    # Ensure parents are different
    while (p1 == p2):  
        parents = np.random.choice(len(fit_scores), size=K)
        p2 = select_tournament(parents)
        
    # Crossover
    if np.random.rand() < p_cross:
        c1, c2 = crossover(strings[parents[0]], strings[parents[1]])

    # Mutation
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

    # Replace parents with better children (if better)
    fitc1 = fitness(c1, goal_string)
    fitc2 = fitness(c2, goal_string)
    if fitc1 > fit_scores[parents[0]]:
        strings[parents[0]] = c1
        fit_scores[parents[0]] = fitc1
    if fitc2 > fit_scores[parents[1]]:
        strings[parents[1]] = c2
        fit_scores[parents[1]] = fitc2

    # Store better max
    if max(fit_scores) > prev_fit:
        prev_fit = max(fit_scores)
        
    fittest[gen] = prev_fit

plt.plot(np.arange(gens), fittest)
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.title('String search GA')
plt.show()
