import random
import numpy as np
import matplotlib.pyplot as plt
from os import path
from tqdm import tqdm
from scipy.stats import norm

''' Assignment 4.4: Exploitation vs exploration '''

goal_string = 'group two cool'
K = 2 # Tournament parameter
mu = 0.01 # Mutation rate

l = len(goal_string) # Length of strings
alphabet = 'abcdefghijklmnopqrstuvwxyz '
N = 1000 # Population size
gens = 1500 # Generations
p_cross = 1 # Crossover prob

prev_fit = 0
fittest = []

# Defining fitness score (closeness to goal sequence)
def fitness(s, goal_string):
    '''
    Fraction of correct characters
    '''
    bools = [[*s][i] == [*goal_string][i] for i in range(len(goal_string))]
    return np.sum(bools*1)

# apply 1 cut crossover between two strings
def crossover(p1, p2):
    '''Applies crossover permutation
    Input:
        p1: parent
        p2: parent
    Output:
        c1: children
        c2: children
    '''
    p1 = [*p1]
    p2 = [*p2]
    
    c1 = np.empty_like(p1)
    c2 = np.empty_like(p2)
    # Choose two cut points
    cut = np.random.randint(1, len(p1)-1)
    # Keep middle piece
    c1 = p2[0:cut]+p1[cut:len(p1)]
    c2 = p1[0:cut]+p2[cut:len(p1)]

    return c1, c2

# tournement selection over the population
def tournament_selection(fit_scores, K):
    tour_population_idx = random.sample(range(len(fit_scores)), K)
    best_participant = -1
    best_fitness = -1
    for participant_idx in tour_population_idx:
        fitness_participant = fit_scores[participant_idx]
        if best_fitness < fitness_participant:
            best_fitness = fitness_participant
            best_participant = participant_idx
    parent_idx = best_participant
    return parent_idx

# position-wise random character with rate mu
def mutation(canidate, mu, alphabet):
    if np.random.rand(1) < mu: # if mutation happens
        switch = np.random.randint(0, len(canidate))
        mutated_char = random.choice(alphabet)
        canidate[switch] = mutated_char
        return ''.join(canidate)
    else:
        return ''.join(canidate)
    

def GA(alphabet, K, mu, goal_string, gens, n_runs, N, l, p_coss):
    avg_runs_duration = []
    for run in tqdm(range(n_runs)):
        prev_fit = 0
        fittest = []
        fit_scores = np.zeros((N))
        population = [''.join(random.choices(alphabet, k=l)) for s in range(N)]
        for gen in range(gens):
            # fitness calculation
            fit_scores = np.zeros((N))
            for i, s in enumerate(population):
                fit_scores[i] = fitness(s, goal_string)
            
            fittest.append(np.max(fit_scores))

            if np.max(fit_scores) == l:
                avg_runs_duration.append(gen)
                break

            new_population = []
            for n in range(0, N, 2):
                # tournament selection
                parent1_idx = tournament_selection(fit_scores, K)
                parent2_idx = tournament_selection(fit_scores, K)

                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]

                # Crossover
                if np.random.rand() < p_cross:
                    c1, c2 = crossover(parent1, parent2)

                # Mutation
                c1 = mutation(c1, mu, alphabet)
                c2 = mutation(c2, mu, alphabet)
                new_population.append(c1)
                new_population.append(c2)
            
            population = new_population

        plt.plot(np.arange(gen+1), fittest, '--')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('String search GA')
    return np.mean(avg_runs_duration), np.std(avg_runs_duration)

n_runs = 20
mu=0.1
gens=200
avg, std = GA(alphabet, K, mu, goal_string, gens, n_runs, N, l, p_cross)

plt.axvline(x=avg, c='k', label='Average $t_{finish}$')
plt.legend()
plt.savefig('imgs/ex4-3.png')
plt.show()

# x_axis = np.arange(avg-3*std, avg+3*std, 0.001)
# plt.figure()
# plt.plot(x_axis, norm.pdf(x_axis, avg, std))
# plt.title('Distribution of $t_{finish}$ after 20 runs')
# plt.xlabel('$t_{finish}$')
# plt.ylabel('pdf')
# plt.savefig('imgs/ex4-2-dist.png')
# plt.show()
