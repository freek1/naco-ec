import numpy as np
import matplotlib.pyplot as plt
import os.path
from os import path

''' Assignment 4.2: Counting Ones problem '''

l = 100 # Length of bit strings
mu = 1./l # Mutation rate
gens = 1500 # Generations

x_goal = np.random.randint(2, size=(l,)) # Generate random goal sequence

best_fitness = np.zeros((gens+1,)) # For saving the best fitness for each gen

# Probability function
def decision(mu, l):
    '''Returns true with probability mu for l indices'''
    return np.random.random(l) < mu

# Defining fitness score (closeness to goal sequence)
def fitness(x, x_goal):
    '''
    Computes closeness to goal sequence (i.e. fitness score).
    Each bit is compared to the goal state, and if they are correct, a 1 is assigned. 
    The resulting list is summed, which is used as a fitness measure. So a perfect match would have 
    the maximum fitness level of l (length of x_goal). The most wrong bit string would have a fitness
    score of 0.
    '''
    bools = x == x_goal
    return np.sum(bools*1)

def GA(l, mu, gens):
    '''
    The (1+1)GA for bit strings.
    Returns: best_fitness scores for each generation.
    '''
    gen = 0 # Generations counter

    # Step 1
    x = np.random.randint(2, size=(l,))

    while gen <= gens:

        # Step 2
        x_copy = x.copy()
        to_flip = decision(mu, l)
        # Flip if to_flip is True
        for i in range(len(x_copy)):
            if to_flip[i] == True:
                x_copy[i] = int(not x_copy[i])
        xm = x_copy

        # Step 3
        if fitness(xm, x_goal) > fitness(x, x_goal):
            x = xm

        # Save best fitness
        best_fitness[gen] = fitness(x, x_goal)

        # Step 4: repeat steps 2 and 3 until goal is reached.
        if fitness(x, x_goal) == l:
            best_fitness[gen:gens+1] = fitness(x, x_goal) # And fill the remaining best_fitness with the current score.
            gen = gens+1 # Make gen larger than gens (while condition no longer holds).

        gen += 1
    
    return best_fitness

'''1. '''
best_fitness = GA(l, mu, gens)
plt.figure()
plt.plot(np.arange(gens+1), best_fitness)
plt.title('Evolution of fitness over generations')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.savefig('imgs/ex2-1.png')
plt.show()

'''2. '''
runs = 10
plt.figure()
for i in range(runs):
    best_fitness = GA(l, mu, gens)
    plt.plot(np.arange(gens+1), best_fitness, label=f'run {i+1}')
plt.title(f'Evolution of fitness over generations for {runs} runs')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend(loc=4) # bottom right
if not path.exists('imgs/ex2-2.png'):
    plt.savefig('imgs/ex2-2.png')
plt.show()

# Question:
#   How many times does the algorithm find the optimum?
# Ans:
#   In our runs, every run found the goal bit string, so 10 times.

'''3. We first create a new GA where x is always replaced by the new xm '''
def GA_always_replace(l, mu, gens):
    '''
    The (1+1)GA for bit strings.
    Altered from GA() such that x is always replaced with xm.
    Returns: best_fitness scores for each generation.
    '''
    gen = 0 # Generations counter

    # Step 1
    x = np.random.randint(2, size=(l,))

    while gen <= gens:

        # Step 2
        x_copy = x.copy()
        to_flip = decision(mu, l)
        # Flip if to_flip is True
        for i in range(len(x_copy)):
            if to_flip[i] == True:
                x_copy[i] = int(not x_copy[i])
        xm = x_copy

        # Step 3: Now always replacing the new xm
        x = xm

        # Save best fitness
        best_fitness[gen] = fitness(x, x_goal)

        # Step 4: repeat steps 2 and 3 until goal is reached.
        if fitness(x, x_goal) == l:
            best_fitness[gen:gens+1] = fitness(x, x_goal) # And fill the remaining best_fitness with the current score.
            gen = gens+1 # Make gen larger than gens (while condition no longer holds).

        gen += 1
    
    return best_fitness

runs = 10
plt.figure()
for i in range(runs):
    best_fitness = GA_always_replace(l, mu, gens)
    plt.plot(np.arange(gens+1), best_fitness, label=f'run {i+1}')
plt.title(f'Evolution of fitness over generations for {runs} runs')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend(loc=4) # bottom right
if not path.exists('imgs/ex2-3.png'):
    plt.savefig('imgs/ex2-3.png')
plt.show()

# Question:
#   Is there a difference in performance when using this modification? Justify your answer.
# Ans:
#   The difference in performance when using this modification is that now, every iteration, the random changes
#   (dependent on mu) are set as the string for the next generation. This means that progress (fitness) is not taken
#   into account. This results in the performance hovering around chance level: 50 out of 100 bits correct (due to there
#   being 2 options for each bit: 0 or 1).
