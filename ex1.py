import numpy as np
import matplotlib.pyplot as plt

f1 = lambda x: np.abs(x)
f2 = lambda x: x**2
f3 = lambda x: 2*x**2
f4 = lambda x: x**2 + 20

def prop_select(x, f, xs):
    '''
    Fitness proportional selection algorithm
    Input:
        x, int: individual to be evaluated
        f, function: fitness function
        a, list: list of all possible values for x
    Ouput:
        ans, double: probability of selecting individual x using fitness-proportional selection
    '''
    ans = f(x)/np.sum(f(xs))
    return ans

# Individuals to be evaluated
xs = np.array([2, 3, 4])
# Functions
funcs = [f1, f2, f3, f4]
# Storing results
prop_list = np.zeros((len(funcs), len(xs)))

# Computing fitness-proportional selection
for i, func in enumerate(funcs):
    for j, x in enumerate(xs):
        prop_list[i, j] = prop_select(x, func, xs)

print('x-axis: the candidate xs, y-axis: the functions f1 -> f4')
print(prop_list)

# Making pie-charts
for i, func in enumerate(funcs):
    plt.subplot(2,2,i+1)
    plt.title(f'Function $f_{i+1}$')
    plt.pie(prop_list[i], labels=[f'x={x}' for x in xs])
plt.suptitle('Pie charts with selection probability using \n fitness-proportional selection')
plt.tight_layout()
plt.savefig('imgs/ex1.png')
plt.show()

# Question:
#   What can you conclude about the effects of fitness scaling on selection pressure?
# Ans:
#   Different fitness functions can alter the candidate selection.