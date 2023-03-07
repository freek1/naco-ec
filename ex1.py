import numpy as np

f1 = lambda x: np.abs(x)
f2 = lambda x: x**2
f3 = lambda x: 2*x**2
f4 = lambda x: x**2 + 20

def prop_select(x, f, a):
    '''
    Fitness proportional selection algorithm
    Input:
        x, int: individual to be evaluated
        f, function: fitness function
        a, list: list of all possible values for x
    Ouput:
        ans, double: probability of selecting individual x using fitness-proportional selection
    '''
    ans = f(x)/np.sum(f(a))
    return ans

# List of all possible individuals
a = np.arange(10)
# Individuals to be evaluated
xs = [2, 3, 4]
for i, func in enumerate([f1, f2, f3, f4]):
    for x in xs:
        print(f'function {i+1} with x = {x}')
        print(prop_select(x, func, a))
