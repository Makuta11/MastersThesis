#%%
from joblib import Parallel, delayed
import math
import time

def sqrt_func(i):
    time.sleep(1)
    return {i: i**2} 

r = Parallel(n_jobs=5, verbose=1)(delayed(sqrt_func)(i) for i in range(5))

# %%
print(type(r))
print(r)
# %%
dictionary = dict()
for d in r:
    dictionary.update(d)
dictionary
# %%
