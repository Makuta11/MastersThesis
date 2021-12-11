"""
    Testing Kristoffers Kernel Indexing Method

This scripts generates a random data matrix and investigates the indexing technique proposed by Kristoffer 
Haugaard for the use of precomputed kernels.
"""

import numpy as np

# Create data - (10,5) matrix with random ints
X = np.random.randint(0,10,(10,5))

# Create precompute linear kernel
precomputed_kernel = np.dot(X,X.T)

# Create train and test idx
train_idx = np.arange(0,7)
test_idx = np.arange(7,10)

# Create train and test kernel 
train_kernel = np.dot(X[train_idx],X[train_idx].T)
test_kernel = np.dot(X[test_idx],X[train_idx].T)

# Assert that test_kernel is the same as slicing technique given by Kristoffer
print(f'Computed test kernel: \n{test_kernel} n\Precomputed kernel indexed: \n{precomputed_kernel[test_idx][:,test_idx]}')
print(f'These matrices are the same: {np.all(test_kernel == precomputed_kernel[test_idx][:,test_idx])}')