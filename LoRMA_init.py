### adopt random initialzation of the entries in  A  and  B 
###followed by a normalization process to make  M^  
### satisfy the following properties:
### 1. Zero mean
### 2. <= unit variance (i.e., (1/mn)\sum_{i,j}(M^)_{i,j}^2\leq 1)
### note: guarantee this by having each |M^_{i,j}|\leq 1 by normalizing
### each a_i and b_j so they have unit norm. Since inner product of all
### a_i \cdot b_j is in [-1,1] and by Cauchy inequality, |a\cdot b|\leq
### ||a||||b|| --> |M^_{i,j}|\leq 1.

# Given m, n, k, initialize and normalize A, B as per the guideline above
#
# parameters
# m, n, k:  shapes for A, B
#
# output
# A:        m x k 2-D numpy array
# B:        k x n 2-D numpy array

def lorma_init(m, n, k):
    # intialize A, B using a zero-mean unit-variance Gaussian (ie a normal distribution with mean 0 and variance 1) per entry
    A = np.random.normal(0, 1, (m, k))
    B = np.random.normal(0, 1, (k, n))
    # normalize the rows of A and columns of B
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=0, keepdims=True)
    return A, B

# test correctness of init
def check_lorma_init():
    A, B = lorma_init(10, 7, 3)
    assert(np.linalg.norm(np.diag(A @ A.T) - np.ones(10), 1) < 1e-6)
    assert(np.linalg.norm(np.diag(B.T @ B) - np.ones(7), 1) < 1e-6)
    print("Function {} is working fine!".format('lorma_init()'))
    return

check_lorma_init()
