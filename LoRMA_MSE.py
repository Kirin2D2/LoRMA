### Approximate a partially observed large matrix using a product of low rank matrices
### Measure loss as mean squared error over observed entries 

import matplotlib.pyplot as plt
import numpy as np

# Given M, M_approx, O, return the average squared loss over the observed entries.
#
# parameters
# M:        m x n 2-D numpy array
#           containing the observed entries (with arbitrary values in the unobserved entries)
#
# M_approx: m x n 2-D numpy array
#           representing the low-dimensional approximation
#
# O:        m x n 2-D numpy array
#           containing 0 if the entry is unobserved and 1 if the entry is observed in M.
#           O tells you which entries in M were observed (corresponds to \Omega in the equations above)
#
# output
# loss:     average of squared loss over observed entries

def loss(M, M_approx, O):
    #define a matrix to hold square losses for each observed position
    square_loss_matrix = O * (M - M_approx) ** 2
    loss = np.sum(square_loss_matrix) / np.sum(O)

    return loss



# test correctness of loss implementation
def check_loss():
    M = np.array([[2, 4, 1], [1, -3, 1]])
    M_approx = np.zeros((2, 3))
    O = np.array([[1, 0, 0], [0, 1, 1]])
    answer = 14 / 3
    output = loss(M, M_approx, O)
    assert(np.isclose(answer, output))
    print("Function {} is working fine!".format('loss()'))

check_loss()
