### Gradients calculated as follows:
### apply chain rule for matrix differentiation to get
### dL/dA = dL/dM^ * dM^/dA
### where dL/dM^ is matrix in R^{mxn}
### and dM^/dA = B^T

# Given M, O, A, B return dL/dA, dL/dB.
#
# parameters
# M:        m x n 2-D numpy array
#           containing the observed entries (with arbitrary values in the unobserved entries)
#
# O:        m x n 2-D numpy array
#           containing 0 if the entry is unobserved and 1 if the entry is observed in M.
#           O tells you which entries in M were observed (corresponds to \Omega in the equations above)
#
# A:        m x k 2-D numpy array
# B:        k x n 2-D numpy array
#
# output
# dL/dA:    m x k 2-D numpy array
# dL/dB:    k x n 2-D numpy array

def lorma_grad(M, O, A, B):
    dL = 2 * O * (lorma(A, B) - M) / np.sum(O)

    dA = dL @ np.transpose(B)
    dB = np.transpose(A) @ dL

    return dA, dB
