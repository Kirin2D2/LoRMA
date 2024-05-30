### low rank approx of form M ~= M^ = AB

# Given A, B, the low-rank matrices, calculate and return M_approx (M hat above)
#
# parameters
# A:        m x k 2-D numpy array
# B:        k x n 2-D numpy array
#
# output
# M_approx: m x n 2-D numpy array

def lorma(A, B):
    M_approx = A @ B
    return M_approx
