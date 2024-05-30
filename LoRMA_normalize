### preprocessing: normalize matrix M since M^ entries likely to follow normal distribution
## normalize M such that sum over (i,j) in omega of M[i,j] = 0 and 1/|omega| * sum over (i,j) in omega of M[i,j]^2 = 1
## mean over observed entries is 0 and std dev is 1

# Given M, O, return the normalized version of M. Does not mutate parameters.
#
# parameters
# M:        m x n 2-D numpy array
#           containing the observed entries (with arbitrary values in the unobserved entries)
#
# O:        m x n 2-D numpy array
#           containing 0 if the entry is unobserved and 1 if the entry is observed in M.
#           O tells you which entries in M were observed (corresponds to \Omega in the equations above)
#
# output
# M:        m x n 2-D numpy array
#           normalized copy of the input matrix M
#
# NOTE: Vectorized implementation optimized for python without loops. 
def get_normalized_matrix(M, O):
    num_observed_entries = np.sum(O)
    a = np.sum(O * M) / num_observed_entries
    s = (np.sum((O * (M - a)) ** 2) / num_observed_entries) ** (1/2)

    M_normalized = (M - a) / s
    return M_normalized

# test for correctness of normalization
def check_normalization():
    M = np.random.rand(10, 3)
    O = (np.random.rand(10, 3) > 0.5) + 0
    Mn = get_normalized_matrix(M, O)
    assert(abs(np.sum(M * O)) > 1e-6)
    assert(abs(np.sum(Mn * O)) < 1e-6)
    assert(abs(np.sum(Mn**2 * O) / np.sum(O) - 1) < 1e-6)
    print("Function {} is working fine!".format('get_normalized_matrix()'))

check_normalization()
