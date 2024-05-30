# Given M, O, params, run gradient descent to compute the low-dimensional
# approximation A, B and losses, the list of approximation errors

import LoRMA_MSE
import LoRMA_init
import LoRMA_gradient

# parameters
# M:        m x n 2-D numpy array
#           containing the observed entries (with arbitrary values in the unobserved entries)
#
# O:        m x n 2-D numpy array
#           containing 0 if the entry is unobserved and 1 if the entry is observed in M.
#           O tells you which entries in M were observed (corresponds to \Omega in the equations above)
#
# params:   tuple of 3 parameters (k, num_epochs, etas)
#.          k is the rank of your LORMA model
#           num_epochs is the number of epochs to run gradient descent
#           etas is a list of floats, with the learning rate for each epoch
#           len(etas) = num_epochs
#
# output
# A:        m x k 2-D numpy array
# B:        k x n 2-D numpy array
# losses:   list of approximation errors evaluated at every 10 epochs
def lorma_learn(M, O, params):
    k, num_epochs, etas = params
    m, n = M.shape
    A, B = lorma_init(m, n, k)
    losses = []
    for e in range(num_epochs):
        dA, dB = lorma_grad(M, O, A, B)
        A = A - etas[e] * dA
        B = B - etas[e] * dB
        if e % 10 == 0:
          losses.append(loss(M, lorma(A, B), O))
          print(losses[-1].round(4))
    return A, B, losses

m, n, k = 100, 40, 5
rand_seed = 10
np.random.seed(rand_seed)

def check_lorma_learn():
    from numpy.random import binomial, randn, uniform
    mockA, mockB = uniform(1, 2, (m, k)), uniform(-2, -1, (k, n))
    M = mockA @ mockB + 0.01 * np.random.randn(m, n)
    O = binomial(1, 0.5, size=M.shape)
    num_epochs = 100
    etas = 2.0 * np.ones(num_epochs)
    params = k, num_epochs, etas
    A, B, losses = lorma_learn(M, O, params)
    plt.plot(losses, '-o')
    return M, A, B

M, A, B = check_lorma_learn()
