# visualize a random matrix from the same distribution we used for initialization

import LoRMA_init

def show_mat(X, str, ind):
    plt.subplot(1,3,ind)
    plt.imshow(X, cmap='hot')
    plt.axis('off')
    plt.title(str)

Ar, Br = lorma_init(m, n, k)
fig = plt.figure(figsize=(10,60))
show_mat(M, 'Original Matrix', 1)
show_mat(lorma(Ar, Br), 'Initial Approximation', 2)
show_mat(lorma(A, B), 'Low Rank Approximation', 3)
