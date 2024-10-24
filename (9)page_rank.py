import numpy as np

def pagerank(M, num_iterations=100, d=0.85):

    N = M.shape[1]  # Number of pages
    ranks = np.ones(N) / N  # Initialize rank of all pages to be equal
    print(f"Initial ranks: {ranks}\n")
    
    for i in range(num_iterations):
        new_ranks = (1 - d) / N + d * M.dot(ranks)
        print(f"Iteration {i+1}: {new_ranks}\n")
        ranks = new_ranks
    return ranks

M = np.array([[0, 0, 1, 0.5],
              [0.33, 0, 0, 0],
              [0.33, 0.5, 0, 0.5],
              [0.33, 0.5, 0, 0]])

final_ranks = pagerank(M, num_iterations=10)
print(f"Final ranks after all iterations: {final_ranks}")
