import numpy as np

num_iterations= int(input('Enter number of iterations:'))
adj_matrix = np.array([[0, 1, 1, 0],  
                       [0, 0, 1, 1], 
                       [1, 0, 0, 0], 
                       [1, 0, 1, 0]]) 

def hits(adj_matrix, num_iterations):
    n = adj_matrix.shape[0]  
    
    hubs = np.ones(n)
    authorities = np.ones(n)
    
    for iteration in range(num_iterations):
        authorities = adj_matrix.T @ hubs
        hubs = adj_matrix @ authorities
        
        authorities = authorities / np.linalg.norm(authorities)
        hubs = hubs / np.linalg.norm(hubs)
        
        print(f"Iteration {iteration + 1}:")
        for i in range(n):
            print(f"  Page {i + 1} - Hub Score: {hubs[i]:.4f}, Authority Score: {authorities[i]:.4f}")
        print("-" * 40)

hits(adj_matrix, num_iterations)
