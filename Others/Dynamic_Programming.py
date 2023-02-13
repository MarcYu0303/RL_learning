# Dynamic programming example
# yuran 2023/2/13
# optimal control of drone with given conditions


import numpy as np


if __name__ == '__main__':
    h_init, v_init = 0, 0  # initial condition of speed and velocity
    h_final, v_final = 10, 0  # final condtion of speed and velocity
    v_max, v_min = 3, 0  # condition of speed
    a_max, a_min = 2, -3  # condition of acceleration
    h_n, v_n = 4, 4  # discretization of velocity and speed
    v_vector = np.array([i*(v_max+v_min)/3 for i in range(v_n)])
    h_diff = (h_final - h_init) / (h_n + 1)
    
    cost_matrix = np.zeros((h_n + 2, v_n))  # cost matrix, in this example, cost == time
    input_matrix = np.zeros((h_n + 2, v_n))  # in this example, input == acceleration
    # ---
    
    for i in range(1, h_n + 2):  # plus 2 here is adding final and initial discretization levels
        for j in range(v_n):
            if i == 1:
                v_avg = (v_final + v_vector[j]) / 2
                cost = h_diff / v_avg
                cost_matrix[i, j] = cost
                input_matrix[i, j] = (v_final - v_vector[j]) / cost
            else:
                v_avg_vector = (v_vector + v_vector[j] * np.ones(v_n)) / 2
                cost_vector = h_diff / v_avg_vector
                a_vector = (v_vector - v_vector[j] * np.ones(v_n)) / cost_vector
                cost_vector = cost_vector + cost_matrix[i-1]
                cost_vector[a_vector > a_max] = 'inf'
                cost_vector[a_vector < a_min] = 'inf'
                
                cost_matrix[i, j] = np.min(cost_vector)
                input_matrix[i, j] = a_vector[np.argmin(cost_vector)]
    input_matrix[input_matrix.shape[0]-1] *= np.eye(1, v_n, k=v_init).squeeze(0)  # by using the given initial condition, set other value to be zero 
    cost_matrix[cost_matrix.shape[0]-1] *= np.eye(1, v_n, k=v_init).squeeze(0)
            
    print(cost_matrix)
    print(input_matrix)
            
                
                