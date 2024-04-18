import numpy as np

# Define the matrices
S_2_2 = np.array([[2, 0], [0, 2]])
S_0_5_0_5 = np.array([[0.5, 0], [0, 0.5]])
S_1_minus_1 = np.array([[1, 0], [0, -1]])
S_minus_1_1 = np.array([[-1, 0], [0, 1]])

# Define a vector v
v = np.array([1, 1])

# Perform the geometric actions
result_a = np.dot(S_2_2, v)
result_b = np.dot(S_0_5_0_5, v)
result_c = np.dot(S_1_minus_1, v)
result_d = np.dot(S_minus_1_1, v)

# Print the results
print("Result (a) S2,2 with λ = 2 and µ = 2:", result_a)
print("Result (b) S0.5,0.5 with λ = 0.5 and µ = 0.5:", result_b)
print("Result (c) S1,−1 with λ = 1 and µ = −1:", result_c)
print("Result (d) S−1,1 with λ = −1 and µ = 1:", result_d)
