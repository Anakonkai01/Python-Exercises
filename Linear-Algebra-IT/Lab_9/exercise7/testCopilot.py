import numpy as np
import matplotlib.pyplot as plt

def plot_transformation(matrix, title):
    # Create a grid of points
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    X, Y = np.meshgrid(x, y)
    Z = np.array([X.flatten(), Y.flatten()])

    # Apply the transformation
    transformed_Z = np.dot(matrix, Z)

    # Plot the original points
    plt.scatter(Z[0], Z[1], color='blue', label='Original points')
    # Plot the transformed points
    plt.scatter(transformed_Z[0], transformed_Z[1], color='red', label='Transformed points')
    plt.title(title)
    plt.legend()
    plt.axis('equal')
    plt.show()

# Define the matrices
S2_2 = np.array([[2, 0], [0, 2]])
S0_5_0_5 = np.array([[0.5, 0], [0, 0.5]])
S1_m1 = np.array([[1, 0], [0, -1]])
Sm1_1 = np.array([[-1, 0], [0, 1]])

# Plot the transformations
plot_transformation(S2_2, 'S2,2 with λ = 2 and µ = 2')
plot_transformation(S0_5_0_5, 'S0.5,0.5 with λ = 0.5 and µ = 0.5')
plot_transformation(S1_m1, 'S1,−1 with λ = 1 and µ = −1')
plot_transformation(Sm1_1, 'S−1,1 with λ = −1 and µ = 1')
