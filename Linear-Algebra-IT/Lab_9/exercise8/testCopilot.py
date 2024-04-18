import numpy as np
import matplotlib.pyplot as plt

def plot_rotation(theta, title):
    # Create a grid of points
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    X, Y = np.meshgrid(x, y)
    Z = np.array([X.flatten(), Y.flatten()])

    # Define the rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # Apply the transformation
    transformed_Z = np.dot(R, Z)

    # Plot the original points
    plt.scatter(Z[0], Z[1], color='blue', label='Original points')
    # Plot the transformed points
    plt.scatter(transformed_Z[0], transformed_Z[1], color='red', label='Transformed points')
    plt.title(title)
    plt.legend()
    plt.axis('equal')
    plt.show()

# Plot the transformations
plot_rotation(np.pi, 'Rθ with θ = π')
plot_rotation(np.pi/3, 'Rθ with θ = π/3')
