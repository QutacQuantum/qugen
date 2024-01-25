import numpy as np
import matplotlib.pyplot as plt

# Define parameters
num_samples = 100000
dimension = 4

'''
mu = 1  # Mean of the Gaussian distribution
sigma = 1e-3  # Variance of the Gaussian distribution
center = np.zeros((num_samples, dimension))  # Assuming "center" is a 4-dimensional point at the origin
'''
def makeO(mu, sigma, center):
    # Generate samples
    np.random.seed(42)  # Setting seed for reproducibility
    samples = np.random.normal(mu, sigma, (num_samples, 1))  # Radial distance from origin
    directions = np.random.randn(num_samples, dimension)  # Random direction vectors
    norms = np.linalg.norm(directions, axis=1, keepdims=True)  # Normalize direction vectors

    # Generate final samples
    random_samples = center + samples * (directions / norms)

    # Print or use random_samples as needed
    np.save("O_4D.npy", random_samples)


def makeX():
    # Generate samples
    np.random.seed(42)  # Setting seed for reproducibility
    samples = np.random.uniform(0,2, size=num_samples) # point on the line (diagonal of the one cube in 4D has length 2)
    samples = np.vstack([samples, samples, samples, samples])
    for i in range(samples.shape[1] // 8):
        # A[:, 3i] remains the same

        # A[:, 3i+1] gets the first sign switched
        samples[0, 8 * i + 1] *= -1
        samples[:, 8 * i + 1] += np.array([0,1,1,1])

        # A[:, 3i+2] gets the first two signs switched
        samples[1, 8 * i + 2] *= -1
        samples[:, 8 * i + 1] += np.array([1,0,1,1])
        
        # A[:, 3i+2] gets the first two signs switched
        samples[2, 8 * i + 3] *= -1
        samples[:, 8 * i + 1] += np.array([1,1,0,1])
        
        # A[:, 3i+2] gets the first two signs switched
        samples[3, 8 * i + 4] *= -1
        samples[:, 8 * i + 1] += np.array([1,1,1,0])
        
        # A[:, 3i+2] gets the first two signs switched
        samples[0, 8 * i + 5] *= -1
        samples[1, 8 * i + 5] *= -1
        samples[:, 8 * i + 1] += np.array([0,0,1,1])
        
        # A[:, 3i+2] gets the first two signs switched
        samples[0, 8 * i + 6] *= -1
        samples[2, 8 * i + 6] *= -1
        samples[:, 8 * i + 1] += np.array([0,1,0,1])
        
        # A[:, 3i+2] gets the first two signs switched
        samples[0, 8 * i + 6] *= -1
        samples[3, 8 * i + 6] *= -1
        samples[:, 8 * i + 1] += np.array([0,1,1,0])
    np.save("X_4D.npy", samples.T)


makeX()
