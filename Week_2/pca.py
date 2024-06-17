import numpy as np
import matplotlib.pyplot as plt

def data1():
    x = [np.random.rand() for i in range(1000)]
    y = [x[i] + 0.05*np.random.rand() for i in range(1000)]
    return [x, y]

def data2():
    x = [np.random.rand() for i in range(1000)]
    y = [(x[i])**2 + 0.05*np.random.rand() for i in range(1000)]
    return [x, y]

# Standardisation of Data
def std_data(nparray):
    mean = np.mean(nparray)
    std_dev = np.std(nparray)
    standardized_array = (nparray - mean) / std_dev
    return standardized_array

def DimReduction(arr):

    data_set = np.array(arr)
    std_data_set = std_data(data_set)
    
    # Compute the covariance matrix
    cov_matrix = np.cov(std_data_set)
    
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Get the principal component (eigenvector with the largest eigenvalue)
    principal_component = eigenvectors[:, np.argmax(eigenvalues)]
    
    # Slope (m) and intercept (c) of the best fit line
    m = principal_component[1] / principal_component[0]
    mean_x = np.mean(data_set[0])
    mean_y = np.mean(data_set[1])
    c = mean_y - m * mean_x
    
    # Displaying the result using matplotlib
    plt.scatter(data_set[0], data_set[1], color="red", s=1)
    plt.plot(data_set[0], m * np.array(data_set[0]) + c, color="blue")
    print("Slope =", m, "Intercept =", c)
    plt.title("Best Fit Line")
    plt.show()
    
DimReduction(data1())
DimReduction(data2())