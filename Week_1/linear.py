import numpy as np
import copy,time
import math
import matplotlib.pyplot as plt


# Load the data from the CSV file
data = np.genfromtxt('data1.csv', delimiter=',', dtype=str, skip_header=1)

mask_non_empty_rows = ~np.any(data == '', axis=1)
data_non_empty = data[mask_non_empty_rows]

# Extract the fourth column into a separate variable y, excluding empty rows
y = data_non_empty[:, 3].astype(float)

# Remove the first column
data_without_first_and_fourth_col = np.delete(data_non_empty, [0, 3], axis=1)

# Convert cleaned data to float

for row in data_without_first_and_fourth_col:
    if row[1].lower() == 'developed':
        row[1] = 1
    elif row[1].lower() == 'developing':
        row[1] = 0

# Print the modified data
# print(data_without_first_and_fourth_col)
# print(y)
data_cleaned = data_without_first_and_fourth_col.astype(float)
x_train = data_cleaned
y_train = y

def compute_cost(X, y, w, b): 
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b           #(n,)(n,) = scalar (see np.dot)
        cost = cost + (f_wb_i - y[i])**2       #scalar
    cost = cost / (2 * m)                      #scalar    
    return cost

#print(x_train.shape)
#print(y_train.shape)

num_features = x_train.shape[1]

# Initialize weights (w) and bias (b)
w_init = np.random.randn(num_features) * 0.01  # small random values
b_init = 0.0  # initializing bias to zero

cost = compute_cost(x_train,y_train,w_init,b_init)
# print(cost)

def zscore_normalize(x_train):
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_norm = (x_train - mean) / std
    return x_norm, mean, std

def zscore_normalize_y(y, mean_y, std_y):
    y_normalized = (y - mean_y) / std_y
    return y_normalized

mean_y = np.mean(y)
std_y = np.std(y)

x_norm, mean, std = zscore_normalize(x_train)
y_norm = zscore_normalize_y(y_train,mean_y,std_y)

def compute_gradient(X, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw

tmp_dj_db, tmp_dj_dw = compute_gradient(x_norm, y_norm, w_init, b_init)
#print(tmp_dj_db)
#print(tmp_dj_dw)


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None
        

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None

        #if i==0:
            #print(dj_dw)
            #print(dj_db)
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
            print(b)
        
    return w, b, J_history #return final w,b and J history for graphing

#print(x_norm)
#print(y_norm)
initial_w = np.random.randn(num_features)
initial_b = 1.
iterations = 1200
alpha = 0.05
#print(initial_w)
w_final, b_final, J_hist = gradient_descent(x_norm, y_norm, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)
#print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = x_norm.shape
y_pred_norm = np.dot(x_norm, w_final) + b_final
y_pred = y_pred_norm * std_y + mean_y
#for i in range(m):
    #print(f"prediction: {y_pred[i]}, target value: {y_train[i]}")
    
train_rmse = np.sqrt(np.mean((y_train - y_pred)**2))
print(f'Training RMSE: {train_rmse}')