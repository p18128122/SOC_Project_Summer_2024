import numpy as np
import copy, math

# Load the data from the CSV file
data = np.genfromtxt('data2.csv', delimiter=',', dtype=str, skip_header=1)
mask_non_empty_rows = ~np.any(data == '', axis=1)
data_non_empty = data[mask_non_empty_rows]
mask = np.any(data == 'NA', axis=1)
data_cleaned = data[~mask]

y = data_cleaned[:, 15]
data_without_first_and_fourth_col = np.delete(data_cleaned, [15], axis=1)
x_train = data_without_first_and_fourth_col.astype(float)
y_train = y.astype(float)

def zscore_normalize(x_train):
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_norm = (x_train - mean) / std
    return x_norm, mean, std

def zscore_normalize_y(y, mean_y, std_y):
    y_normalized = (y - mean_y) / std_y
    return y_normalized

mean_y = np.mean(y_train)
std_y = np.std(y_train)

x_norm, mean, std = zscore_normalize(x_train)
y_norm = zscore_normalize_y(y_train, mean_y, std_y)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost_logistic(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i] * np.log(f_wb_i + epsilon) - (1 - y[i]) * np.log(1 - f_wb_i + epsilon)
    cost = cost / m
    return cost

def compute_gradient_logistic(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.0

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        err_i = f_wb_i - y[i]
        dj_dw += err_i * X[i]
        dj_db += err_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, alpha, num_iters, min_cost):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    
    for i in range(num_iters):
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        cost = compute_cost_logistic(X, y, w, b)
        
        if abs(cost)<0.01 :
            break
        if i < 100000:
            J_history.append(cost)

        if i % (num_iters // 10) == 0:
            print(f"Iteration {i:4d}: Cost {cost}, dj_dw: {dj_dw}, dj_db: {dj_db}")
        
    return w, b, J_history

num_features = x_train.shape[1]
epsilon = 1e-15

initial_w = np.random.randn(num_features)
initial_b = 0.1
iterations = 2500
alpha = 0.0025  # Lower the learning rate
min_cost = 1.0
w_final, b_final, _= gradient_descent(x_norm, y_norm, initial_w, initial_b, alpha, iterations, min_cost)

#cost = compute_cost_logistic(x_norm, y_norm, w_min, b_min)
#print(cost)

y_pred_norm = sigmoid(np.dot(x_norm, w_final) + b_final)
y_pred = y_pred_norm * std_y +mean_y

m,_ = x_norm.shape
#for i in range(m):
    #print(f"prediction: {y_pred[i]}, target value: {y_train[i]}")

def predict_binary(y_pred, threshold = 0.3):
    return y_pred >= threshold

def count_false_positives(y_train, y_pred):
    false_positives = np.sum((y_pred == 1) & (y_train == 0))
    return false_positives

def count_false_negatives(y_train, y_pred):
    false_negatives = np.sum((y_pred == 0) & (y_train == 1))
    return false_negatives

def calculate_accuracy(y_true, y_pred, threshold=0.3):
    correct_predictions = sum([1 if pred == true else 0 for pred, true in zip(y_pred, y_true)])
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy * 100

y_pred_binary = predict_binary(y_pred)
false_positives = count_false_positives(y_train, y_pred_binary)
false_negatives = count_false_negatives(y_train, y_pred_binary)
accuracy = calculate_accuracy(y_train, y_pred_binary)
print("Number of false positives:", false_positives)
print("Number of false negatives:", false_negatives)
print(accuracy)
