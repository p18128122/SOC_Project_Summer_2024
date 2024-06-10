import numpy as np
import matplotlib.pyplot as plt

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

# Fit polynomial regression model
degree = 6  # Degree of the polynomial

# Create the polynomial features matrix
x_poly_train = np.hstack([x_norm**d for d in range(1, degree + 1)])

# Add a bias term to the features matrix
x_poly_train_with_bias = np.c_[np.ones((x_poly_train.shape[0], 1)), x_poly_train]

# Calculate the weights using the normal equation
weights = np.linalg.pinv(x_poly_train_with_bias.T.dot(x_poly_train_with_bias)).dot(x_poly_train_with_bias.T).dot(y_norm)

# Make predictions
y_train_pred_norm = x_poly_train_with_bias.dot(weights)

y_train_pred = y_train_pred_norm * std_y + mean_y

# Evaluate the model
train_rmse = np.sqrt(np.mean((y_train - y_train_pred)**2))
print(f'Training RMSE: {train_rmse}')

m,_ = x_norm.shape
#for i in range(m) :
    #print(f"Prediction:{y_train_pred[i]}    Actual:{y_train[i]}")


# Plot the results
x_axis = x_train[:, 0]
plt.scatter(x_axis, y_train, color='blue', label='Actual data')
plt.scatter(x_axis, y_train_pred, color='red', label='Predicted train data')
plt.legend()
plt.show()
