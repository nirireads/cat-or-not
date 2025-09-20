from copy import deepcopy
import numpy as np

#------------------------------------------
# Helper functions
#------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b

#------------------------------------------
# Forward and Backward propagation
#------------------------------------------
def propagate(w, b, X, Y):
    m = X.shape[1]
    
    # Forward propagation
    A = sigmoid(np.dot(w.T, X) + b)
    cost = - (1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    
    # Backward propagation
    dZ = A - Y
    dw = (1/m) * np.dot(X, dZ.T)
    db = (1/m) * np.sum(dZ)

    cost = np.squeeze(np.array(cost))  # Ensure cost is a scalar
    grads = {"dw": dw, "db": db}
    
    return grads, cost

#------------------------------------------
# Optimization (gradient descent)
#------------------------------------------
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    w= deepcopy(w)
    b= deepcopy(b)

    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        
        #Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"Cost after iteration {i}: {cost}")

    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    
    return params, grads, costs

#------------------------------------------
# Prediction
#------------------------------------------
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w=w.reshape(X.shape[0], 1)
    
    A = sigmoid(np.dot(w.T, X) + b)
    
    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    return Y_prediction

#------------------------------------------
# Model function
#------------------------------------------
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_zeros(X_train.shape[0])
    
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    w = params["w"]
    b = params["b"]
    
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations
    }
    
    return d