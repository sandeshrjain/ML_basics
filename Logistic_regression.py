# -*- coding: utf-8 -*-
# Logistic Regression

import sys
import os
path = '/your_path'
sys.path.append(path)
#synthetic dataset
logistic_x_data_path = os.path.join(path, 'data/logistic_x_.txt')
logistic_y_data_path = os.path.join(path, 'data/logistic_y_.txt')
def feature_normalize(X):
    # feature_normalize: Normalizes the features in X 
    mu     = 0
    sigma  = 0
    mu     = np.mean(X, 0)
    sigma  = np.std(X, 0)
    X      = (X - mu) / sigma
    X_norm = X
  
    return X_norm, mu, sigma
import numpy as np
from sklearn.model_selection import train_test_split
X = np.loadtxt(logistic_x_data_path) 
y = np.loadtxt(logistic_y_data_path).reshape(-1, 1) 
print(np.shape(X))
X, mu, std = feature_normalize(X) #normalize synthetic set
m = np.shape(X)[0]
X = np.concatenate((np.ones((m, 1)), X), axis=1)
X, test_data, y, test_labels = train_test_split(X, y, test_size=0.2) #keep 20% for test and 80% for train
#   Digits dataset import
from sklearn.datasets import load_digits
digits = load_digits(n_class= 2)
X_digit = digits.data
y_digit = digits.target.reshape((-1,1))
y_digit[y_digit==0] = -1    # apply mods to convert y labels
X_digit = np.concatenate((np.ones((X_digit.shape[0], 1)), X_digit), axis=1)
X_digit, test_X_digit, y_digit, test_y_digit = train_test_split(X_digit, y_digit, 
test_size=0.2)
#   Cancer dataset import
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_cancer = cancer.data
y_cancer = cancer.target.reshape((-1,1))
y_cancer[y_cancer==0] = -1  # apply mods to convert y labels
X_cancer = np.concatenate((np.ones((X_cancer.shape[0], 1)), X_cancer), axis=1)
X_cancer, test_X_cancer, y_cancer, test_y_cancer = train_test_split(X_cancer, 
y_cancer, test_size=0.2)
print(y)
import matplotlib.pyplot as plt
plt.plot(X[np.where(y==1), 1], X[np.where(y==1), 2], 'rx')
plt.plot(X[np.where(y==-1), 1], X[np.where(y==-1), 2], 'bo')  
plt.xlabel('x2')
plt.ylabel('x1')
plt.show()
def sigmoid(z):
    g = 1/(1+np.exp(-z))
    return g
def compute_cost(X, y, theta):
    J = 0;
    J = -np.mean(np.log(0.0001+sigmoid(y*(X@theta))))  # add small value so log is not inf
    return J
def compute_gradient(X, y, theta):          
    gradient = (((sigmoid(-y*(X@theta)))*(-y)*X))
    gradient = (1/np.shape(X)[0])*np.sum(gradient, axis=0)
    gradient.resize(np.shape(X)[1],1)
    #print(np.shape(gradient_))
    return gradient
def gradient_descent_logistic(X, y, theta, alpha, num_iters):
    J_history = []
    for iter in range(num_iters):
        theta = theta - (alpha)*compute_gradient(X, y, theta)
        J = compute_cost(X, y, theta)
        #print(J)
        J_history.append(J)
    return theta, J_history
def sg_descent_logistic(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []
    for iter in range(num_iters):
        for (x_single, y_single) in zip(X,y):
          theta = theta - (alpha)*compute_gradient(np.array(x_single).reshape((1, 
len(x_single))), np.array(y_single).reshape((1, 1)), theta)
        J = compute_cost(X, y, theta)
        #print(J)
        J_history.append(J)
        print("Accuracy on training set: ", evaluate_accuracy_digit(X, y, theta))
        indices = np.arange(m)
        rand = np.random.shuffle(indices)
        X = X[rand]
        y = y[rand]
    return theta, J_history
def scikit_logistic(X, y):
    from sklearn.linear_model import LogisticRegression 
    clf = LogisticRegression(fit_intercept=True, C = 1e15) 
    clf.fit(X, y) 
    acc = clf.score(X,y)
    return clf.intercept_, clf.coef_, acc
# Train model on scikit logistic regression
theta = np.zeros((X.shape[1], 1))
alpha = 0.1;
num_iters = 1;
theta0, theta, acc = scikit_logistic(X, y)
print(theta.T)
# Train your model on our own implementation
theta = np.zeros((X.shape[1], 1))
alpha = 0.1;
num_iters = 4;
theta, J_history = sg_descent_logistic(X, y, theta, alpha, num_iters)
print(theta)
plt.plot(list(range(0, len(J_history))), J_history, '-b')  
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()
plt.plot(X[np.where(y==1), 1], X[np.where(y==1), 2], 'rx')
plt.plot(X[np.where(y==-1), 1], X[np.where(y==-1), 2], 'bo')
plt.plot([np.max(X), np.min(X)], [-theta[0]/theta[2] -theta[1]/theta[2] * 
np.max(X), -theta[0]/theta[2] -theta[1]/theta[2] * np.min(X)], 'k')
print(-theta[0]/theta[2] -theta[1]/theta[2] * np.max(X))
print(np.shape(X))
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
y
def predict(X, theta):
    y_hat = sigmoid(X@theta)
    return y_hat
def evaluate_accuracy(X, y, theta):
    y_pred = predict(X, theta)
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = -1
    return np.mean(y_pred == y)
print("Accuracy on training set: ", evaluate_accuracy(X, y, theta))
print("Accuracy on testing set: ", evaluate_accuracy(test_data, test_labels, 
theta))
# Train model for different alphas.
accuracies_diff_alphas = []
alphas = [0.001,0.01,0.1,1,10,100]
for alpha in alphas:
    theta = np.zeros((X.shape[1], 1))
    num_iters = 5;
    theta, J_history = sg_descent_logistic(X, y, theta, alpha, num_iters)
    accuracies_diff_alphas.append(evaluate_accuracy(X, y, theta))
plt.plot(np.log(alphas), accuracies_diff_alphas)
plt.xlabel('ln(alpha)')
plt.ylabel("Accuracy on Training Set")
plt.title("Accuracy v/s Alpha")
y_digit
def evaluate_accuracy_digit(X, y, theta):
    y_pred = predict(X, theta)
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = -1
    return np.mean(y_pred == y)
# Train model for digit dataset.
theta = np.zeros((X_digit.shape[1], 1))
alpha = 0.01;
num_iters = 5;
theta, J_history = sg_descent_logistic(X_digit, y_digit, theta, alpha, num_iters)
print("Accuracy on training set: ", evaluate_accuracy_digit(X_digit, y_digit, 
theta))
print("Accuracy on testing set: ", evaluate_accuracy_digit(test_X_digit, 
test_y_digit, theta))
# Train your model for cancer dataset.
theta = np.zeros((X_cancer.shape[1], 1))
alpha = 0.1;
num_iters = 5;
theta, J_history = sg_descent_logistic(X_cancer, y_cancer, theta, alpha, num_iters)
print("Accuracy on training set: ", evaluate_accuracy(X_cancer, y_cancer, theta))
print("Accuracy on testing set: ", evaluate_accuracy(test_X_cancer, test_y_cancer, 
theta))
# Load and train on wine dataset
import numpy as np
X_wine = np.loadtxt(os.path.join(path, 'data/wine_train_X.txt'))
y_train = np.loadtxt(os.path.join(path, 'data/wine_train_y.txt')).reshape(-1, 1)
X_test = np.loadtxt(os.path.join(path, 'data/wine_test_X.txt'))
y_test = np.loadtxt(os.path.join(path, 'data/wine_test_y.txt')).reshape(-1, 1)
X_wine,_,_ = feature_normalize(X_wine)
X_test,_,_ = feature_normalize(X_test)
X_wine = np.concatenate((np.ones((X_wine.shape[0], 1)), X_wine), axis=1)
X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)
#theta_init = np.ones((np.shape(X_wine)[1], 1))
#theta_1, J_history_1 = gradient_descent_logistic(X_wine, y_train, theta_init, alpha, num_iters)
#plt.plot(list(range(0, len(J_history_1))), J_history_1, '-b')  
#plt.xlabel('Number of iterations')
#plt.ylabel('Cost J')
#plt.show()
theta = np.zeros((X_wine.shape[1], 1))
alpha = 0.1;
num_iters = 5;
theta, J_history = sg_descent_logistic(X_wine, y_train, theta, alpha, num_iters)
print("Accuracy on training set: ", evaluate_accuracy(X_wine, y_train, theta))
print("Accuracy on testing set: ", evaluate_accuracy(X_test, y_test, theta))