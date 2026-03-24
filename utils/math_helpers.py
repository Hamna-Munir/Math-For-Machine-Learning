# utils/math_helpers.py

import numpy as np


# =========================================================
# 🔢 Basic Math Functions
# =========================================================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    exp_z = np.exp(z - np.max(z))  # stability trick
    return exp_z / np.sum(exp_z)


# =========================================================
# 📉 Loss Functions
# =========================================================
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * np.log(1 - y_pred)
    )


def cross_entropy(y_true, y_pred):
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]


# =========================================================
# 🔄 Gradients (for learning algorithms)
# =========================================================
def mse_gradient(X, y_true, y_pred):
    n = len(y_true)
    return (-2 / n) * np.dot(X.T, (y_true - y_pred))


def logistic_gradient(X, y_true, y_pred):
    n = len(y_true)
    return (1 / n) * np.dot(X.T, (y_pred - y_true))


# =========================================================
# ⚖️ Normalization / Scaling
# =========================================================
def min_max_scale(X):
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0) + 1e-10)


def standardize(X):
    return (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-10)


# =========================================================
# ➕ Vector / Matrix Operations
# =========================================================
def add_bias(X):
    return np.c_[np.ones((X.shape[0], 1)), X]


def dot_product(a, b):
    return np.dot(a, b)


def transpose(X):
    return np.transpose(X)


# =========================================================
# 📊 Metrics
# =========================================================
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    return TP / (TP + FP + 1e-10)


def recall(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP / (TP + FN + 1e-10)


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r + 1e-10)


# =========================================================
# 🧠 PCA Helper Functions
# =========================================================
def compute_mean(X):
    return np.mean(X, axis=0)


def compute_covariance(X):
    mean = compute_mean(X)
    X_centered = X - mean
    return np.dot(X_centered.T, X_centered) / (X.shape[0] - 1)


def eigen_decomposition(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    return eigenvalues, eigenvectors


# =========================================================
# 🔁 Gradient Descent Step
# =========================================================
def gradient_descent_step(weights, gradients, learning_rate):
    return weights - learning_rate * gradients
