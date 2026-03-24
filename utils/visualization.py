# utils/visualization.py

import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# 🎨 Global Style
# =========================================================
def set_plot_style():
    plt.style.use('default')
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "axes.grid": True,
        "font.size": 11
    })


# =========================================================
# 📈 Line Plot
# =========================================================
def plot_line(x, y, title="Line Plot", xlabel="X-axis", ylabel="Y-axis"):
    set_plot_style()
    plt.figure()
    plt.plot(x, y, linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# =========================================================
# 🔵 Scatter Plot
# =========================================================
def plot_scatter(x, y, title="Scatter Plot", xlabel="X-axis", ylabel="Y-axis"):
    set_plot_style()
    plt.figure()
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# =========================================================
# 📊 Histogram
# =========================================================
def plot_histogram(data, bins=10, title="Histogram"):
    set_plot_style()
    plt.figure()
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.show()


# =========================================================
# 📉 Loss Curve
# =========================================================
def plot_loss_curve(losses):
    set_plot_style()
    plt.figure()
    plt.plot(range(len(losses)), losses, linewidth=2)
    plt.title("Training Loss Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()


# =========================================================
# 📊 Multiple Lines
# =========================================================
def plot_multiple_lines(x, ys, labels, title="Comparison Plot"):
    set_plot_style()
    plt.figure()

    for y, label in zip(ys, labels):
        plt.plot(x, y, label=label)

    plt.title(title)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.show()


# =========================================================
# 📦 Bar Chart
# =========================================================
def plot_bar(categories, values, title="Bar Chart"):
    set_plot_style()
    plt.figure()
    plt.bar(categories, values)
    plt.title(title)
    plt.xlabel("Categories")
    plt.ylabel("Values")
    plt.show()


# =========================================================
# 🧠 Decision Boundary (2D)
# =========================================================
def plot_decision_boundary(X, y, model, resolution=100):
    set_plot_style()

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title("Decision Boundary")
    plt.show()


# =========================================================
# 🎥 Gradient Descent Animation (Basic)
# =========================================================
def animate_gradient_descent(x, y, predictions_history):
    """
    predictions_history: list of predicted y values per iteration
    """
    set_plot_style()

    for i, preds in enumerate(predictions_history):
        plt.clf()
        plt.scatter(x, y)
        plt.plot(x, preds)
        plt.title(f"Iteration {i}")
        plt.pause(0.1)

    plt.show()


# =========================================================
# 📊 Confusion Matrix
# =========================================================
def plot_confusion_matrix(cm, classes):
    set_plot_style()

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, cm[i, j], ha="center")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# =========================================================
# 📈 ROC Curve
# =========================================================
def plot_roc_curve(y_true, y_scores):
    set_plot_style()

    thresholds = np.linspace(0, 1, 100)
    tpr = []
    fpr = []

    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)

        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        TN = np.sum((y_true == 0) & (y_pred == 0))

        tpr.append(TP / (TP + FN + 1e-10))
        fpr.append(FP / (FP + TN + 1e-10))

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()


# =========================================================
# 📊 PCA Visualization (2D)
# =========================================================
def plot_pca_2d(X_transformed, y=None):
    set_plot_style()

    plt.figure()

    if y is not None:
        plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y)
    else:
        plt.scatter(X_transformed[:, 0], X_transformed[:, 1])

    plt.title("PCA Projection (2D)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()
