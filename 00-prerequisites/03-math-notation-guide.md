# Mathematical Notation Guide for Machine Learning

**Author:** Hamna Munir  
**Repository:** Math-for-Machine-Learning  

---

##  Purpose of This Guide

Mathematical notation is the *language of Machine Learning*.  
This guide explains commonly used symbols and notations so you can:
- Read ML research papers confidently
- Understand algorithms mathematically
- Connect equations with Python code

---

##  Scalars, Vectors, and Matrices

| Symbol | Meaning | Example |
|------|--------|--------|
| \( a \) | Scalar (single number) | \( a = 5 \) |
| \( \mathbf{x} \) | Vector | \( \mathbf{x} = [1, 2, 3] \) |
| \( \mathbf{X} \) | Matrix | \( 3 \times 3 \) matrix |
| \( x_i \) | i-th element of a vector | \( x_1 = 1 \) |
| \( X_{ij} \) | Element at row i, column j | Matrix entry |

---

##  Basic Mathematical Operations

| Notation | Meaning |
|--------|--------|
| \( + \) | Addition |
| \( - \) | Subtraction |
| \( \times \) | Multiplication |
| \( \div \) | Division |
| \( a^2 \) | Power |
| \( \sqrt{a} \) | Square root |

---

##  Summation & Products

| Notation | Meaning |
|--------|--------|
| \( \sum_{i=1}^{n} x_i \) | Sum of elements |
| \( \prod_{i=1}^{n} x_i \) | Product of elements |

**ML Example:**  
Loss over dataset  
\[
\sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

---

##  Vector & Matrix Operations

| Notation | Meaning |
|--------|--------|
| \( \mathbf{x}^T \) | Transpose |
| \( \mathbf{X}^{-1} \) | Inverse |
| \( \mathbf{A}\mathbf{B} \) | Matrix multiplication |
| \( \| \mathbf{x} \| \) | Vector norm |

---

##  Norms (Magnitude)

| Notation | Name | Usage |
|--------|------|------|
| \( \|x\|_1 \) | L1 norm | Sparsity (Lasso) |
| \( \|x\|_2 \) | L2 norm | Regularization (Ridge) |

---

##  Probability Notation

| Notation | Meaning |
|--------|--------|
| \( P(A) \) | Probability of event A |
| \( P(A|B) \) | Conditional probability |
| \( \mathbb{E}[X] \) | Expected value |
| \( \text{Var}(X) \) | Variance |

---

##  Derivatives & Calculus

| Notation | Meaning |
|--------|--------|
| \( \frac{d}{dx} f(x) \) | Derivative |
| \( \nabla f(x) \) | Gradient |
| \( \frac{\partial f}{\partial x} \) | Partial derivative |

**ML Example:**  
Gradient Descent update:
\[
\theta = \theta - \alpha \nabla J(\theta)
\]

---

##  Optimization Symbols

| Symbol | Meaning |
|------|--------|
| \( \min \) | Minimize |
| \( \arg\min \) | Value that minimizes |
| \( \alpha \) | Learning rate |

---

##  Common ML Symbols

| Symbol | Meaning |
|------|--------|
| \( x \) | Input feature |
| \( y \) | True label |
| \( \hat{y} \) | Predicted output |
| \( \theta \) | Model parameters |
| \( J(\theta) \) | Loss / Cost function |

---

##  ML Connection

Every Machine Learning algorithm is expressed using mathematical notation.  
Understanding these symbols allows you to:
- Decode ML equations
- Implement algorithms correctly
- Transition from beginner to advanced ML concepts

---

##  Summary

- Mathematical notation is essential for ML
- Scalars, vectors, and matrices form the foundation
- Calculus and probability drive learning algorithms
- This guide acts as a reference throughout the repository

---

**Developed by — Hamna Munir  **
