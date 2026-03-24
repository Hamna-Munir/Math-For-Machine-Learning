# 🛠️ Utils Module

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)
![Maintained](https://img.shields.io/badge/Maintained-Yes-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

👩‍💻 **Author:** Hamna Munir  
📦 **Repository:** *Math-for-Machine-Learning*  

---

## 📌 Overview

The `utils/` module provides reusable helper functions for:

- 🔢 Mathematical computations  
- 📊 Data visualization  

It helps keep your notebooks **clean, modular, and professional**.

---

## 📂 Folder Structure

```
utils/
├── visualization.py # 📊 Plotting utilities
├── math_helpers.py # 🔢 Math functions
└── init.py # 📦 Package initializer
```


---

## 📊 visualization.py

### ✨ Features

- 📈 Line plots  
- 🔵 Scatter plots  
- 📊 Histograms  
- 📉 Loss curves  
- 🧠 Decision boundary visualization  

### 🚀 Example

```python
from utils.visualization import plot_line

plot_line(x, y, title="Training Loss")

```
## 🔢 math_helpers.py

### ✨ Features

- 📌 Sigmoid function
- 📉 Mean Squared Error (MSE)
- 🔄 Gradient calculations
- ⚖️ Normalization
- ➕ Vector operations
 
### 🚀 Example

```python
from utils.math_helpers import sigmoid

print(sigmoid(0.5))
```
---

## 📦 init.py

Marks the folder as a Python package and enables clean imports:

```python
from utils import visualization
from utils import math_helpers
```

---

## 💡 Why Use This Module?

- ✅ Avoid code duplication
- ✅ Improve readability
- ✅ Reuse logic across notebooks
- ✅ Keep your project organized

---

##  🔗 Where It's Used

These utilities power multiple parts of the project:

- 📉 Linear Regression
- 📊 Logistic Regression
- 🔁 Gradient Descent
- 🧠 Neural Networks
- 📦 PCA
  
---
  
## 🚀 Future Improvements

- 🎥 Animated visualizations
- 📊 3D plotting
- ⚙️ Optimization algorithms
- 📏 Evaluation metrics (Accuracy, Precision, Recall)
- 🧮 Advanced matrix operations

---

## 🧾 Summary

The utils/ module acts as the backbone of the project:

⚡ Reusable, clean, and efficient tools for ML development

---

✨ Clean code = Better understanding = Better models


