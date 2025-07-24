# 📘 Linear and Logistic Regression From Scratch

This repository contains a Jupyter notebook (`Linear_&_Logistic_Regression.ipynb`) that implements **Linear Regression** and **Logistic Regression** from scratch in Python, without relying on machine learning libraries like scikit-learn.  
It includes detailed explanations, mathematical foundations, code implementations using gradient descent, evaluation metrics, and example outputs.

---

## 📚 Overview

This notebook is an educational resource for understanding two fundamental supervised learning algorithms:

- **Linear Regression** – For predicting continuous outcomes.
- **Logistic Regression** – For binary classification tasks.

> Both implementations use pure Python with minimal dependencies (only the `math` library for logistic regression).  
> Includes gradient descent optimization, error metrics, and detailed line-by-line code explanations.

---

## 🔹 1. Linear Regression

Linear Regression predicts a **continuous output** based on one or more input features.  
It models the relationship between inputs and outputs as a straight line.

### 📌 Key Components

**Equation:**

```math
y = a + bX + \varepsilon
```

- $y$: Predicted output (dependent variable)  
- $a$: Intercept  
- $b$: Slope coefficient (weight)  
- $X$: Input feature  
- $\varepsilon$: Error term

### 🎯 Goal

Minimize the difference between actual and predicted values by finding optimal $a$ and $b$.

### ⚙️ Optimization

Uses **gradient descent** to minimize **Mean Squared Error (MSE)**.

### 🧪 Implementation Details

- **Input data**: `x = [1, 2, 3, 4, 5]`, `y = [3, 5, 7, 9, 11]`
- **True equation**: $y = 2x + 1$
- **Parameters**: weight $w$ and bias $b$ initialized to 0
- **Hyperparameters**: learning rate = 0.01, max iterations = 10,000, tolerance = 1e-6
- **Method**: Gradient descent updates $w$ and $b$ iteratively

### 📏 Evaluation Metrics

- **MAE** (Mean Absolute Error):

  ```math
  \text{MAE} = \frac{1}{n} \sum |y_i - \hat{y}_i|
  ```

- **MSE** (Mean Squared Error):

  ```math
  \text{MSE} = \frac{1}{n} \sum (y_i - \hat{y}_i)^2
  ```

- **RMSE** (Root Mean Squared Error):

  ```math
  \text{RMSE} = \sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}
  ```

- **R² Score** (Coefficient of Determination):

  ```math
  R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
  ```

### ✅ Example Output

```
Final Model:
y = 2.0078 * x + 0.9718
Final MSE: 0.000146
Final MAE: 0.010377
Final R2 Score: 0.999982
```

The model closely approximates the true relationship $y = 2x + 1$.

---

## 🔹 2. Logistic Regression

Logistic Regression is a **binary classification** algorithm predicting the probability that an input belongs to class 1.

### 📌 Key Components

**Equations:**

```math
z = wX + b
```

```math
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
```

- $X$: Input feature vector  
- $w$: Weight  
- $b$: Bias  
- $\sigma$: Sigmoid function  
- $\hat{y}$: Predicted probability of class 1

### 🔍 Prediction Rule

- If $\hat{y} \geq 0.5$, predict **class 1**
- If $\hat{y} < 0.5$, predict **class 0**

### ❌ Loss Function: Binary Cross-Entropy (Log Loss)

```math
L = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
```

Penalizes **confident but incorrect** predictions.

### ⚙️ Optimization

Uses **gradient descent** to minimize the loss.

### 🧪 Implementation Details

- **Input data**:  
  `x_data = [1, 2, 3, 4, 5]`  
  `y_data = [0, 0, 0, 1, 1]` *(e.g., pass/fail)*

- **Parameters**: weight $w$, bias $b$ initialized to 0  
- **Hyperparameters**: learning rate = 0.1, epochs = 1000  
- **Sigmoid function**: implemented using `math.exp`  
- **Gradient descent** updates $w$ and $b$ using error $\hat{y} - y$

### ✅ Example Output

```
Epoch 900 | Loss: 0.1807 | w: 1.6875 | b: -5.6662

Testing predictions:
Input: 1 => Prediction: 0  
Input: 2 => Prediction: 0  
Input: 3 => Prediction: 0  
Input: 4 => Prediction: 1  
Input: 5 => Prediction: 1  
Input: 6 => Prediction: 1
```

---

## 🔹 3. Gradient Descent (Both Models)

### 📘 Linear Regression Gradients

```math
\frac{\partial \text{MSE}}{\partial w} = -\frac{2}{n} \sum x_i (y_i - \hat{y}_i)
```

```math
\frac{\partial \text{MSE}}{\partial b} = -\frac{2}{n} \sum (y_i - \hat{y}_i)
```

### 📘 Logistic Regression Gradients

```math
\frac{\partial L}{\partial w} = \frac{1}{n} \sum x_i (\hat{y}_i - y_i)
```

```math
\frac{\partial L}{\partial b} = \frac{1}{n} \sum (\hat{y}_i - y_i)
```

### 🔁 Parameter Update Rule

```math
w = w - \alpha \cdot \frac{\partial L}{\partial w}, \quad b = b - \alpha \cdot \frac{\partial L}{\partial b}
```

Where $\alpha$ is the learning rate.

---

## 🔹 4. Linear vs Logistic Regression Comparison

| Aspect         | Linear Regression              | Logistic Regression               |
|----------------|--------------------------------|-----------------------------------|
| Problem Type   | Regression (continuous output) | Classification (binary output)    |
| Output         | Continuous values              | Probability (0 to 1)              |
| Model          | $y = a + bX + \varepsilon$     | $\hat{y} = \sigma(wX + b)$        |
| Loss           | Mean Squared Error (MSE)       | Binary Cross-Entropy              |
| Prediction     | Direct output                  | Threshold at 0.5                  |
| Use Cases      | Predicting prices, scores      | Spam detection, pass/fail tests   |

---

## 🚀 Getting Started

### ✅ Prerequisites

- Python 3.x  
- Jupyter Notebook  
- `math` library (built-in)

### 💾 Installation

```bash
https://github.com/Umair86450/Linear-Logistic-Regression-From-Scratch.git
cd Linear-Logistic-Regression-From-Scratch
pip install jupyter
jupyter notebook
```

Open `Linear_&_Logistic_Regression.ipynb` in the Jupyter interface.

### ▶️ Running the Notebook

- Run each cell in order to see the outputs.
- Sample data is hardcoded — no external datasets needed.
- Includes markdown and comments throughout.

---

## 📈 Results

- **Linear Regression**: Learns the relationship $y \approx 2x + 1$ with high accuracy ($R^2 \approx 0.999982$)  
- **Logistic Regression**: Correctly classifies inputs (e.g., predicts pass for $x \geq 4$, fail for $x < 4$) with decreasing loss over time

---

## 📝 Notes

- Designed to be **minimal and educational**
- Uses **simple, synthetic datasets**
- Tune **learning rate** and **epochs** for better performance


