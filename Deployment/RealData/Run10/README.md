
## Run10

* deployed models: forward and inverse
* adjustable alpha
* seed = 42
* https://rcalix1.github.io/ConstraintsMLprediction/Deployment/RealData/Run10/index.html

---

# 🔁 Invertibility Analysis of a Neural Network Forward Model

## 🧠 Problem Setup

We are given:



- **Forward model**: `x ∈ ℝ⁷ → y ∈ ℝ⁴`
- **Inverse model**: `y ∈ ℝ⁴ → x ∈ ℝ⁷`


We have already:

* Trained a forward and inverse neural network model
* Enforced a consistency loss:

  `F( I(y) ) ~ y `

Now the goal is to **analyze whether the forward model is invertible**.

---

## 🔍 What Does "Invertibility" Mean?

A function ( f(x) ) is **invertible** if every output ( y = f(x) ) has a **unique** input ( x ).

Since exact invertibility is rare in neural networks, we can ask:

> **Is the mapping *locally invertible* at least?**
> (Do small changes in ( y ) correspond to small, unique changes in ( x )?)

This is where the **Jacobian matrix** becomes essential.

---

## 🧠 Step-by-Step Plan

### 1. **Compute the Jacobian of the Forward Model**

The **Jacobian** is the matrix of partial derivatives of ( y = f(x) ):




`J(x) = dy/dx ∈ ℝ⁴ˣ⁷`


In PyTorch:

```python
import torch
from torch.autograd.functional import jacobian

def forward_model(x):
    return model(x)  # Your trained forward NN

x = torch.randn(1, 7, requires_grad=True)
J = jacobian(forward_model, x)
print(J.shape)  # Should be [1, 4, 7]
```

---

### 2. **Use Linear Approximation**

From first-order Taylor expansion:



`Δy ≈ A · Δx` where `A = J(x)`


This approximates the model locally as a linear system.

---

### 3. **Analyze the Rank of the Jacobian Matrix**

The **rank** of ( J(x) ) tells us how much information from ( x ) is preserved in ( y ).

* **Full row rank (4)** → mapping covers output space, **locally invertible**
* **Rank < 4** → collapse in dimensions, **not invertible** at that point

In PyTorch:

```python
u, s, v = torch.svd(J[0])  # J[0] is the [4,7] matrix
rank = (s > 1e-5).sum()
print("Local Jacobian rank:", rank.item())
```

---

## ✅ Steps to the Process

Run this analysis across many points in your dataset:

1. Sample 100+ `x` values
2. For each:

   * Compute ( J(x) )
   * Calculate its **rank**
3. Visualize: histogram of ranks

### Interpretation:

* If most ranks = 4 → model is locally invertible
* If many ranks < 4 → info is lost, inverse may be ambiguous

---

## 🧠 Additional

### ✅ Condition Number

Check numerical stability of inversion:

```python
cond = s.max() / s.min()
```

### ✅ Heatmap

Plot rank or condition number against different inputs to identify fragile zones.

---

## 🔚 Summary

* ✔ Compute Jacobian ( J(x) )
* ✔ Analyze rank of ( J(x) )
* ✔ If rank = 4 → locally invertible
* ❌ If rank < 4 → not invertible in that region

This provides a way to assess whether a neural network's forward function can support a useful inverse mapping.


---

## Condition Number

# Jacobian Rank and Condition Number Analysis

This README explains the code used to compute the **Jacobian rank** and **condition number** of a neural network model's input-output mapping using PyTorch. These metrics help determine whether the model is locally invertible and how stable that inversion is.

---

## ⚙️ Code Purpose

The code:

1. Computes the **Jacobian matrix** of the model's output with respect to its input.
2. Applies **Singular Value Decomposition (SVD)**.
3. Calculates:

   * The **rank** of the Jacobian (i.e., number of significant directions)
   * The **condition number**, which indicates how well-conditioned or invertible the system is

---

## 🔎 Line-by-Line Breakdown

```python
x_point = x_point.detach().clone().requires_grad_(True)  # shape: [7]
```

* Prepares the input point to allow gradient computation.
* `detach().clone()` ensures it's not connected to any previous computation graph.
* `requires_grad_(True)` enables gradient tracking.

```python
J = jacobian(wrapped_model, x_point)  # shape: [output_dim, input_dim]
```

* Computes the **Jacobian matrix**:
  
$$ J_{ij} = \frac{\partial \text{output}_i}{\partial \text{input}_j} $$


  
* If output = 4 and input = 7, then `J` has shape `[4, 7]`.

```python
u, s, v = torch.svd(J)
```

* Performs **Singular Value Decomposition**:
  
  $$ J = U \cdot \text{diag}(s) \cdot V^T $$
  
* `s` contains the singular values, which tell you how much the input space is stretched or squashed.

```python
rank = (s > 1e-5).sum().item()
```

* Counts how many singular values are significantly non-zero.
* This gives the **numerical rank** of the Jacobian.

```python
if s.min().item() < 1e-12:
    cond_number = float('inf')
else:
    cond_number = s.max().item() / s.min().item()
```

* Computes the **condition number**:
  
  $$ \kappa(J) = \frac{\sigma_{\text{max}}}{\sigma_{\text{min}}} $$
  
* A high condition number (e.g., > 10,000) means the system is **ill-conditioned** and nearly non-invertible.
* If the smallest singular value is close to zero, it sets the condition number to `inf`.

---

## 📊 Why It Matters

* **Low condition number (e.g., < 100)**: The model is stable and easily invertible.
* **High condition number (e.g., > 10,000)**: Small changes in output can cause large changes in input.
* **Infinite condition number**: The system is effectively non-invertible at that point.

This analysis is critical for tasks like **Neural Input Optimization**, **control systems**, and **inverse problems**, where you rely on reversing the model to find inputs that produce desired outputs.

---







---






