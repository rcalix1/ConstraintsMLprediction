
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
* Enforced a consistency loss: ( F(I(y)) \approx y )

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

[
\Delta y \approx A \cdot \Delta x \quad \text{where } A = J(x)
]

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
