## Inverse Models

* NIO
* Min norm solutions

# Minimum-Norm Inverse Solution

This README outlines the use of a **minimum-norm inverse** to solve the inverse problem of a PCA-compressed neural forward model. The objective is to recover **7-dimensional input vectors** that produce desired **4-dimensional outputs**, using the local Jacobian at each test sample.

---

## 📦 Problem Setup

* **Original data**: input ( x \in \mathbb{R}^7 ), output ( y \in \mathbb{R}^4 )
* **PCA**: Compress ( x \to x_{PCA} \in \mathbb{R}^4 )
* **Model**: Train forward map ( y = f(x_{PCA}) )
* **Goal**: Given a desired change ( \Delta y ), find a corresponding ( \Delta x ) such that ( f(x + \Delta x) \approx y + \Delta y )

---

## 🧠 Minimum-Norm Method

### Optimization Objective

At each test sample ( t ), solve:

[ \min | x^{(t)} | \quad \text{subject to} \quad G^{(t)} x^{(t)} = p^{(t)} ]

Where ( G = J ) is the Jacobian of the forward model at ( x^{(t)} ), and ( p^{(t)} = y^{(t)} + \Delta y ).

### Solution via Pseudoinverse

The closed-form minimum-norm solution is:

[ \Delta x = J^{\dagger} \Delta y \quad \text{where} \quad J^{\dagger} = (J^T J)^{-1} J^T ]

This selects the unique solution with the smallest ( \ell_2 ) norm among all ( \Delta x ) that satisfy ( J \Delta x = \Delta y ).

---

## 🧪 Example Code Snippet

```python
def get_min_norm_delta_x(x_point, delta_y):
    x_point = x_point.detach().clone().requires_grad_(True)
    J = jacobian(wrapped_model, x_point)
    J_pinv = torch.linalg.pinv(J)
    delta_x = J_pinv @ delta_y.view(-1, 1)
    return delta_x.view(-1)
```

This operates in PCA space. To recover full input vector ( x \in \mathbb{R}^7 ), reverse the PCA transform after computing ( \Delta x ).

---

## 🧩 Interpretation

* The **minimum-norm** refers to the input perturbation ( \Delta x ) having the smallest magnitude needed to achieve ( \Delta y ).
* Useful when the Jacobian is not square or is ill-conditioned.
* Diagnostic metrics: **rank**, **condition number**, and **output error**.

---

## ✅ Summary

Use the Jacobian pseudoinverse at each test point to compute local minimum-norm input deltas. This gives a unique, efficient inverse—ideal for linearized models in PCA space.

---
