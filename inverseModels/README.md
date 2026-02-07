## Inverse Models

* NIO
* Min norm solutions

## Jacobians

* https://github.com/rcalix1/ConstraintsMLprediction/tree/main/Deployment/RealData/Run10

# 🧮 Minimum-Norm Inverse Solution

This README explains how to compute a **minimum-norm input** that causes a desired change in model output using the local Jacobian of a neural forward model. The model maps PCA-compressed input vectors (size 4) to output vectors (size 4), originally derived from a 7D input space.

---

## 🔧 Problem Setup

* Original input: `x ∈ ℝ⁷`
* PCA-compressed input: `x_pca ∈ ℝ⁴`
* Forward model: `y = f(x_pca)`
* Objective: For a small change in output `Δy`, find the smallest `Δx` such that:

$$
J \cdot \Delta x = \Delta y
$$

---

## ✅ Solution: Minimum-Norm Input

To solve for `Δx`, we use the **Moore–Penrose pseudoinverse** of the Jacobian `J`:

$$
\Delta x = J^{\dagger} \cdot \Delta y
$$

This gives the input change `Δx` with the smallest Euclidean norm (`‖Δx‖`) that still satisfies the desired output change.

* This works **locally**, per test sample.
* The Jacobian `J` is computed via autograd.
* After solving in PCA space (4D), the result can be **reversed back to the original 7D** input space.

---

## 🧪 Example Code (Clean)

```python
def get_min_norm_delta_x(x_point, delta_y):
    x_point = x_point.detach().clone().requires_grad_(True)
    J = jacobian(wrapped_model, x_point)  # shape: [4, 4] in PCA space
    delta_x = torch.linalg.pinv(J) @ delta_y.view(-1, 1)
    return delta_x.view(-1)
```

---

## 📈 Diagnostic Metrics

For each sample, you may also compute:

* **Rank of J**: to detect singularities
* **Condition number**: `cond = σ_max / σ_min` from SVD
* **Error**: `‖f(x + Δx) − (y + Δy)‖`

---

## 📌 Interpretation

* The **minimum-norm** refers to choosing the input perturbation `Δx` with smallest norm among all possible solutions.
* It finds a unique solution even when multiple inputs could produce the same output change.
* Especially helpful when the Jacobian is non-invertible or poorly conditioned.

---

## 🧠 Reminder

This operates in **PCA space**, so after `Δx` is found, use the PCA decoder to recover the full 7D input if needed.

---
