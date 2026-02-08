## Inverse Models

* NIO
* Min norm solutions

## Questions

* With this min norm solution you need to have an initial x,y, and delta?

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

## A code example

```

import torch
from torch.autograd.functional import jacobian

def solve_minimum_norm_update(model, x0, target_y):
    """
    Computes a minimum-norm update step for the input x0 towards the target_y.
    
    Args:
        model (nn.Module): The neural network.
        x0 (torch.Tensor): Initial guess for the input.
        target_y (torch.Tensor): The desired output.

    Returns:
        torch.Tensor: The minimum-norm update to x0 (Delta x).
    """
    # Ensure the input requires gradients
    x = x0.detach().clone().requires_grad_(True)
    
    # Define a function to compute the output for use with jacobian()
    def func(input_x):
        return model(input_x)

    # Compute the Jacobian matrix at the current input x
    # 'create_graph=True' is often needed if you want to backpropagate through this process
    J = jacobian(func, x, create_graph=True, vectorize=True)
    
    # Calculate the current residual (difference between desired and actual output)
    current_y = func(x)
    delta_y = target_y - current_y
    
    # Flatten the Jacobian and residual if necessary (depends on problem dimensions)
    # This example assumes J is 2D and delta_y is 1D for simplicity
    if J.dim() > 2:
        J = J.view(-1, x.numel())
        delta_y = delta_y.view(-1)
        
    # Compute the pseudoinverse of the Jacobian using SVD
    J_pinv = torch.pinverse(J)
    
    # Calculate the minimum norm update: Delta x = J_pinv @ Delta y
    delta_x = J_pinv @ delta_y
    
    # Reshape delta_x to match the original input shape
    delta_x = delta_x.view_as(x0)
    
    return delta_x

# Example usage with a simple model (ensure model is defined)
# model = MyNeuralNetwork()
# x0 = torch.randn(1, input_dim)
# target_y = torch.randn(1, output_dim)
# delta_x = solve_minimum_norm_update(model, x0, target_y)
# x_new = x0 + delta_x



```



## Neural Input Optimization (NIO)

* https://ieeexplore.ieee.org/abstract/document/11337483
* 




