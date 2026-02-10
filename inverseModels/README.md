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


## Notes

G is the Jacobian matrix of f(x) evaluated at some point

p' = Gx

min || x ||

subjected to:

p Gx

p(t) = G(t) x(t), t =1, 2, 3, ..., 1000

then at each t, we need to solve the above optimization problem 

for x(t)

min || x(t) ||

## Papers

* https://d1wqtxts1xzle7.cloudfront.net/34612234/iksurvey-libre.pdf?1409714427=&response-content-disposition=inline%3B+filename%3DInverse_Kinematics.pdf&Expires=1770521683&Signature=DDFz-WzL40pYo25T2NVFvMuGvGgxMWBWEo3d0C5W8MUDWeAQcUDrVVr-g0~YZifL8FMAa3qZ7pUzO-gB5Mf8dxgm7xu38L3jKQoGKPTqHqRHKzW7WoH-kGm0iIr5~3LztdTQ9aZNu7r~6Tu1Al9gUbcO56YDhtK24tOfyxZke45ommbcBIFmWW0GsWAfvgi~8ftFls0STKaTDM0BeLAYf-bP7bWnCdJ1dymBF6nqCrGldAL7Yzmgg-xji-4yvbSzVki-Dr1zTOhvnfnHHVhPVfnlpbftjhPUufBw4vk8oV0znen8SC0OXiKVwit~Prytkuu0YZvB6hjV46~R6VDVpQ__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA

* 

---




# Minimum-Norm Solutions and Neural Input Optimization

This document explores four key techniques for solving inverse problems using minimum-norm solutions, starting from classical linear algebra to nonlinear neural network inversion. Each section includes a conceptual explanation and working code examples.

---

## 1. Analytical Regression (Solving for $w$ in $y = Xw$)

In classical linear regression, we solve for weights $w$ that minimize squared error:

$$
\min_w |y - Xw|^2
$$

This gives the closed-form solution:

$$
w = (X^T X)^{-1} X^T y
$$

If $X^T X$ is not invertible, we use the Moore–Penrose pseudoinverse:

$$
w = X^+ y
$$

### Code Example:

```python
import numpy as np

X = np.array([[1, 1], [1, 2], [1, 3]])  # 3 samples, 2 features (bias + slope)
y = np.array([1, 2, 2.5])

w = np.linalg.pinv(X) @ y

print("Weights w:", w)
print("Predicted y:", X @ w)
```

---

## 2. Minimum-Norm Inverse (Solving $Ax = y$)

When $A$ is underdetermined (more variables than equations), there are infinite $x$ that solve $Ax = y$. The minimum-norm solution is:

$$
x = A^+ y
$$

This minimizes $|x|$ among all valid solutions.

### Code Example:

```python
A = np.array([[1, 2, 3], [4, 5, 6]])  # Shape (2, 3)
y = np.array([7, 8])

x_min_norm = np.linalg.pinv(A) @ y

print("Minimum-norm solution x:", x_min_norm)
print("Norm of x:", np.linalg.norm(x_min_norm))
```

---

## 3. Neural Input Optimization (NIO)

For a nonlinear function $f(x)$ (e.g., a neural network), we want to find $x$ such that:

$$
\min_x |f(x) - y|^2 + \lambda |x|^2
$$

This is the gradient-based analog of the minimum-norm inverse.

### Code Example (PyTorch):

```python
import torch
import torch.nn as nn
import torch.optim as optim

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
    def forward(self, x):
        return self.net(x)

model = TinyNet()
for param in model.parameters():
    param.requires_grad = False

y_target = torch.tensor([1.0, 2.0])
x = torch.randn(3, requires_grad=True)
optimizer = optim.Adam([x], lr=0.01)

for step in range(300):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = ((y_pred - y_target)**2).sum() + 0.01 * (x**2).sum()
    loss.backward()
    optimizer.step()

print("Recovered input x:", x.data)
print("Output f(x):", model(x).data)
```

---

## 4. Jacobian-Based Linear Inversion

We linearize the neural network $f(x)$ near a point $x_0$ using a first-order Taylor approximation:

$$
f(x) \approx f(x_0) + J(x_0)(x - x_0)
$$

Where:

* $f(x)$ is a vector-valued function (e.g., neural net output)
* $J(x_0)$ is the **Jacobian matrix** of shape output_dim and input_dim containing all partial derivatives:

---



$$
J_{ij} = \frac{\partial f_i}{\partial x_j}
$$

To solve for an update that moves the output closer to a target $y$, we compute:

$$
x = x_0 + J(x_0)^+ (y - f(x_0))
$$

This is a **minimum-norm correction** to $x_0$ based on the local linear approximation of the network.

---

### ✅ Code Example: Compute Jacobian Using Autograd

```python
import torch
import torch.nn as nn
import numpy as np

# Define model: f: R^3 → R^2
model = nn.Sequential(
    nn.Linear(3, 5),
    nn.Tanh(),
    nn.Linear(5, 2)
)
for p in model.parameters():
    p.requires_grad = False

y_target = torch.tensor([1.0, 2.0])
x0 = torch.randn(3, requires_grad=True)
y0 = model(x0)

# --- Compute full Jacobian J(x0) ---
def compute_jacobian(fx, x):
    """
    Returns Jacobian J where J[i, j] = ∂f_i / ∂x_j
    Assumes: 
      - fx is the output tensor f(x), shape [m]
      - x is the input tensor, shape [n]
    Returns:
      - J: shape [m, n]
    """
    return torch.autograd.functional.jacobian(lambda x_: model(x_), x)

J = compute_jacobian(y0, x0)  # Shape [2, 3]

# --- Solve for minimum-norm input correction ---
residual = y_target - y0.detach()
J_np = J.detach().numpy()
residual_np = residual.numpy()

dx = torch.tensor(np.linalg.pinv(J_np) @ residual_np)

# --- Update x ---
x_new = x0.detach() + dx
print("Updated x:", x_new)
print("New f(x):", model(x_new))
```

---

### ✅ Notes:

* `torch.autograd.functional.jacobian()` is recommended for full Jacobians.
* Assumes a single input vector (not batched). For batched inputs, use `vmap` or loop.
* For better numerical stability, cast tensors and model to `.double()`.



---

## Summary Table

|  # | Technique                 | Solve For | Model Type   | Solution Style    |
| -: | ------------------------- | --------- | ------------ | ----------------- |
|  1 | Analytical Regression     | $w$       | Linear       | Closed-form       |
|  2 | Minimum-Norm Inverse      | $x$       | Linear       | Pseudoinverse     |
|  3 | Neural Input Optimization | $x$       | Nonlinear NN | Iterative (grad)  |
|  4 | Jacobian Linearization    | $x$       | Nonlinear NN | Local linear step |

Use this as a reference when comparing inverse strategies — whether for traditional systems or modern deep learning models.

---

## Limitations


* https://en.wikipedia.org/wiki/Gauss–Newton_algorithm

