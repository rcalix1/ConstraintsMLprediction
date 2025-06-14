## Run 6

## ✅ Input Optimization Results (Constraint Satisfaction)

Each block below shows:
- The unscaled **target outputs**
- The **input clamping bounds**
- A feasible **input vector** (generated via gradient-based optimization)
- The resulting **predicted output**
- The constraint **mask** used (selecting which outputs to optimize)

---

### 🔧 Case 1 — Mild High Temperature Target

```python
target_output_not_scaled = tensor([[1900., 1700., 100.]])
clamp_min = tensor([[   0,    0,    0,    0,   21, 1200, 150]])
clamp_max = tensor([[ 100, 1500, 300, 300,   40, 1500, 250]])

Input:            tensor([[   2.9200,  482.2400,   85.7100,    9.2100,   23.9300, 1451.3400,  214.0900]])
Predicted Output: tensor([[2085.0601, 1859.2100, 134.8300]])
mask = [3, 2, 2]
```

---

### 🔧 Case 2 — Constrained Geometry

```python
target_output_not_scaled = tensor([[1900., 1700., 100.]])
clamp_min = tensor([[   0,    0,    0,    0,   21, 1300, 150]])
clamp_max = tensor([[   0,    0,  200,  200,   32, 1500, 250]])

Input:            tensor([[   0.0000,    0.0000,   63.6100,    3.4100,   21.7900, 1320.4700, 233.9900]])
Predicted Output: tensor([[2108.8999, 1715.8800, 212.0000]])
mask = [3, 2, 2]
```

---

### 🔧 Case 3 — High FTA and Low TGT

```python
target_output_not_scaled = tensor([[2600., 1800., 70.]])
clamp_min = tensor([[   0,    0,    0,    0,   21, 1200, 150]])
clamp_max = tensor([[ 100, 1500, 300, 300,   40, 1500, 250]])

Input:            tensor([[   8.8900, 1203.0000,   44.7800,  136.0100,   31.0000, 1423.0300, 239.7300]])
Predicted Output: tensor([[2608.3301, 1847.9500, 69.6700]])
mask = [3, 2, 2]
```

---

### 🔧 Case 4 — Tighter Clamp, High FTA

```python
target_output_not_scaled = tensor([[2600., 1800., 70.]])
clamp_min = tensor([[   0,    0,    0,    0,   21, 1300, 150]])
clamp_max = tensor([[   0,    0,  200,  200,   32, 1500, 250]])

Input:            tensor([[   0.0000,    0.0000,   19.8900,   38.0200,   27.5700, 1467.5699, 232.3900]])
Predicted Output: tensor([[2630.3701, 2114.0500, 71.1400]])
mask = [3, 2, 2]
```

---

🧠 **Notes:**
- `mask = [3, 2, 2]` means: Raceway Flame Temp > 1900K, Hot Metal Temp > 1700K, Top Gas Temp > 100°C
- Inputs are bounded within realistic process ranges
- Gradient-based optimization performed on standardized latent input `z`
