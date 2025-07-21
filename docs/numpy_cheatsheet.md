# 🧠 NumPy Cheat Sheet – Quick Reference

This cheat sheet captures key NumPy operations across array creation, manipulation, math, and linear algebra — ideal for interviews, coding, and ML workflows.

---

## 📦 Array Creation

```python
np.array([1, 2, 3])             # From list
np.zeros((2, 3))                # All zeros
np.ones((3, 3))                 # All ones
np.full((2, 2), 7)              # Constant value
np.empty((3, 3))                # Uninitialized
np.arange(0, 10, 2)             # Even range
np.linspace(0, 1, 5)            # Evenly spaced floats
np.eye(3)                       # Identity matrix
np.random.rand(2, 3)            # Uniform [0, 1)
np.random.randn(2, 3)           # Normal (mean=0)
np.random.randint(1, 10, (2, 2))# Random integers
```

---

## 🔍 Indexing & Slicing

```python
arr[2]               # Element
arr[1:4]             # Slice
arr[::-1]            # Reverse
arr[mask]            # Boolean indexing
arr[[0, 2, 4]]       # Fancy indexing
arr[:, 1]            # Column from 2D
arr[1, :]            # Row from 2D
np.where(arr > 5)    # Condition locations
```

---

## 🔁 Reshape & Transform

```python
arr.reshape(3, 2)
arr.flatten()                # Copy
arr.ravel()                  # View
np.expand_dims(arr, axis=0)
np.squeeze(arr)
np.transpose(arr)
arr.T
np.swapaxes(arr, 0, 1)
```

---

## 🧮 Math & Aggregation

```python
arr + 1, arr - 2, arr * 3
np.add(a, b), np.multiply(a, b)
np.sum(arr), np.mean(arr)
np.min(arr), np.max(arr)
np.std(arr), np.var(arr)
np.round(arr, 2)
np.exp(arr), np.log(arr), np.sqrt(arr)
np.sin(arr), np.cos(arr)
np.clip(arr, 0, 1)
np.cumsum(arr), np.cumprod(arr)
```

---

## 🎲 Random Sampling

```python
np.random.seed(42)
np.random.rand(3)
np.random.randn(3)
np.random.choice(arr, 4)
np.random.permutation(5)
np.random.shuffle(arr)
```

---

## 📊 Statistics

```python
np.median(arr)
np.percentile(arr, 75)
np.quantile(arr, [0.25, 0.5])
np.corrcoef(x, y)
np.cov(x, y)
np.histogram(arr, bins=5)
np.bincount(arr)
```

---

## 🧠 Masking & Logic

```python
arr[arr > 0]
np.where(arr > 0, 1, -1)
np.select([cond1, cond2], [val1, val2])
np.count_nonzero(arr)
np.any(arr > 5)
np.all(arr > 0)
np.isfinite(arr)
np.isnan(arr)
```

---

## 📐 Linear Algebra

```python
np.dot(a, b)               # Dot product
a @ b                      # Matrix multiplication
np.linalg.inv(A)
np.linalg.det(A)
np.linalg.eig(A)
np.linalg.norm(a)
np.linalg.solve(A, b)
np.linalg.svd(X)
np.linalg.pinv(X)
```

---

## 💾 File I/O

```python
np.save("file.npy", arr)
np.load("file.npy")
np.savez("file.npz", x=a, y=b)
np.savetxt("file.txt", arr, delimiter=",")
np.loadtxt("file.txt", delimiter=",")
np.genfromtxt("file.csv", delimiter=",")
```

---

## 🧠 Views vs Copies

```python
view = arr.view()      # Shares memory
copy = arr.copy()      # Independent
view.base is arr       # True
```

---

## 📦 Memory Mapping

```python
np.memmap("data.dat", dtype='float32', mode='w+', shape=(100, 100))
```

---

### ✅ Tip

- Use `.shape`, `.dtype`, `.ndim`, `.nbytes` to inspect arrays
- Broadcasting enables implicit expansion of dimensions during operations

---

© 2025 Satvik Praveen – _NumPyMasterPro_
