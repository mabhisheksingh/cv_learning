import numpy as np
import time

from sympy.sets.handlers import functions

n = 1_000_000
weight_array = list(range(1, n + 1))
feature_array = list(range(1, n + 1))
weight_vector = np.asarray(weight_array, dtype=np.float64)
feature_vector = np.asarray(feature_array, dtype=np.float64)

runs = 5

# --- Python loop ---
loop_times = []
for _ in range(runs):
    prediction = 0.0
    start = time.perf_counter()
    for i in range(n):
        prediction += feature_array[i] * weight_array[i]
    end = time.perf_counter()
    loop_times.append((end - start) * 1000)

print(f"Python loop   | result: {prediction:.0f} | avg time: {sum(loop_times)/len(loop_times):.3f} ms")

# --- NumPy dot ---
np_times = []
for _ in range(runs):
    start = time.perf_counter()
    prediction = np.dot(weight_vector, feature_vector)
    end = time.perf_counter()
    np_times.append((end - start) * 1000)

print(f"NumPy dot     | result: {prediction:.0f} | avg time: {sum(np_times)/len(np_times):.3f} ms")

speedup = (sum(loop_times)/len(loop_times)) / (sum(np_times)/len(np_times))
print(f"\nNumPy is ~{speedup:.1f}x faster than the Python loop for {n:,} elements")


# sklearn, numpy functions and

