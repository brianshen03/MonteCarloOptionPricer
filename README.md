# CUDA Branch

## 📈 Overview

This branch runs the same pricing logic as the OpenMP version (European via Monte Carlo; American via Longstaff–Schwartz), but executes the simulations on an NVIDIA GPU using CUDA kernels.

- Monte Carlo path simulation and LSM regression steps are executed on the GPU.
- The OpenMP `THREADS` argument is **not used** here.
- Default **threads-per-block (TPB)** is **256**; the grid size is computed from `NUM_SIMULATIONS`.
- Note you need a **NVIDIA GPU with CUDA support** (desktop/laptop or cloud instance) to run this 

---

## 🧪 Usage

```bash
.pricer.exe --symbol TICKER --paths NUM_SIMULATIONS
```

Parameters:
- `TICKER`: Stock symbol (e.g., AAPL)
- `NUM_SIMULATIONS`: Number of Monte Carlo paths to generate