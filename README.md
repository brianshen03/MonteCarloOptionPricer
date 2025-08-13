# High Performance Computing: European Option Pricer (CUDA branch)

## 🧪 Usage

.pricer.exe TICKER NUM_SIMULATIONS

- `TICKER`: Stock symbol (e.g., AAPL)
- `NUM_SIMULATIONS`: Number of Monte Carlo paths to generate

💡 Currently, the simulations are parallelized per option, not per stock. So the performance benefits scale better with fewer options and more simulations.  
Example:  
- ✅ 20 options × 10M simulations → Good parallelism  
- ❌ 5000 options × 100 simulations → Poor GPU/CPU utilization

---

## ⚙️ Project Structure

There are two branches:

- `main` — CPU version using OpenMP  
- `cuda_port` — GPU version using CUDA  

Each branch contains the same interface and functionality, but leverages different hardware acceleration models for comparison and performance benchmarking.

---

## 📦 APIs Used

- Tradier API - Options Chain: https://documentation.tradier.com/brokerage-api/markets/get-options-chains  
- Tradier API - Stock Quotes: https://documentation.tradier.com/brokerage-api/markets/get-quotes  
- FRED API - 3-Month Treasury Bill Rate: https://fred.stlouisfed.org/docs/api/fred/

---

## 🚀 Future Plans

- Expand GPU acceleration (CUDA) to support US Monte Carlo Sim
- Add visualization and performance benchmarks comparing OpenMP and CUDA performance

---
