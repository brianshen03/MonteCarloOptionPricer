# High Performance Computing: European Option Pricer

This project implements a high-performance Monte Carlo simulation to price European call options using real-time market data. It supports both CPU-parallelized (OpenMP) and GPU-accelerated (CUDA) implementations.

## 📈 Overview

The program calculates the fair price of a European call option based on:

- `S`: Stock price (live from Tradier API)
- `X`: Strike price (live from Tradier API)
- `T`: Time to expiration (in years)
- `σ` (sigma): Volatility (live from Tradier API)
- `r`: Risk-free interest rate (live from FRED 3-month Treasury Bill rate)

⚠️ Note: Since `r` is based on the 3-month T-bill rate, this pricing model is most accurate when the option's time to maturity `T` is under 3 months. For longer expirations (e.g. 6 months or 1 year), `r` should ideally be adjusted accordingly.

---

## 🧪 Usage

./pricer TICKER THREADS NUM_SIMULATIONS


- `TICKER`: Stock symbol (e.g., AAPL)
- `THREADS`: Number of CPU threads to parallelize simulations (only applies to the OpenMP version)
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

- Expand GPU acceleration (CUDA) to support pricing thousands of options efficiently
- Implement American option pricing using Least-Squares Monte Carlo (LSMC)
- Integrate variance reduction techniques (e.g., antithetic variates, control variates)
- Add visualization and performance benchmarks comparing OpenMP and CUDA performance

---

## 📎 Author

Feel free to explore the `cuda_port` branch for the latest GPU-accelerated implementation. 