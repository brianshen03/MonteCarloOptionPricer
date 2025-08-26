#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include "live_data.hpp"
#include <curand_kernel.h>


//stock option parameters
// S = stock price
// X = strike price
// T = time to expiration in years
// r = risk-free interest rate
// sigma = volatility of the stock price
struct OptionGPU {
    double S, X, T, r, sigma;
};

enum class DataSource { Live, CSV };


struct config {
    std::string ticker;
    int num_simulations = 1000000; // default to 1 million simulations
    DataSource  source = DataSource::Live;
    std::optional<std::string> csv_path; // required for CSV
};

struct optionPrices {
    double call_price;
    double put_price;
};

#define TIME_STEPS 20 // number of time steps for American option pricing
#define min_dte_days 30 // minimum days to expiration
#define max_dte_days 180 // maximum days to expiration
#define max_expiries 3 // maximum number of expiries to fetch


std::vector<optionParams> trades;

//helper function to calculate CDF


#define CUDA_CHECK(expr) do { \
  cudaError_t _e = (expr); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    std::abort(); \
  } \
} while(0)

#define KERNEL_OK() do { \
  CUDA_CHECK(cudaPeekAtLastError()); \
  CUDA_CHECK(cudaDeviceSynchronize()); \
} while(0)

double phi(double x) {
    return 0.5 * std::erfc(-x/std::sqrt(2.0));
}

//calculate option price using Black-Scholes formula
optionPrices calc_call(const optionParams& params) {

    double d1 = (std::log(params.S/params.X) + (params.r + (params.sigma*params.sigma)/2) * params.T)/
                (params.sigma*std::sqrt(params.T));

    double d2 = (std::log(params.S/params.X) + (params.r - (params.sigma*params.sigma)/2) * params.T)/
             (params.sigma*std::sqrt(params.T));

             
    double CDF_d1_c = phi(d1);
    double CDF_d2_c = phi(d2);
    double CDF_d1_p = phi(-d1);
    double CDF_d2_p = phi(-d2);
    
    double call = params.S * CDF_d1_c - params.X * std::exp(-params.r * params.T) * CDF_d2_c;
    double put = params.X * std::exp(-params.r * params.T) * CDF_d2_p - params.S * CDF_d1_p;
    optionPrices option_price = {call, put};
    return option_price;

}

int validation_check(double price1, double price2) {
    int pass = 0;
    const double rel_tol = 0.02;   
    const double eps     = 1e-12;  // guard for zero BS
    const double abs_tol = 0.02;

    double abs_err = std::abs(price1 - price2);
    double rel_err = abs_err / std::max(std::abs(price1), eps);

    if (rel_err <= rel_tol || abs_err <= abs_tol) {
        std::cout << "✅ within tol\n";
    } else {
        std::cout << "❌ outside tol\n";
        pass = 1;
    }

    return pass;
}

// Cox-Ross-Rubinstein (CRR) binomial tree pricing method for validation check 
optionPrices crr_price(const optionParams &params, int time_steps, double q) {

    const double dt = params.T / time_steps;
    const double disc = std::exp(-params.r * dt);
    const double u = std::exp(params.sigma * std::sqrt(dt));
    const double d = 1.0 / u;
    const double a = std::exp((params.r - q) * dt);
    const double p = (a - d) / (u - d);

    if (!(p > 0.0 && p < 1.0)) {
        // CRR requires 0<p<1; if violated, increase N or adjust inputs.
        throw std::runtime_error("CRR probability out of (0,1); increase N or adjust params");
    }

    std::vector<double> call_prices(time_steps + 1);
    std::vector<double> put_prices(time_steps + 1);

    double Sjd = params.S * std::pow(d, time_steps);
    const double ud = (u/d);
    for (int j = 0; j <= time_steps; ++j) {
        call_prices[j] = std::max(Sjd - params.X, 0.0);
        put_prices[j] = std::max(params.X - Sjd, 0.0);
        Sjd *= ud; 
    }

    for (int i = time_steps - 1; i >= 0; --i) {
        double S_i = params.S * std::pow(d, i);
        for (int j = 0; j <= i; ++j) {
            const double cont_call = disc * (p * call_prices[j + 1] + (1.0 - p) * call_prices[j]);
            const double cont_put  = disc * (p * put_prices[j + 1] + (1.0 - p) * put_prices[j]);

            const double intr_call = std::max(S_i - params.X, 0.0);
            const double intr_put  = std::max(params.X - S_i, 0.0);

            call_prices[j] = std::max(intr_call, cont_call); // American step
            put_prices[j] = std::max(intr_put,  cont_put);

            S_i *= ud; // move across the level
        }
    }

    return {call_prices[0], put_prices[0]};
}
// Monte Carlo simulation to estimate the option price
//each cuda thread does a simulation of the option price
__global__ void eu_monte_carlo_simulation(const OptionGPU* trades_pointer, optionPrices* trades_results, int num_simulations, unsigned long long seed) {

    //initialize thread index & stride for each thread 
    int tid = threadIdx.x;
    int stride = blockDim.x;
    int optId = blockIdx.x;         

 
    double call_sum = 0.0;
    double put_sum = 0.0;
    curandStatePhilox4_32_10_t state;
    curand_init(seed + optId, tid, 0, &state);

    const OptionGPU p = trades_pointer[optId];


    const double drift     = (p.r - 0.5 * p.sigma * p.sigma) * p.T;
    const double diffusion = p.sigma * sqrt(p.T);

    for (int i = tid; i < num_simulations; i+=stride) {

        double Z = curand_normal_double(&state); 
        //stock price at expiration 
        double ST = p.S * exp(drift + diffusion * Z);
        //if strike price is greater than stock price at expiration, then payoff is zero
        //otherwise, payoff is stock price at expiration minus strike price (For call option)
        call_sum += fmax(ST - p.X, 0.0); 
        put_sum += fmax(p.X - ST, 0.0);
    }

    extern __shared__ double sdata[];  
    double* call_buf = sdata;
    double* put_buf  = sdata + blockDim.x;
    call_buf[tid] = call_sum;
    put_buf[tid]  = put_sum;
    __syncthreads(); 

    for (int offset = stride >> 1; offset > 0; offset >>= 1) {
    if (tid < offset)       
        {
                call_buf[tid] += call_buf[tid + offset];
                put_buf[tid]  += put_buf[tid + offset];
        }         
    __syncthreads();   

    }
    if (tid == 0) {                    
        // average of payoffs over number of simulations , then discounting back to present value
        trades_results[optId].call_price = exp(-p.r * p.T) * call_buf[0] / static_cast<double>(num_simulations);
        trades_results[optId].put_price = exp(-p.r * p.T) * put_buf[0] / static_cast<double>(num_simulations);
    }
}

//STEP 1: generate all paths for each option
__global__ void gen_paths(OptionGPU p, int N, int M, unsigned long long seed, double* __restrict__ Sgrid) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;                  

    curandStatePhilox4_32_10_t state;
    curand_init(seed, i, 0, &state);
    const double dt    = p.T / M;
    const double drift = (p.r - 0.5 * p.sigma * p.sigma) * dt;
    const double vol   = p.sigma * sqrt(dt);
    double St = p.S;
    Sgrid[0ull * N + i] = St;            // t=0


    for (int t = 1; t <= M; ++t) {
        double Z = curand_normal_double(&state);
        St *= exp(drift + vol * Z);
        Sgrid[(size_t)t * N + i] = St;   
    }
}

//STEP 2 : calculate cash_flow at maturity 
__global__ void terminal_payoffs(const double* __restrict__ S_T,double X, int N,double* __restrict__ cf_call, double* __restrict__ cf_put)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const double ST = S_T[i];
    cf_call[i] = fmax(ST - X, 0.0);
    cf_put[i]  = fmax(X - ST, 0.0);
}

// BEGINNING OF STEP 3: Backward induction to calculate option price

//creating regression statistics 
__global__ void masked_stats_call(const double* __restrict__ S_t, const double* __restrict__ cf,  double K, int N, double disc, double* __restrict__ stats)      // 8 doubles (initialized to 0 before launch)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double X = S_t[i];
    if (X > K) {
        double Y  = disc * cf[i];
        double X2 = X * X;
        atomicAdd(&stats[0], 1.0);
        atomicAdd(&stats[1], X);
        atomicAdd(&stats[2], X2);
        atomicAdd(&stats[3], X2 * X);
        atomicAdd(&stats[4], X2 * X2);
        atomicAdd(&stats[5], Y);
        atomicAdd(&stats[6], X * Y);
        atomicAdd(&stats[7], X2 * Y);
    }
}

//creating regression statistics 
__global__ void masked_stats_put( const double* __restrict__ S_t,const double* __restrict__ cf,   double K, int N, double disc, double* __restrict__ stats)      // 8 doubles (initialized to 0 before launch)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double X = S_t[i];
    if (X < K) {
        double Y  = disc * cf[i];
        double X2 = X * X;
        atomicAdd(&stats[0], 1.0);
        atomicAdd(&stats[1], X);
        atomicAdd(&stats[2], X2);
        atomicAdd(&stats[3], X2 * X);
        atomicAdd(&stats[4], X2 * X2);
        atomicAdd(&stats[5], Y);
        atomicAdd(&stats[6], X * Y);
        atomicAdd(&stats[7], X2 * Y);
    }
}

// func to perform polynomial regression 
__global__ void solve_beta_from_stats( const double* __restrict__ stats, double* __restrict__ beta, int* __restrict__ has_beta)       
{
    // single thread
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    double n  = stats[0];
    double S1 = stats[1], S2 = stats[2], S3 = stats[3], S4 = stats[4];
    double T0 = stats[5], T1 = stats[6], T2 = stats[7];

    if (n < 3.0) { // not enough points for quadratic
        *has_beta = 0;
        beta[0] = beta[1] = beta[2] = 0.0;
        return;
    }

    // Build normal equations G * beta = h, where G = A^T A, h = A^T b
    double G00 = n;
    double G10 = S1,  G11 = S2;
    double G20 = S2,  G21 = S3,  G22 = S4;

    double h0 = T0, h1 = T1, h2 = T2;

    // Cholesky factorization G = L * L^T (3x3)
    // L lower-triangular: [l00 0   0; l10 l11 0; l20 l21 l22]
    auto safe_sqrt = [](double x)->double { return (x > 0.0 ? sqrt(x) : 0.0); };
    // __device__ inline double safe_sqrt_d(double x) { return x > 0.0 ? sqrt(x) : 0.0; }


    double l00 = safe_sqrt(G00);
    if (l00 <= 0.0) { *has_beta = 0; beta[0]=beta[1]=beta[2]=0.0; return; }

    double l10 = G10 / l00;
    double l20 = G20 / l00;

    double a11 = G11 - l10*l10;
    double a21 = G21 - l20*l10;
    double a22 = G22 - l20*l20;

    double l11 = safe_sqrt(a11);
    if (l11 <= 0.0) { *has_beta = 0; beta[0]=beta[1]=beta[2]=0.0; return; }

    double l21 = a21 / l11;

    double a22_ = a22 - l21*l21;
    double l22 = safe_sqrt(a22_);
    if (l22 <= 0.0) { *has_beta = 0; beta[0]=beta[1]=beta[2]=0.0; return; }

    // Solve L * y = h  (forward)
    double y0 = h0 / l00;
    double y1 = (h1 - l10*y0) / l11;
    double y2 = (h2 - l20*y0 - l21*y1) / l22;

    // Solve L^T * beta = y  (backward)
    double b2 = y2 / l22;
    double b1 = (y1 - l21*b2) / l11;
    double b0 = (y0 - l10*b1 - l20*b2) / l00;

    beta[0] = b0; beta[1] = b1; beta[2] = b2;
    *has_beta = 1;
}

// comparing continuation value with immediate payoff 
__global__ void update_cf_call( const double* __restrict__ S_t, double K, int N, double disc, const double* __restrict__ beta, const int* __restrict__ has_beta, double* __restrict__ cf) // in-place update
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double X = S_t[i];
    double imm = fmax(X - K, 0.0);

    if (imm > 0.0) {
        if (*has_beta) {
            double cont = beta[0] + beta[1]*X + beta[2]*X*X;
            cf[i] = (imm > cont) ? imm : (disc * cf[i]);
        } else {
            cf[i] = disc * cf[i]; // no regression this step
        }
    } else {
        cf[i] = disc * cf[i];     // OTM: just discount
    }
}

//comparing continuation value with immediate payoff 
__global__ void update_cf_put( const double* __restrict__ S_t,double K, int N, double disc,  const double* __restrict__ beta, const int* __restrict__ has_beta, double* __restrict__ cf) // in-place update
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double X = S_t[i];
    double imm = fmax(K - X, 0.0);

    if (imm > 0.0) {
        if (*has_beta) {
            double cont = beta[0] + beta[1]*X + beta[2]*X*X;
            cf[i] = (imm > cont) ? imm : (disc * cf[i]);
        } else {
            cf[i] = disc * cf[i];
        }
    } else {
        cf[i] = disc * cf[i];
    }
}

//END OF STEP 3 functions 
//master function call all of the kernels 
optionPrices us_mc_cuda_lsm(const optionParams& P, const config& C, int M)
{
    const int    N    = C.num_simulations;
    const double dt   = P.T / M;
    const double disc = std::exp(-P.r * dt);

    // device buffers
    double *all_paths = nullptr, *d_cf_call = nullptr, *d_cf_put = nullptr;
    CUDA_CHECK(cudaMalloc(&all_paths,  (size_t)(M + 1) * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cf_call, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cf_put,  N * sizeof(double)));

    // launch config
    const int block = 256;
    const int grid  = (N + block - 1) / block;

    // Path generation
    gen_paths<<<grid, block>>>(OptionGPU{P.S, P.X, P.T, P.r, P.sigma}, N, M, /*seed=*/1234ULL, all_paths);
    KERNEL_OK();  

    // Terminal payoffs (t = M)
    const double* d_S_T = all_paths + (size_t)M * N;
    terminal_payoffs<<<grid, block>>>(d_S_T, P.X, N, d_cf_call, d_cf_put);
    KERNEL_OK();  

    // scratch for stats/beta per side
    double *d_stats_call=nullptr, *d_stats_put=nullptr, *d_beta_call=nullptr, *d_beta_put=nullptr;
    int *d_has_beta_call=nullptr, *d_has_beta_put=nullptr;
    CUDA_CHECK(cudaMalloc(&d_stats_call, 8*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_stats_put,  8*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_beta_call,  3*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_beta_put,   3*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_has_beta_call, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_has_beta_put,  sizeof(int)));

    //  Backward induction t = M-1 .. 1
    for (int t = M-1; t >= 1; --t) {
        const double* d_S_t = all_paths + (size_t)t * N;

        // ---- CALL ----
        CUDA_CHECK(cudaMemset(d_stats_call, 0, 8*sizeof(double)));
        masked_stats_call<<<grid, block>>>(d_S_t, d_cf_call, P.X, N, disc, d_stats_call);
        KERNEL_OK();
        solve_beta_from_stats<<<1,1>>>(d_stats_call, d_beta_call, d_has_beta_call);
        KERNEL_OK();
        update_cf_call<<<grid, block>>>(d_S_t, P.X, N, disc, d_beta_call, d_has_beta_call, d_cf_call);
        KERNEL_OK();

        // ---- PUT ----
        CUDA_CHECK(cudaMemset(d_stats_put, 0, 8*sizeof(double)));
        masked_stats_put<<<grid, block>>>(d_S_t, d_cf_put, P.X, N, disc, d_stats_put);
        KERNEL_OK();
        solve_beta_from_stats<<<1,1>>>(d_stats_put, d_beta_put, d_has_beta_put);
        KERNEL_OK();
        update_cf_put<<<grid, block>>>(d_S_t, P.X, N, disc, d_beta_put, d_has_beta_put, d_cf_put);
        KERNEL_OK();
    }

    //  Average discounted cash flows at t=0
    std::vector<double> h_call(N), h_put(N);
    CUDA_CHECK(cudaMemcpy(h_call.data(), d_cf_call, N*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_put.data(),  d_cf_put,  N*sizeof(double), cudaMemcpyDeviceToHost));

    double sum_c = 0.0, sum_p = 0.0;
    for (int i = 0; i < N; ++i) { 
        sum_c += h_call[i]; 
        sum_p += h_put[i];
    }

    optionPrices option_price = { sum_c / N, sum_p / N };

    cudaFree(all_paths);
    cudaFree(d_cf_call);
    cudaFree(d_cf_put);
    cudaFree(d_stats_call);
    cudaFree(d_stats_put);
    cudaFree(d_beta_call);
    cudaFree(d_beta_put);
    cudaFree(d_has_beta_call);
    cudaFree(d_has_beta_put);

    return option_price;
}

static void run_us_pricer(const std::vector<optionParams>& trades, const config& cfg, int M)
{
    int pass_call = 0;
    int pass_put = 0;
    for (size_t i = 0; i < trades.size(); ++i) {
        const auto& opt = trades[i];

        std::cout << "Option " << i+1 << ": " << "S: " << opt.S << ", X: " << opt.X << ", Expiration date: " << 
        opt.expiration_date << ", T: " << opt.T << ", r: " << opt.r << ", sigma: " << opt.sigma << "\n";

        optionPrices mc = us_mc_cuda_lsm(opt, cfg, M);
        optionPrices eu_bs = calc_call(opt); 
        optionPrices crr = crr_price(opt, 1500, 0.0);

         std::cout << " BS call price: " << eu_bs.call_price << " | American Monte Carlo call price: " << mc.call_price;
        int pass = validation_check(eu_bs.call_price, mc.call_price);
        if (pass == 0) ++pass_call;

        std::cout << " Cox-Ross-Rubinstein put price: " << crr.put_price << " | American Monte Carlo put price: " << mc.put_price;
        pass = validation_check(crr.put_price, mc.put_price);
        if (pass == 0) ++pass_put;

        std::cout << "\n";
    }

    std::cout << pass_call << " call prices passed out of " << trades.size() << "\n";
    std::cout << pass_put  << " put prices passed out of " << trades.size() << "\n";
}

static void run_eu_pricer(const std::vector<optionParams>&trades, const config& options) {

    //num threads
    int blockSize = 256;
    //number of blocks (in a grid) (one per option)
    int numBlocks = trades.size(); 
    size_t shmem    = 2 * blockSize * sizeof(double);

    std::vector<OptionGPU> gpuTrades(trades.size());
    for (size_t i=0; i<trades.size(); ++i) {
        gpuTrades[i] = { trades[i].S, trades[i].X, trades[i].T, trades[i].r, trades[i].sigma };
    }

    //allocate memory on GPU and copy data from CPU to GPU
    OptionGPU* d_trades;
    cudaMallocManaged(&d_trades, gpuTrades.size()*sizeof(OptionGPU));
    cudaMemcpy(d_trades, gpuTrades.data(),gpuTrades.size()*sizeof(OptionGPU),cudaMemcpyHostToDevice);

    //allocate memory on GPU for results of each simulation 
    optionPrices *trades_results;
    cudaMallocManaged(&trades_results, trades.size() * sizeof(optionPrices));
    
    eu_monte_carlo_simulation <<<numBlocks, blockSize, shmem>>> (d_trades, trades_results, options.num_simulations,  /*seed=*/1234ULL);

    cudaDeviceSynchronize(); 

    int pass_call = 0;
    int pass_put = 0;
    for (size_t i = 0; i < trades.size(); ++i) {
        const auto& opt = trades[i];
        std::cout << "Option " << i+1 << ": " << "S: " << opt.S << ", X: " << opt.X << ", Expiration date: " << 
        opt.expiration_date << ", T: " << opt.T << ", r: " << opt.r << ", sigma: " << opt.sigma << "\n";

        optionPrices eu_bs = calc_call(opt);

        std::cout << " Analytical call price: " << eu_bs.call_price << " | European Monte Carlo call price: " << trades_results[i].call_price;

        int pass = validation_check(eu_bs.call_price, trades_results[i].call_price  );
        if (pass == 0) ++pass_call;

        std::cout << " Analytical put price: " << eu_bs.put_price << " | European Monte Carlo put price: " << trades_results[i].put_price;
        pass = validation_check(eu_bs.put_price, trades_results[i].put_price);
        if (pass == 0) ++pass_put;

        std::cout << "\n";

    }

    std::cout << pass_call << " call prices passed out of " << trades.size() << "\n";
    std::cout << pass_put  << " put prices passed out of " << trades.size() << "\n";

    cudaFree(d_trades);
    cudaFree(trades_results);

}

static void print_usage() {
    std::cout <<
        "Usage:\n"
        "  pricer --symbol TICKER [--paths N] \n"
        "  pricer --csv FILE       [--paths N] \n"
        "\n"
        "Notes:\n"
        "  - If --csv is provided, data is loaded from FILE and --symbol is ignored.\n"
        "  - Defaults: --paths 1000000, --threads 4.\n";
}

config parse_cmd_args(int argc, char* argv[]) {
    config c; // defaults set in struct

    auto need = [&](int& i) -> char* {
        if (++i == argc)
            throw std::runtime_error("missing value for " + std::string(argv[i-1]));
        return argv[i];
    };

    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];

        if      (a == "--symbol")   c.ticker          = need(i);
        else if (a == "--paths")    c.num_simulations = std::stol(need(i));
        else if (a == "--csv") {
            c.source   = DataSource::CSV;
            c.csv_path = std::string(need(i));
        }
        else if (a == "--help" || a == "-h") {
            print_usage();
            std::exit(0);
        }
        else {
            throw std::runtime_error("unknown flag: " + std::string(a));
        }
    }

    // Validate
    if (c.source == DataSource::CSV) {
        if (!c.csv_path || c.csv_path->empty())
            throw std::runtime_error("--csv requires a path (try --help)");
        // When CSV is used, ticker is optional; ignore if present.
    } else { // Live
        if (c.ticker.empty())
            throw std::runtime_error("--symbol is required for live fetch (try --help)");
    }
    if (c.num_simulations <= 0)
        throw std::runtime_error("--paths must be positive");

    return c;
}

std::vector<optionParams> load_csv(const std::string& filename) {
    std::vector<optionParams> trades;
    std::ifstream in(filename);
    std::string line;

    if (!in.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return trades;
    }

    while (std::getline(in, line)) {
        std::istringstream ss(line);
        optionParams trade;
        char comma;
        ss >> trade.S >> comma >> trade.X >> comma >> trade.T >> comma >> trade.r >> comma >> trade.sigma;
        trades.push_back(trade);
    }
    return trades;
}

int main(int argc, char *argv[]) {

    config options;
    try {
        options = parse_cmd_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Argument error: " << e.what() << "\n";
        return 2;
    }

    double r = 0.0; // live daily rate (only needed for live fetch)

    if (options.source == DataSource::Live) {
        try {
            r = fetch_risk_free_rate();
        } catch (const std::exception& e) {
            std::cerr << "Error fetching risk-free rate: " << e.what() << "\n";
            return 1;
        }
        std::cout << "Risk-free rate used (DGS3MO): " << r << "\n\n";
    }

    std::vector<optionParams> trades;
    try {
        if (options.source == DataSource::CSV) {
            trades = load_csv(*options.csv_path);
        } else {
            trades = fetch_chain(options.ticker, r, min_dte_days, max_dte_days, max_expiries, true, true);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error obtaining chain data: " << e.what() << "\n";
        return 1;
    }

    auto start = std::chrono::steady_clock::now();
    // run_eu_pricer(trades, options);
        run_us_pricer(trades, options, TIME_STEPS);


    auto stop = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = stop - start;   



    double seconds = elapsed.count();          
    double throughput = options.num_simulations / seconds;          


    std::cout << "Simulation Summary\n"
     << "------------------\n" 
     << "Contracts processed: " << trades.size() << " options in " << seconds << " seconds \n"
    << "Paths per contract: " << options.num_simulations << "\n"
    << "Time steps (for US) : " << TIME_STEPS << "\n"
    << "Throughput: " << throughput << " paths/second\n";


    

    return 0;
}
