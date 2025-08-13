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

struct config {
    std::string ticker;
    int num_simulations = 1000000; // default to 1 million simulations
};

struct optionPrices {
    double call_price;
    double put_price;
};

std::vector<optionParams> trades;

//helper function to calculate CDF
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

// Monte Carlo simulation to estimate the option price
//each cuda thread does a simulation of the option price
__global__ void monte_carlo_simulation(const OptionGPU* trades_pointer, optionPrices* trades_results, int num_simulations, unsigned long long seed) {

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

//optional function to load trades from a CSV file
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

static void run_pricer(const std::vector<optionParams>&trades, const config& options) {

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
    
    monte_carlo_simulation <<<numBlocks, blockSize, shmem>>> (d_trades, trades_results, options.num_simulations,  /*seed=*/1234ULL);


    cudaDeviceSynchronize(); 


    for (size_t i = 0; i < trades.size(); ++i) {
        const auto& opt = trades[i];
        std::cout << "Option " << i+1 << ": " << "S: " << opt.S << ", X: " << opt.X << ", Expiration date: " << 
        opt.expiration_date << ", T: " << opt.T << ", r: " << opt.r << ", sigma: " << opt.sigma << "\n";

        optionPrices eu_bs = calc_call(opt);

        std::cout << " Analytical call price: " << eu_bs.call_price << " | Monte Carlo call price: " << trades_results[i].call_price << "\n";
        std::cout << " Analytical put price: " << eu_bs.put_price << " | Monte Carlo put price: " << trades_results[i].put_price << "\n\n";

    }

    cudaFree(d_trades);
    cudaFree(trades_results);


}

config parse_cmd_args(int argc, char *argv[]) {
    config c;
    auto need = [&](int& i) -> char* {
        //value for flag not provided
        if (++i == argc)
            throw std::runtime_error("missing value for " + std::string(argv[i-1]));
        return argv[i];
    };

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        if      (a == "--symbol")  c.ticker  = need(i);
        else if (a == "--paths")   c.num_simulations   = std::stol(need(i));
        else if (a == "--help") {
            std::cout << "Usage: ./pricer --symbol TICKER --paths N\n";
            std::cout << "if --paths is not specified, default is 1 million paths\n";
            std::exit(0);
        }
        else
            throw std::runtime_error("unknown flag: " + std::string(a));
    }
    if (c.ticker.empty())
        throw std::runtime_error("--symbol is required (try --help)");
    return c;
}

int main(int argc, char *argv[]) {

    config options;
    try {
        options = parse_cmd_args(argc, argv);
    }
    catch (const std::exception& e) {
        std::cerr << "Error parsing command line arguments: " << e.what() << std::endl;
        return 1;
    }

    double r = 0.0;
    try {
        r = fetch_risk_free_rate();
        std::cout << "Risk-free rate used (DGS3MO): " << r << "\n\n";
    } catch (const std::exception& e) {
        std::cerr << "Error in fetch risk free rate " << e.what() << std::endl;  
    }
    
    std::vector<optionParams> trades;
    try {
        trades = fetch_chain(options.ticker, r); 
    } catch (const std::exception& e) {
    std::cerr << "Error in fetch chain: " << e.what() << std::endl; 
    }

    auto start = std::chrono::steady_clock::now();
    run_pricer(trades, options);
    auto stop = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = stop - start;   


    double seconds = elapsed.count();          
    double throughput = options.num_simulations / seconds;          


    std::cout << "Simulation Summary\n"
     << "------------------\n" 
     << "Contracts processed: " << trades.size() << " options in " << seconds << " seconds \n"
    << "Paths per contract: " << options.num_simulations << "\n"
    << "Throughput: " << throughput << " paths/second\n";


    

    return 0;
}
