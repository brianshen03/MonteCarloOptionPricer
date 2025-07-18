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

struct config {
    std::string ticker;
    int num_simulations = 1000000; // default to 1 million simulations
};

std::vector<optionParams> trades;

//helper function to calculate CDF
double phi(double x) {
    return 0.5 * std::erfc(-x/std::sqrt(2.0));
}

//calculate option price using Black-Scholes formula
double calc_option_price(const optionParams& params) {

    double d1 = (std::log(params.S/params.X) + (params.r + (params.sigma*params.sigma)/2) * params.T)/
                (params.sigma*std::sqrt(params.T));

    double d2 = (std::log(params.S/params.X) + (params.r - (params.sigma*params.sigma)/2) * params.T)/
             (params.sigma*std::sqrt(params.T));

             
    double CDF_d1 = phi(d1);
    double CDF_d2 = phi(d2);
    
    double option_price = params.S * CDF_d1 - params.X * std::exp(-params.r * params.T) * CDF_d2;

    return option_price;

}

// Monte Carlo simulation to estimate the option price
//each cuda thread does a simulation of the option price
__global__ void monte_carlo_simulation(optionParams* trades_pointer, double* trades_results, int num_simulations, unsigned long long seed) {


    //initialize thread index & stride for each thread 
    // int i = blockDim.x * blockId.x + threadId.x;
    int stride = blockDim.x * gridDim.x;
    int optId = blockIdx.x;          // one block → one option

 
    double sum_payoffs = 0.0;
    curandStatePhilox4_32_10_t state;
    curand_init(seed + optId, /*threadId=*/0, 0, &state);

    const auto&p = trades_pointer[optId];


    for (int i = 0; i < num_simulations; ++i) {

        double Z = curand_normal_double(&state);  // one N(0,1) sample

        //stock price at expiration 
        double ST = p.S * std::exp((p.r - 0.5 * p.sigma * p.sigma) * p.T + p.sigma * std::sqrt(p.T) * Z);
        //if strike price is greater than stock price at expiration, then payoff is zero
        //otherwise, payoff is stock price at expiration minus strike price
        double payoff = fmax(ST - p.X, 0.0); 
        sum_payoffs += payoff;
    }
    // average of payoffs over number of simulations , then discounting back to present value
    double average = std::exp(-p.r * p.T) * (sum_payoffs / num_simulations);
    trades_results[optId] = average;
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
    //number of blocks (in a grid)
    int numBlocks = (options.num_simulations + blockSize - 1) / blockSize;

        //allocate memory on GPU and copy data from CPU to GPU
        optionParams* trades_pointer;
        cudaMallocManaged(&trades_pointer, trades.size() * sizeof(optionParams));
        std::memcpy(trades_pointer, trades.data(), sizeof(optionParams) * trades.size());

        //allocate memory on GPU for results of each simulation 
        double *trades_results;
        cudaMallocManaged(&trades_results, trades.size() * sizeof(double));

        monte_carlo_simulation <<<1, 1>>> (trades_pointer, trades_results, options.num_simulations,  /*seed=*/1234ULL);
        cudaDeviceSynchronize(); 


        for (size_t i = 0; i < trades.size(); ++i) {
            const auto& opt = trades[i];
            std::cout << "Option " << i+1 << ": " << "S: " << opt.S << ", X: " << opt.X << ", Expiration date: " << 
            opt.expiration_date << ", T: " << opt.T << ", r: " << opt.r << ", sigma: " << opt.sigma << "\n";

            double analytical = calc_option_price(opt);
            // double mc = monte_carlo_simulation(opt, options);

            std::cout << "  Analytical: " << analytical << " | Monte Carlo: " << trades_results[i] << "\n\n";
        }

        cudaFree(trades_pointer);
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
        std::string_view a = argv[i];

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

    std::cout << "[DEBUG] Trades fetched: " << trades.size() << '\n';

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
