#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <random>
#include <chrono>

#include <omp.h>
#include "live_data.hpp"

//stock option parameters
// S = stock price
// X = strike price
// T = time to expiration in years
// r = risk-free interest rate
// sigma = volatility of the stock price
// struct optionParams {
//     double S, X, T, r, sigma;
// };

struct config {
    std::string ticker;
    int thread_count;
    int num_simulations;
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
double monte_carlo_simulation(const optionParams& params, const config& config) {

    std::mt19937_64 rng(42 + omp_get_thread_num());
    std::normal_distribution<double> distribution(0.0, 1.0);
    double sum_payoffs = 0.0;
 
    #pragma omp parallel for num_threads(config.thread_count) reduction(+:sum_payoffs)
    for (int i = 0; i < config.num_simulations; ++i) {
        double Z = distribution(rng);
        //stock price at expiration 
        double ST = params.S * std::exp((params.r - 0.5 * params.sigma * params.sigma) * params.T + params.sigma * std::sqrt(params.T) * Z);
        //if strike price is greater than stock price at expiration, then payoff is zero
        //otherwise, payoff is stock price at expiration minus strike price
        double payoff = std::max(ST - params.X, 0.0); 
        sum_payoffs += payoff;
    }
    // average of payoffs over number of simulations , then discounting back to present value
    return std::exp(-params.r * params.T) * (sum_payoffs / config.num_simulations);
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

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " TICKER THREADS NUM_SIMULATIONS\n";
        return 1;
    }

    std::string ticker = argv[1];
    config options;
    options.thread_count = std::stoi(argv[2]);
    options.num_simulations = std::stoi(argv[3]);

    const char* api_key = std::getenv("TRADIER_API_KEY");
    if (!api_key) {
        std::cerr << "Missing TRADIER_API_KEY environment variable\n";
        return 1;
    }

    double r = 0.05;    // risk-free rate — could fetch later

    std::vector<optionParams> trades;
    try {
        trades = fetch_chain(ticker, api_key, r);
    } catch (const std::exception& e) {
        std::cerr << "Error fetching options: " << e.what() << '\n';
        return 1;
    }

    for (size_t i = 0; i < trades.size(); ++i) {
        const auto& opt = trades[i];
        std::cout << "Option " << i+1 << ": "
                  << "S: " << opt.S << ", X: " << opt.X << ", Expiration date: " << opt.expiration_date << ", T: " << opt.T << ", r: " << opt.r << ", sigma: " << opt.sigma << "\n";

        double analytical = calc_option_price(opt);
        double mc = monte_carlo_simulation(opt, options);

        std::cout << "  Analytical: " << analytical << " | Monte Carlo: " << mc << "\n\n";
    }

    return 0;
}