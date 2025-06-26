#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <random>
#include <omp.h>
#include <chrono>


//stock option parameters
// S = stock price
// X = strike price
// T = time to expiration in years
// r = risk-free interest rate
// sigma = volatility of the stock price
struct optionParams {
    double S, X, T, r, sigma;
};

struct config {
    int thread_count;
    int num_simulations;
};

std::vector<optionParams> trades;

#define THREADS 2

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

//helper function to load trades from a CSV file
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
        std::cerr << "Usage: " << argv[0] << " file_name thread_count num_simulations" << std::endl;
        return 1;
    }

    const char *filename = argv[1];
    config options;
    options.thread_count = std::stoi(argv[2]);
    options.num_simulations = std::stoi(argv[3]);
    trades = load_csv(filename);

    for (int i = 0; i < trades.size(); i++) {
        std::cout << "Trade " << i + 1 << ": "
                  << "S: " << trades[i].S << ", "
                  << "X: " << trades[i].X << ", "
                  << "T: " << trades[i].T << ", "
                  << "r: " << trades[i].r << ", "
                  << "sigma: " << trades[i].sigma << std::endl;

        double analytical_price = calc_option_price(trades[i]);
        std::cout << "Analytical Price: " << analytical_price;

        double price = monte_carlo_simulation(trades[i], options);
        std::cout << " MC Price: " << price << "\n\n";
    }

    return 0;
}