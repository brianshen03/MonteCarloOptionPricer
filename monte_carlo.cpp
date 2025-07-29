#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <random>
#include <chrono>
#include <Eigen/Dense>
#include <omp.h>
#include "live_data.hpp"

//stock option parameters
// S = stock price
// X = strike price
// T = time to expiration in years
// r = risk-free interest rate
// sigma = volatility of the stock price

struct config {
    std::string ticker;
    int thread_count = 4; // default to 4 threads
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

// Monte Carlo simulation to estimate the european option price
double eu_monte_carlo_simulation(const optionParams& params, const config& config) {

    double sum_payoffs = 0.0;
    #pragma omp parallel num_threads(config.thread_count)
    {
        // each thread gets its own RNG & distribution
        std::mt19937_64 rng(42 + omp_get_thread_num());
        std::normal_distribution<double> dist(0.0, 1.0);

        double local_sum = 0.0;

        #pragma omp for
        for (int i = 0; i < config.num_simulations; ++i) {
            double Z  = dist(rng);
            double drift = (params.r - 0.5 * params.sigma * params.sigma) * params.T;
            double diffusion = params.sigma * std::sqrt(params.T);
            double ST = params.S * std::exp(drift + diffusion * Z);

            // double ST = params.S * std::exp((params.r - 0.5 * params.sigma*params.sigma)*params.T + params.sigma*std::sqrt(params.T)*Z);
            local_sum += std::max(ST - params.X, 0.0);
        }
        // reduction by hand (atomic or critical) or use OpenMP reduction on local_sum
        #pragma omp atomic
        sum_payoffs += local_sum;
    }

    double price = std::exp(-params.r * params.T) * (sum_payoffs / config.num_simulations);
    return price;
}

//helper func to perform polynomial regression
Eigen::VectorXd perform_regression(const std::vector<double>& X, const std::vector<double>& Y) {
    int N = X.size();
    Eigen::MatrixXd A(N, 3);
    Eigen::VectorXd b(N);

    for (int i = 0; i < N; ++i) {
        A(i, 0) = 1.0;
        A(i, 1) = X[i];
        A(i, 2) = X[i] * X[i];
        b(i) = Y[i];
    }

    // Solve for beta using least squares
    Eigen::VectorXd beta = (A.transpose() * A).ldlt().solve(A.transpose() * b);
    return beta; // beta[0] = b0, beta[1] = b1, beta[2] = b2
}

// monte carlo simulation to estimate the american option price (M is time steps)
double us_monte_carlo_simulation(const optionParams& params, const config& config, int M) {

    //2d array to store stock prices at each time step
    std::vector<std::vector<double>> all_paths(config.num_simulations, std::vector<double>(M+1));

    std::mt19937_64 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    double dt = params.T / M; // time step size
    double drift = (params.r - 0.5 * params.sigma * params.sigma) * dt;

    // step 1: generate all paths
    for (int i = 0; i < config.num_simulations; ++i) {
        all_paths[i][0] = params.S;
        //saving each stock price from 1 to M time paths 
        for (int t = 1; t <= M; ++t) {
            double Z = dist(rng);
            double diffusion = params.sigma * std::sqrt(dt) * Z;
            all_paths[i][t] = all_paths[i][t-1] * std::exp(drift + diffusion);
        }

    }

    // step 2: calculate cash_flow (payoff) at maturity
    std::vector<double> cash_flow(config.num_simulations);
    for (int i = 0; i < config.num_simulations; ++i) {
        double ST = all_paths[i][M];
        cash_flow[i] = std::max(ST - params.X, 0.0); // final payoff
    }

    // step 3: backward induction to calculate option price
    std::vector<int> itm_indices;
    for (int t = M - 1; t >= 1; --t) {
        // find in-the-money paths at time t
        for (int i = 0; i < config.num_simulations; ++i) {
            double ST = all_paths[i][t];
            if (ST > params.X) { 
                itm_indices.push_back(i); // in-the-money paths
            }
        }

        // build regression input 
        std::vector<double> X, Y;
        for (int i : itm_indices) {
            double ST = all_paths[i][t];
            double discounted_cash_flow = cash_flow[i] * std::exp(-params.r * dt);
            X.push_back(ST);
            Y.push_back(discounted_cash_flow);
        }

        // need at least 3 points for regression
        if (X.size() < 3) continue;

        // perform regression to estimate continuation value
        Eigen::VectorXd beta = perform_regression(X, Y);

        //compare continuation value with immediate payoff
        for (int i : itm_indices) {
            double ST = all_paths[i][t];
            double continuation_value = beta[0] + beta[1] * ST + beta[2] * ST * ST;
            double immediate_payoff = std::max(ST - params.X, 0.0);

            if (immediate_payoff > continuation_value) {
                cash_flow[i] = immediate_payoff; // exercise option
            } else {
                cash_flow[i] = std::exp(-params.r * dt) * cash_flow[i];
            }
        }
    }

    // step 4: average discounted cash flows
    double sum_payoffs = 0.0;
    for (int i = 0; i < config.num_simulations; ++i) {
        sum_payoffs += cash_flow[i];
    }
    double price = sum_payoffs / config.num_simulations;

    return price;
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

        for (size_t i = 0; i < trades.size(); ++i) {
            const auto& opt = trades[i];
            std::cout << "Option " << i+1 << ": " << "S: " << opt.S << ", X: " << opt.X << ", Expiration date: " << 
            opt.expiration_date << ", T: " << opt.T << ", r: " << opt.r << ", sigma: " << opt.sigma << "\n";

            double analytical = calc_option_price(opt);
            // double eu_mc = eu_monte_carlo_simulation(opt, options);
            double us_mc = us_monte_carlo_simulation(opt, options, 50);

            // std::cout << "  Analytical: " << analytical << " | European Monte Carlo: " << eu_mc << " | American Monte Carlo: " << us_mc << "\n\n";
            std::cout << " Analytical price: " << analytical << "\n";
            std::cout << " American Monte Carlo price: " << us_mc << "\n";
            // std::cout << " European Monte Carlo price: " << eu_mc << "\n\n";
    }
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
        else if (a == "--threads") c.thread_count = std::stoi(need(i));
        else if (a == "--help") {
            std::cout << "Usage: ./pricer --symbol TICKER --paths N --threads N\n";
            std::cout << "if --paths and --threads are not specified, defaults are 1 million paths and 4 threads.\n";
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

    config options = parse_cmd_args(argc, argv);

    double r = fetch_risk_free_rate();          // live daily rate
    std::cout << "Risk-free rate used (DGS3MO): " << r << "\n\n";

    std::vector<optionParams> trades;
    try {
        trades = fetch_chain(options.ticker, r); 
    } catch (const std::exception& e) {
        std::cerr << "Error fetching chain data: " << e.what() << "\n";
        return 1;
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
    << "Threads used: " << options.thread_count << "\n"
    << "Throughput: " << throughput << " paths/second\n";

    

    return 0;
}