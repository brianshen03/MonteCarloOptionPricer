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

#define TIME_STEPS 20 // number of time steps for American option pricing
struct config {
    std::string ticker;
    int thread_count = 4; // default to 4 threads
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

// Monte Carlo simulation to estimate the european option price
optionPrices eu_mc_sim(const optionParams& params, const config& config) {

    double call_sum = 0.0;
    double put_sum = 0.0;
    #pragma omp parallel num_threads(config.thread_count)
    {
        // each thread gets its own RNG & distribution
        std::mt19937_64 rng(42 + omp_get_thread_num());
        std::normal_distribution<double> dist(0.0, 1.0);

        double local_call_sum = 0.0;
        double local_put_sum = 0.0;

        #pragma omp for
        for (int i = 0; i < config.num_simulations; ++i) {
            double Z  = dist(rng);
            double drift = (params.r - 0.5 * params.sigma * params.sigma) * params.T;
            double diffusion = params.sigma * std::sqrt(params.T);
            double ST = params.S * std::exp(drift + diffusion * Z);
            local_call_sum += std::max(ST - params.X, 0.0);
            local_put_sum += std::max(params.X - ST, 0.0);
        }
        // reduction by hand (atomic or critical) or use OpenMP reduction on local_sum
        #pragma omp atomic
        call_sum += local_call_sum;
        #pragma omp atomic
        put_sum += local_put_sum;
    }

    double call_price = std::exp(-params.r * params.T) * (call_sum / config.num_simulations);
    double put_price = std::exp(-params.r * params.T) * (put_sum / config.num_simulations);
    optionPrices price = {call_price, put_price};
    return price;
}

//helper func to perform polynomial regression
Eigen::VectorXd perform_regression(const std::vector<double>& X, const std::vector<double>& Y) {
    std::vector<double> X_sub = X, Y_sub = Y;
    const int MAX_POINTS = 20000;

    if (X.size() > MAX_POINTS) {
        X_sub.clear();
        Y_sub.clear();
        std::sample(X.begin(), X.end(), std::back_inserter(X_sub), MAX_POINTS, std::mt19937{std::random_device{}()});
        std::sample(Y.begin(), Y.end(), std::back_inserter(Y_sub), MAX_POINTS, std::mt19937{std::random_device{}()});
    }

    int N = X_sub.size();
    Eigen::MatrixXd A(N, 3);
    Eigen::VectorXd b(N);

    // #pragma omp parallel for
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
optionPrices us_mc_sim(const optionParams& params, const config& config, int M) {

    //2d array to store stock prices at each time step
    std::vector<std::vector<double>> all_paths(config.num_simulations, std::vector<double>(M+1));

    double dt = params.T / M; // time step size
    double drift = (params.r - 0.5 * params.sigma * params.sigma) * dt;

    auto t0 = std::chrono::steady_clock::now();

    // STEP 1: generate all paths
    #pragma omp parallel for num_threads(config.thread_count)
    for (int i = 0; i < config.num_simulations; ++i) {
        std::mt19937_64 rng(42+i);
        std::normal_distribution<double> dist(0.0, 1.0);

        all_paths[i][0] = params.S;
        //saving each stock price from 1 to M time paths 
        for (int t = 1; t <= M; ++t) {
            double Z = dist(rng);
            double diffusion = params.sigma * std::sqrt(dt) * Z;
            all_paths[i][t] = all_paths[i][t-1] * std::exp(drift + diffusion);
        }
    }

    auto t1 = std::chrono::steady_clock::now();

    // STEP 2: calculate cash_flow (payoff) at maturity
    std::vector<double> cash_flow_call(config.num_simulations);
    std::vector<double> cash_flow_put(config.num_simulations);
    for (int i = 0; i < config.num_simulations; ++i) {
        double ST = all_paths[i][M];
        cash_flow_call[i] = std::max(ST - params.X, 0.0); // final call payoff
        cash_flow_put[i] = std::max(params.X - ST, 0.0); // put option payoff
    }

    auto t2 = std::chrono::steady_clock::now();

    // STEP 3: backward induction to calculate option price
    std::vector<int> itm_call;
    std::vector<int> itm_put;
    std::vector<double> X_call, Y_call, X_put, Y_put;

    long long total_regression_time_call_ms = 0;
    long long total_regression_time_put_ms = 0;
    const double disc = std::exp(-params.r * dt);

    //  Eigen::setNbThreads(1); 
    // std::cout << "Eigen threads: " << Eigen::nbThreads() << "\n";

    for (int t = M - 1; t >= 1; --t) {

        itm_call.clear(); itm_put.clear();
        X_call.clear();   Y_call.clear();
        X_put.clear();    Y_put.clear();

            // find in-the-money paths at time t & build regression input
            for (int i = 0; i < config.num_simulations; ++i) {

                double ST = all_paths[i][t];
                if (ST > params.X) { 
                    itm_call.push_back(i); //in the money paths for call 
                    X_call.push_back(ST);
                    Y_call.push_back(cash_flow_call[i] * disc); // discounted cash flow
                }

                if (params.X > ST) {
                    itm_put.push_back(i); // in the money paths for put
                    X_put.push_back(ST);
                    Y_put.push_back(cash_flow_put[i] * disc); // discounted cash flow
                }
            }

        // need at least 3 points for regression
        if (X_call.size() < 3) continue;

        // perform regression to estimate continuation value
        auto r1 = std::chrono::high_resolution_clock::now();
        Eigen::VectorXd beta_call = perform_regression(X_call, Y_call);
        auto r2 = std::chrono::high_resolution_clock::now();

        long long call_duration = std::chrono::duration_cast<std::chrono::milliseconds>(r2 - r1).count();
        total_regression_time_call_ms += call_duration;

        //compare continuation value with immediate payoff
        #pragma omp parallel for num_threads(config.thread_count)
        for (int j = 0; j < itm_call.size(); ++j) {
            int i = itm_call[j];
            double ST = all_paths[i][t];
            double continuation_value = beta_call[0] + beta_call[1] * ST + beta_call[2] * ST * ST;
            double immediate_payoff = std::max(ST - params.X, 0.0);
            if (immediate_payoff > continuation_value)
                cash_flow_call[i] = immediate_payoff; // exercise
            else
                cash_flow_call[i] = disc * cash_flow_call[i]; // continue holding
        }

        if (X_put.size() < 3) continue;
        // perform regression for put option
        auto r3 = std::chrono::high_resolution_clock::now();
        Eigen::VectorXd beta_put = perform_regression(X_put, Y_put);
        auto r4 = std::chrono::high_resolution_clock::now();

        long long put_duration = std::chrono::duration_cast<std::chrono::milliseconds>(r4 - r3).count();
        total_regression_time_put_ms += put_duration;

        #pragma omp parallel for num_threads(config.thread_count)
        for (int j = 0; j < itm_put.size(); ++j) {
            int i = itm_put[j];
            double ST = all_paths[i][t];
            double continuation_value = beta_put[0] + beta_put[1] * ST + beta_put[2] * ST * ST;
            double immediate_payoff = std::max(params.X - ST, 0.0);
            if (immediate_payoff > continuation_value)
                cash_flow_put[i] = immediate_payoff; // exercise
            else
                cash_flow_put[i] = disc * cash_flow_put[i]; // continue holding
        }
    }

    auto t3 = std::chrono::steady_clock::now();

    // STEP 4: average discounted cash flows
    double call_sum = 0.0, put_sum = 0.0;
    for (int i = 0; i < config.num_simulations; ++i) {
        call_sum += cash_flow_call[i];
        put_sum += cash_flow_put[i];
    }
    double call_price = call_sum / config.num_simulations;
    double put_price = put_sum / config.num_simulations;

    auto t4 = std::chrono::steady_clock::now();
    // std::cout << "Total regression time for call: " << total_regression_time_call_ms << " ms\n";
    // std::cout << "Total regression time for put: " << total_regression_time_put_ms << " ms\n\n";

    // std::cout << "Time taken for each step:\n\n"
    //           << "Path generation: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms\n"
    //           << "Cash flow calculation: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms\n"
    //           << "Backward induction: " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << " ms\n"
    //           << "Final averaging: " << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << " ms\n\n";

    optionPrices price = {call_price, put_price};
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

static void run_eu_pricer(const std::vector<optionParams>& trades, const config& options) {

    //for normal sized option prices ($5, $10), relative tolerance of 2% is reasonable
    //for small option prices ($0.05, $0.10), absolute tolerance is within 2 cents of black schole price

    const double rel_tol = 0.02;   
    const double eps     = 1e-12;  // guard for zero BS
    const double abs_tol_call = 0.02;
    const double abs_tol_put  = 0.02;

    int pass_call = 0;
    int pass_put  = 0;

    for (size_t i = 0; i < trades.size(); ++i) {
        const auto& opt = trades[i];
        std::cout << "Option " << i+1 << ": " << "S: " << opt.S << ", X: " << opt.X << ", Expiration date: " << 
        opt.expiration_date << ", T: " << opt.T << ", r: " << opt.r << ", sigma: " << opt.sigma << "\n\n";

        optionPrices eu_bs = calc_call(opt);
        optionPrices eu_mc = eu_mc_sim(opt, options);


        std::cout << " BS call price: " << eu_bs.call_price << " | European Monte Carlo call price: " << eu_mc.call_price;
        double abs_err_call = std::abs(eu_bs.call_price - eu_mc.call_price);
        double rel_err_call = abs_err_call / std::max(std::abs(eu_bs.call_price), eps);

        if (rel_err_call <= rel_tol || abs_err_call <= abs_tol_call) {
            ++pass_call;
            std::cout << "  ✅ within tol\n";
        } else {
            std::cout << "  ❌ outside tol\n";
        }

        std::cout << " BS put price: " << eu_bs.put_price << " | European Monte Carlo put price: " << eu_mc.put_price;
        double abs_err_put  = std::abs(eu_bs.put_price  - eu_mc.put_price);
        double rel_err_put  = abs_err_put  / std::max(std::abs(eu_bs.put_price),  eps);

        if (rel_err_put <= rel_tol || abs_err_put <= abs_tol_put) {
            ++pass_put;
            std::cout << "  ✅ within tol\n\n";
        } else {
            std::cout << "  ❌ outside tol\n\n";
        }

        std::cout << "----------------------------------------\n";
    }

    std::cout << pass_call << " call prices passed out of " << trades.size() << "\n";
    std::cout << pass_put  << " put prices passed out of " << trades.size() << "\n";
}


static void run_us_pricer(const std::vector<optionParams>&trades, const config& options) {

        for (size_t i = 0; i < trades.size(); ++i) {
            const auto& opt = trades[i];
            std::cout << "Option " << i+1 << ": " << "S: " << opt.S << ", X: " << opt.X << ", Expiration date: " << 
            opt.expiration_date << ", T: " << opt.T << ", r: " << opt.r << ", sigma: " << opt.sigma << "\n\n";

            optionPrices eu_bs = calc_call(opt);
            optionPrices us_mc = us_mc_sim(opt, options, TIME_STEPS);

            std::cout << " Analytical call price: " << eu_bs.call_price << "\n";
            std::cout << " Analytical put price: " << eu_bs.put_price << "\n";


            std::cout << " American Monte Carlo call price: " << us_mc.call_price << "\n";
            std::cout << "American Monte Carlo put price: " << us_mc.put_price << "\n";

            std::cout << "----------------------------------------\n";
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
    double r; // live daily rate

    try {
        r = fetch_risk_free_rate();         
    } catch (const std::exception& e) {
        std::cerr << "Error fetching risk-free rate: " << e.what() << "\n";
        return 1;
    }
    std::cout << "Risk-free rate used (DGS3MO): " << r << "\n\n";

    std::vector<optionParams> trades;
    try {
        trades = fetch_chain(options.ticker, r); 
    } catch (const std::exception& e) {
        std::cerr << "Error fetching chain data: " << e.what() << "\n";
        return 1;
    }

    auto start = std::chrono::steady_clock::now();
    run_eu_pricer(trades, options);
    // run_us_pricer(trades, options);
    auto stop = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = stop - start;   


    double seconds = elapsed.count();          
    double throughput = options.num_simulations / seconds;          




    std::cout << "Simulation Summary\n"
     << "------------------\n" 
     << "Contracts processed: " << trades.size() << " options in " << seconds << " seconds \n"
    << "Total paths: " << options.num_simulations << "\n"
    << "Threads used: " << options.thread_count << "\n"
    << "Throughput: " << throughput << " paths/second\n";

    

    return 0;
}