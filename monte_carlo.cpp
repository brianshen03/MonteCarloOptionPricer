#include <iostream>
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
double S;
double X;
double T;
double r;
double sigma;
double d1, d2;
double CDF_d1, CDF_d2;
double option_price;

#define THREADS 4
//10 million simulations
#define NUM_SIMULATIONS 100000000

//helper function to calculate CDF
double phi(double x) {
    return 0.5 * std::erfc(-x/std::sqrt(2.0));
}

double calc_option_price(double S, double X, double T, double r, double sigma) {



    d1 = (std::log(S/X) + (r + (sigma*sigma)/2) * T)/
                (sigma*std::sqrt(T));

    d2 = (std::log(S/X) + (r - (sigma*sigma)/2) * T)/
             (sigma*std::sqrt(T));

             
    CDF_d1 = phi(d1);
    CDF_d2 = phi(d2);
    
    option_price = S * CDF_d1 - X * std::exp(-r * T) * CDF_d2;

    return option_price;

}

double monte_carlo_simulation(double S, double X, double T, double r, double sigma, int num_simulations) {

    std::mt19937_64 rng(42 + omp_get_thread_num());
    std::normal_distribution<double> distribution(0.0, 1.0);
    double sum_payoffs = 0.0;
 
    #pragma omp parallel for num_threads(THREADS) reduction(+:sum_payoffs)
    for (int i = 0; i < num_simulations; ++i) {
        double Z = distribution(rng);
        //stock price at expiration 
        double ST = S * std::exp((r - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * Z);
        //if strike price is greater than stock price at expiration, then payoff is zero
        //otherwise, payoff is stock price at expiration minus strike price
        double payoff = std::max(ST - X, 0.0); 
        sum_payoffs += payoff;
    }

    // average of payoffs over number of simulations , then discounting back to present value
    return std::exp(-r * T) * (sum_payoffs / num_simulations);
}

int main() {

    // Initialize stock option parameters
    S = 125.0;     
    X = 100.0;     
    T = 3.0;       
    r = 0.05;  
    sigma = 0.2;    


    std::cout << "Parameters:" << std::endl;
    std::cout << "Stock Price (S): " << S << std::endl;
    std::cout << "Strike Price (X): " << X << std::endl;
    std::cout << "Time to Expiration (T): " << T << " years" << std::endl;
    std::cout << "Risk-Free Interest Rate (r): " << r * 100 << "%" << std::endl;
    std::cout << "Volatility (sigma): " << sigma * 100 << "%"  << std::endl;
    std::cout << "The Black schole option price is: " << calc_option_price(S, X, T, r, sigma) << "\n\n";

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Monte Carlo Simulation Price: " << monte_carlo_simulation(S, X, T, r, sigma, NUM_SIMULATIONS) << std::endl;
    auto end = std::chrono::high_resolution_clock::now();

    double secs = std::chrono::duration<double>(end - start).count();
    std::cout << "Time taken for Monte Carlo Simulation: " << secs << " seconds with " << THREADS << " threads and "
              << NUM_SIMULATIONS << " simulations" << std::endl;


    return 0;
}