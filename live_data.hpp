#pragma once
#include <string>
#include <vector>

struct optionParams {
    double S, X, T, r, sigma;
    std::string expiration_date;
};

double fetch_risk_free_rate();       
double fetch_spot_tradier(const std::string& ticker, const std::string& bearer);
std::vector<optionParams> fetch_chain(const std::string& ticker, double r);
std::vector<optionParams> fetch_chain( const std::string& ticker, double r, int min_dte_days ,  int max_dte_days , int max_expiries , bool include_calls ,bool include_puts);
