#pragma once
#include <string>
#include <vector>

struct optionParams {
    double S, X, T, r, sigma;
    std::string expiration_date;
};

double fetch_spot_tradier(const std::string& ticker, const std::string& bearer_token);
std::vector<optionParams> fetch_chain(const std::string& ticker, const std::string& bearer_token, double r);
