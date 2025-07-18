#include "httplib.h"
#include "nlohmann/json.hpp"
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

struct optionParams {
    double S, X, T, r, sigma;
    std::string expiration_date;
};

// HTTP GET request to fetch JSON data
json http_get_json(const std::string& path, const std::string& bearer) {
    httplib::SSLClient cli("api.tradier.com", 443);

    httplib::Headers headers = {
        { "Authorization", "Bearer " + bearer },
        { "Accept", "application/json" },
        { "User-Agent", "pricer-cpp/1.0" }
    };

    auto res = cli.Get(path.c_str(), headers);

    if (!res || res->status != 200) {
        throw std::runtime_error("HTTP error: " + std::to_string(res ? res->status : 0));
    }

    return json::parse(res->body);
}

//convert a date string in YYYY-MM-DD format to seconds until expiration
 double time_to_expiration(const std::string& ymd) {
    std::tm tm{};
    std::istringstream(ymd) >> std::get_time(&tm, "%Y-%m-%d");
    std::time_t exp = std::mktime(&tm);
    std::time_t now = std::time(nullptr);
    return std::difftime(exp, now) / (365.0 * 24 * 60 * 60);
}

/*───────────────────────────── Tradier helpers ───────────────────────────*/

//fetches live spot price for a ticker
double fetch_spot_tradier(const std::string& ticker, const std::string& bearer) {
    std::string url = "/v1/markets/quotes?symbols=" + ticker + "&greeks=true";
    auto j = http_get_json(url, bearer);
    return j["quotes"]["quote"]["last"].get<double>();
}

//fetches option chain for a ticker 
std::vector<optionParams> fetch_chain(const std::string& ticker, double r) {

    std::cout << "Fetching option chain for " << ticker << "...\n";
    
    const char* token_env = std::getenv("TRADIER_API_KEY");
    if (!token_env || std::string(token_env).empty())
        throw std::runtime_error("TRADIER_API_KEY is not set");
    std::string bearer = token_env;

    auto exp_json = http_get_json("/v1/markets/options/expirations?symbol=" + ticker, bearer);
    std::string exp_date;

    //pickest nearest expiration date in the future
    for (const auto& d : exp_json["expirations"]["date"]) {
        if (time_to_expiration(d) > 0) { exp_date = d; break; }
    }
    if (exp_date.empty()) throw std::runtime_error("no future expirations for " + ticker);


    auto chain = http_get_json("/v1/markets/options/chains?symbol=" + ticker + "&expiration=" + exp_date + "&greeks=true", bearer);

    double S = fetch_spot_tradier(ticker, bearer);
    std::vector<optionParams> out;

    for (const auto& o : chain["options"]["option"]) {
        //only call options
        if (o["option_type"] != "call") continue;
        double sigma = 0.0;

        if (o.contains("greeks")) {
            const auto& g = o["greeks"];
            for (const char* field : {"mid_iv", "smv_vol", "bid_iv", "ask_iv"}) {
                if (g.contains(field) && !g[field].is_null()) { 
                    sigma = g[field].get<double>(); 
                    std::cout << "[DEBUG] Using " << field << " = " << sigma << " for " << o["symbol"] << '\n';
                    break; 
                }
            }
        }
        if (sigma <= 0.0) continue;
        double X = o["strike"].get<double>();
        double T = time_to_expiration(o["expiration_date"].get<std::string>());
        if (T <= 0) continue;

        out.push_back({S, X, T, r, sigma, o["expiration_date"].get<std::string>()});
    }
    return out;
}

/*───────────────────────────── FRED helpers ───────────────────────────*/

 double fetch_risk_free_rate() {
    const char* key = std::getenv("FRED_KEY");
    if (!key) throw std::runtime_error("FRED_KEY not set");

    httplib::Client cli("https://api.stlouisfed.org");
    std::string url = "/fred/series/observations?series_id=DGS3MO&file_type=json&sort_order=desc&limit=1&api_key=" + std::string(key);
    auto res = cli.Get(url.c_str());
    if (!res || res->status != 200)
        throw std::runtime_error("HTTP error: FRED " + (res ? std::to_string(res->status) : "no response"));

    auto j = json::parse(res->body);
    return std::stod(j["observations"][0]["value"].get<std::string>()) / 100.0;
}