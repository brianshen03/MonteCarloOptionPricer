#include <nlohmann/json.hpp>
#include <curl/curl.h>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>
#include <cstdlib>               

struct optionParams {
    double S, X, T, r, sigma;
    std::string expiration_date;
};

// Helper: curl → string
static size_t write_cb(char* ptr, size_t sz, size_t nmemb, void* userdata) {
    auto* s = static_cast<std::string*>(userdata);
    s->append(ptr, sz * nmemb);
    return sz * nmemb;
}

static std::string http_get(const std::string& url, const std::string& bearer_token) {
    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("curl init failed");

    std::string buffer;
    struct curl_slist* headers = nullptr;
    if (!bearer_token.empty())
        headers = curl_slist_append(headers, ("Authorization: Bearer " + bearer_token).c_str());
    headers = curl_slist_append(headers, "Accept: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);

    CURLcode rc = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    if (rc != CURLE_OK) throw std::runtime_error("curl failed");
    return buffer;
}

// Convert "YYYY-MM-DD" to T in years
double timeToExpiration(const std::string& expiration_date) {
    std::tm tm_exp = {};
    std::istringstream ss(expiration_date);
    ss >> std::get_time(&tm_exp, "%Y-%m-%d");

    //  Force expiration to end-of-day
    // tm_exp.tm_hour = 23;
    // tm_exp.tm_min = 59;
    // tm_exp.tm_sec = 59;

    std::time_t now = std::time(nullptr);
    std::time_t exp_time = std::mktime(&tm_exp);

    double seconds = std::difftime(exp_time, now);
    return seconds / (365.0 * 24 * 60 * 60);
}

double fetch_spot_tradier(const std::string& ticker, const std::string& bearer_token) {
    std::string url = "https://api.tradier.com/v1/markets/quotes";

    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("curl init failed");

    std::string buffer;
    std::string post_fields = "symbols=" + ticker + "&greeks=true";

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, ("Authorization: Bearer " + bearer_token).c_str());
    headers = curl_slist_append(headers, "Accept: application/json");
    headers = curl_slist_append(headers, "Content-Type: application/x-www-form-urlencoded");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, post_fields.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);

    CURLcode rc = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    if (rc != CURLE_OK) throw std::runtime_error("curl failed");

    auto j = nlohmann::json::parse(buffer);
    if (!j.contains("quotes") || !j["quotes"].contains("quote")) {
        throw std::runtime_error("No quote found for " + ticker);
    }

    const auto& quote = j["quotes"]["quote"];
    if (!quote.contains("last")) {
        throw std::runtime_error("Missing 'last' field in quote.");
    }

    return quote["last"].get<double>();
}


// Main function: fetch optionParams from Tradier
std::vector<optionParams> fetch_chain(const std::string& ticker, const std::string& bearer_token, double r) {
    // Step 1: Get expiration dates
    std::string exp_url = "https://api.tradier.com/v1/markets/options/expirations?symbol=" + ticker;
    auto exp_json = nlohmann::json::parse(http_get(exp_url, bearer_token));
    auto exp_list = exp_json["expirations"]["date"];
    if (exp_list.empty()) throw std::runtime_error("No expiration dates found.");
    std::string chosen_exp;
    // Find the first expiration date that is in the future
        for (const auto& date : exp_list) {
        if (timeToExpiration(date) > 0) {
            chosen_exp = date;
            break;
        }
    }

    // Step 2: Get option chain for that expiration
    std::string chain_url = "https://api.tradier.com/v1/markets/options/chains?symbol=" + ticker + "&expiration=" + chosen_exp + "&greeks=true";
    auto chain_json = nlohmann::json::parse(http_get(chain_url, bearer_token));
    if (!chain_json.contains("options") || !chain_json["options"].contains("option"))
        throw std::runtime_error("No options returned for " + ticker);

    // Step 3: Inject spot price
    double spot = fetch_spot_tradier(ticker, bearer_token);
    std::vector<optionParams> result;

    for (const auto& opt : chain_json["options"]["option"]) {
        if (opt["option_type"] != "call") continue;

        double strike = opt["strike"].get<double>();
        double T = timeToExpiration(opt["expiration_date"].get<std::string>());
        if (T <= 0) continue; 

        double sigma = 0.0;
        if (opt.contains("greeks")) {
            const auto& g = opt["greeks"];

            /* Preferred: midpoint IV */
            if (g.contains("mid_iv") && !g["mid_iv"].is_null()) {
                sigma = g["mid_iv"].get<double>();
            }
            /* Fallbacks */
            else if (g.contains("smv_vol") && !g["smv_vol"].is_null()) {
                sigma = g["smv_vol"].get<double>();
            } else if (g.contains("bid_iv") && !g["bid_iv"].is_null()) {
                sigma = g["bid_iv"].get<double>();
            } else if (g.contains("ask_iv") && !g["ask_iv"].is_null()) {
                sigma = g["ask_iv"].get<double>();
            }

            /* Normalise only if the API ever switches to % */
            if (sigma > 1.0) sigma /= 100.0;
        }
        if (sigma <= 0.0) continue;


        result.push_back(optionParams {
            .S = spot,
            .X = strike,
            .T = T,
            .r = r,
            .sigma = sigma,
            .expiration_date = opt["expiration_date"].get<std::string>()
        });
    }

    return result;
}

/* ───────── LIVE risk-free rate from FRED (DGS3MO) ───────── */
double fetch_risk_free_rate()
{
    const char* key = std::getenv("FRED_KEY");
    if (!key)
        throw std::runtime_error("FRED_KEY not set");

    std::string url =
        "https://api.stlouisfed.org/fred/series/observations?"
        "series_id=DGS3MO&file_type=json&sort_order=desc&limit=1&api_key=" +
        std::string(key);

    auto j = nlohmann::json::parse(http_get(url, ""));            

    double pct = std::stod(j["observations"][0]["value"].get<std::string>());
    return pct / 100.0;   // 5.19 → 0.0519
}