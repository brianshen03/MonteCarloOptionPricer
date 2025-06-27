#include "live_data.hpp"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <iostream>

// Write callback for libcurl
static size_t write_cb(char* ptr, size_t sz, size_t nmemb, void* userdata) {
    auto* s = static_cast<std::string*>(userdata);
    s->append(ptr, sz * nmemb);
    return sz * nmemb;
}

// Perform HTTP GET request and return response body
static std::string http_get(const std::string& url) {
    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("curl init failed");

    std::string buffer;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_USERAGENT,
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36");
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);

    CURLcode rc = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(curl);

    if (rc != CURLE_OK)
        throw std::runtime_error("curl failed: " + std::string(curl_easy_strerror(rc)));
    if (http_code != 200)
        throw std::runtime_error("HTTP status " + std::to_string(http_code));

    return buffer;
}

// Get the current stock price from AlphaVantage
double liveSpot(const std::string& ticker) {
    const char* key = std::getenv("ALPHAADVANTAGE_API_KEY");
    if (!key) throw std::runtime_error("Missing ALPHAADVANTAGE_API_KEY env var");

    std::string url =
        "https://www.alphavantage.co/query?function=GLOBAL_QUOTE"
        "&symbol=" + ticker +
        "&apikey=" + key;

    auto body = http_get(url);
    auto j = nlohmann::json::parse(body);
    auto quote = j["Global Quote"];
    if (quote.empty())
        throw std::runtime_error("No quote data for “" + ticker + "”");

    return std::stod(quote["05. price"].get<std::string>());
}
