This is my personal project parallelizing a Monte Carlo Simulation with OpenMP and CUDA!

Right now, this is based on a European call option pricer, but I plan on expanding to barrier and asian options.

This is the API I am using: 
https://documentation.tradier.com/brokerage-api/reference/response/quotes
https://documentation.tradier.com/brokerage-api/markets/post-quotes
https://fred.stlouisfed.org/docs/api/fred/



talk about how risk free rate is 3 month treasury 
not super accurate for t <= 3 months

elaborate on what you are pulling from each API is pulling 