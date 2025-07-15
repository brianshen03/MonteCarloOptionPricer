High Performance Computing European option pricer 

This program takes in real time data to calculate the option price of a stock. Paramters include S(stock price), X(strike price), T(time to expiration of the option), r(risk-free rate), and sigma (volatility of the stock) 

Data: 
S, X, T and sigma are taken live as soon as you run from the program from Tradier API.
r is pulled live from the current 3 month treasury bill rate from FRED API.
The option price is only presumably accurate for a T that is less than 3 months since r is pulled from the 3 month treasurey bill.
If t happened to be 6 months or 1 year, r would change as the best risk-free-rate would accordingly change depending on the amount of time you have.

Usage:
./pricer TICKER THREADS NUM_SIMULATIONS

Ticker is the symbol for the stock. e.x. (AAPL)

THREADS is an integer representing how many threads you want to run the program. Currently, the threads are parallelizing the number of simulations, and not optimized for a large amount of options. For example, for a stock with 5000 options but only 100 simulations versus a stock with 20 options and 10M simulations, the latter would be more optimized. 

NUM_SIMULATIONS is an integer representing the amount of simulations/paths the stock can take. In my monte carlo simulation, I take the average of the payoffs over number of simulations. 


Future Plans:
I plan on coding the program with CUDA to use GPU acceleration and comparison with OpenMP (CPU acceleration)
Right now, this is based on a European call option pricer, but I plan on expanding to different options such as American and Asian.


Links for the API I am using.
https://documentation.tradier.com/brokerage-api/markets/get-options-chains
https://documentation.tradier.com/brokerage-api/markets/get-quotes
https://fred.stlouisfed.org/docs/api/fred/



