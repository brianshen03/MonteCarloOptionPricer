CXX = /opt/homebrew/bin/g++-15
CXXFLAGS = -fopenmp -std=c++17 -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk

all: monte_carlo

monte_carlo: monte_carlo.cpp
	$(CXX) $(CXXFLAGS) -o pricer monte_carlo.cpp
