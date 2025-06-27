CXX      := /opt/homebrew/bin/g++-15
CXXFLAGS := -std=c++17 -fopenmp \
             -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
			            -I/opt/homebrew/include
LDFLAGS  := -lcurl

SRC      := monte_carlo.cpp live_data.cpp
OBJ      := $(SRC:.cpp=.o)
TARGET   := pricer

.PHONY: all clean

all: $(TARGET)

# Link step
$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile step
%.o: %.cpp live_data.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
