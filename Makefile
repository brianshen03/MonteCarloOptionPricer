CXX      := /opt/homebrew/bin/g++-15
OPENSSL  := $(shell brew --prefix openssl@3)

CXXFLAGS := -std=c++20 -fopenmp \
            -DCPPHTTPLIB_OPENSSL_SUPPORT \
            -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk \
            -I/opt/homebrew/include \
            -I$(OPENSSL)/include \
            -Iinclude \
            -Iinclude/eigen-3.4.0

LDFLAGS  := -L$(OPENSSL)/lib -lssl -lcrypto

SRC      := monte_carlo.cpp live_data.cpp
OBJ      := $(SRC:.cpp=.o)
TARGET   := pricer

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
