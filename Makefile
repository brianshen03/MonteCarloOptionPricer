# ---------- user paths ----------
OPENSSL_INC = C:\Users\Suraj\vcpkg\installed\x64-windows\include
OPENSSL_LIB = C:\Users\Suraj\vcpkg\installed\x64-windows\lib

# ---------- CUDA settings ----------
CUDA_ARCH   = sm_89          # RTX 4070 / Ada
NVCC        = nvcc
NVCCFLAGS = -std=c++20 -O2 -arch=$(CUDA_ARCH) -Iinclude -I"$(OPENSSL_INC)" -DCPPHTTPLIB_OPENSSL_SUPPORT -Xcompiler "/EHsc"


# ---------- libraries (one line for nmake) ----------
LDFLAGS = "$(OPENSSL_LIB)\libssl.lib" "$(OPENSSL_LIB)\libcrypto.lib" -lcurand

# ---------- sources / objects ----------
SRC    = monte_carlo.cu live_data.cpp
OBJ    = monte_carlo.obj live_data.obj
TARGET = pricer.exe

# ---------- default target ----------
all: $(TARGET)

# ---------- link step ----------
$(TARGET): $(OBJ)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(OBJ) $(LDFLAGS)

# ---------- compile rules ----------
monte_carlo.obj: monte_carlo.cu live_data.hpp
	$(NVCC) $(NVCCFLAGS) -c monte_carlo.cu -o monte_carlo.obj

live_data.obj: live_data.cpp live_data.hpp
	$(NVCC) $(NVCCFLAGS) -c live_data.cpp -o live_data.obj

clean:
	del /Q *.obj *.exe
