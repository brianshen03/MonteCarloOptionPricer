#include <Eigen/Dense>
#include <iostream>
#include <chrono>

int main() {
    using namespace std::chrono;
    Eigen::setNbThreads(1);  // or 8

    int rows = 200000;
    int cols = 10;

    Eigen::MatrixXd A(rows, cols);
    Eigen::VectorXd b(rows);

    A.setRandom();
    b.setRandom();

    auto t1 = high_resolution_clock::now();
    Eigen::VectorXd beta = (A.transpose() * A).ldlt().solve(A.transpose() * b);
    auto t2 = high_resolution_clock::now();

    std::cout << "Time: " << duration_cast<milliseconds>(t2 - t1).count() << " ms\n";
    std::cout << "Threads: " << Eigen::nbThreads() << "\n";
}

// g++-15 microbenchmark.cpp -O3 -march=native -fopenmp -Iinclude/eigen-3.4.0 -o eigen_benchmark


