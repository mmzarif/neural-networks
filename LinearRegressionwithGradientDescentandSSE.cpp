#include <iostream>
#include <vector>
#include <cmath>

// Function to calculate the predicted value
double predict(double x, double w, double b) {
    return w * x + b;
}

// Function to calculate the summed squared error
double computeSSE(const std::vector<double>& x, const std::vector<double>& y, double w, double b) {
    double sse = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double error = y[i] - predict(x[i], w, b);
        sse += error * error;
    }
    return sse;
}

// Function to perform gradient descent
void gradientDescent(const std::vector<double>& x, const std::vector<double>& y, double& w, double& b, double learningRate, int iterations) {
    size_t n = x.size();

    for (int iter = 0; iter < iterations; ++iter) {
        double w_gradient = 0.0;
        double b_gradient = 0.0;

        // Compute gradients
        for (size_t i = 0; i < n; ++i) {
            double error = y[i] - predict(x[i], w, b);
            w_gradient += -2 * x[i] * error;
            b_gradient += -2 * error;
        }

        // Update parameters
        w -= learningRate * w_gradient / n;
        b -= learningRate * b_gradient / n;

        // Optional: Print the SSE for monitoring
        if (iter % 100 == 0) {
            std::cout << "Iteration " << iter << ": SSE = " << computeSSE(x, y, w, b) << std::endl;
        }
    }
}

int main() {
    // Example dataset: y = 2x + 1
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {3, 5, 7, 9, 11};

    // Initial guesses for w and b
    double w = 0.0;
    double b = 0.0;

    // Hyperparameters
    double learningRate = 0.01;
    int iterations = 10000;

    // Perform gradient descent
    gradientDescent(x, y, w, b, learningRate, iterations);

    // Output the final parameters
    std::cout << "Final parameters: w = " << w << ", b = " << b << std::endl;

    return 0;
}
