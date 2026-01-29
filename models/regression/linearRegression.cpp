#include <iostream>
#include <vector>
#include <cmath>
using std::vector;

class LinearRegression {
public:
    // LinearRegression() : {}
    void fit();
    void summary();
    void predict();
private:
     

};

void LinearRegression::fit() {
    std::cout << "Model fitted" << std::endl;

}

void LinearRegression::summary() {

}

void LinearRegression::predict() {

}

int main(void) {

    LinearRegression slr = LinearRegression();
    slr.fit();

    // Data 
    vector<int> X = {2,4,5,7,9,11,12,15,18,22};
    vector<int> y = {1,2,3,4,5,6,7,8,9,10};

    // Compute means from the sample data 
    double ybar = 0; 
    double xbar = 0;
    size_t n = y.size();
    for (auto b=y.begin(); b != y.end(); ++b)
        ybar += *b;
    ybar = ybar / n;
    for (auto b=X.begin(); b != X.end(); ++b)
        xbar += *b;
    xbar = xbar / n;

    // Find the sample std
    double resid_sum_sqr = 0;
    for (auto b=y.begin(); b != y.end(); ++b)
        resid_sum_sqr += (*b - ybar) * (*b - ybar);

    double sample_var = resid_sum_sqr / (n-1);
    double sample_std = std::sqrt(sample_var);

    // Calculate Beta1 

    double numerator = 0;
    double denominator = 0;
    for (auto yb=y.begin(), xb=X.begin(); yb != y.end(); yb++, xb++) {
        numerator += (*xb - xbar)*(*yb - ybar);
        denominator += (*xb - xbar) * (*xb - xbar);
    }

    double beta_1 = numerator / denominator;
    double beta_0 = ybar - beta_1*xbar;

    std::cout << "Equation: E(y) = " << beta_0 << 
        ((beta_1 > 0.0) ? " + " : " ") << beta_1 << "*X" << std::endl;

    return 0;
}
